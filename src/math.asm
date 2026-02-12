; ==========================================================================
; Fixed-Point Math Library (Q8.8) for SNES GPT
; ==========================================================================
; All arithmetic for the transformer forward pass.
; Uses the SNES PPU hardware 8x8 unsigned multiplier at $4202/$4203/$4216.

.p816
.smart

.include "snes.inc"

.segment "ZEROPAGE"
.importzp dp_tmp0, dp_tmp1, dp_tmp2, dp_tmp3, dp_tmp4
.importzp dp_acc, dp_acc_hi

.segment "CODE"

.export FP_Mul, FP_Div, FP_Exp, FP_InvSqrt
.import ExpTable, InvSqrtTable

; --- Constants ---
FP_ONE   = $0100
FP_ZERO  = $0000

; ==========================================================================
; FP_Mul — Signed Q8.8 multiply
;   Input:  A = multiplicand (Q8.8), dp_tmp0 = multiplier (Q8.8)
;   Output: A = product (Q8.8)
;   Clobbers: dp_tmp1, dp_tmp2, dp_acc, dp_acc_hi
; ==========================================================================

.proc FP_Mul
    .a16
    .i16

    sta dp_tmp1             ; save multiplicand

    ; Determine result sign
    eor dp_tmp0
    and #$8000
    sta dp_tmp2             ; sign flag

    ; Absolute value of A
    lda dp_tmp1
    bpl @a_pos
    eor #$FFFF
    inc a
    sta dp_tmp1
@a_pos:

    ; Absolute value of B
    lda dp_tmp0
    bpl @b_pos
    eor #$FFFF
    inc a
    sta dp_tmp0
@b_pos:

    ; Clear 32-bit local accumulator (dp_tmp3:dp_tmp4, NOT dp_acc!)
    ; Using dp_tmp3/dp_tmp4 so callers' dp_acc/dp_acc_hi are preserved
    stz dp_tmp3
    stz dp_tmp4

    ; Switch to 8-bit accumulator for hardware multiply
    sep #$20
    .a8

    ; --- Partial product 1: AL * BL ---
    lda dp_tmp1             ; AL (low byte)
    sta WRMPYA              ; $4202
    lda dp_tmp0             ; BL (low byte)
    sta WRMPYB              ; $4203 — triggers multiply
    nop                     ; wait 8 cycles for result
    nop
    nop
    nop
    rep #$20
    .a16
    lda $4216               ; AL*BL (16-bit result)
    sta dp_tmp3             ; store as bytes 0-1

    ; --- Partial product 2: AH * BL ---
    sep #$20
    .a8
    lda dp_tmp1+1           ; AH (high byte)
    sta WRMPYA
    lda dp_tmp0             ; BL
    sta WRMPYB
    nop
    nop
    nop
    nop
    rep #$20
    .a16
    lda $4216               ; AH*BL
    ; Add shifted left 8: add to bytes 1-2 of result
    clc
    adc dp_tmp3+1
    sta dp_tmp3+1
    lda #0
    adc dp_tmp4
    sta dp_tmp4

    ; --- Partial product 3: AL * BH ---
    sep #$20
    .a8
    lda dp_tmp1             ; AL
    sta WRMPYA
    lda dp_tmp0+1           ; BH
    sta WRMPYB
    nop
    nop
    nop
    nop
    rep #$20
    .a16
    lda $4216               ; AL*BH
    clc
    adc dp_tmp3+1           ; add to bytes 1-2
    sta dp_tmp3+1
    lda #0
    adc dp_tmp4
    sta dp_tmp4

    ; --- Partial product 4: AH * BH ---
    ; This contributes to bytes 2-3 (overflow detection mostly)
    sep #$20
    .a8
    lda dp_tmp1+1           ; AH
    sta WRMPYA
    lda dp_tmp0+1           ; BH
    sta WRMPYB
    nop
    nop
    nop
    nop
    rep #$20
    .a16
    lda $4216               ; AH*BH
    clc
    adc dp_tmp4             ; add to bytes 2-3
    sta dp_tmp4

    ; Q8.8 result = bytes 1-2 of 32-bit product (>> 8)
    lda dp_tmp3+1

    ; Apply sign
    ldx dp_tmp2
    beq @done
    eor #$FFFF
    inc a
@done:
    rts
.endproc

; ==========================================================================
; FP_Div — Q8.8 signed division (A / dp_tmp0)
;   Input:  A = dividend (Q8.8), dp_tmp0 = divisor (Q8.8)
;   Output: A = quotient (Q8.8)
; ==========================================================================

.proc FP_Div
    .a16
    .i16

    sta dp_tmp1             ; save dividend

    ; Determine result sign
    eor dp_tmp0
    and #$8000
    sta dp_tmp2             ; sign flag

    ; Absolute value of dividend
    lda dp_tmp1
    bpl @d_pos
    eor #$FFFF
    inc a
    sta dp_tmp1
@d_pos:

    ; Absolute value of divisor
    lda dp_tmp0
    bpl @v_pos
    eor #$FFFF
    inc a
    sta dp_tmp0
@v_pos:

    ; Shift dividend left 8 for Q8.8 precision
    ; 24-bit dividend: dp_acc_hi:dp_acc = dp_tmp1 << 8
    lda dp_tmp1
    xba                     ; swap bytes = effectively << 8 for low, >> 8 for high
    and #$FF00
    sta dp_acc              ; low word (only high byte set)
    lda dp_tmp1
    xba
    and #$00FF
    sta dp_acc_hi           ; high word (low byte from original high byte)

    ; 16-iteration restoring division
    ldx #16
    stz dp_tmp3             ; quotient = 0

@div_loop:
    ; Shift dividend left 1 bit (24-bit shift)
    asl dp_acc
    rol dp_acc_hi

    ; Try subtract divisor from high word
    lda dp_acc_hi
    sec
    sbc dp_tmp0
    bcc @no_sub             ; borrow = divisor > remainder

    ; Commit subtraction
    sta dp_acc_hi

    ; Set quotient bit
    lda dp_tmp3
    ora #1
    sta dp_tmp3
    bra @shift_q

@no_sub:
    lda dp_tmp3

@shift_q:
    ; Shift quotient left for next iteration (except last)
    dex
    beq @div_done_loop
    asl dp_tmp3
    bra @div_loop

@div_done_loop:
    lda dp_tmp3             ; quotient

    ; Apply sign
    ldx dp_tmp2
    beq @div_done
    eor #$FFFF
    inc a
@div_done:
    rts
.endproc

; ==========================================================================
; FP_Exp — Q8.8 exp(x) via lookup table
;   Input:  A = x (Q8.8 signed)
;   Output: A = exp(x) (Q8.8, always positive)
;   Table covers [-4.0, +4.0) with 256 entries.
; ==========================================================================

.proc FP_Exp
    .a16
    .i16

    ; Clamp: x >= 4.0 => saturate
    cmp #$0400
    bcc @check_low
    ; Check if positive (signed compare)
    bit #$8000
    bne @check_low          ; negative values fall through
    lda #$7FFF              ; saturate to max Q8.8
    rts

@check_low:
    ; Clamp: x < -4.0 => return 0
    ; Signed comparison: if x < $FC00 (-4.0)
    cmp #$FC00
    bcs @in_range           ; if x >= -4.0 (unsigned, but works for neg range)
    bit #$8000
    beq @in_range           ; positive values are in range
    lda #FP_ZERO
    rts

@in_range:
    ; Index = ((x + 0x0400) >> 3) & 0xFF
    ; This maps [-4.0, +4.0) to [0, 255]
    clc
    adc #$0400              ; x + 4.0, now in [0, 8.0)
    ; Divide by 8.0/256 = 1/32: shift right 5... no wait
    ; We have 8.0 range mapped to 256 entries
    ; Each entry covers 8.0/256 = 0.03125
    ; x_shifted is in Q8.8, range [0, $0800)
    ; Index = x_shifted / (8.0/256) = x_shifted * 256/8 = x_shifted * 32
    ; But x_shifted is Q8.8, so in integer form it's x*256
    ; Index = (x_shifted_q8) / (8*256/256) = x_shifted_q8 / 8
    ; Actually: x_shifted ranges from 0 to $07FF (0.0 to 7.996)
    ; We want index 0..255, so index = x_shifted >> 3 (divide by 8, but x is Q8.8)
    ; Wait: $0800 >> 3 = $0100 = 256. So index = (x+4.0 in Q8.8) >> 3
    lsr a
    lsr a
    lsr a
    and #$01FE              ; mask to even value (0..510), but we want 0..255 then *2
    ; Actually we need index 0..255, then *2 for word table
    ; After >>3, value is 0..$0100 (0..256)
    ; We want to clamp to 0..255 and multiply by 2
    cmp #$0200              ; 256*2
    bcc @idx_ok
    lda #$01FE              ; clamp to index 255
@idx_ok:
    and #$01FE              ; ensure even (word-aligned)
    tax
    lda ExpTable,x
    rts
.endproc

; ==========================================================================
; FP_InvSqrt — Q8.8 1/sqrt(x) via lookup table
;   Input:  A = x (Q8.8, must be positive)
;   Output: A = 1/sqrt(x) (Q8.8)
;   Table: 256 entries, index = (x >> 2), covers x in [0, 4.0)
; ==========================================================================

.proc FP_InvSqrt
    .a16
    .i16

    ; Clamp to valid range
    cmp #$0001
    bcs @not_zero
    lda #$7FFF              ; 1/sqrt(0) -> max
    rts
@not_zero:

    ; Index = (x_q8 >> 2) - 1, clamped to [0, 255]
    ; x_q8 >> 2 maps x in [0, 4.0) to [0, 256)
    lsr a
    lsr a
    ; Now value is 0..255 range (for x in [0, 4.0))
    beq @use_zero           ; if 0, use index 0
    sec
    sbc #1                  ; table is indexed from (i+1)/64, so subtract 1
    bra @lookup
@use_zero:
    ; x was very small, use first entry
@lookup:
    cmp #$0100
    bcc @idx_ok
    lda #$00FF              ; clamp to 255
@idx_ok:
    asl a                   ; *2 for word table
    tax
    lda InvSqrtTable,x
    rts
.endproc
