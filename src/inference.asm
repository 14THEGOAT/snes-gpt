; ==========================================================================
; Inference Loop — Generate samples and display on SNES
; ==========================================================================

.p816
.smart

.include "snes.inc"

.segment "ZEROPAGE"
.importzp dp_tmp0, dp_tmp1, dp_tmp2, dp_tmp3, dp_tmp4
.importzp dp_acc, dp_acc_hi
.importzp dp_i, dp_j, dp_k, dp_n
.importzp dp_src_lo, dp_dst_lo
.importzp dp_token_id, dp_pos_id, dp_seq_len

.segment "BSS"
.import vec_logits, vec_probs, vec_attn_w
.import kv_keys, kv_values
.import gen_tokens, gen_string
.import rng_state

; Cursor position for display
.export cursor_row, cursor_col
cursor_row:     .res 2
cursor_col:     .res 2

.segment "CODE"

.export GenerateSample, PrintSample, PRNG_Next, InitDisplay
.import GPT_Forward
.import FP_Mul
.import Softmax
.importzp dp_out_lo
.import FontTiles

; --- Constants ---
N_EMBD     = 16
N_HEAD     = 4
BLOCK_SIZE = 8
VOCAB_SIZE = 27
VEC_SIZE   = N_EMBD * 2
BOS_TOKEN  = 0

; Temperature: 1/0.6 ≈ 1.667 in Q8.8 = $01AB
TEMP_INV   = $01AB

; Tilemap base address in VRAM (words)
TILEMAP_BASE = $0400        ; tilemap at VRAM $0400
; Font tile data base in VRAM (words)
FONT_BASE    = $1000        ; tiles at VRAM $1000

; ==========================================================================
; GenerateSample — Generate one name autoregressively
;   Output: gen_string filled with generated characters (null-terminated)
; ==========================================================================

.proc GenerateSample
    .a16
    .i16

    ; Clear KV cache
    ldx #0
    lda #0
@clear_kv:
    sta kv_keys,x
    sta kv_values,x
    inx
    inx
    cpx #(BLOCK_SIZE * VEC_SIZE)    ; 256 bytes
    bcc @clear_kv

    ; Start with BOS token
    lda #BOS_TOKEN
    sta dp_token_id
    stz dp_pos_id
    stz dp_seq_len

    ldx #0                  ; output string index
    stx dp_k                ; reuse dp_k as string index

@gen_loop:
    ; Update sequence length (= pos + 1)
    lda dp_pos_id
    ; inc a
    ; sta dp_seq_len
    ; Actually seq_len = how many KV entries we have = pos_id (0-indexed, before we store)
    ; After storing in GPT_Forward, seq_len should be pos+1
    sta dp_seq_len

    ; Forward pass
    jsr GPT_Forward

    ; Apply temperature scaling: logits[i] *= (1/temperature)
    ldy #0
@temp_loop:
    lda vec_logits,y
    sta dp_tmp0
    lda #TEMP_INV           ; 1/0.6 ≈ 1.667
    phy
    jsr FP_Mul
    ply
    sta vec_logits,y
    iny
    iny
    cpy #(VOCAB_SIZE * 2)
    bcc @temp_loop

    ; Softmax
    lda #vec_logits
    sta dp_src_lo
    lda #vec_probs
    sta dp_dst_lo
    lda #(VOCAB_SIZE * 2)
    sta dp_n
    jsr Softmax

    ; Sample from distribution
    jsr PRNG_Next           ; A = random Q8.8 in [0, 1.0)
    sta dp_tmp0             ; random threshold

    ; Cumulative sum sampling
    stz dp_acc
    ldy #0
@sample_loop:
    lda dp_acc
    clc
    adc vec_probs,y
    sta dp_acc
    cmp dp_tmp0
    bcs @sampled
    iny
    iny
    cpy #(VOCAB_SIZE * 2)
    bcc @sample_loop

    ; Fallback: pick last token
    ldy #((VOCAB_SIZE - 1) * 2)

@sampled:
    ; Convert byte offset to token index
    tya
    lsr a                   ; / 2
    sta dp_token_id

    ; Check for BOS (end of name)
    cmp #BOS_TOKEN
    beq @gen_done

    ; Decode token to character
    ; Token mapping: 0=BOS, 1='.', 2='a', ... 27='z'
    tax
    lda TokenToChar,x
    and #$00FF              ; mask to byte

    ; Store in output string
    ldx dp_k
    sep #$20
    .a8
    sta gen_string,x
    rep #$20
    .a16
    inx
    stx dp_k

    ; Advance position
    inc dp_pos_id

    ; Check max context length
    lda dp_pos_id
    cmp #BLOCK_SIZE
    bcc @gen_loop

@gen_done:
    ; Null-terminate string
    ldx dp_k
    sep #$20
    .a8
    lda #0
    sta gen_string,x
    rep #$20
    .a16

    rts
.endproc

; ==========================================================================
; PRNG_Next — 32-bit xorshift PRNG
;   Output: A = random Q8.8 value in [0, 1.0)
; ==========================================================================

.proc PRNG_Next
    .a16

    ; 16-bit xorshift with full-period triplet (7, 9, 13)
    ; Period = 2^16 - 1 = 65535 (maximal)
    lda rng_state

    ; state ^= state << 7
    pha
    .repeat 7
        asl a
    .endrepeat
    sta dp_tmp2
    pla
    eor dp_tmp2

    ; state ^= state >> 9
    pha
    .repeat 9
        lsr a
    .endrepeat
    sta dp_tmp2
    pla
    eor dp_tmp2

    ; state ^= state << 13
    pha
    .repeat 13
        asl a
    .endrepeat
    sta dp_tmp2
    pla
    eor dp_tmp2

    sta rng_state

    ; Return fractional value in [0, 1.0) as Q8.8
    and #$00FF
    rts
.endproc

; ==========================================================================
; InitDisplay — Set up SNES PPU for text display
;   Sets up Mode 0, BG1, loads font tiles, sets palette
; ==========================================================================

.proc InitDisplay
    .a16

    ; Switch to 8-bit A for PPU register writes
    sep #$20
    .a8

    ; Force blank (screen off)
    lda #$80
    sta INIDISP

    ; BG Mode 0 (4 BG layers, 2bpp tiles)
    lda #$00
    sta BGMODE

    ; BG1 tilemap at VRAM $0400 (word address), 32x32
    ; Register value: (addr >> 8) | size
    ; $0400 >> 8 = $04, size = 0 (32x32)
    lda #$04
    sta BG1SC

    ; BG1 tile data at VRAM $1000 (word address)
    ; Register value: addr >> 12
    ; $1000 >> 12 = $01
    lda #$01
    sta BG12NBA

    ; Enable BG1 on main screen
    lda #$01
    sta TM

    ; --- Set up VRAM for font tile upload ---
    ; VRAM auto-increment on low byte write (mode for word writes)
    lda #$80
    sta VMAIN               ; increment after writing high byte

    rep #$30
    .a16
    .i16

    ; Set VRAM address to tile data area
    lda #FONT_BASE
    sta VMADDL              ; VRAM address (low + high)

    ; Upload font tiles to VRAM
    ; Font is 2bpp, 16 bytes per tile
    ; We need to write words to VRAM
    ldx #0
@upload_font:
    lda FontTiles,x
    sta VMDATAL             ; write word to VRAM
    inx
    inx
    cpx #(38 * 16)          ; 38 tiles * 16 bytes each = 608 bytes
    bcc @upload_font

    ; Clear tilemap (fill with space tiles = 0)
    lda #TILEMAP_BASE
    sta VMADDL
    lda #$0000              ; tile 0 = space, palette 0
    ldx #0
@clear_map:
    sta VMDATAL
    inx
    inx
    cpx #(32 * 32 * 2)      ; 32x32 tilemap, 2 bytes per entry
    bcc @clear_map

    ; --- Set palette (CGRAM) ---
    sep #$20
    .a8

    ; Palette 0: color 0 = black (BG), color 3 = white (text)
    lda #$00
    sta CGADD               ; palette index 0

    ; Color 0: black ($0000 in 15-bit BGR)
    lda #$00
    sta CGDATA
    lda #$00
    sta CGDATA

    ; Color 1: dark gray
    lda #$08
    sta CGDATA
    lda #$21
    sta CGDATA

    ; Color 2: light gray
    lda #$10
    sta CGDATA
    lda #$42
    sta CGDATA

    ; Color 3: white ($7FFF in 15-bit BGR)
    lda #$FF
    sta CGDATA
    lda #$7F
    sta CGDATA

    ; Set scroll to 0
    lda #$00
    sta BG1HOFS
    sta BG1HOFS
    sta BG1VOFS
    sta BG1VOFS

    ; Screen on, full brightness
    lda #$0F
    sta INIDISP

    ; NMI stays disabled — we poll RDNMI directly for VBlank sync
    lda #$00
    sta NMITIMEN

    rep #$20
    .a16

    ; Initialize cursor position
    stz cursor_row
    stz cursor_col

    rts
.endproc

; ==========================================================================
; PrintSample — Write generated name to screen
;   Reads gen_string, writes to BG1 tilemap in VRAM
; ==========================================================================

.proc PrintSample
    .a16

    ; Wait for VBlank before writing VRAM
    jsr WaitVBlank

    ; Force blank for VRAM access
    sep #$20
    .a8
    lda #$80
    sta INIDISP
    rep #$20
    .a16

    ; Compute tilemap address: TILEMAP_BASE + row * 32 + col
    lda cursor_row
    asl a
    asl a
    asl a
    asl a
    asl a                   ; row * 32
    clc
    adc #TILEMAP_BASE
    clc
    adc cursor_col

    ; Set VRAM address
    sta VMADDL

    sep #$20
    .a8
    lda #$80
    sta VMAIN               ; increment after high byte write
    rep #$20
    .a16

    ; Write characters
    ldx #0
@print_loop:
    lda gen_string,x
    and #$00FF
    beq @print_done         ; null terminator

    ; Map ASCII to tile index
    ; Our font: tile 0=space, 1='.', 2=A, ..., 27=Z, 28=0, ..., 37=9
    ; ASCII 'a'-'z' -> tile 2-27 (uppercase glyphs)
    ; ASCII 'A'-'Z' -> tile 2-27
    ; ASCII '.' -> tile 1
    ; ASCII '0'-'9' -> tile 28-37

    cmp #$2E                ; '.'
    bne @not_dot
    lda #$0001              ; tile 1
    bra @write_tile
@not_dot:
    cmp #$61                ; 'a'
    bcc @check_upper
    cmp #$7B                ; 'z'+1
    bcs @check_digit
    sec
    sbc #$5F                ; 'a' - 2 = tile 2 for 'a'
    bra @write_tile
@check_upper:
    cmp #$41                ; 'A'
    bcc @check_digit
    cmp #$5B                ; 'Z'+1
    bcs @check_digit
    sec
    sbc #$3F                ; 'A' - 2 = tile 2 for 'A'
    bra @write_tile
@check_digit:
    cmp #$30                ; '0'
    bcc @use_space
    cmp #$3A                ; '9'+1
    bcs @use_space
    sec
    sbc #$30                ; '0' -> 0
    clc
    adc #28                 ; tile 28 for '0'
    bra @write_tile
@use_space:
    lda #$0000              ; space tile

@write_tile:
    sta VMDATAL             ; write tile index (low = tile, high = attributes)

    inx
    cpx #BLOCK_SIZE
    bcc @print_loop

@print_done:
    ; Advance cursor to next row
    inc cursor_row
    stz cursor_col

    ; Screen back on
    sep #$20
    .a8
    lda #$0F
    sta INIDISP
    rep #$20
    .a16

    rts
.endproc

; ==========================================================================
; WaitVBlank — Wait for vertical blanking period
; ==========================================================================

.proc WaitVBlank
    .a16
    sep #$20
    .a8
@wait:
    lda RDNMI               ; $4210 - read NMI flag
    and #$80
    beq @wait
    rep #$20
    .a16
    rts
.endproc

; ==========================================================================
; Token to character lookup table
; ==========================================================================

.segment "RODATA"

.export TokenToChar
TokenToChar:
    .byte $00               ; 0 = BOS (null)
    .byte 'a'               ; 1
    .byte 'b'               ; 2
    .byte 'c'               ; 3
    .byte 'd'               ; 4
    .byte 'e'               ; 5
    .byte 'f'               ; 6
    .byte 'g'               ; 7
    .byte 'h'               ; 8
    .byte 'i'               ; 9
    .byte 'j'               ; 10
    .byte 'k'               ; 11
    .byte 'l'               ; 12
    .byte 'm'               ; 13
    .byte 'n'               ; 14
    .byte 'o'               ; 15
    .byte 'p'               ; 16
    .byte 'q'               ; 17
    .byte 'r'               ; 18
    .byte 's'               ; 19
    .byte 't'               ; 20
    .byte 'u'               ; 21
    .byte 'v'               ; 22
    .byte 'w'               ; 23
    .byte 'x'               ; 24
    .byte 'y'               ; 25
    .byte 'z'               ; 26
