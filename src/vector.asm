; ==========================================================================
; Vector Operations for SNES GPT
; ==========================================================================
; VecDot, Linear, RMSNorm, Softmax, VecAdd, VecCopy

.p816
.smart

.include "snes.inc"

.segment "ZEROPAGE"
.importzp dp_tmp0, dp_tmp1, dp_tmp2, dp_tmp3, dp_tmp4
.importzp dp_acc, dp_acc_hi
.importzp dp_i, dp_j, dp_k, dp_n
.importzp dp_src_lo, dp_dst_lo, dp_mat_lo, dp_vec_lo
.importzp dp_max_val, dp_sum, dp_scale

; Extra zero-page vars for Linear output pointer (avoids clobbering dp_dst_lo)
.exportzp dp_out_lo
dp_out_lo:      .res 2

.segment "CODE"

.export VecDot, Linear, RMSNorm, Softmax, VecAdd, VecCopy
.import FP_Mul, FP_Div, FP_Exp, FP_InvSqrt

; --- Constants ---
N_EMBD    = 16
VEC_SIZE  = N_EMBD * 2     ; 32 bytes

; ==========================================================================
; VecDot — Dot product of two Q8.8 vectors
;   Input:  dp_src_lo = pointer to vec A
;           dp_dst_lo = pointer to vec B
;           dp_n = vector length in bytes (elements * 2)
;   Output: A = dot product (Q8.8)
;   Uses 32-bit accumulator to avoid overflow during summation.
; ==========================================================================

.proc VecDot
    .a16
    .i16

    stz dp_acc              ; clear 32-bit accumulator
    stz dp_acc_hi
    ldy #0

@loop:
    lda (dp_src_lo),y       ; A[i]
    sta dp_tmp0
    lda (dp_dst_lo),y       ; B[i]
    phy                     ; save Y (FP_Mul may clobber)
    jsr FP_Mul              ; A[i] * B[i] -> Q8.8 in A
    ply

    ; Sign-extend and add to 32-bit accumulator
    tax                     ; save result
    bpl @pos
    ; Negative: sign extend
    clc
    txa
    adc dp_acc
    sta dp_acc
    lda dp_acc_hi
    adc #$FFFF              ; sign extend carry
    sta dp_acc_hi
    bra @next
@pos:
    txa
    clc
    adc dp_acc
    sta dp_acc
    lda dp_acc_hi
    adc #0
    sta dp_acc_hi

@next:
    iny
    iny
    cpy dp_n
    bcc @loop

    lda dp_acc              ; return Q8.8 result
    rts
.endproc

; ==========================================================================
; Linear — Matrix-vector multiply: out[r] = dot(W[r], x) for each row r
;   Input:  dp_mat_lo = pointer to weight matrix W [nout][nin]
;           dp_vec_lo = pointer to input vector x [nin]
;           dp_out_lo = pointer to output vector [nout]  (use dp_out_lo!)
;           dp_i = nout (number of output elements, in BYTES = nout*2)
;           dp_j = nin (number of input elements, in BYTES = nin*2)
;   Output: result written to [dp_out_lo]
; ==========================================================================

.proc Linear
    .a16
    .i16

    ldy #0                  ; output byte offset

@row:
    ; Set up dot product: src=current matrix row, dst=input vector
    lda dp_mat_lo
    sta dp_src_lo
    lda dp_vec_lo
    sta dp_dst_lo
    lda dp_j
    sta dp_n

    phy                     ; save output offset
    phx                     ; save X
    jsr VecDot              ; A = dot product
    plx
    ply

    ; Store result at output[row]
    sta (dp_out_lo),y

    ; Advance matrix pointer by one row (nin bytes)
    lda dp_mat_lo
    clc
    adc dp_j
    sta dp_mat_lo

    ; Advance output offset
    iny
    iny
    cpy dp_i                ; done all rows?
    bcc @row

    rts
.endproc

; ==========================================================================
; RMSNorm — Root Mean Square normalization, in-place
;   Input:  dp_src_lo = pointer to vector x [N_EMBD]
;           dp_n = vector length in bytes
;   Output: x normalized in-place
; ==========================================================================

.proc RMSNorm
    .a16
    .i16

    ; Step 1: Compute sum of squares
    stz dp_acc
    stz dp_acc_hi
    ldy #0

@sq_loop:
    lda (dp_src_lo),y       ; x[i]
    sta dp_tmp0             ; also use as second operand (x[i] * x[i])
    phy
    jsr FP_Mul              ; x[i]^2
    ply

    ; Add to 32-bit accumulator (sign-extend)
    ; Note: x^2 is always positive, so no sign extension needed
    clc
    adc dp_acc
    sta dp_acc
    lda dp_acc_hi
    adc #0
    sta dp_acc_hi

    iny
    iny
    cpy dp_n
    bcc @sq_loop

    ; Step 2: Divide by N_EMBD to get mean square
    ; dp_n is in bytes, so elements = dp_n / 2
    ; For N_EMBD=16: shift right 4
    lda dp_acc
    lsr dp_acc_hi
    ror a
    lsr dp_acc_hi
    ror a
    lsr dp_acc_hi
    ror a
    lsr dp_acc_hi
    ror a                   ; ms = sum_sq / 16

    ; Step 3: 1/sqrt(ms)
    jsr FP_InvSqrt          ; A = scale = 1/sqrt(ms)
    sta dp_scale

    ; Step 4: x[i] *= scale for each element
    ldy #0
@scale_loop:
    lda (dp_src_lo),y       ; x[i]
    sta dp_tmp0
    lda dp_scale
    phy
    jsr FP_Mul              ; x[i] * scale
    ply
    sta (dp_src_lo),y       ; write back
    iny
    iny
    cpy dp_n
    bcc @scale_loop

    rts
.endproc

; ==========================================================================
; Softmax — Convert logits to probabilities
;   Input:  dp_src_lo = pointer to logits
;           dp_dst_lo = pointer to output probs (can be same as src)
;           dp_n = number of elements in bytes
;   Output: probabilities written to dp_dst_lo
; ==========================================================================

.proc Softmax
    .a16
    .i16

    ; --- Find max for numerical stability ---
    lda #$8000              ; most negative Q8.8
    sta dp_max_val
    ldy #0

@max_loop:
    lda (dp_src_lo),y
    sec
    sbc dp_max_val
    ; Signed comparison: if (src - max) > 0, new max
    bmi @not_max
    lda (dp_src_lo),y
    sta dp_max_val
@not_max:
    iny
    iny
    cpy dp_n
    bcc @max_loop

    ; --- Compute exp(x[i] - max) and sum ---
    stz dp_sum
    stz dp_sum+2
    ldy #0

@exp_loop:
    lda (dp_src_lo),y       ; logit[i]
    sec
    sbc dp_max_val          ; logit[i] - max (always <= 0)
    phy
    jsr FP_Exp              ; exp(logit[i] - max)
    ply
    sta (dp_dst_lo),y       ; store exp value

    ; Add to sum (unsigned, exp values are positive)
    clc
    adc dp_sum
    sta dp_sum
    lda dp_sum+2
    adc #0
    sta dp_sum+2

    iny
    iny
    cpy dp_n
    bcc @exp_loop

    ; --- Normalize: prob[i] = exp[i] / sum ---
    lda dp_sum
    sta dp_tmp0             ; divisor = sum
    ldy #0

@norm_loop:
    lda (dp_dst_lo),y       ; exp[i]
    phy
    jsr FP_Div              ; exp[i] / sum
    ply
    sta (dp_dst_lo),y       ; prob[i]
    iny
    iny
    cpy dp_n
    bcc @norm_loop

    rts
.endproc

; ==========================================================================
; VecAdd — Element-wise vector add: out[i] = A[i] + B[i]
;   Input:  dp_src_lo = pointer to A
;           dp_dst_lo = pointer to B (also output)
;           dp_n = length in bytes
;   Output: result written to dp_dst_lo
; ==========================================================================

.proc VecAdd
    .a16
    .i16

    ldy #0
@loop:
    lda (dp_src_lo),y
    clc
    adc (dp_dst_lo),y
    sta (dp_dst_lo),y
    iny
    iny
    cpy dp_n
    bcc @loop
    rts
.endproc

; ==========================================================================
; VecCopy — Copy vector: dst[i] = src[i]
;   Input:  dp_src_lo = source pointer
;           dp_dst_lo = destination pointer
;           dp_n = length in bytes
;   Output: dp_dst_lo filled with copy of dp_src_lo
; ==========================================================================

.proc VecCopy
    .a16
    .i16

    ldy #0
@loop:
    lda (dp_src_lo),y
    sta (dp_dst_lo),y
    iny
    iny
    cpy dp_n
    bcc @loop
    rts
.endproc
