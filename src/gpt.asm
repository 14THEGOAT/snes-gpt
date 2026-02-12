; ==========================================================================
; GPT Forward Pass — The Heart of SNES GPT
; ==========================================================================
; Implements the full GPT forward pass:
;   embedding -> RMSNorm -> attention -> residual -> MLP -> residual -> logits

.p816
.smart

.include "snes.inc"

.segment "ZEROPAGE"
.importzp dp_tmp0, dp_tmp1, dp_tmp2, dp_tmp3, dp_tmp4
.importzp dp_acc, dp_acc_hi
.importzp dp_i, dp_j, dp_k, dp_n, dp_head
.importzp dp_src_lo, dp_dst_lo, dp_mat_lo, dp_vec_lo
.importzp dp_token_id, dp_pos_id, dp_seq_len
.importzp dp_max_val, dp_sum, dp_scale

; Local zero-page vars for this module
dp_head_off:    .res 2      ; byte offset for current head (h * HEAD_DIM * 2)
dp_attn_t:      .res 2      ; attention position counter
dp_attn_j:      .res 2      ; attention element counter within head
dp_kv_ptr:      .res 2      ; pointer into KV cache

.segment "CODE"

; BSS imports (WRAM working buffers)
.import vec_x, vec_x_res, vec_q, vec_k, vec_v
.import vec_attn_out, vec_mlp_hidden, vec_logits
.import vec_attn_w, vec_scratch
.import kv_keys, kv_values

; ROM parameter imports (read directly from ROM)
.import param_wte, param_wpe
.import param_attn_wq, param_attn_wk, param_attn_wv, param_attn_wo
.import param_mlp_fc1, param_mlp_fc2, param_lm_head

.export GPT_Forward

.import FP_Mul, FP_Div
.import VecDot, Linear, RMSNorm, Softmax, VecAdd, VecCopy
.importzp dp_out_lo

; --- Constants ---
N_EMBD     = 16
N_HEAD     = 4
HEAD_DIM   = 4
BLOCK_SIZE = 8
VOCAB_SIZE = 27
VEC_SIZE   = N_EMBD * 2            ; 32 bytes
VEC4_SIZE  = HEAD_DIM * 2          ; 8 bytes
MLP_HIDDEN = 4 * N_EMBD            ; 64 elements
MLP_HID_BYTES = MLP_HIDDEN * 2     ; 128 bytes

; ==========================================================================
; GPT_Forward — Run one forward pass
;   Input:  dp_token_id = current token index (0..27)
;           dp_pos_id = current position (0..7)
;           dp_seq_len = current sequence length in KV cache
;   Output: vec_logits filled with 28 logit values (Q8.8)
; ==========================================================================

.proc GPT_Forward
    .a16
    .i16
    ; ==================================================================
    ; 1. Embedding: x = wte[token_id] + wpe[pos_id]
    ; ==================================================================

    ; Compute pointer to wte[token_id]: param_wte + token_id * 32
    lda dp_token_id
    asl a
    asl a
    asl a
    asl a
    asl a                   ; * 32
    clc
    adc #param_wte
    sta dp_src_lo           ; pointer to wte[token]

    ; Compute pointer to wpe[pos_id]: param_wpe + pos_id * 32
    lda dp_pos_id
    asl a
    asl a
    asl a
    asl a
    asl a                   ; * 32
    clc
    adc #param_wpe
    sta dp_dst_lo           ; pointer to wpe[pos]

    ; x[i] = wte[token][i] + wpe[pos][i]
    ldy #0
@emb_loop:
    lda (dp_src_lo),y       ; wte[token][i]
    clc
    adc (dp_dst_lo),y       ; + wpe[pos][i]
    sta vec_x,y
    iny
    iny
    cpy #VEC_SIZE
    bcc @emb_loop

    ; ==================================================================
    ; 2. Pre-attention: save residual, then RMSNorm
    ; ==================================================================

    ; Save residual BEFORE normalizing: x_res = copy(x)
    lda #vec_x
    sta dp_src_lo
    lda #vec_x_res
    sta dp_dst_lo
    lda #VEC_SIZE
    sta dp_n
    jsr VecCopy

    ; RMSNorm on x (in-place)
    lda #vec_x
    sta dp_src_lo
    lda #VEC_SIZE
    sta dp_n
    jsr RMSNorm

    ; ==================================================================
    ; 3. Attention block
    ; ==================================================================

    ; --- Q = Wq @ x ---
    lda #param_attn_wq
    sta dp_mat_lo
    lda #vec_x
    sta dp_vec_lo
    lda #vec_q
    sta dp_out_lo
    lda #VEC_SIZE           ; nout = 16 elements = 32 bytes
    sta dp_i
    lda #VEC_SIZE           ; nin = 16 elements = 32 bytes
    sta dp_j
    jsr Linear

    ; --- K = Wk @ x ---
    lda #param_attn_wk
    sta dp_mat_lo
    lda #vec_x
    sta dp_vec_lo
    lda #vec_k
    sta dp_out_lo
    lda #VEC_SIZE
    sta dp_i
    sta dp_j
    jsr Linear

    ; --- V = Wv @ x ---
    lda #param_attn_wv
    sta dp_mat_lo
    lda #vec_x
    sta dp_vec_lo
    lda #vec_v
    sta dp_out_lo
    lda #VEC_SIZE
    sta dp_i
    sta dp_j
    jsr Linear

    ; --- Store K, V into KV cache at current position ---
    ; kv_keys[pos] = k, kv_values[pos] = v
    lda dp_pos_id
    asl a
    asl a
    asl a
    asl a
    asl a                   ; pos * 32 = pos * VEC_SIZE
    tax                     ; X = offset into KV cache

    ldy #0
@store_kv:
    lda vec_k,y
    sta kv_keys,x
    lda vec_v,y
    sta kv_values,x
    inx
    inx
    iny
    iny
    cpy #VEC_SIZE
    bcc @store_kv

    ; ==================================================================
    ; 3a. Multi-head attention
    ; ==================================================================
    ; For each head h=0..3:
    ;   For each cached position t=0..pos:
    ;     logit[t] = dot(q[h*4:h*4+4], kv_keys[t][h*4:h*4+4]) / 2.0
    ;   weights = softmax(logits[0..pos])
    ;   For j=0..3:
    ;     attn_out[h*4+j] = sum_t(weights[t] * kv_values[t][h*4+j])

    stz dp_head

@head_loop:
    ; Compute head byte offset: h * HEAD_DIM * 2 = h * 8
    lda dp_head
    asl a
    asl a
    asl a                   ; h * 8
    sta dp_head_off

    ; --- Compute attention logits for this head ---
    stz dp_attn_t           ; t = 0

@attn_score_t:
    ; dot(q_h, kv_keys[t]_h) over HEAD_DIM elements
    stz dp_acc
    stz dp_acc_hi

    ; Compute KV cache offset: t * VEC_SIZE + head_offset
    lda dp_attn_t
    asl a
    asl a
    asl a
    asl a
    asl a                   ; t * 32
    clc
    adc dp_head_off         ; + head byte offset
    sta dp_kv_ptr           ; offset into kv_keys

    ; Dot product over HEAD_DIM=4 elements
    ldx #0                  ; element counter (in bytes, 0..6)
@attn_dot:
    ; q[h*hd + j]
    txa
    clc
    adc dp_head_off
    tay
    lda vec_q,y             ; q_h[j]
    sta dp_tmp0

    ; kv_keys[t][h*hd + j]
    txa
    clc
    adc dp_kv_ptr
    tay
    lda kv_keys,y           ; k_cached[t][h*hd+j]

    phx
    jsr FP_Mul              ; q * k element
    plx

    ; Accumulate (16-bit, safe for HEAD_DIM=4)
    clc
    adc dp_acc
    sta dp_acc

    inx
    inx
    cpx #VEC4_SIZE          ; 8 bytes = 4 elements
    bcc @attn_dot

    ; Divide by sqrt(HEAD_DIM) = sqrt(4) = 2 => arithmetic shift right 1
    lda dp_acc
    cmp #$8000
    ror a                   ; signed >> 1
    pha                     ; save logit value

    ; Store logit[t]: vec_attn_w[t] = dot / 2
    lda dp_attn_t
    asl a                   ; t * 2 for word offset
    tay
    pla
    sta vec_attn_w,y

    ; Next t: loop while t < pos+1
    inc dp_attn_t
    lda dp_attn_t
    cmp dp_pos_id
    bcc @attn_go_t
    beq @attn_go_t          ; include pos itself
    bra @attn_scores_done
@attn_go_t:
    jmp @attn_score_t

@attn_scores_done:

    ; --- Softmax over attention logits [0..pos] ---
    lda #vec_attn_w
    sta dp_src_lo
    sta dp_dst_lo
    lda dp_pos_id
    inc a                   ; pos + 1 elements
    asl a                   ; * 2 for bytes
    sta dp_n
    jsr Softmax

    ; --- Weighted sum of values for this head ---
    ; attn_out[h*hd+j] = sum_t(attn_w[t] * kv_values[t][h*hd+j])

    ldx #0                  ; j = 0 (byte offset within head)
@val_j_loop:
    stx dp_attn_j

    stz dp_acc
    stz dp_acc_hi
    stz dp_attn_t           ; t = 0

@val_t_loop:
    ; Load attn_w[t]
    lda dp_attn_t
    asl a
    tay
    lda vec_attn_w,y
    sta dp_tmp0

    ; Load kv_values[t][h*hd + j]
    lda dp_attn_t
    asl a
    asl a
    asl a
    asl a
    asl a                   ; t * 32
    clc
    adc dp_head_off         ; + head offset
    clc
    adc dp_attn_j           ; + j (byte offset within head)
    tay
    lda kv_values,y         ; values[t][h*hd+j]

    jsr FP_Mul              ; attn_w[t] * value

    ; Accumulate
    clc
    adc dp_acc
    sta dp_acc
    bcc @val_no_carry
    inc dp_acc_hi
@val_no_carry:

    ; Next t
    inc dp_attn_t
    lda dp_attn_t
    cmp dp_pos_id
    bcc @val_t_loop
    beq @val_t_loop         ; include pos itself (pos+1 entries)

    ; Store: attn_out[h*hd + j] = accumulated value
    lda dp_attn_j
    clc
    adc dp_head_off
    tay
    lda dp_acc
    sta vec_attn_out,y

    ; Next j
    ldx dp_attn_j
    inx
    inx
    cpx #VEC4_SIZE
    bcc @val_j_loop

    ; --- Next head ---
    inc dp_head
    lda dp_head
    cmp #N_HEAD
    bcs @head_done
    jmp @head_loop
@head_done:

    ; ==================================================================
    ; 3b. Attention output projection: x = Wo @ attn_out
    ; ==================================================================

    lda #param_attn_wo
    sta dp_mat_lo
    lda #vec_attn_out
    sta dp_vec_lo
    lda #vec_x
    sta dp_out_lo
    lda #VEC_SIZE
    sta dp_i
    sta dp_j
    jsr Linear

    ; Residual: x = x + x_res
    lda #vec_x_res
    sta dp_src_lo
    lda #vec_x
    sta dp_dst_lo
    lda #VEC_SIZE
    sta dp_n
    jsr VecAdd

    ; ==================================================================
    ; 4. MLP block
    ; ==================================================================

    ; Save residual
    lda #vec_x
    sta dp_src_lo
    lda #vec_x_res
    sta dp_dst_lo
    lda #VEC_SIZE
    sta dp_n
    jsr VecCopy

    ; RMSNorm
    lda #vec_x
    sta dp_src_lo
    lda #VEC_SIZE
    sta dp_n
    jsr RMSNorm

    ; fc1: hidden = W1 @ x  (16 -> 64)
    lda #param_mlp_fc1
    sta dp_mat_lo
    lda #vec_x
    sta dp_vec_lo
    lda #vec_mlp_hidden
    sta dp_out_lo
    lda #MLP_HID_BYTES      ; nout = 128 bytes
    sta dp_i
    lda #VEC_SIZE            ; nin = 32 bytes
    sta dp_j
    jsr Linear

    ; Activation: ReLU²(x) = max(0, x)²
    ldy #0
@relu_sq:
    lda vec_mlp_hidden,y
    bpl @relu_pos
    ; Negative: clamp to 0
    lda #0
    sta vec_mlp_hidden,y
    bra @relu_next
@relu_pos:
    ; Positive: square it
    sta dp_tmp0
    phy
    jsr FP_Mul              ; x * x
    ply
    sta vec_mlp_hidden,y
@relu_next:
    iny
    iny
    cpy #MLP_HID_BYTES
    bcc @relu_sq

    ; fc2: x = W2 @ hidden  (64 -> 16)
    lda #param_mlp_fc2
    sta dp_mat_lo
    lda #vec_mlp_hidden
    sta dp_vec_lo
    lda #vec_x
    sta dp_out_lo
    lda #VEC_SIZE            ; nout = 32 bytes
    sta dp_i
    lda #MLP_HID_BYTES       ; nin = 128 bytes
    sta dp_j
    jsr Linear

    ; MLP residual: x = x + x_res
    lda #vec_x_res
    sta dp_src_lo
    lda #vec_x
    sta dp_dst_lo
    lda #VEC_SIZE
    sta dp_n
    jsr VecAdd

    ; ==================================================================
    ; 5. Output: logits = lm_head @ x
    ; ==================================================================

    lda #param_lm_head
    sta dp_mat_lo
    lda #vec_x
    sta dp_vec_lo
    lda #vec_logits
    sta dp_out_lo
    lda #(VOCAB_SIZE * 2)    ; nout = 56 bytes
    sta dp_i
    lda #VEC_SIZE            ; nin = 32 bytes
    sta dp_j
    jsr Linear

    rts
.endproc
