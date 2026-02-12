; ==========================================================================
; SNES GPT — Main Entry Point
; ==========================================================================
; "The most atomic way to inference a GPT on a Super Nintendo"
;
; A tiny GPT-2 transformer running on 65816 assembly.
; Generates names using a 1-layer, 16-dim, 4-head model.
;
; Parameters live in ROM and are read directly (they're read-only).
; Only working buffers and KV cache use WRAM.

.p816
.smart

.include "snes.inc"

; ==========================================================================
; Constants
; ==========================================================================

N_EMBD     = 16
N_HEAD     = 4
N_LAYER    = 1
BLOCK_SIZE = 8
HEAD_DIM   = N_EMBD / N_HEAD
VOCAB_SIZE = 27             ; BOS + a-z (no period in names dataset)
VEC_SIZE   = N_EMBD * 2    ; 32 bytes per vector
BOS_TOKEN  = 0

; ==========================================================================
; ZERO PAGE variables
; ==========================================================================

.segment "ZEROPAGE"

.exportzp dp_tmp0, dp_tmp1, dp_tmp2, dp_tmp3, dp_tmp4
.exportzp dp_acc, dp_acc_hi
.exportzp dp_i, dp_j, dp_k, dp_n, dp_head
.exportzp dp_src_lo, dp_dst_lo, dp_mat_lo, dp_vec_lo
.exportzp dp_token_id, dp_pos_id, dp_seq_len
.exportzp dp_max_val, dp_sum, dp_scale

dp_tmp0:       .res 2
dp_tmp1:       .res 2
dp_tmp2:       .res 2
dp_tmp3:       .res 2
dp_tmp4:       .res 2
dp_acc:        .res 2
dp_acc_hi:     .res 2

dp_i:          .res 2
dp_j:          .res 2
dp_k:          .res 2
dp_n:          .res 2
dp_head:       .res 2

dp_src_lo:     .res 2
dp_dst_lo:     .res 2
dp_mat_lo:     .res 2
dp_vec_lo:     .res 2

dp_token_id:   .res 2
dp_pos_id:     .res 2
dp_seq_len:    .res 2

dp_max_val:    .res 2
dp_sum:        .res 4
dp_scale:      .res 2

dp_sample_cnt: .res 2

; ==========================================================================
; BSS — WRAM working buffers only (parameters stay in ROM)
; ==========================================================================

.segment "BSS"

.export vec_x, vec_x_res, vec_q, vec_k, vec_v
.export vec_attn_out, vec_mlp_hidden, vec_logits
.export vec_probs, vec_attn_w, vec_scratch
.export kv_keys, kv_values
.export gen_tokens, gen_string
.export rng_state

; Working vectors (~480 bytes)
vec_x:         .res VEC_SIZE               ; 32 - current hidden state
vec_x_res:     .res VEC_SIZE               ; 32 - residual copy
vec_q:         .res VEC_SIZE               ; 32 - query vector
vec_k:         .res VEC_SIZE               ; 32 - key vector
vec_v:         .res VEC_SIZE               ; 32 - value vector
vec_attn_out:  .res VEC_SIZE               ; 32 - attention output
vec_mlp_hidden:.res 4 * N_EMBD * 2        ; 128 - MLP hidden layer
vec_logits:    .res VOCAB_SIZE * 2         ; 54 - output logits
vec_probs:     .res VOCAB_SIZE * 2         ; 54 - softmax probabilities
vec_attn_w:    .res BLOCK_SIZE * 2         ; 16 - attention weights
vec_scratch:   .res VEC_SIZE               ; 32 - scratch space

; KV cache (512 bytes)
kv_keys:       .res N_LAYER * BLOCK_SIZE * VEC_SIZE   ; 256
kv_values:     .res N_LAYER * BLOCK_SIZE * VEC_SIZE   ; 256

; Generation output (13 bytes)
gen_tokens:    .res BLOCK_SIZE
gen_string:    .res BLOCK_SIZE + 1

; PRNG (4 bytes)
rng_state:     .res 4

; Total BSS: ~1009 bytes — fits easily in $0200-$1FFF

; ==========================================================================
; SNES Header and Vectors
; ==========================================================================

.segment "HEADER"
    .byte "SNES GPT             "  ; 21-byte title (padded)
    .byte $20                      ; LoROM
    .byte $00                      ; ROM only
    .byte $08                      ; ROM size: 256KB
    .byte $00                      ; No SRAM
    .byte $01                      ; North America
    .byte $00                      ; Developer ID
    .byte $00                      ; Version
    .word $0000                    ; Checksum complement
    .word $0000                    ; Checksum

.segment "VECTORS"
    ; Native mode vectors
    .word $0000                    ; unused
    .word $0000                    ; unused
    .word $0000                    ; COP
    .word $0000                    ; BRK
    .word $0000                    ; ABORT
    .word NMI_Handler              ; NMI
    .word $0000                    ; unused
    .word $0000                    ; IRQ
    ; Emulation mode vectors
    .word $0000                    ; unused
    .word $0000                    ; unused
    .word $0000                    ; COP
    .word $0000                    ; unused
    .word $0000                    ; ABORT
    .word NMI_Handler              ; NMI
    .word Reset_Handler            ; RESET
    .word $0000                    ; IRQ/BRK

; ==========================================================================
; CODE
; ==========================================================================

.segment "CODE"

.import InitDisplay, GenerateSample, PrintSample, PRNG_Next
.importzp dp_out_lo
.import cursor_row, cursor_col

; ==========================================================================
; Reset_Handler — SNES boot entry point
; ==========================================================================

Reset_Handler:
    sei                     ; disable interrupts
    clc
    xce                     ; switch to native 65816 mode

    rep #$30                ; 16-bit A and X/Y
    .a16
    .i16

    ldx #$01FF
    txs                     ; set stack

    lda #$0000
    tcd                     ; set direct page

    jsr ClearRegisters
    jsr ClearWRAM
    jsr InitDisplay

    ; Seed PRNG
    lda #$ACE1
    sta rng_state
    lda #$1337
    sta rng_state+2

    ; Display title
    jsr DisplayTitle

    ; Generate 20 samples
    ldx #20
@gen_loop:
    phx
    jsr GenerateSample
    jsr PrintSample
    plx
    dex
    bne @gen_loop

@forever:
    bra @forever

; ==========================================================================
; NMI_Handler
; ==========================================================================

NMI_Handler:
    rti

; ==========================================================================
; ClearRegisters
; ==========================================================================

.proc ClearRegisters
    .a16
    sep #$20
    .a8
    lda #$80
    sta INIDISP             ; force blank
    lda #$00
    ldx #$01
@ppu:
    sta $2100,x
    inx
    cpx #$34
    bcc @ppu
    sta NMITIMEN
    sta MDMAEN
    sta HDMAEN
    sta MEMSEL
    rep #$20
    .a16
    rts
.endproc

; ==========================================================================
; ClearWRAM
; ==========================================================================

.proc ClearWRAM
    .a16
    lda #$0000
    ldx #$0200
@loop:
    sta $00,x
    inx
    inx
    cpx #$2000
    bcc @loop
    rts
.endproc

; ==========================================================================
; DisplayTitle — Write "SNES GPT" on screen
; ==========================================================================

.proc DisplayTitle
    .a16
    sep #$20
    .a8
@vbl:
    lda RDNMI
    and #$80
    beq @vbl
    lda #$80
    sta INIDISP
    rep #$20
    .a16

    ; Row 1, col 11 (centered "SNES GPT")
    lda #($0400 + 43)
    sta VMADDL

    sep #$20
    .a8
    lda #$80
    sta VMAIN
    rep #$20
    .a16

    ; S=20, N=15, E=6, S=20, sp=0, G=8, P=17, T=21
    lda #20
    sta VMDATAL
    lda #15
    sta VMDATAL
    lda #6
    sta VMDATAL
    lda #20
    sta VMDATAL
    lda #0
    sta VMDATAL
    lda #8
    sta VMDATAL
    lda #17
    sta VMDATAL
    lda #21
    sta VMDATAL

    ; Set cursor for generated names (row 3)
    lda #3
    sta cursor_row
    stz cursor_col

    ; Screen on
    sep #$20
    .a8
    lda #$0F
    sta INIDISP
    rep #$20
    .a16
    rts
.endproc

; ==========================================================================
; ROM Parameters — trained weights embedded directly in ROM
; ==========================================================================

.segment "RODATA"

; Export parameter labels so GPT forward pass can reference them
.export param_wte, param_wpe
.export param_attn_wq, param_attn_wk, param_attn_wv, param_attn_wo
.export param_mlp_fc1, param_mlp_fc2, param_lm_head

; Each weight matrix gets its own label via .incbin offset/size
; Binary layout: wte, wpe, wq, wk, wv, wo, fc1, fc2, lm_head
param_wte:
    .incbin "trained_weights_q8x8.bin", 0, 864          ; [27][16]
param_wpe:
    .incbin "trained_weights_q8x8.bin", 864, 256         ; [8][16]
param_attn_wq:
    .incbin "trained_weights_q8x8.bin", 1120, 512        ; [16][16]
param_attn_wk:
    .incbin "trained_weights_q8x8.bin", 1632, 512        ; [16][16]
param_attn_wv:
    .incbin "trained_weights_q8x8.bin", 2144, 512        ; [16][16]
param_attn_wo:
    .incbin "trained_weights_q8x8.bin", 2656, 512        ; [16][16]
param_mlp_fc1:
    .incbin "trained_weights_q8x8.bin", 3168, 2048       ; [64][16]
param_mlp_fc2:
    .incbin "trained_weights_q8x8.bin", 5216, 2048       ; [16][64]
param_lm_head:
    .incbin "trained_weights_q8x8.bin", 7264, 864        ; [27][16]
; Total: 8128 bytes
