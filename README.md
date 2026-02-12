# SNES GPT

A tiny transformer running on a Super Nintendo, in 65816 assembly.

Generates names autoregressively using a 1-layer, 16-dimensional, 4-head model. All arithmetic is Q8.8 signed fixed-point. The SNES PPU hardware multiplier at `$4202`/`$4203` handles every multiply. Trained weights (8KB) live directly in ROM. The entire forward pass — embedding, RMSNorm, multi-head attention with KV cache, ReLU^2 MLP, and output projection — runs on the 3.58 MHz 65816 CPU.

Based on Andrej Karpathy's [micro-gpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), a minimal GPT implementation using only scalar `Value` operations (no tensors, no PyTorch).

## Build

Requires [cc65](https://cc65.github.io/) (ca65/ld65 assembler/linker) and Python 3.

```bash
make
```

This trains the model (~500 steps), exports Q8.8 weights, generates lookup tables and font data, assembles all source files, and links the ROM.

Output: `build/snes_gpt.sfc` — open in any SNES emulator (bsnes, Mesen, Snes9x).

Tested on Snes9x.

## How it works

### Model

```
Decoder-only causal transformer, 1 layer
  n_embd=16, n_head=4, head_dim=4
  block_size=8, vocab_size=27 (BOS + a-z)
  Activation: ReLU^2, Normalization: RMSNorm
  Learned position embeddings, no biases
  4064 parameters
```

### Forward pass

Each token runs the full transformer pipeline:

1. **Embedding** — `x = wte[token] + wpe[position]`
2. **RMSNorm** — normalize, save residual
3. **Multi-head attention** — Q/K/V projections via matrix-vector multiply, 4 heads with head_dim=4, scaled dot-product attention with KV cache, output projection
4. **Residual add**
5. **RMSNorm** — normalize, save residual
6. **MLP** — fc1 (16 -> 64), ReLU^2 activation, fc2 (64 -> 16)
7. **Residual add**
8. **Output projection** — `logits = lm_head @ x` (16 -> 27)

### Math

All arithmetic is Q8.8 signed fixed-point (16-bit: 8 integer bits, 8 fractional bits, range -128.0 to +127.996).

- **Multiply**: Four 8x8 unsigned partial products via the SNES PPU hardware multiplier, combined into a 32-bit result, then shifted right 8 for the Q8.8 product. Sign handled separately.
- **Divide**: 16-iteration restoring division with the dividend pre-shifted left 8 bits.
- **exp() and 1/sqrt()**: 256-entry ROM lookup tables. exp covers [-4.0, +4.0), 1/sqrt covers (0, 4.0).

### Inference

Generation is autoregressive with temperature sampling (T=0.6). A 16-bit xorshift PRNG (triplet 7,9,13, period 65535) provides randomness. Softmax converts logits to probabilities, then cumulative-sum sampling picks the next token. Each name terminates on BOS or after 8 tokens.

The ROM generates 20 names on boot and displays them using SNES Mode 0 BG1 with a 2bpp bitmap font.

### Memory layout

Parameters are read directly from ROM (LoROM at `$8000`+). Working buffers use ~1KB of WRAM:

| Buffer | Size | Purpose |
|--------|------|---------|
| `vec_x`, `vec_x_res`, `vec_q`, `vec_k`, `vec_v`, `vec_attn_out`, `vec_scratch` | 32B each | Working vectors |
| `vec_mlp_hidden` | 128B | MLP hidden layer (64 elements) |
| `vec_logits`, `vec_probs` | 54B each | Output logits and probabilities |
| `vec_attn_w` | 16B | Attention weights per head |
| `kv_keys`, `kv_values` | 256B each | KV cache (8 positions x 16 dims) |

ROM parameters (8128 bytes):

| Weight | Shape | Bytes |
|--------|-------|-------|
| `wte` | [27][16] | 864 |
| `wpe` | [8][16] | 256 |
| `attn_wq/wk/wv/wo` | [16][16] each | 2048 |
| `mlp_fc1` | [64][16] | 2048 |
| `mlp_fc2` | [16][64] | 2048 |
| `lm_head` | [27][16] | 864 |

## Files

```
snes-gpt/
├── src/                        65816 assembly source
│   ├── main.asm                Entry point, SNES init, header/vectors, weight embedding
│   ├── gpt.asm                 Full GPT forward pass
│   ├── math.asm                Q8.8 multiply, divide, exp, inverse sqrt
│   ├── vector.asm              VecDot, Linear, RMSNorm, Softmax, VecAdd, VecCopy
│   ├── inference.asm           Generation loop, PRNG, display routines
│   ├── snes.inc                SNES hardware register definitions
│   └── lorom.cfg               ld65 LoROM linker configuration
├── tools/                      Python tooling
│   ├── export_weights.py       Train model (based on micro-gpt) + export Q8.8 weights
│   ├── gen_tables.py           Generate exp/invsqrt lookup tables
│   ├── gen_font.py             Generate 2bpp font tile data
│   └── micro-gpt.py            Karpathy's reference GPT (not used in build)
├── build/                      Build output (gitignored)
└── Makefile
```

## Notable bugs found during development

**Missing `.i16` assembler directive** — The 65816 CPU uses the same opcodes for 8-bit and 16-bit index register operations, differing only in instruction length (2 vs 3 bytes for immediates). ca65 tracks register width via `.a16`/`.i16` directives, but each `.asm` file is assembled independently. Several files had `.a16` but not `.i16`, causing every `ldy #imm`, `ldx #imm`, `cpy #imm`, and `cpx #imm` to assemble as 2-byte (8-bit) instructions when the CPU was running in 16-bit index mode. This misaligned the entire instruction stream after the first such instruction, causing the ROM to crash.

**PRNG degenerate cycle** — The initial PRNG used the 65816 `xba` instruction (byte swap) as a substitute for `<< 8` in the xorshift algorithm. But `xba` is not a left shift — it creates a degenerate subspace where high and low bytes always mirror each other (e.g. `$D3D3` -> `$7474` -> ...). The state cycled back to the exact same value after 8 calls (one name's worth of tokens), so every generated name was identical. Fixed by using a proper xorshift(7,9,13) with repeated `asl a` instructions for the shifts.

**FP_Mul clobbering accumulator** — The fixed-point multiply routine used `dp_acc`/`dp_acc_hi` for its internal 32-bit accumulator, which are the same zero-page variables used by `VecDot` for accumulating dot products. Every multiply inside a dot product would zero out the running sum. Fixed by using `dp_tmp3`/`dp_tmp4` as FP_Mul's internal accumulator instead.

**Linear clobbering string index** — The `Linear` routine had a stale `sty dp_k` instruction that overwrote `dp_k`, which `GenerateSample` uses as its output string index. Every matrix-vector multiply during the forward pass would reset the string write position to 0, so only the last character of each name survived.
