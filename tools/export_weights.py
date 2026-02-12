"""
Export trained micro-GPT weights to Q8.8 binary format for SNES ROM.

This script:
1. Trains the micro GPT model (from micro-gpt.py logic)
2. Quantizes all float32 weights to Q8.8 signed 16-bit fixed-point
3. Exports as a binary blob matching the SNES WRAM layout

The output file `trained_weights_q8x8.bin` is included in the ROM via .incbin.
"""

import os
import math
import random
import struct
import sys

# =====================================================================
# Training (adapted from micro-gpt.py)
# =====================================================================

outdir = sys.argv[1] if len(sys.argv) > 1 else "."

random.seed(42)

# Load dataset
input_path = os.path.join(outdir, 'input.txt')
if not os.path.exists(input_path):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, input_path)

docs = [l.strip() for l in open(input_path).read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer
chars = ['<BOS>'] + sorted(set(''.join(docs)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
BOS = stoi['<BOS>']
print(f"vocab size: {vocab_size}")

# Autograd
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def __repr__(self): return f"Value(data={self.data}, grad={self.grad})"

# Model architecture
n_embd = 16
n_head = 4
n_layer = 1
block_size = 8
head_dim = n_embd // n_head

matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd, std=0)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd, std=0)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# Model functions
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() ** 2 for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

# Training
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

num_steps = 500
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [stoi[ch] for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (step + 1) % 50 == 0 or step == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# Quick inference test
print("\n--- inference test (Python) ---")
temperature = 0.6
for sample_idx in range(5):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    name = ""
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        name += itos[token_id]
    print(f"  sample {sample_idx+1}: {name}")

# =====================================================================
# Export weights to Q8.8 binary
# =====================================================================

def quantize_q8_8(value):
    """Convert float to Q8.8 signed 16-bit integer."""
    clamped = max(-128.0, min(127.996, value))
    q = int(round(clamped * 256))
    # Clamp to signed 16-bit range
    q = max(-32768, min(32767, q))
    return q & 0xFFFF  # unsigned representation

def export_matrix(matrix, f):
    """Export a 2D list of Values as Q8.8 little-endian binary."""
    for row in matrix:
        for val in row:
            data = val.data if isinstance(val, Value) else val
            q = quantize_q8_8(data)
            f.write(struct.pack('<H', q))

# Weight statistics
print("\n--- weight statistics ---")
for name, mat in state_dict.items():
    vals = [v.data for row in mat for v in row]
    print(f"  {name:25s}: min={min(vals):+.4f} max={max(vals):+.4f} mean={sum(vals)/len(vals):+.6f}")

# Export in the order matching SNES WRAM layout
output_file = os.path.join(outdir, 'trained_weights_q8x8.bin')
with open(output_file, 'wb') as f:
    export_matrix(state_dict['wte'], f)                    # param_wte
    export_matrix(state_dict['wpe'], f)                    # param_wpe
    export_matrix(state_dict['layer0.attn_wq'], f)         # param_attn_wq
    export_matrix(state_dict['layer0.attn_wk'], f)         # param_attn_wk
    export_matrix(state_dict['layer0.attn_wv'], f)         # param_attn_wv
    export_matrix(state_dict['layer0.attn_wo'], f)         # param_attn_wo
    export_matrix(state_dict['layer0.mlp_fc1'], f)         # param_mlp_fc1
    export_matrix(state_dict['layer0.mlp_fc2'], f)         # param_mlp_fc2
    export_matrix(state_dict['lm_head'], f)                # param_lm_head

file_size = os.path.getsize(output_file)
print(f"\nExported {file_size} bytes to {output_file}")
# Size depends on actual vocab_size (27 for a-z + BOS, no period)
expected = (vocab_size * n_embd + block_size * n_embd + n_embd * n_embd * 4 + 4 * n_embd * n_embd + n_embd * 4 * n_embd + vocab_size * n_embd) * 2
print(f"Expected {expected} bytes (vocab_size={vocab_size})")
assert file_size == expected, f"Expected {expected} bytes, got {file_size}"
print("Weight export successful!")
