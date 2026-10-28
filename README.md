# triton-mla

A fused Triton decode kernel for DeepSeek's **Multi-head Latent Attention (MLA)**,
as used in DeepSeek-V2 / V3 / R1. Drop-in API similar to FlashMLA, works with
paged or contiguous KV caches, and ships with a reference MLA layer and a
benchmark against `torch.compile`.

```bash
pip install triton-mla                                    # once published to PyPI
pip install git+https://github.com/lvadlamudi/triton-mla.git   # straight from GitHub
```

## Usage

```python
import torch
from triton_mla import mla_decode

B, H = 32, 128                     # batch, query heads
q = torch.randn(B, H, 576, device="cuda", dtype=torch.bfloat16)        # [absorbed q_nope | q_pe]
kv_cache = torch.randn(num_blocks, 64, 576, device="cuda", dtype=torch.bfloat16)  # [c_kv | k_pe]
block_table = ...                  # [B, max_blocks] int32
cache_seqlens = ...                # [B] int32

out = mla_decode(q, kv_cache, cache_seqlens, block_table, softmax_scale=192 ** -0.5)
# out: [B, H, 512] in latent space; multiply by W_UV to get per-head values
```

A full MLA layer that does the projections, RoPE, cache append, weight
absorption and calls the kernel:

```python
from triton_mla import MLA, MLAConfig

layer = MLA(MLAConfig()).cuda().bfloat16()            # DeepSeek-V3 dimensions by default
y, cache_seqlens = layer(x, kv_cache, cache_seqlens, block_table)
```

## How it works

MLA caches one 576-dim vector per token instead of per-head K and V: a 512-dim
compressed latent `c_kv` plus a 64-dim decoupled RoPE key `k_pe`. At decode the
up-projections are absorbed:

```
score_h = (q_nope_h W_UK_h) . c_kv  +  q_pe_h . k_pe        # W_UK folded into the query
out_h   = (softmax(score_h) . c_kv) W_UV_h                   # W_UV applied after attention
```

so every head attends over the *same* cache vector. MLA decode becomes MQA with
`head_dim_qk = 576`, `head_dim_v = 512`, and is bound by reading the latent cache.

The kernel (`src/triton_mla/kernel.py`) is flash-decoding style:

1. **Stage 1** - grid `(batch, head_groups, kv_splits)`. A program loads
   `BLOCK_H` query heads once, then streams its slice of the cache through
   paged block-table lookups. Each `[BLOCK_N, 576]` tile is loaded once and
   reused by all `BLOCK_H` heads via tensor-core `tl.dot`s (latent and RoPE
   parts handled as separate 512 + 64 dots, no padding to 1024). Online softmax
   in `exp2` space with fp32 accumulators.
2. **Stage 2** - log-sum-exp merge of the per-split partial outputs.

The split count is chosen so `batch x head_groups x splits` fills the SMs
without making splits shorter than ~256 tokens, which keeps small-batch,
long-context decode bandwidth bound instead of occupancy bound. Tile sizes,
warps and pipeline stages are autotuned per GPU.

The torch baseline (`mla_decode_torch`) gathers the paged cache, materializes the
`[B, H, S]` score matrix and runs two bmm's; `torch.compile` fuses the
softmax/masking but still writes the scores and the gathered cache to HBM,
which is where the fused kernel wins.

## Benchmark

```bash
triton-mla-bench --batch 1 32 128 --seqlen 4096 16384
```

Prints a markdown table of Triton vs eager vs `torch.compile` latency, achieved
memory bandwidth and speedup. Results depend on the GPU; run it on yours.

## Tests

```bash
pip install -e ".[test]"
pytest                                # on a CUDA GPU, bf16
TRITON_INTERPRET=1 pytest             # on CPU via the Triton interpreter, fp32
```

Tests check the kernel against a per-sequence fp32 reference (dense and paged
caches, uneven and empty sequences, several split counts, head counts that are
not multiples of the tile), and check that the weight-absorbed MLA layer matches
an un-absorbed per-head MHA computation.

## Limitations

- One query token per sequence (no speculative / MTP multi-token decode yet).
- bf16 / fp16 cache only; no FP8 KV cache.
- `num_splits=None` reads `cache_seqlens.max()` on the host; pass `num_splits`
  for CUDA-graph capture.

## Publishing

```bash
pip install build twine
python -m build
twine upload dist/*
```
