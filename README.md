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

Head groups that share a KV slice are adjacent in launch order, so only the
first one reads it from HBM and the rest hit L2. The split count targets one
wave of SMs (`batch x head_groups x splits ~= num_SMs`); more splits only add
fp32 partial-output traffic. Both choices, and the autotune search space, come
from a measured sweep of every tile config x split count
(`scripts/tune_splits.py`). The autotune key includes batch and context-size
buckets so a config tuned on a tiny problem is not reused for a large one.

The torch baseline (`mla_decode_torch`) gathers the paged cache, materializes the
`[B, H, S]` score matrix and runs two bmm's; `torch.compile` fuses the
softmax/masking but still writes the scores and the gathered cache to HBM,
which is where the fused kernel wins.

## Results (H100 80GB SXM)

128 query heads (DeepSeek-V3, one GPU), bf16, paged KV cache with 64-token
pages, one query token per sequence. `torch.compile` uses
`mode="max-autotune-no-cudagraphs"`. Timed with `triton.testing.do_bench`;
every case is checked for numerical agreement before timing.

![Speedup over torch.compile](assets/speedup.png)

| batch | context | Triton | torch.compile | speedup | Triton TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 1 | 4K | 24.4 us | 29.1 us | 1.19x | 47 |
| 1 | 32K | 67.9 us | 86.9 us | 1.28x | 135 |
| 8 | 1K | 30.7 us | 29.1 us | 0.95x | 74 |
| 8 | 32K | 326 us | 538 us | 1.65x | 224 |
| 32 | 32K | 1.20 ms | 4.32 ms | 3.59x | 242 |
| 128 | 32K | 4.69 ms | 16.86 ms | 3.60x | 249 |

Full sweep: [`results/bench_h100.csv`](results/bench_h100.csv). Faster than
`torch.compile` in 15 of 16 shapes; the one loss is batch 8 at 1K tokens, where
every implementation is launch-latency bound.

**Why TFLOPS, not bandwidth.** With 128 heads sharing one latent vector, every
cached byte feeds ~240 FLOPs of tensor-core work, which puts MLA decode near the
H100's compute roof rather than its memory roof. The kernel reaches ~250 TFLOPS,
about 25% of H100 dense bf16 peak. At batch >= 32 `torch.compile` produced no
gain over eager PyTorch, so the large-batch speedups are effectively over the
plain PyTorch bmm-softmax-bmm path.

![Achieved TFLOPS](assets/tflops_h100.png)

Reproduce (locally on a GPU, or on Modal with `modal run modal_app.py --cmd "..."`):

```bash
triton-mla-bench --batch 1 8 32 128 --seqlen 1024 4096 16384 32768 --csv results/bench.csv
python scripts/plot_bench.py results/bench.csv --out assets
```

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
- **Blackwell (B200) is not working yet**: the kernel hits a misaligned-address
  fault on B200 with Triton 3.8 while being correct on H100. Under investigation
  (`scripts/sweep_configs.py`).
- `num_splits=None` reads `cache_seqlens.max()` on the host; pass `num_splits`
  for CUDA-graph capture.

## Publishing

```bash
pip install build twine
python -m build
twine upload dist/*
```
