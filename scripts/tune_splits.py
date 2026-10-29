"""Time every (tile config, num_splits) pair on representative decode shapes.

Used to pick the autotune search space and the default split heuristic.
"""

import itertools

import torch
import triton

import triton_mla.kernel as k
from triton_mla.bench import make_inputs

SHAPES = [(1, 4096), (1, 32768), (8, 4096), (8, 32768), (32, 4096), (32, 16384), (128, 4096), (128, 16384)]
SPLITS = [1, 2, 4, 8, 16, 32, 64]
CONFIGS = [
    triton.Config({"BLOCK_H": bh, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bh, bn, nw, ns in itertools.product((16, 32, 64), (32, 64), (4, 8), (2, 3))
    if bh * 512 // (nw * 32) <= 128
]


def main():
    tuner = k._mla_decode_stage1
    print(f"# {torch.cuda.get_device_name()}  {len(CONFIGS)} configs")
    for batch, seqlen in SHAPES:
        q, kv, seqlens, bt = make_inputs(batch, seqlen, 128, 64, torch.bfloat16, "cuda")
        nbytes = batch * seqlen * 576 * 2
        results = []
        for cfg in CONFIGS:
            tuner.configs = [cfg]
            tuner.cache = {}
            name = f"H{cfg.kwargs['BLOCK_H']}/N{cfg.kwargs['BLOCK_N']}/w{cfg.num_warps}/s{cfg.num_stages}"
            for splits in SPLITS:
                if seqlen // splits < 64:
                    continue
                try:
                    us = triton.testing.do_bench(lambda: k.mla_decode(q, kv, seqlens, bt, num_splits=splits)) * 1e3
                except triton.runtime.errors.OutOfResources:
                    continue
                results.append((us, name, splits))
        results.sort()
        best_us, best_cfg, best_splits = results[0]
        print(f"\n## batch={batch} seqlen={seqlen}  best {best_us:.1f}us {nbytes / best_us / 1e6:.2f} TB/s  {best_cfg} splits={best_splits}")
        for us, name, splits in results[:6]:
            print(f"   {us:8.1f}us  {nbytes / us / 1e6:5.2f} TB/s  {name:16s} splits={splits}")
        by_splits = {}
        for us, name, splits in results:
            by_splits.setdefault(splits, (us, name))
        print("   best per splits: " + "  ".join(f"{s}:{u:.0f}us" for s, (u, _) in sorted(by_splits.items())))
        del q, kv, seqlens, bt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
