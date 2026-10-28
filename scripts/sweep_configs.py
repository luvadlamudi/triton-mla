"""Run every stage-1 autotune config in its own process and report which ones fail.

A misaligned-address fault poisons the CUDA context, so configs are isolated.
"""

import subprocess
import sys

CHILD = r"""
import sys, torch, triton
import triton_mla.kernel as k
from triton_mla import mla_decode, mla_decode_ref
bh, bn, nw, ns, paged = map(int, sys.argv[1:])
k._mla_decode_stage1.configs = [triton.Config({"BLOCK_H": bh, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)]
torch.manual_seed(0)
seqlens = torch.tensor([0, 1, 513, 4096], dtype=torch.int32, device="cuda")
B, H, S, P = 4, 128, 4096, 64
q = torch.randn(B, H, 576, device="cuda", dtype=torch.bfloat16)
if paged:
    kv = torch.randn(B * S // P, P, 576, device="cuda", dtype=torch.bfloat16)
    bt = torch.randperm(B * S // P, device="cuda").view(B, -1).int()
else:
    kv = torch.randn(B, S, 576, device="cuda", dtype=torch.bfloat16)
    bt = None
out = mla_decode(q, kv, seqlens, bt, num_splits=4)
torch.cuda.synchronize()
ref, _ = mla_decode_ref(q, kv, seqlens, bt)
print("maxerr", (out.float() - ref.float()).abs().max().item())
"""


def main():
    from triton_mla.kernel import _stage1_configs

    for paged in (0, 1):
        for cfg in _stage1_configs():
            args = [cfg.kwargs["BLOCK_H"], cfg.kwargs["BLOCK_N"], cfg.num_warps, cfg.num_stages, paged]
            r = subprocess.run([sys.executable, "-c", CHILD, *map(str, args)], capture_output=True, text=True)
            if r.returncode == 0:
                status = r.stdout.strip().splitlines()[-1]
            else:
                err = [l for l in r.stderr.splitlines() if "Error" in l]
                status = "FAIL " + (err[-1][:120] if err else r.stderr[-120:])
            print(f"paged={paged} BLOCK_H={args[0]:>2} BLOCK_N={args[1]:>2} warps={args[2]} stages={args[3]}  {status}", flush=True)


if __name__ == "__main__":
    main()
