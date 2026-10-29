"""Benchmark the Triton MLA decode kernel against PyTorch eager and torch.compile.

    python -m triton_mla.bench --batch 1 32 128 --seqlen 1024 4096 16384
"""

import argparse
import csv
import os

import torch
import triton

from .kernel import _mla_decode_stage1, _num_sms, get_num_splits, mla_decode
from .reference import mla_decode_torch

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
CSV_FIELDS = ["gpu", "batch", "seqlen", "heads", "num_splits", "config", "bytes", "triton_us", "eager_us", "compile_us"]


def make_inputs(batch, seqlen, num_heads, page_size, dtype, device, d_latent=512, d_rope=64):
    blocks_per_seq = triton.cdiv(seqlen, page_size)
    num_blocks = batch * blocks_per_seq
    q = torch.randn(batch, num_heads, d_latent + d_rope, device=device, dtype=dtype)
    kv_cache = torch.randn(num_blocks, page_size, d_latent + d_rope, device=device, dtype=dtype)
    block_table = torch.randperm(num_blocks, device=device).view(batch, blocks_per_seq).int()
    seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=device)
    return q, kv_cache, seqlens, block_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    parser.add_argument("--seqlen", type=int, nargs="+", default=[1024, 4096, 8192, 16384, 32768])
    parser.add_argument("--heads", type=int, default=128, help="128 for one GPU, 16 per GPU under TP8")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--no-eager", action="store_true")
    parser.add_argument("--csv", help="also append results to this CSV file")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    elt = torch.finfo(dtype).bits // 8
    scale = 192 ** -0.5
    compiled = torch.compile(mla_decode_torch, mode=args.compile_mode, dynamic=False)

    gpu = torch.cuda.get_device_name()
    writer = None
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        csv_file = open(args.csv, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

    print(f"# {gpu}  heads={args.heads}  page={args.page_size}  {args.dtype}")
    cols = ["batch", "seqlen", "splits", "config", "triton us", "TB/s", "TFLOPS", "compile us", "speedup"]
    if not args.no_eager:
        cols.insert(7, "eager us")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))

    for batch in args.batch:
        for seqlen in args.seqlen:
            q, kv, seqlens, bt = make_inputs(batch, seqlen, args.heads, args.page_size, dtype, device)
            splits = get_num_splits(batch, args.heads, seqlen, _num_sms(device))

            def run_triton():
                return mla_decode(q, kv, seqlens, bt, softmax_scale=scale, num_splits=splits)

            def run_compiled():
                return compiled(q, kv, seqlens, bt, softmax_scale=scale)

            torch.testing.assert_close(run_triton().float(), run_compiled().float(), atol=2e-2, rtol=2e-2)
            cfg = _mla_decode_stage1.best_config
            config = f"H{cfg.kwargs['BLOCK_H']}/N{cfg.kwargs['BLOCK_N']}/w{cfg.num_warps}/s{cfg.num_stages}"

            t_triton = triton.testing.do_bench(run_triton) * 1e3
            t_compile = triton.testing.do_bench(run_compiled) * 1e3
            nbytes = (batch * seqlen * kv.shape[-1] + q.numel() + batch * args.heads * 512) * elt
            flops = batch * seqlen * args.heads * 2 * (kv.shape[-1] + 512)
            row = [batch, seqlen, splits, config, f"{t_triton:.1f}", f"{nbytes / t_triton / 1e6:.2f}", f"{flops / t_triton / 1e6:.0f}"]
            t_eager = None
            if not args.no_eager:
                t_eager = triton.testing.do_bench(
                    lambda: mla_decode_torch(q, kv, seqlens, bt, softmax_scale=scale)
                ) * 1e3
                row.append(f"{t_eager:.1f}")
            row += [f"{t_compile:.1f}", f"{t_compile / t_triton:.2f}x"]
            print("| " + " | ".join(str(c) for c in row) + " |", flush=True)
            if writer:
                writer.writerow(
                    {
                        "gpu": gpu,
                        "batch": batch,
                        "seqlen": seqlen,
                        "heads": args.heads,
                        "num_splits": splits,
                        "config": config,
                        "bytes": nbytes,
                        "triton_us": t_triton,
                        "eager_us": t_eager,
                        "compile_us": t_compile,
                    }
                )
                csv_file.flush()
            del q, kv, seqlens, bt
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
