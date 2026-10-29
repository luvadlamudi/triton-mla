"""Plot benchmark CSVs from `triton-mla-bench --csv`.

    python scripts/plot_bench.py results/bench_h100.csv [results/bench_b200.csv] --out assets
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GPUS = ["H100", "H200", "B200"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("batch", "seqlen", "heads", "num_splits", "bytes"):
            r[k] = int(r[k])
        for k in ("triton_us", "eager_us", "compile_us"):
            r[k] = float(r[k]) if r[k] else None
    return rows


def short_gpu(name):
    return next((k for k in GPUS if k in name), name)


def seqlen_axis(ax, seqlens):
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqlens)
    ax.set_xticklabels([f"{s // 1024}K" for s in seqlens])
    ax.set_xlabel("context length (tokens)")


def plot_speedup(datasets, out):
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4), sharey=True, squeeze=False)
    for ax, (gpu, rows) in zip(axes[0], datasets):
        by_batch = defaultdict(list)
        for r in rows:
            by_batch[r["batch"]].append(r)
        seqlens = sorted({r["seqlen"] for r in rows})
        ax.axhline(1.0, color=INK_2, linewidth=1, linestyle="--")
        ends = []
        for color, (batch, rs) in zip(SERIES, sorted(by_batch.items())):
            rs.sort(key=lambda r: r["seqlen"])
            x = [r["seqlen"] for r in rs]
            y = [r["compile_us"] / r["triton_us"] for r in rs]
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6, label=f"batch {batch}")
            ends.append((y[-1], x[-1]))
        ymax = max(y for y, _ in ends)
        last = None
        for y, x in sorted(ends, reverse=True):
            ty = y if last is None else min(y, last - 0.07 * ymax)
            ax.annotate(f"{y:.2f}x", (x, y), xytext=(x * 1.12, ty), textcoords="data", va="center",
                        color=INK_2, fontsize=9, ha="left")
            last = ty
        seqlen_axis(ax, seqlens)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{gpu}: speedup over torch.compile", loc="left")
    axes[0][0].set_ylabel("torch.compile time / Triton time")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.08, 0.97, 1))
    fig.savefig(os.path.join(out, "speedup.png"), dpi=160)


def plot_throughput(gpu, rows, out):
    """Achieved tensor-core TFLOPS. With 128 heads every KV byte feeds ~240 FLOPs,
    so MLA decode sits near the compute roof and TFLOPS is the honest yardstick."""
    batches = sorted({r["batch"] for r in rows})
    seqlens = sorted({r["seqlen"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(3.6 * len(batches), 3.8), sharey=True, squeeze=False)
    impls = [("triton_us", "Triton (fused)"), ("compile_us", "torch.compile"), ("eager_us", "PyTorch eager")]
    for ax, batch in zip(axes[0], batches):
        rs = sorted((r for r in rows if r["batch"] == batch), key=lambda r: r["seqlen"])
        for color, (key, label) in zip(SERIES, impls):
            pts = [(r["seqlen"], r["batch"] * r["seqlen"] * r["heads"] * 2 * 1088 / r[key] / 1e6) for r in rs if r[key]]
            if pts:
                ax.plot(*zip(*pts), color=color, linewidth=2, marker="o", markersize=6, label=label)
        seqlen_axis(ax, seqlens)
        ax.set_title(f"batch {batch}", loc="left")
    top = max(line.get_ydata().max() for ax in axes[0] for line in ax.get_lines())
    axes[0][0].set_ylim(0, top * 1.1)
    axes[0][0].set_ylabel("achieved TFLOPS (bf16)")
    axes[0][0].legend(loc="upper left")
    fig.suptitle(f"{gpu}: MLA decode, 128 heads, bf16, paged KV cache", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"tflops_{gpu.lower()}.png"), dpi=160)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--out", default="assets")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    datasets = []
    for path in args.csvs:
        rows = load(path)
        datasets.append((short_gpu(rows[0]["gpu"]), rows))
    plot_speedup(datasets, args.out)
    for gpu, rows in datasets:
        plot_throughput(gpu, rows, args.out)


if __name__ == "__main__":
    main()
