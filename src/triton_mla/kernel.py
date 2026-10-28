"""Fused split-KV decode kernel for DeepSeek Multi-head Latent Attention (MLA).

At decode time MLA is run in "weight-absorbed" form: W_UK is folded into the
query and W_UV into the output projection, so every head attends directly over
the shared compressed KV cache. Each cached token is a single 576-dim vector:
512 latent dims (c_kv, used as both K and V) followed by 64 decoupled RoPE dims
(k_pe, used only for K). That makes MLA decode look like MQA with
head_dim_qk = 576 and head_dim_v = 512, and the kernel is purely bandwidth bound
on reading the latent cache.

The kernel is flash-decoding style:
  stage 1: grid (batch, head_groups, kv_splits). Each program loads a block of
           query heads once, streams its slice of the KV cache (paged or
           contiguous), and keeps an online softmax in registers. Because all
           heads share one KV vector, each cache tile is loaded once and reused
           by BLOCK_H heads via tensor-core matmuls.
  stage 2: log-sum-exp merge of the per-split partial outputs.
"""

import math
import os

import torch
import triton
import triton.language as tl

_INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"
_LOG2E = 1.4426950408889634
_LN2 = tl.constexpr(0.6931471805599453)


def _stage1_configs():
    if _INTERPRET:
        return [triton.Config({"BLOCK_H": 16, "BLOCK_N": 16}, num_warps=4, num_stages=1)]
    configs = []
    for block_h in (16, 32, 64):
        for block_n in (16, 32, 64):
            for num_warps in (4, 8):
                for num_stages in (1, 2, 3):
                    # [BLOCK_H, 512] fp32 accumulator must fit in registers.
                    if block_h * 512 // (num_warps * 32) > 256:
                        continue
                    configs.append(
                        triton.Config(
                            {"BLOCK_H": block_h, "BLOCK_N": block_n},
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    )
    return configs


@triton.autotune(
    configs=_stage1_configs(),
    key=["num_heads", "NUM_SPLITS", "PAGED", "PAGE_SIZE", "D_LATENT", "D_ROPE"],
)
@triton.jit
def _mla_decode_stage1(
    Q,
    KV,
    BlockTable,
    SeqLens,
    MidO,
    MidLse,
    qk_scale,
    stride_q_b,
    stride_q_h,
    stride_kv_blk,
    stride_kv_tok,
    stride_bt_b,
    stride_mo_b,
    stride_mo_h,
    stride_mo_s,
    stride_ml_b,
    stride_ml_h,
    num_heads,
    PAGED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    D_LATENT: tl.constexpr,
    D_ROPE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    seq_len = tl.load(SeqLens + pid_b)
    split_len = tl.cdiv(seq_len, NUM_SPLITS)
    start = pid_s * split_len
    end = tl.minimum(start + split_len, seq_len)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_c = tl.arange(0, D_LATENT)
    offs_r = tl.arange(0, D_ROPE)

    q_row = Q + pid_b.to(tl.int64) * stride_q_b + offs_h[:, None] * stride_q_h
    q_nope = tl.load(q_row + offs_c[None, :], mask=mask_h[:, None], other=0.0)
    q_pe = tl.load(q_row + D_LATENT + offs_r[None, :], mask=mask_h[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_LATENT], dtype=tl.float32)

    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < end
        if PAGED:
            page = tl.load(
                BlockTable + pid_b * stride_bt_b + offs_n // PAGE_SIZE, mask=mask_n, other=0
            ).to(tl.int64)
            kv_row = KV + page * stride_kv_blk + (offs_n % PAGE_SIZE) * stride_kv_tok
        else:
            kv_row = KV + pid_b.to(tl.int64) * stride_kv_blk + offs_n.to(tl.int64) * stride_kv_tok

        c_kv = tl.load(kv_row[:, None] + offs_c[None, :], mask=mask_n[:, None], other=0.0)
        k_pe = tl.load(kv_row[:, None] + D_LATENT + offs_r[None, :], mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q_nope, tl.trans(c_kv))
        qk += tl.dot(q_pe, tl.trans(k_pe))
        qk = tl.where(mask_n[None, :], qk * qk_scale, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(c_kv.dtype), c_kv)
        m_i = m_new

    has_tokens = l_i > 0
    out = acc / tl.where(has_tokens, l_i, 1.0)[:, None]
    lse = tl.where(has_tokens, m_i + tl.log2(tl.where(has_tokens, l_i, 1.0)), float("-inf"))

    mo_ptr = (
        MidO
        + pid_b.to(tl.int64) * stride_mo_b
        + offs_h[:, None] * stride_mo_h
        + pid_s * stride_mo_s
        + offs_c[None, :]
    )
    tl.store(mo_ptr, out, mask=mask_h[:, None])
    tl.store(MidLse + pid_b * stride_ml_b + offs_h * stride_ml_h + pid_s, lse, mask=mask_h)


@triton.jit
def _mla_decode_stage2(
    MidO,
    MidLse,
    O,
    Lse,
    stride_mo_b,
    stride_mo_h,
    stride_mo_s,
    stride_ml_b,
    stride_ml_h,
    stride_o_b,
    stride_o_h,
    stride_lse_b,
    D_LATENT: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLITS_P2: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_s = tl.arange(0, SPLITS_P2)
    offs_c = tl.arange(0, D_LATENT)

    ml_ptr = MidLse + pid_b * stride_ml_b + pid_h * stride_ml_h
    lse = tl.load(ml_ptr + offs_s, mask=offs_s < NUM_SPLITS, other=float("-inf"))
    m = tl.max(lse, 0)
    m = tl.where(m == float("-inf"), 0.0, m)

    mo_ptr = MidO + pid_b.to(tl.int64) * stride_mo_b + pid_h * stride_mo_h + offs_c
    acc = tl.zeros([D_LATENT], dtype=tl.float32)
    w_sum = 0.0
    for s in range(NUM_SPLITS):
        w = tl.exp2(tl.load(ml_ptr + s) - m)
        acc += w * tl.load(mo_ptr + s * stride_mo_s)
        w_sum += w

    out = acc / tl.where(w_sum > 0, w_sum, 1.0)
    tl.store(O + pid_b.to(tl.int64) * stride_o_b + pid_h * stride_o_h + offs_c, out.to(O.dtype.element_ty))
    final_lse = tl.where(w_sum > 0, (m + tl.log2(tl.where(w_sum > 0, w_sum, 1.0))) * _LN2, float("-inf"))
    tl.store(Lse + pid_b * stride_lse_b + pid_h, final_lse)


def _num_sms(device):
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 132


def get_num_splits(batch, num_heads, max_seqlen, num_sms, min_tokens_per_split=256):
    """Pick enough KV splits to fill the GPU without making splits too short."""
    head_groups = triton.cdiv(num_heads, 64)
    target = triton.cdiv(2 * num_sms, max(batch * head_groups, 1))
    max_by_len = max(triton.cdiv(max_seqlen, min_tokens_per_split), 1)
    splits = max(min(target, max_by_len, 64), 1)
    return triton.next_power_of_2(splits)


def mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor = None,
    head_dim_v: int = 512,
    softmax_scale: float = None,
    num_splits: int = None,
    return_lse: bool = False,
):
    """Weight-absorbed MLA decode attention over a shared latent KV cache.

    Args:
        q: [batch, num_heads, head_dim_v + rope_dim]. The first head_dim_v dims
            are the W_UK-absorbed nope query, the rest are the RoPE query.
        kv_cache: paged [num_blocks, page_size, head_dim_v + rope_dim] when
            block_table is given, otherwise contiguous [batch, max_seqlen, ...].
            A singleton kv-head dim ([..., 1, D], FlashMLA layout) is accepted.
        cache_seqlens: [batch] int32 number of valid tokens per sequence.
        block_table: optional [batch, max_blocks_per_seq] int32 page indices.
        head_dim_v: latent rank (512 for DeepSeek-V2/V3).
        softmax_scale: defaults to q.shape[-1] ** -0.5. DeepSeek uses
            (qk_nope_dim + qk_rope_dim) ** -0.5 times a YaRN mscale.
        num_splits: number of KV splits. If None it is picked from the batch
            size and max sequence length (requires one host sync); pass it
            explicitly for CUDA-graph capture.
        return_lse: also return the natural-log log-sum-exp per head.

    Returns:
        out [batch, num_heads, head_dim_v] (and lse [batch, num_heads] fp32).
    """
    if kv_cache.dim() == 4:
        assert kv_cache.shape[2] == 1, "MLA uses a single shared latent KV head"
        kv_cache = kv_cache.squeeze(2)
    assert q.dim() == 3 and kv_cache.dim() == 3
    batch, num_heads, d_qk = q.shape
    d_rope = d_qk - head_dim_v
    assert kv_cache.shape[-1] == d_qk
    assert q.stride(-1) == 1 and kv_cache.stride(-1) == 1
    assert triton.next_power_of_2(head_dim_v) == head_dim_v
    assert triton.next_power_of_2(d_rope) == d_rope
    assert q.dtype == kv_cache.dtype

    if softmax_scale is None:
        softmax_scale = d_qk ** -0.5
    cache_seqlens = cache_seqlens.to(torch.int32)

    paged = block_table is not None
    if paged:
        block_table = block_table.to(torch.int32)
        page_size = kv_cache.shape[1]
        stride_bt_b = block_table.stride(0)
    else:
        assert kv_cache.shape[0] == batch
        block_table = cache_seqlens
        page_size = 1
        stride_bt_b = 0

    if num_splits is None:
        max_seqlen = int(cache_seqlens.max().item()) if batch > 0 else 0
        num_splits = get_num_splits(batch, num_heads, max_seqlen, _num_sms(q.device))

    mid_o = torch.empty(batch, num_heads, num_splits, head_dim_v, dtype=torch.float32, device=q.device)
    mid_lse = torch.empty(batch, num_heads, num_splits, dtype=torch.float32, device=q.device)
    out = torch.empty(batch, num_heads, head_dim_v, dtype=q.dtype, device=q.device)
    lse = torch.empty(batch, num_heads, dtype=torch.float32, device=q.device)

    def grid(meta):
        return (batch, triton.cdiv(num_heads, meta["BLOCK_H"]), num_splits)

    _mla_decode_stage1[grid](
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        mid_o,
        mid_lse,
        softmax_scale * _LOG2E,
        q.stride(0),
        q.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        stride_bt_b,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        num_heads,
        PAGED=paged,
        PAGE_SIZE=page_size,
        D_LATENT=head_dim_v,
        D_ROPE=d_rope,
        NUM_SPLITS=num_splits,
    )
    _mla_decode_stage2[(batch, num_heads)](
        mid_o,
        mid_lse,
        out,
        lse,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        D_LATENT=head_dim_v,
        NUM_SPLITS=num_splits,
        SPLITS_P2=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
    if return_lse:
        return out, lse
    return out
