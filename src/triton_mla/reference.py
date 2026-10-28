"""PyTorch MLA decode implementations used as correctness oracle and baseline."""

import torch


def gather_kv(kv_cache, cache_seqlens, block_table=None):
    """Return a contiguous [batch, max_seqlen, D] view of a paged or dense cache."""
    if kv_cache.dim() == 4:
        kv_cache = kv_cache.squeeze(2)
    if block_table is None:
        return kv_cache
    batch = block_table.shape[0]
    return kv_cache[block_table.long()].reshape(batch, -1, kv_cache.shape[-1])


def mla_decode_ref(q, kv_cache, cache_seqlens, block_table=None, head_dim_v=512, softmax_scale=None):
    """Per-sequence fp32 reference. Slow but obviously correct."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    kv = gather_kv(kv_cache, cache_seqlens, block_table)
    out = torch.zeros(q.shape[0], q.shape[1], head_dim_v, dtype=torch.float32, device=q.device)
    lse = torch.full((q.shape[0], q.shape[1]), float("-inf"), dtype=torch.float32, device=q.device)
    for b in range(q.shape[0]):
        n = int(cache_seqlens[b])
        if n == 0:
            continue
        k = kv[b, :n].float()
        s = q[b].float() @ k.T * softmax_scale
        lse[b] = torch.logsumexp(s, dim=-1)
        out[b] = torch.softmax(s, dim=-1) @ k[:, :head_dim_v]
    return out.to(q.dtype), lse


def mla_decode_torch(q, kv_cache, cache_seqlens, block_table=None, head_dim_v=512, softmax_scale=None):
    """Batched, masked PyTorch implementation: the eager / torch.compile baseline."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    kv = gather_kv(kv_cache, cache_seqlens, block_table)
    scores = torch.matmul(q, kv.transpose(1, 2)).float() * softmax_scale
    pos = torch.arange(kv.shape[1], device=q.device)
    mask = pos[None, :] >= cache_seqlens[:, None]
    scores = scores.masked_fill(mask[:, None, :], float("-inf"))
    p = torch.softmax(scores, dim=-1).to(q.dtype)
    return torch.matmul(p, kv[..., :head_dim_v])
