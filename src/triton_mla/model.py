"""DeepSeek-V2/V3 style MLA layer with a weight-absorbed decode path."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import mla_decode
from .reference import gather_kv


@dataclass
class MLAConfig:
    hidden_size: int = 7168
    num_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    @property
    def qk_head_dim(self):
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_cache_dim(self):
        return self.kv_lora_rank + self.qk_rope_head_dim


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def apply_rope(x, positions, theta):
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    freqs = positions.float()[..., None] * inv_freq
    cos, sin = freqs.cos(), freqs.sin()
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


class MLA(nn.Module):
    """Multi-head Latent Attention.

    Only the compressed latent c_kv (kv_lora_rank) and the shared RoPE key
    (qk_rope_head_dim) are cached per token: 576 values instead of
    2 * num_heads * head_dim for standard MHA.
    """

    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.num_heads
        self.wq_a = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps)
        self.wq_b = nn.Linear(cfg.q_lora_rank, h * cfg.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(cfg.hidden_size, cfg.kv_cache_dim, bias=False)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps)
        self.wkv_b = nn.Linear(cfg.kv_lora_rank, h * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.wo = nn.Linear(h * cfg.v_head_dim, cfg.hidden_size, bias=False)
        self.softmax_scale = cfg.qk_head_dim ** -0.5

    def _project(self, x, positions):
        cfg = self.cfg
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(x.shape[0], cfg.num_heads, cfg.qk_head_dim)
        q_nope, q_pe = q.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
        q_pe = apply_rope(q_pe, positions[:, None], cfg.rope_theta)
        c_kv, k_pe = self.wkv_a(x).split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_pe = apply_rope(k_pe, positions, cfg.rope_theta)
        return q_nope, q_pe, c_kv, k_pe

    @staticmethod
    def _append_kv(kv_cache, c_kv, k_pe, cache_seqlens, block_table):
        entry = torch.cat([c_kv, k_pe], dim=-1).to(kv_cache.dtype)
        pos = cache_seqlens.long()
        rows = torch.arange(pos.shape[0], device=pos.device)
        if block_table is None:
            kv_cache[rows, pos] = entry
        else:
            page_size = kv_cache.shape[1]
            pages = block_table[rows, pos // page_size].long()
            kv_cache[pages, pos % page_size] = entry

    def _wkv_b(self):
        cfg = self.cfg
        w = self.wkv_b.weight.view(cfg.num_heads, cfg.qk_nope_head_dim + cfg.v_head_dim, cfg.kv_lora_rank)
        return w[:, : cfg.qk_nope_head_dim], w[:, cfg.qk_nope_head_dim :]

    def forward(self, x, kv_cache, cache_seqlens, block_table=None, num_splits=None):
        """One decode step for x [batch, hidden].

        Appends the new token to kv_cache in place and returns
        (output [batch, hidden], updated cache_seqlens).
        """
        q_nope, q_pe, c_kv, k_pe = self._project(x, cache_seqlens)
        self._append_kv(kv_cache, c_kv, k_pe, cache_seqlens, block_table)
        seqlens = cache_seqlens + 1

        w_uk, w_uv = self._wkv_b()
        q_abs = torch.einsum("bhd,hdc->bhc", q_nope, w_uk)
        q = torch.cat([q_abs, q_pe], dim=-1).to(kv_cache.dtype).contiguous()
        o = mla_decode(
            q,
            kv_cache,
            seqlens,
            block_table,
            head_dim_v=self.cfg.kv_lora_rank,
            softmax_scale=self.softmax_scale,
            num_splits=num_splits,
        )
        o = torch.einsum("bhc,hdc->bhd", o.to(w_uv.dtype), w_uv)
        return self.wo(o.flatten(1)), seqlens

    def forward_naive(self, x, kv_cache, cache_seqlens, block_table=None):
        """Same decode step without absorption: decompress the cache into per-head K/V."""
        cfg = self.cfg
        q_nope, q_pe, c_kv, k_pe = self._project(x, cache_seqlens)
        self._append_kv(kv_cache, c_kv, k_pe, cache_seqlens, block_table)
        seqlens = cache_seqlens + 1

        kv = gather_kv(kv_cache, seqlens, block_table).to(x.dtype)
        latent, kpe = kv.split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1)
        kvb = self.wkv_b(latent).view(*latent.shape[:2], cfg.num_heads, -1)
        k_nope, v = kvb.split([cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1)
        k = torch.cat([k_nope, kpe[:, :, None].expand(-1, -1, cfg.num_heads, -1)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)

        scores = torch.einsum("bhd,bshd->bhs", q.float(), k.float()) * self.softmax_scale
        mask = torch.arange(kv.shape[1], device=x.device)[None, :] >= seqlens[:, None]
        scores = scores.masked_fill(mask[:, None, :], float("-inf"))
        o = torch.einsum("bhs,bshd->bhd", scores.softmax(-1), v.float()).to(x.dtype)
        return self.wo(o.flatten(1)), seqlens
