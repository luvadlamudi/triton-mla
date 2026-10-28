import pytest
import torch

from conftest import INTERPRET
from triton_mla import mla_decode, mla_decode_ref, mla_decode_torch

D_LATENT, D_ROPE = 512, 64
HEADS = [16, 20] if INTERPRET else [16, 128]
SPLITS = [1, 4] if INTERPRET else [1, 4, 16]


def tol(dtype):
    return dict(atol=1e-4, rtol=1e-4) if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)


def make_paged(kv_dense, seqlens, page_size, generator):
    batch, max_len, d = kv_dense.shape
    blocks_per_seq = -(-max_len // page_size)
    num_blocks = batch * blocks_per_seq
    perm = torch.randperm(num_blocks, generator=generator).to(kv_dense.device)
    block_table = perm.view(batch, blocks_per_seq).int()
    padded = torch.zeros(batch, blocks_per_seq * page_size, d, dtype=kv_dense.dtype, device=kv_dense.device)
    padded[:, :max_len] = kv_dense
    kv_cache = torch.empty(num_blocks, page_size, d, dtype=kv_dense.dtype, device=kv_dense.device)
    kv_cache[block_table.long()] = padded.view(batch, blocks_per_seq, page_size, d)
    return kv_cache, block_table


@pytest.mark.parametrize("num_heads", HEADS)
@pytest.mark.parametrize("num_splits", SPLITS)
@pytest.mark.parametrize("paged", [False, True])
def test_matches_reference(device, dtype, num_heads, num_splits, paged):
    g = torch.Generator().manual_seed(0)
    seqlens = torch.tensor([0, 1, 37, 100] if INTERPRET else [0, 1, 513, 4096], dtype=torch.int32)
    batch, max_len = len(seqlens), int(seqlens.max())
    q = torch.randn(batch, num_heads, D_LATENT + D_ROPE, generator=g).to(device, dtype)
    kv = torch.randn(batch, max_len, D_LATENT + D_ROPE, generator=g).to(device, dtype)
    seqlens = seqlens.to(device)

    block_table = None
    if paged:
        kv, block_table = make_paged(kv, seqlens, 16 if INTERPRET else 64, g)

    scale = 192 ** -0.5
    out, lse = mla_decode(q, kv, seqlens, block_table, softmax_scale=scale, num_splits=num_splits, return_lse=True)
    ref_out, ref_lse = mla_decode_ref(q, kv, seqlens, block_table, softmax_scale=scale)

    torch.testing.assert_close(out.float(), ref_out.float(), **tol(dtype))
    valid = seqlens > 0
    torch.testing.assert_close(lse[valid], ref_lse[valid], atol=1e-3, rtol=1e-3)
    assert torch.all(out[~valid] == 0)


def test_default_splits_and_flashmla_layout(device, dtype):
    g = torch.Generator().manual_seed(1)
    seqlens = torch.tensor([50, 75], dtype=torch.int32, device=device)
    q = torch.randn(2, 16, D_LATENT + D_ROPE, generator=g).to(device, dtype)
    kv = torch.randn(2, 80, 1, D_LATENT + D_ROPE, generator=g).to(device, dtype)
    out = mla_decode(q, kv, seqlens)
    ref = mla_decode_torch(q, kv, seqlens)
    torch.testing.assert_close(out.float(), ref.float(), **tol(dtype))
