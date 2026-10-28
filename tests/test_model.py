import pytest
import torch

from conftest import INTERPRET
from triton_mla import MLA, MLAConfig


@pytest.mark.parametrize("paged", [False, True])
def test_absorbed_decode_matches_naive(device, dtype, paged):
    torch.manual_seed(0)
    cfg = MLAConfig(hidden_size=256, num_heads=16, q_lora_rank=128)
    model = MLA(cfg).to(device, dtype)
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    model.q_norm.weight.data.fill_(1.0)
    model.kv_norm.weight.data.fill_(1.0)

    batch, max_len, page_size = 3, 48 if INTERPRET else 256, 16
    seqlens = torch.tensor([0, 5, max_len // 2], dtype=torch.int32, device=device)
    if paged:
        blocks = max_len // page_size
        block_table = torch.randperm(batch * blocks, device=device).view(batch, blocks).int()
        kv_cache = torch.randn(batch * blocks, page_size, cfg.kv_cache_dim, device=device, dtype=dtype)
    else:
        block_table = None
        kv_cache = torch.randn(batch, max_len, cfg.kv_cache_dim, device=device, dtype=dtype)

    x = torch.randn(batch, cfg.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        out, new_lens = model(x, kv_cache, seqlens, block_table)
        ref, _ = model.forward_naive(x, kv_cache, seqlens, block_table)

    assert torch.equal(new_lens, seqlens + 1)
    atol = 1e-4 if dtype == torch.float32 else 3e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=atol)
