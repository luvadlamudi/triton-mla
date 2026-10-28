import os

import pytest
import torch

INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"


@pytest.fixture
def device():
    if INTERPRET:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (or run with TRITON_INTERPRET=1)")
    return torch.device("cuda")


@pytest.fixture
def dtype(device):
    return torch.float32 if device.type == "cpu" else torch.bfloat16
