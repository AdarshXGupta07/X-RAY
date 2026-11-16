"""Shared test setup.

All unit tests run on the CPU. PyTorch's Adam/AdamW ``step()`` still runs a
GPU "graph capture" health check whenever a GPU exists, and that check opens
a CUDA context. On this shared 4 GB card (Ollama and the browser also use it)
that intermittently failed with "CUDA error: out of memory"; hiding the GPU
with CUDA_VISIBLE_DEVICES made the same call crash instead. The check is only
relevant to CUDA-graph capture, which no test uses, so it is switched off for
the test session only. Real training (``tbshift.training``) is unaffected.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _no_gpu_health_check(monkeypatch):
    monkeypatch.setattr(
        torch.optim.Optimizer, "_accelerator_graph_capture_health_check", lambda self: None
    )
