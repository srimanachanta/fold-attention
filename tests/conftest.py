"""Test-suite hygiene for GPU sweeps.

The suite compiles many kernel configurations in one process. Without cleanup
between tests the CUDA context grows until compilation itself starts to fail,
which surfaces as `ptxas` crashes and spurious mismatches that do not reproduce
when a test runs alone. A bitwise claim must never rest on one of those.
"""

from __future__ import annotations

import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def _release_gpu_between_tests():
    yield
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
