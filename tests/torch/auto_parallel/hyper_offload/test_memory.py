# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Integration tests: memory limit enforcement with real accelerator.

Verifies that offloading reduces peak device memory consumption and
that online eviction during warmup preserves gradient correctness.
"""

import logging
import os
from typing import Any

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import pytest
import torch
from torch import nn

from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession


logger = logging.getLogger(__name__)


def _compare_grads(
    model_ref: nn.Module,
    model_off: nn.Module,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> None:
    """Compare gradients of two models.

    Args:
        model_ref: Reference model (without offload).
        model_off: Offload model.
        rtol: Relative tolerance for gradient comparison.
        atol: Absolute tolerance for gradient comparison.
    """
    for p_ref, p_off in zip(model_ref.parameters(), model_off.parameters(), strict=False):
        assert p_off.grad is not None, "offload param missing grad"
        torch.testing.assert_close(p_off.grad, p_ref.grad, rtol=rtol, atol=atol)


def _run_step_with_ref(
    model_ref: nn.Module,
    model_off: nn.Module,
    x: torch.Tensor,
    session: Any,
) -> tuple[int, int]:
    """Run one forward+backward on both models, verify gradients, return peak memory.

    Offload step runs first (clean CUDA state), then ref step.

    Args:
        model_ref: Reference model (without offload).
        model_off: Offload model.
        x: Input tensor.
        session: Offload session.

    Returns:
        Tuple of (peak_ref, peak_off).
    """
    # Offload step
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()
    opt_off = torch.optim.SGD(model_off.parameters(), lr=0.1)
    opt_off.zero_grad()
    with session:
        loss_off = model_off(x).sum()
        loss_off.backward()
    opt_off.step()
    peak_off = torch.accelerator.max_memory_allocated()

    # Ref step
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()
    opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.1)
    opt_ref.zero_grad()
    loss_ref = model_ref(x).sum()
    loss_ref.backward()
    opt_ref.step()
    peak_ref = torch.accelerator.max_memory_allocated()

    logger.info("peak_ref: %.2f MiB, peak_off: %.2f MiB", peak_ref / 1024**2, peak_off / 1024**2)
    _compare_grads(model_ref, model_off)

    return peak_ref, peak_off


def _make_models(dim: int = 1024, n_layers: int = 4, device=None) -> tuple[nn.Module, nn.Module]:
    """Create reference and offload models with identical weights.

    Args:
        dim: Hidden dimension size.
        n_layers: Number of linear+relu pairs.
        device: Target device. If None, uses current accelerator.

    Returns:
        Tuple of (model_off, model_ref).
    """
    if device is None:
        device = torch.accelerator.current_accelerator()
    layers = []
    for _ in range(n_layers):
        layers.extend([nn.Linear(dim, dim), nn.ReLU()])
    model = nn.Sequential(*layers).to(device)
    model_ref = nn.Sequential(*layers).to(device)
    model_ref.load_state_dict(model.state_dict())
    return model, model_ref


def test_peak_memory_reduced_in_replay():
    """Peak memory should be lower with offload enabled (replay phase).

    Runs a warmup pass, then compares peak memory consumption between
    offload and reference on the replay pass.
    """
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config = OffloadConfig(max_resident_activation_mb=64)
    dim = 1024
    bs = 8192

    model_off, model_ref = _make_models(dim=dim, n_layers=4, device=device)
    session = OffloadSession(config)
    x = torch.randn(bs, dim, device=device)

    # Warmup
    _run_step_with_ref(model_ref, model_off, x, session)

    # Replay — compare peak memory
    model_off.load_state_dict(model_ref.state_dict())
    peak_ref, peak_off = _run_step_with_ref(model_ref, model_off, x, session)
    assert peak_off < peak_ref, f"peak_off={peak_off} >= peak_ref={peak_ref}"


def test_peak_memory_reduced_in_warmup():
    """Even the warmup (tracing) phase should reduce peak memory via online eviction."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config = OffloadConfig(max_resident_activation_mb=64)
    dim = 1024
    bs = 8192

    model_off, model_ref = _make_models(dim=dim, n_layers=4, device=device)
    session = OffloadSession(config)
    x = torch.randn(bs, dim, device=device)

    # First pass is warmup — verify memory reduction
    peak_ref, peak_off = _run_step_with_ref(model_ref, model_off, x, session)
    assert peak_off < peak_ref, f"peak_off={peak_off} >= peak_ref={peak_ref}"


def test_aggressive_eviction_preserves_gradients():
    """With a tight 1 MiB budget, gradients should still be correct.

    Total forward activations: ~1.06 MiB (bs=256, 3× Linear+ReLU + 1× Linear).
    Budget 1 MiB forces the warmup executor to evict at least one activation,
    verifying that gradient flow is preserved under eviction pressure.
    """
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    torch.manual_seed(0)
    config = OffloadConfig(max_resident_activation_mb=1)

    model_off = nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 64),
    ).to(device)
    model_ref = nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 64),
    ).to(device)
    model_ref.load_state_dict(model_off.state_dict())

    session = OffloadSession(config)
    x = torch.randn(256, 128, device=device)

    model_off.zero_grad()
    with session:
        loss_off = model_off(x).square().mean()
        loss_off.backward()

    model_ref.zero_grad()
    loss_ref = model_ref(x).square().mean()
    loss_ref.backward()

    _compare_grads(model_ref, model_off, rtol=1e-4, atol=1e-6)


def test_tight_budget_still_reduces_memory():
    """With a tight budget, memory should still be reduced vs no offload.

    Each activation is ~8 MiB (4096 * 512 * 4 bytes).  The 32 MiB budget
    allows at most 4 simultaneous resident activations, while the 3-layer
    MLP produces 6 activations total, forcing eviction during warmup.
    """
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config = OffloadConfig(max_resident_activation_mb=32)
    dim = 512
    bs = 4096

    model_off, model_ref = _make_models(dim=dim, n_layers=3, device=device)
    session = OffloadSession(config)
    x = torch.randn(bs, dim, device=device)

    # Warmup
    _run_step_with_ref(model_ref, model_off, x, session)

    # Replay
    model_off.load_state_dict(model_ref.state_dict())
    peak_ref, peak_off = _run_step_with_ref(model_ref, model_off, x, session)
    assert peak_off < peak_ref, f"peak_off={peak_off} >= peak_ref={peak_ref}"
