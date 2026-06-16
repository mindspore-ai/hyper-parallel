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
"""Integration tests: offload precision with real accelerator.

Verifies that offloading produces numerically identical results to
a reference forward+backward pass, across common module types.
"""

import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import pytest
import torch
from torch import nn

from hyper_parallel.auto_parallel.hyper_offload import OffloadConfig, OffloadSession, skip_offload


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
        assert p_ref.grad is not None, "ref param missing grad"
        torch.testing.assert_close(p_off.grad, p_ref.grad, rtol=rtol, atol=atol)


def _compare_params(
    model_ref: nn.Module,
    model_off: nn.Module,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> None:
    """Compare parameters of two models.

    Args:
        model_ref: Reference model (without offload).
        model_off: Offload model.
        rtol: Relative tolerance for parameter comparison.
        atol: Absolute tolerance for parameter comparison.
    """
    for p_ref, p_off in zip(model_ref.parameters(), model_off.parameters(), strict=False):
        torch.testing.assert_close(p_off, p_ref, rtol=rtol, atol=atol)


def _run_step(
    model: nn.Module,
    x: torch.Tensor,
    use_offload: bool,
    session: OffloadSession | None = None,
) -> tuple[float, list[torch.Tensor]]:
    """Run one forward+backward step, return loss and grads.

    Args:
        model: Module to run.
        x: Input tensor.
        use_offload: Whether to run inside an offload session.
        session: Offload session (required when use_offload=True).

    Returns:
        Tuple of (loss as float, list of gradient tensors).
    """
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    opt.zero_grad()
    if use_offload:
        assert session is not None
        with session:
            loss = model(x).sum()
            loss.backward()
    else:
        loss = model(x).sum()
        loss.backward()
    opt.step()
    grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
    return float(loss.item()), grads


def _warmup_offload(
    model: nn.Module, session: OffloadSession, bs: int, dim: int, device
) -> torch.Tensor:
    """Run a warmup forward+backward pass inside the offload session.

    Args:
        model: Module to warm up.
        session: Offload session.
        bs: Batch size.
        dim: Feature dimension.
        device: Target device.

    Returns:
        The random input tensor used for warmup.
    """
    x = torch.randn(bs, dim, device=device)
    with session:
        loss = model(x).sum()
        loss.backward()
    return x


def test_forward_output_matches():
    """Forward output with offload matches reference."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config_debug = OffloadConfig(max_resident_activation_mb=1)

    model_ref = nn.Sequential(nn.Linear(256, 256), nn.ReLU()).to(device)
    model_off = nn.Sequential(nn.Linear(256, 256), nn.ReLU()).to(device)
    model_off.load_state_dict(model_ref.state_dict())

    session = OffloadSession(config_debug)
    _warmup_offload(model_off, session, 16, 256, device)
    model_off.load_state_dict(model_ref.state_dict())

    x = torch.randn(16, 256, device=device)
    loss_ref, _ = _run_step(model_ref, x, use_offload=False)
    loss_off, _ = _run_step(model_off, x, use_offload=True, session=session)
    torch.testing.assert_close(loss_off, loss_ref)
    _compare_params(model_ref, model_off)


def test_gradient_close():
    """Gradients with offload match reference."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config_debug = OffloadConfig(max_resident_activation_mb=1)

    model_ref = nn.Sequential(nn.Linear(256, 256), nn.ReLU()).to(device)
    model_off = nn.Sequential(nn.Linear(256, 256), nn.ReLU()).to(device)
    model_off.load_state_dict(model_ref.state_dict())

    session = OffloadSession(config_debug)
    _warmup_offload(model_off, session, 16, 256, device)
    model_off.load_state_dict(model_ref.state_dict())

    x = torch.randn(16, 256, device=device)
    _, grads_ref = _run_step(model_ref, x, use_offload=False)
    _, grads_off = _run_step(model_off, x, use_offload=True, session=session)
    for g_ref, g_off in zip(grads_ref, grads_off, strict=False):
        assert g_off is not None, "offload grad is None"
        torch.testing.assert_close(g_off, g_ref, rtol=1e-4, atol=1e-6)


def test_gradient_close_three_layers():
    """Gradients match for a 3-layer MLP."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config_debug = OffloadConfig(max_resident_activation_mb=1)

    model_ref = nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 64),
    ).to(device)
    model_off = nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 64),
    ).to(device)
    model_off.load_state_dict(model_ref.state_dict())

    session = OffloadSession(config_debug)
    _warmup_offload(model_off, session, 32, 128, device)
    model_off.load_state_dict(model_ref.state_dict())

    x = torch.randn(32, 128, device=device)
    _run_step(model_ref, x, use_offload=False)
    _run_step(model_off, x, use_offload=True, session=session)
    _compare_grads(model_ref, model_off)


def test_inplace_relu_gradient_close():
    """In-place ReLU with offload preserves gradients."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config_debug = OffloadConfig(max_resident_activation_mb=1)

    model_ref = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True)).to(device)
    model_off = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True)).to(device)
    model_off.load_state_dict(model_ref.state_dict())

    session = OffloadSession(config_debug)
    _warmup_offload(model_off, session, 16, 256, device)
    model_off.load_state_dict(model_ref.state_dict())

    x = torch.randn(16, 256, device=device)
    _run_step(model_ref, x, use_offload=False)
    _run_step(model_off, x, use_offload=True, session=session)
    _compare_grads(model_ref, model_off)


def test_inplace_add_gradient_close():
    """In-place add_ with offload preserves gradients."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config_debug = OffloadConfig(max_resident_activation_mb=1)

    class AddBlock(nn.Module):
        """Module with in-place add."""

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(256, 256)

        def forward(self, x):
            x = self.linear(x)
            return x.add_(0.5)

    model_ref = AddBlock().to(device)
    model_off = AddBlock().to(device)
    model_off.load_state_dict(model_ref.state_dict())

    session = OffloadSession(config_debug)
    _warmup_offload(model_off, session, 16, 256, device)
    model_off.load_state_dict(model_ref.state_dict())

    x = torch.randn(16, 256, device=device)
    _run_step(model_ref, x, use_offload=False)
    _run_step(model_off, x, use_offload=True, session=session)
    _compare_grads(model_ref, model_off)


def test_view_op_gradient_close():
    """View ops (reshape) with offload preserve gradients."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config_debug = OffloadConfig(max_resident_activation_mb=1)

    model_ref = nn.Sequential(
        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128),
    ).to(device)
    model_off = nn.Sequential(
        nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128),
    ).to(device)
    model_off.load_state_dict(model_ref.state_dict())

    session = OffloadSession(config_debug)
    _warmup_offload(model_off, session, 16, 256, device)
    model_off.load_state_dict(model_ref.state_dict())

    x = torch.randn(16, 256, device=device)
    _run_step(model_ref, x, use_offload=False)
    _run_step(model_off, x, use_offload=True, session=session)
    _compare_grads(model_ref, model_off)


def test_skip_offload_with_model():
    """skip_offload region interleaved with model preserves gradients."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config_debug = OffloadConfig(max_resident_activation_mb=1)

    model_ref = nn.Sequential(nn.Linear(256, 256), nn.ReLU()).to(device)
    model_off = nn.Sequential(nn.Linear(256, 256), nn.ReLU()).to(device)
    model_off.load_state_dict(model_ref.state_dict())

    @skip_offload
    def metric_region(x: torch.Tensor) -> torch.Tensor:
        """Compute norm outside offload session.

        Args:
            x: Input tensor.

        Returns:
            Norm of the input tensor (detached).
        """
        return x.detach().norm()

    session = OffloadSession(config_debug)
    x_warm = torch.randn(32, 256, device=device)
    with session:
        out = model_off(x_warm)
        _ = metric_region(out)
        loss = out.sum()
        loss.backward()
    model_off.zero_grad()
    model_off.load_state_dict(model_ref.state_dict())

    x = torch.randn(16, 256, device=device)
    loss_ref, _ = _run_step(model_ref, x, use_offload=False)

    opt = torch.optim.SGD(model_off.parameters(), lr=0.1)
    opt.zero_grad()
    with session:
        out = model_off(x)
        _ = metric_region(out)
        loss = out.sum()
        loss.backward()
    opt.step()

    torch.testing.assert_close(float(loss.detach()), loss_ref, rtol=1e-4, atol=1e-6)
    _compare_params(model_ref, model_off)


def test_skip_offload_swish_gradient_flow():
    """skip_offload with swish preserves parameter gradients."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config_debug = OffloadConfig(max_resident_activation_mb=1)

    model_ref = nn.Sequential(nn.Linear(256, 256), nn.ReLU()).to(device)
    model_off = nn.Sequential(nn.Linear(256, 256), nn.ReLU()).to(device)
    model_off.load_state_dict(model_ref.state_dict())

    @skip_offload
    def swish(x: torch.Tensor) -> torch.Tensor:
        """Swish activation outside offload session.

        Args:
            x: Input tensor.

        Returns:
            ``x * sigmoid(x)``.
        """
        return x * torch.sigmoid(x)

    session = OffloadSession(config_debug)
    x = torch.randn(16, 256, device=device)

    # Warmup
    with session:
        _ = swish(model_off(x)).sum().backward()
    model_off.zero_grad()
    model_off.load_state_dict(model_ref.state_dict())

    # Reference
    loss_ref = swish(model_ref(x)).sum()
    loss_ref.backward()
    grads_ref = [p.grad.clone() for p in model_ref.parameters()]

    # Offload replay
    with session:
        loss_off = swish(model_off(x)).sum()
        loss_off.backward()
    grads_off = [p.grad.clone() for p in model_off.parameters()]

    for g_ref, g_off in zip(grads_ref, grads_off, strict=True):
        torch.testing.assert_close(g_off, g_ref, rtol=1e-4, atol=1e-6)


def test_skip_offload_identity_region_gradients():
    """Identity region (output == input) preserves gradients."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()
    config_debug = OffloadConfig(max_resident_activation_mb=1)

    model_ref = nn.Sequential(nn.Linear(256, 256)).to(device)
    model_off = nn.Sequential(nn.Linear(256, 256)).to(device)
    model_off.load_state_dict(model_ref.state_dict())

    @skip_offload
    def identity_region(x: torch.Tensor) -> torch.Tensor:
        """Identity function outside offload session.

        Args:
            x: Input tensor.

        Returns:
            The input tensor unchanged.
        """
        return x

    session = OffloadSession(config_debug)
    x = torch.randn(16, 256, device=device)

    # Warmup
    with session:
        _ = identity_region(model_off(x)).sum().backward()
    model_off.zero_grad()
    model_off.load_state_dict(model_ref.state_dict())

    # Reference
    loss_ref = identity_region(model_ref(x)).sum()
    loss_ref.backward()
    grads_ref = [p.grad.clone() for p in model_ref.parameters()]

    # Offload replay
    with session:
        loss_off = identity_region(model_off(x)).sum()
        loss_off.backward()
    grads_off = [p.grad.clone() for p in model_off.parameters()]

    for g_ref, g_off in zip(grads_ref, grads_off, strict=True):
        torch.testing.assert_close(g_off, g_ref, rtol=1e-4, atol=1e-6)


def test_buffer_replaced_by_pre_hook_produces_correct_gradients():
    """When a forward pre-hook replaces a buffer, gradients should still be correct."""
    if not torch.accelerator.is_available():
        pytest.skip("accelerator is not available")
    device = torch.accelerator.current_accelerator()

    class BufferMatmul(nn.Module):
        """Module using a buffer."""

        def __init__(self, device: torch.device):
            super().__init__()
            self.register_buffer("proj", torch.eye(16, device=device))

        def forward(self, x):
            return x @ self.proj

    model_ref = BufferMatmul(device).to(device)
    model_off = BufferMatmul(device).to(device)
    model_off.load_state_dict(model_ref.state_dict())

    def replace_buffer(module, args):  # pylint: disable=unused-argument
        """Replace buffer with random values.

        Args:
            module: The module whose buffer to replace.
            args: Arguments to the forward call (unused).
        """
        with torch._C._DisableTorchDispatch():  # pylint: disable=protected-access
            module.proj = torch.randn_like(module.proj)

    model_off.register_forward_pre_hook(replace_buffer)
    model_ref.register_forward_pre_hook(replace_buffer)

    session = OffloadSession(OffloadConfig())
    x = torch.randn(4, 16, device=device, requires_grad=True)

    # Warmup
    with session:
        model_off(x).sum().backward()
    model_off.zero_grad()

    # Reference
    loss_ref = model_ref(x).sum()
    loss_ref.backward()
    grads_ref = [p.grad.clone() for p in model_ref.parameters()]

    # Offload replay
    with session:
        loss_off = model_off(x).sum()
        loss_off.backward()
    grads_off = [p.grad.clone() for p in model_off.parameters()]

    for g_ref, g_off in zip(grads_ref, grads_off, strict=True):
        torch.testing.assert_close(g_off, g_ref, rtol=1e-4, atol=1e-6)
