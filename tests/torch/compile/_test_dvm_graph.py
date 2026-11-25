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
"""Worker-side graph-mode DVM precision regression.

The DVM Inductor backend is selected at torch_npu import time, so the
environment must be configured before torch/torch_npu/hyper_parallel are
imported. The launcher only spawns this worker; this module owns the env setup.

The same toy model is trained three ways from identical weights:

- eager (uncompiled reference),
- joint graph with Inductor disabled (graph baseline),
- joint graph with ``inductor_mode="regional"`` + ``inductor_backend="dvm"``.

Loss, gradients and updated parameters must agree, and the generated Inductor
artifact must contain DVM kernel code (not just a silent fallback).
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
    prefix="hyper_parallel_dvm_graph_st_"
)

# pylint: disable=wrong-import-position,import-outside-toplevel,protected-access
import pytest
import torch
from torch import nn

from hyper_parallel.compile import GraphCompiler, PassConfig
from tests.torch.utils import get_device_type

_BATCH = 8
_DIM = 64
_STEPS = 3
_LR = 0.01


class _ElementwiseModel(nn.Module):
    """Tiny trainable model built only from DVM-supported elementwise ops."""

    def __init__(self, dim: int) -> None:
        """Build one trainable weight/bias pair."""
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim) * 0.1)
        self.bias = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run ``exp``/``log``/mul/add pointwise math."""
        hidden = torch.exp(inputs * self.weight + self.bias)
        hidden = torch.log(hidden + 1.0)
        return hidden * inputs + hidden


def _train_fn(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Return scalar MSE so the joint graph carries a mean-free reduction."""
    squared = (model(inputs) - targets) ** 2
    return squared.sum() / targets.numel()


def _make_batch(
    step: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one deterministic batch on the target device and dtype."""
    generator = torch.Generator().manual_seed(1234 + step)
    inputs = torch.randn((_BATCH, _DIM), generator=generator) * 0.1
    targets = torch.randn((_BATCH, _DIM), generator=generator) * 0.1
    return (
        inputs.to(device=device, dtype=dtype),
        targets.to(device=device, dtype=dtype),
    )


def _tolerances(dtype: torch.dtype) -> dict:
    """Return dtype-appropriate assert_close tolerances."""
    if dtype == torch.float32:
        return {"rtol": 1e-5, "atol": 1e-6}
    return {"rtol": 1e-2, "atol": 1e-3}


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype, label: str
) -> None:
    """Compare one loss/gradient/parameter value with a named failure message."""
    torch.testing.assert_close(
        actual,
        expected,
        **_tolerances(dtype),
        msg=lambda message: f"{label}: {message}",
    )


def _make_compiler(
    model: nn.Module, device: torch.device, *, use_dvm: bool
) -> GraphCompiler:
    """Build a single-card joint-graph compiler for the target model."""
    pass_config = PassConfig(
        fsdp_enabled=False,
        enable_overlap=False,
        inductor_mode="regional" if use_dvm else "off",
        inductor_backend="dvm" if use_dvm else "default",
    )
    return GraphCompiler(
        model,
        _train_fn,
        pass_config=pass_config,
        device=device,
    )


def _run_graph_step(
    compiler: GraphCompiler, inputs: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Run one graph forward/backward and return the scalar loss."""
    loss, _ = compiler.forward_backward(inputs=inputs, targets=targets)
    return loss.detach()


def _find_dvm_kernel_code(cache_dir: Path) -> Optional[Path]:
    """Return a generated Inductor artifact containing a DVM kernel decorator."""
    for path in cache_dir.rglob("*"):
        if not path.is_file() or path.suffix not in (".py", ".txt"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "@dvm.kernel" in text:
            return path
    return None


def _assert_dvm_backend(compiler: GraphCompiler) -> None:
    """Verify the DVM backend was active and generated a DVM kernel."""
    graph_module = compiler._joint_graph.graph_module  # pylint: disable=protected-access
    assert graph_module.meta.get("inductor_compiled") == "regional", (
        f"expected regional Inductor compilation, got "
        f"{graph_module.meta.get('inductor_compiled')!r}"
    )
    assert graph_module.meta.get("inductor_backend") == "dvm", (
        f"expected dvm backend meta, got {graph_module.meta.get('inductor_backend')!r}"
    )

    cache_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    kernel_file = _find_dvm_kernel_code(cache_dir)
    assert kernel_file is not None, (
        f"no DVM kernel code found under {cache_dir}; Inductor fell back "
        f"without fusing any DVM-supported region"
    )


def _compare_gradients(
    actual: nn.Module,
    graph: nn.Module,
    reference: nn.Module,
    dtype: torch.dtype,
    step: int,
) -> None:
    """Compare every DVM gradient with the graph and eager baselines."""
    for (name, actual_param), (_, graph_param), (_, ref_param) in zip(
        actual.named_parameters(),
        graph.named_parameters(),
        reference.named_parameters(),
    ):
        _assert_close(
            actual_param.grad,
            graph_param.grad,
            dtype,
            f"step {step} {name} dvm-vs-graph grad",
        )
        _assert_close(
            actual_param.grad,
            ref_param.grad,
            dtype,
            f"step {step} {name} dvm-vs-eager grad",
        )


def _compare_parameters(
    actual: nn.Module,
    graph: nn.Module,
    reference: nn.Module,
    dtype: torch.dtype,
    step: int,
) -> None:
    """Compare every updated DVM parameter with both baselines."""
    for (name, actual_param), (_, graph_param), (_, ref_param) in zip(
        actual.named_parameters(),
        graph.named_parameters(),
        reference.named_parameters(),
    ):
        _assert_close(
            actual_param,
            graph_param,
            dtype,
            f"step {step} {name} dvm-vs-graph param",
        )
        _assert_close(
            actual_param,
            ref_param,
            dtype,
            f"step {step} {name} dvm-vs-eager param",
        )


def _run_case(dtype: torch.dtype) -> None:
    """Compare DVM, plain joint graph and eager training over several steps."""
    if get_device_type() != "npu":
        pytest.skip("graph-mode DVM requires an Ascend NPU")

    import torch_npu  # noqa: F401  # pylint: disable=unused-import

    from hyper_parallel.compile.passes.inductor_pass import (
        _loaded_npu_inductor_backend,
    )

    try:
        loaded_backend = _loaded_npu_inductor_backend()
    except ImportError:
        loaded_backend = None
    if loaded_backend != "dvm":
        pytest.skip("torch_npu build does not provide the DVM Inductor backend")

    torch.manual_seed(0)
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("npu")

    reference = _ElementwiseModel(_DIM).to(device=device, dtype=dtype)
    graph = _ElementwiseModel(_DIM).to(device=device, dtype=dtype)
    dvm = _ElementwiseModel(_DIM).to(device=device, dtype=dtype)
    graph.load_state_dict(reference.state_dict())
    dvm.load_state_dict(reference.state_dict())

    graph_compiler = _make_compiler(graph, device, use_dvm=False)
    dvm_compiler = _make_compiler(dvm, device, use_dvm=True)

    optimizers = [
        torch.optim.SGD(target.parameters(), lr=_LR)
        for target in (reference, graph, dvm)
    ]
    for step in range(_STEPS):
        inputs, targets = _make_batch(step, device, dtype)
        for target in (reference, graph, dvm):
            target.zero_grad(set_to_none=True)

        ref_loss = _train_fn(reference, inputs, targets)
        ref_loss.backward()
        graph_loss = _run_graph_step(graph_compiler, inputs, targets)
        dvm_loss = _run_graph_step(dvm_compiler, inputs, targets)

        _assert_close(dvm_loss, graph_loss, dtype, f"step {step} dvm-vs-graph loss")
        _assert_close(
            dvm_loss, ref_loss.detach(), dtype, f"step {step} dvm-vs-eager loss"
        )
        _compare_gradients(dvm, graph, reference, dtype, step)

        for optimizer in optimizers:
            optimizer.step()
        _compare_parameters(dvm, graph, reference, dtype, step)

    _assert_dvm_backend(dvm_compiler)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_dvm_graph_precision(dtype: torch.dtype) -> None:
    """Feature: Graph-mode DVM Inductor backend.
    Description: Train the same toy model eager, as a plain joint graph, and
        with ``inductor_backend="dvm"`` from identical weights.
    Expectation: DVM loss, gradients and updated parameters match both
        baselines; Inductor generated DVM kernel code.
    """
    _run_case(dtype)
