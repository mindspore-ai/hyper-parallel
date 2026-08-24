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
"""CPU contracts for manifold hyper-connection module replacements."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from hyper_parallel.auto_models.components.model_transform import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)
from hyper_parallel.auto_models.modules import MhcPostModule, MhcPostProcessModule, MhcPreModule
from hyper_parallel.auto_models.ops import mhc_post, mhc_pre
from hyper_parallel.auto_models.ops.mhc_post import mhc_post_process

mhc_pre_module = importlib.import_module("hyper_parallel.auto_models.ops.mhc_pre")


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        mhc_num_stream=2,
        mhc_use_gamma=False,
        mhc_recur_norm=3,
        mhc_hpre_renorm=False,
        use_mhc_ascendc_pre=False,
        use_mhc_ascendc_post=False,
    )


class MhcPreSource(nn.Module):
    def __init__(self) -> None:
        """Build the parameter layout expected by MhcPreModule."""
        super().__init__()
        self.config = _config()
        self.num_stream = 2
        self.phi = nn.Linear(8, 8, bias=False)
        self.branch_alpha = nn.Parameter(torch.ones(3))
        self.branch_beta = nn.Parameter(torch.zeros(8))
        self.hc_eps = 1e-6
        self.norm_eps = 1e-6

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply the eager MHC pre function."""
        return mhc_pre(
            x,
            self.phi.weight,
            self.branch_alpha,
            self.branch_beta,
            self.num_stream,
            self.config.mhc_recur_norm,
            self.norm_eps,
            self.hc_eps,
        )


class MhcPostSource(nn.Module):
    def __init__(self) -> None:
        """Build the configuration expected by MhcPostModule."""
        super().__init__()
        self.config = _config()
        self.num_stream = 2

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the eager MHC post function."""
        return mhc_post(x, residual, h_post, h_res, self.num_stream)


class MhcPostProcessSource(nn.Module):
    def __init__(self) -> None:
        """Build the parameter layout expected by MhcPostProcessModule."""
        super().__init__()
        self.config = _config()
        self.num_stream = 2
        self.phi = nn.Linear(8, 2, bias=False)
        self.branch_alpha = nn.Parameter(torch.ones(1))
        self.branch_beta = nn.Parameter(torch.zeros(2))
        self.hc_eps = 1e-6
        self.norm_eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the eager MHC final merge function."""
        return mhc_post_process(
            x,
            self.phi.weight,
            self.branch_alpha,
            self.branch_beta,
            self.num_stream,
            self.norm_eps,
            self.hc_eps,
        )


def _replace(source: nn.Module, replacement: type[nn.Module]) -> nn.Module:
    model = nn.Sequential(source)
    spec = ModuleReplacementSpec(
        match=("0",),
        factory=replacement,
        module_type=type(source),
        exact_type=True,
    )
    plan = compile_module_replacements(model, [spec])
    apply_module_replacements(model, plan)
    return model[0]


def test_mhc_replacements_match_eager_functions() -> None:
    """All three MHC modules must be usable by the generic replacement path."""
    torch.manual_seed(11)
    hidden_states = torch.randn(2, 3, 8)

    pre_source = MhcPreSource()
    expected_pre = pre_source(hidden_states)
    actual_pre = _replace(pre_source, MhcPreModule)(hidden_states)
    for actual, expected in zip(actual_pre[:3], expected_pre[:3]):
        torch.testing.assert_close(actual, expected)

    post_source = MhcPostSource()
    block_output = torch.randn(2, 3, 4)
    h_post = torch.randn(2, 3, 2)
    h_res = torch.randn(2, 3, 2, 2)
    expected_post = post_source(block_output, hidden_states, h_post, h_res)
    actual_post = _replace(post_source, MhcPostModule)(
        block_output, hidden_states, h_post, h_res
    )
    torch.testing.assert_close(actual_post, expected_post)

    final_source = MhcPostProcessSource()
    expected_final = final_source(hidden_states)
    actual_final = _replace(final_source, MhcPostProcessModule)(hidden_states)
    torch.testing.assert_close(actual_final, expected_final)


def test_mhc_pre_without_gamma_returns_no_gamma_gradient(monkeypatch: Any) -> None:
    """The custom autograd bridge must match an absent optional gamma input."""

    def fake_forward(
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Return tensors with the custom forward output contract."""
        del phi, alpha, bias, kwargs
        return x, x, x, x, x, x

    def fake_backward(
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Return a non-None gamma gradient to exercise the wrapper guard."""
        del args, kwargs
        bias = torch.zeros(3, dtype=x.dtype)
        return (
            torch.ones_like(x),
            torch.ones_like(phi),
            torch.ones_like(alpha),
            bias,
            torch.ones(1, dtype=x.dtype),
        )

    monkeypatch.setattr(mhc_pre_module, "omni_training_custom_ops", object())
    monkeypatch.setattr(
        torch.ops.custom,
        "npu_manifold_constrained_hyper_connection_pre",
        fake_forward,
    )
    monkeypatch.setattr(
        torch.ops.custom,
        "npu_manifold_constrained_hyper_connection_pre_grad",
        fake_backward,
    )
    x = torch.randn(2, 3, requires_grad=True)
    phi = torch.randn(4, 3, requires_grad=True)
    alpha = torch.randn(3, requires_grad=True)
    bias = torch.randn(3, requires_grad=True)

    outputs = mhc_pre_module._MhcPre.apply(x, phi, alpha, bias, None, True, 1e-6, 1e-6)
    sum(output.sum() for output in outputs).backward()

    assert x.grad is not None
    assert phi.grad is not None
    assert alpha.grad is not None
    assert bias.grad is not None
