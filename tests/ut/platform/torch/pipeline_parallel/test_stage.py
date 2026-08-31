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
"""Unit tests for Torch pipeline-stage split backward."""

import os
from types import SimpleNamespace
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch

from hyper_parallel.platform.torch.pipeline_parallel.stage import PipelineStageBase


class _MixedInputModule(torch.nn.Module):
    """Module with a non-differentiable input before its activation."""

    def __init__(self) -> None:
        """Initialize the scalar trainable weight."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Combine a non-gradient ID tensor with a differentiable activation."""
        return self.weight * hidden + input_ids.float()


def _build_middle_stage():
    """Build a middle stage with deterministic forward and gradient buffers."""
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(2.0)
    stage = PipelineStageBase(module, stage_index=1, stage_num=3)
    activation = torch.tensor([[3.0]], requires_grad=True)
    stage.args_recv_info = {
        0: [SimpleNamespace(buffer=activation, requires_grad=True)],
    }
    stage.grad_recv_info = {
        0: [SimpleNamespace(buffer=torch.tensor([[4.0]]))],
    }
    stage.forward_one_chunk(0)
    return stage, module, activation


def test_split_backward_computes_dx_before_dw() -> None:
    """Input backward sends dx before weight backward accumulates parameter gradients."""
    stage, module, activation = _build_middle_stage()

    stage.backward_input_one_chunk(0)

    torch.testing.assert_close(activation.grad, torch.tensor([[8.0]]))
    torch.testing.assert_close(stage.bwd_cache[0][0], torch.tensor([[8.0]]))
    assert module.weight.grad is None

    stage.backward_weight_one_chunk(0)

    torch.testing.assert_close(module.weight.grad, torch.tensor([[12.0]]))
    assert stage.args_recv_info[0][0].buffer is None
    assert stage.grad_recv_info[0][0].buffer is None


def test_split_backward_caches_parameter_tuple() -> None:
    """The parameter tree is enumerated once across the dx and dw phases."""
    stage, module, _ = _build_middle_stage()
    original_parameters = module.parameters

    with patch.object(module, "parameters", wraps=original_parameters) as parameters:
        stage.backward_input_one_chunk(0)
        stage.backward_weight_one_chunk(0)

    parameters.assert_called_once_with()


def test_weight_backward_requires_input_backward_state() -> None:
    """A non-first stage cannot run dw before its matching dx phase."""
    stage, _, _ = _build_middle_stage()

    try:
        stage.backward_weight_one_chunk(0)
    except RuntimeError as exc:
        assert "dw called before dx" in str(exc)
    else:
        raise AssertionError("Expected backward_weight_one_chunk to reject a missing dx state.")


def test_split_backward_selects_inputs_by_gradient_edge_not_position() -> None:
    """
    Feature: Torch split-backward input selection.
    Description: Place an integer input before a differentiable activation.
    Expectation: Only the activation receives dx, while dw is deferred to the weight phase.
    """
    module = _MixedInputModule()
    stage = PipelineStageBase(module, stage_index=1, stage_num=3)
    input_ids = torch.tensor([7], dtype=torch.long)
    hidden = torch.tensor([3.0], requires_grad=True)
    stage.args_recv_info = {
        0: [
            SimpleNamespace(buffer=input_ids, requires_grad=False),
            SimpleNamespace(buffer=hidden, requires_grad=True),
        ],
    }
    stage.grad_recv_info = {
        0: [SimpleNamespace(buffer=torch.tensor([4.0]))],
    }
    stage.forward_one_chunk(0)

    stage.backward_input_one_chunk(0)

    assert input_ids.grad is None, f"Expected no input_ids gradient, got={input_ids.grad}"
    torch.testing.assert_close(hidden.grad, torch.tensor([8.0]))
    assert module.weight.grad is None, f"Expected weight gradient to remain deferred, got={module.weight.grad}"

    stage.backward_weight_one_chunk(0)

    torch.testing.assert_close(module.weight.grad, torch.tensor(12.0))
