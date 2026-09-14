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
"""Model-family shard-aware initialization contracts."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tests.common.mark_utils import arg_mark

from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.models.qwen3_5.adapter.init_weights import initialize_weights as initialize_qwen3_5_weights
from hyper_parallel.models.qwen3_5_moe.adapter.init_weights import (
    initialize_weights as initialize_qwen3_5_moe_weights,
)
from hyper_parallel.models.qwen3_next.adapter.init_weights import initialize_weights as initialize_qwen3_next_weights
from hyper_parallel.models.qwen4_exp.adapter.init_weights import initialize_weights as initialize_qwen4_exp_weights


class _FakeGatedDeltaNet(nn.Module):
    """Gated-delta module whose parameters represent a local head shard."""

    def __init__(self):
        super().__init__()
        self.num_v_heads = 8
        self.dt_bias = nn.Parameter(torch.empty(4))
        self.A_log = nn.Parameter(torch.empty(4))


class _FakeModel(nn.Module):
    """Minimal model exposing the Transformers initialization contract."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="fake", architectures=["FakeForCausalLM"])
        self.gated_delta = _FakeGatedDeltaNet()
        self.projection = nn.Linear(2, 2, bias=False)
        self.native_init_calls = 0
        self.emulate_unsafe_gated_delta_init = True

    def initialize_weights(self) -> None:
        """Initialize remaining state and emulate unsafe native gated-delta init."""
        self.native_init_calls += 1
        if self.emulate_unsafe_gated_delta_init and not getattr(
            self.gated_delta, "_is_hf_initialized", False
        ):
            self.gated_delta.A_log.copy_(torch.empty(self.gated_delta.num_v_heads))
        nn.init.constant_(self.projection.weight, 3.0)


_QWEN_INITIALIZERS = [
    (
        initialize_qwen3_5_weights,
        "hyper_parallel.models.qwen3_5.adapter.init_weights.Qwen3_5GatedDeltaNet",
    ),
    (
        initialize_qwen3_5_moe_weights,
        "hyper_parallel.models.qwen3_5_moe.adapter.init_weights.Qwen3_5MoeGatedDeltaNet",
    ),
    (
        initialize_qwen3_next_weights,
        "hyper_parallel.models.qwen3_next.adapter.init_weights.Qwen3NextGatedDeltaNet",
    ),
    (
        initialize_qwen4_exp_weights,
        "hyper_parallel.models.qwen4_exp.adapter.init_weights.Qwen4ExpTextGatedDeltaNet",
    ),
]


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("initializer,module_type_path", _QWEN_INITIALIZERS)
def test_qwen_initializer_uses_local_parameter_shape(initializer, module_type_path):
    """Each Qwen family initializes the local gated-delta shard before native state."""
    model = _FakeModel()

    torch.manual_seed(1234)
    with mock.patch(module_type_path, _FakeGatedDeltaNet):
        initializer(model)

    assert model.gated_delta.A_log.shape == (4,), "case: local_shape"
    assert torch.equal(model.gated_delta.dt_bias, torch.ones(4)), "case: dt_bias"
    a_values = model.gated_delta.A_log.exp()
    assert torch.all(a_values >= 0.01), "case: a_log_lower_bound"
    assert torch.all(a_values < 16.0), "case: a_log_upper_bound"
    assert getattr(model.gated_delta.dt_bias, "_is_hf_initialized", False), "case: dt_bias_initialized"
    assert getattr(model.gated_delta.A_log, "_is_hf_initialized", False), "case: a_log_initialized"
    assert getattr(model.gated_delta, "_is_hf_initialized", False), "case: hf_initialized"
    assert model.native_init_calls == 1, "case: remaining_native_init"
    assert torch.equal(model.projection.weight, torch.full((2, 2), 3.0)), "case: projection_init"


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_builder_dispatches_registered_initializer_instead_of_native():
    """A registered initializer owns initialization at the builder boundary."""
    model = _FakeModel()

    def initializer(target: nn.Module) -> None:
        """Initialize the model without using its native entry point."""
        nn.init.constant_(target.projection.weight, 5.0)

    provider = mock.Mock(return_value=initializer)
    with mock.patch(
        "hyper_parallel.models._transformers.model_builder._get_init_weights_provider",
        return_value=provider,
    ):
        _initialize_model_weights(model)

    provider.assert_called_once_with()
    assert model.native_init_calls == 0, "case: native_not_called_by_builder"
    assert torch.equal(model.projection.weight, torch.full((2, 2), 5.0)), "case: custom_init"


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_builder_falls_back_to_native_initialization():
    """Models without an adapter preserve direct native initialization."""
    model = _FakeModel()
    model.emulate_unsafe_gated_delta_init = False
    with mock.patch(
        "hyper_parallel.models._transformers.model_builder._get_init_weights_provider",
        return_value=None,
    ):
        _initialize_model_weights(model)

    assert model.native_init_calls == 1, "case: native_call_count"
