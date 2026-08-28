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
"""Regression tests for tied parameters across TP sharding and FSDP unwrap."""

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

import hyper_parallel.auto_models.components.distributed.sharding.apply as sharding_apply_module
import hyper_parallel.auto_models.components.distributed.sharding_applier as sharding_applier_module
from hyper_parallel.core.dtensor.placement_types import Shard


class _FakeDTensorParameter(nn.Parameter):
    """Parameter subclass exposing the DTensor attributes used by sharding."""

    @staticmethod
    def __new__(  # pylint: disable=signature-differs
        cls,
        data: torch.Tensor,
        layout: Any,
    ) -> "_FakeDTensorParameter":
        parameter = nn.Parameter._make_subclass(cls, data, True)
        parameter._test_layout = layout  # pylint: disable=protected-access
        parameter.placements = layout.placements
        return parameter

    @property
    def layout(self) -> Any:
        """Return the test layout associated with this parameter."""
        return self._test_layout  # pylint: disable=protected-access

    def to_local(self) -> torch.Tensor:
        """Return the zero-copy local tensor view."""
        return torch.Tensor.detach(self)


class _TiedCausalLM(nn.Module):
    """Minimal causal LM exposing separate embedding and LM-head paths."""

    def __init__(self) -> None:
        """Create two public parameter paths that should remain tied."""
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


def test_tp_unwrap_preserves_tied_parameter_identity(monkeypatch: Any) -> None:
    """TP replication and local unwrap expose one Parameter to downstream FSDP."""
    model = _TiedCausalLM()
    layout = SimpleNamespace(
        placements=(Shard(0),),
        alias_placements=(Shard(0),),
        mesh=object(),
    )
    model.model.embed_tokens.weight = _FakeDTensorParameter(
        model.model.embed_tokens.weight.detach(),
        layout,
    )
    model.lm_head.weight = _FakeDTensorParameter(
        model.lm_head.weight.detach().clone(),
        layout,
    )
    monkeypatch.setattr(sharding_applier_module, "DTensor", _FakeDTensorParameter)
    monkeypatch.setattr(sharding_apply_module, "DTensor", _FakeDTensorParameter)

    sharding_applier_module._replicate_tied_weights(  # pylint: disable=protected-access
        model,
        [("model.embed_tokens.weight", "lm_head.weight")],
    )
    records = sharding_apply_module._local_params_context(model)  # pylint: disable=protected-access

    assert records == {
        "model.embed_tokens.weight": (Shard(0),),
        "lm_head.weight": (Shard(0),),
    }
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert model.lm_head.weight._sharding_spec is layout  # pylint: disable=protected-access
