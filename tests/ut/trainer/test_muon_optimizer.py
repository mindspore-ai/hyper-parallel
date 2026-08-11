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
"""Unit tests for the YAML-targeted mixed Muon and AdamW optimizer."""

from unittest.mock import patch

from torch import nn

from hyper_models.components.optim import AdamW, Muon


class _MixedModel(nn.Module):
    """Small model containing parameters routed to both optimizer children."""

    def __init__(self) -> None:
        """Create matrix, normalization, embedding, and head parameters."""
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8)

    def forward(self, inputs):
        """Apply the matrix layer used by optimizer-routing tests."""
        return self.linear(inputs)


def test_split_muon_and_adamw_parameters() -> None:
    """
    Feature: Mixed Muon and AdamW parameter routing.
    Description: Route matrix, bias, norm, embedding, and head parameters by policy.
    Expectation: Every trainable parameter belongs to exactly one optimizer child.
    """
    model = _MixedModel()

    muon_params, adamw_params, muon_names, adamw_names = Muon.split_muon_adamw_params(model)

    assert muon_names == ["linear.weight"]
    assert set(adamw_names) == {
        "linear.bias",
        "norm.weight",
        "norm.bias",
        "embed.weight",
        "lm_head.weight",
        "lm_head.bias",
    }
    assert {id(param) for param in muon_params}.isdisjoint({id(param) for param in adamw_params})
    assert len(muon_params) + len(adamw_params) == len(list(model.parameters()))


def test_extra_keyword_routes_matrix_parameter_to_adamw() -> None:
    """
    Feature: Configurable AdamW fallback routing.
    Description: Add a case-insensitive parameter-name keyword for a matrix parameter.
    Expectation: The matching matrix parameter is routed from Muon to AdamW.
    """
    model = _MixedModel()

    _, _, muon_names, adamw_names = Muon.split_muon_adamw_params(
        model,
        extra_adamw_name_keywords=["LINEAR"],
    )

    assert "linear.weight" not in muon_names
    assert "linear.weight" in adamw_names


def test_adamw_builds_decay_and_no_decay_groups() -> None:
    """
    Feature: AdamW decay and no-decay parameter groups.
    Description: Build AdamW with bias and norm names excluded from weight decay.
    Expectation: Parameters are assigned to groups with the configured and zero decay values.
    """
    model = _MixedModel()
    adamw_config = {"adamw_lr": 1e-4, "adamw_weight_decay": 0.07}
    runtime = object()

    with patch(
        "hyper_models.components.optim.optimizer.optimizer.get_hyper_optimizer",
        return_value=runtime,
    ) as build_core_optimizer:
        component = AdamW(
            model=model,
            adamw_config=adamw_config,
            no_decay_params=["bias", "norm"],
        )

    assert component.get_optimizer() is runtime
    groups = build_core_optimizer.call_args.kwargs["adamw_params"]
    assert [group["weight_decay"] for group in groups] == [0.07, 0.0]
    decay_ids = {id(parameter) for parameter in groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}
    assert id(model.linear.weight) in decay_ids
    assert id(model.linear.bias) in no_decay_ids
    assert id(model.norm.weight) in no_decay_ids


def test_empty_no_decay_list_keeps_all_parameters_in_decay_group() -> None:
    """
    Feature: Explicit empty no-decay policy.
    Description: Build AdamW groups with no excluded parameter-name keywords.
    Expectation: Every trainable parameter remains in the decay group.
    """
    model = _MixedModel()

    groups, _ = AdamW.get_adamw_param_groups(
        model,
        weight_decay=0.03,
        no_decay_params=[],
    )

    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0.03
    assert len(groups[0]["params"]) == len(list(model.parameters()))


def test_muon_passes_prefixed_configs_directly_to_core() -> None:
    """
    Feature: Muon component integration with the core optimizer.
    Description: Build mixed optimizer children from prefixed Muon and AdamW configurations.
    Expectation: Core receives unchanged configs and disjoint routed parameter groups.
    """
    model = _MixedModel()
    muon_config = {"muon_lr": 1e-3, "muon_ns_steps": 5}
    adamw_config = {"adamw_lr": 1e-4, "adamw_weight_decay": 0.01}
    runtime = object()

    with patch(
        "hyper_models.components.optim.optimizer.optimizer.get_hyper_optimizer",
        return_value=runtime,
    ) as build_core_optimizer:
        component = Muon(
            model=model,
            muon_config=muon_config,
            adamw_config=adamw_config,
            no_decay_params=["bias", "norm"],
        )

    assert component.get_optimizer() is runtime
    call = build_core_optimizer.call_args.kwargs
    assert call["model"] is model
    assert call["muon_kwargs"] is muon_config
    assert call["adamw_kwargs"] is adamw_config
    assert len(call["muon_params"]) == 1
    assert [group["weight_decay"] for group in call["adamw_params"]] == [0.01, 0.0]
    adamw_param_ids = {
        id(parameter)
        for group in call["adamw_params"]
        for parameter in group["params"]
    }
    assert len(adamw_param_ids) == 6
    assert id(model.linear.weight) not in adamw_param_ids
