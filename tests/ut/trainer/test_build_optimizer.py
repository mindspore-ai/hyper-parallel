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
"""Unit tests for the YAML-targeted AdamW component."""

from unittest.mock import patch

from torch import nn

from hyper_models.components.optim import AdamW


class _Model(nn.Module):
    """Small model covering decay and no-decay parameter names."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)


def test_adamw_builds_core_optimizer_from_prefixed_config() -> None:
    """AdamW forwards its prefixed config and grouped parameters to core."""
    model = _Model()
    runtime = object()
    config = {
        "adamw_lr": 1e-4,
        "adamw_weight_decay": 0.01,
        "adamw_betas": (0.9, 0.999),
        "adamw_eps": 1e-8,
    }

    with patch(
        "hyper_models.components.optim.optimizer.optimizer.get_hyper_optimizer",
        return_value=runtime,
    ) as build_core:
        component = AdamW(
            model=model,
            adamw_config=config,
            no_decay_params=["bias", "norm"],
        )

    assert component.get_optimizer() is runtime
    assert build_core.call_args.kwargs["adamw_kwargs"] is config
    groups = build_core.call_args.kwargs["adamw_params"]
    assert [group["weight_decay"] for group in groups] == [0.01, 0.0]
