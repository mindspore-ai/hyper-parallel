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
"""Focused pre-sharding generic replacement contracts."""

from types import SimpleNamespace

from torch import nn

from hyper_models._transformers.infrastructure import _apply_module_replacement_actions
from hyper_models.components.model_transform import module_replacement
from hyper_models.trainer.config import PlanOverride, Target


@module_replacement
def _identity_replacement(*, module, module_fqn, context):
    del module_fqn, context
    return module


def _setup():
    return SimpleNamespace(
        plan_overrides=[],
        mesh_context=SimpleNamespace(cp_size=1, ep_size=1),
    )


def test_generic_yaml_replacement_runs_without_feature_specific_runtime():
    setup = _setup()
    setup.plan_overrides = [
        PlanOverride(
            match="0",
            module_type="torch.nn.Linear",
            replace_module=Target(
                _identity_replacement,
                target_path=f"{__name__}._identity_replacement",
            ),
        )
    ]
    model = nn.Sequential(nn.Linear(4, 8))

    assert _apply_module_replacement_actions(model, setup) is model


def test_invalid_fqn_fails_during_pre_sharding_replacement_compilation():
    setup = _setup()
    setup.plan_overrides = [
        PlanOverride(
            match="missing",
            module_type="torch.nn.Linear",
            replace_module=Target(
                _identity_replacement,
                target_path=f"{__name__}._identity_replacement",
            ),
        )
    ]

    try:
        _apply_module_replacement_actions(nn.Sequential(nn.Linear(4, 8)), setup)
    except ValueError as exc:
        assert "matched no module" in str(exc)
    else:
        raise AssertionError("invalid replacement FQN must fail before sharding")
