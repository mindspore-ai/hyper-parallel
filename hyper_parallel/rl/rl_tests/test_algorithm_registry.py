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
"""Contract tests for MOLT-style algorithm registries and flat modules."""

from pathlib import Path
from typing import Any, Mapping

import pytest

import rl.algorithm as algorithm_package
from rl.algorithm import build_algorithm, register_algorithm
from rl.algorithm.advantage import (
    GAEAdvantageEstimator,
    GroupRelativeAdvantageEstimator,
    get_advantage_estimator,
    register_advantage_estimator,
)
from rl.algorithm.loss import (
    ALGORITHMS,
    ClippedPolicyObjective,
    get_policy_loss,
    register_policy_loss,
)


def test_algorithm_directory_contains_only_flat_core_modules() -> None:
    """Expose algorithm internals through three flat modules instead of subpackages."""
    algorithm_dir = Path(algorithm_package.__file__).parent
    assert {path.name for path in algorithm_dir.glob("*.py")} == {
        "__init__.py",
        "advantage.py",
        "loss.py",
        "reward.py",
    }
    assert not (algorithm_dir / "components").exists()
    assert not (algorithm_dir / "reward").is_dir()


def test_algorithm_registry_rejects_unknown_and_duplicate_names() -> None:
    """Keep algorithm plugin errors explicit and deterministic."""
    assert ALGORITHMS.names == ("grpo", "ppo")
    assert ALGORITHMS.build(
        "grpo",
        {"name": "grpo", "loss_aggregation": "token-mean"},
    ).name == "grpo"

    with pytest.raises(ValueError, match="Unknown algorithm 'missing'"):
        build_algorithm({"name": "missing", "loss_aggregation": "token-mean"})

    def build_duplicate(config: Mapping[str, Any]) -> Any:
        del config
        raise AssertionError("duplicate builder must not be called")

    with pytest.raises(ValueError, match="Algorithm is already registered: grpo"):
        register_algorithm("grpo")(build_duplicate)
    with pytest.raises(ValueError, match="Algorithm is already registered: grpo"):
        ALGORITHMS.register("grpo")(build_duplicate)


def test_advantage_registry_rejects_unknown_and_duplicate_names() -> None:
    """Select advantage estimators through a MOLT-style registry."""
    assert isinstance(
        get_advantage_estimator("grpo", epsilon=1.0e-5),
        GroupRelativeAdvantageEstimator,
    )
    assert isinstance(get_advantage_estimator("gae"), GAEAdvantageEstimator)

    with pytest.raises(ValueError, match="Unknown advantage estimator 'missing'"):
        get_advantage_estimator("missing")

    with pytest.raises(ValueError, match="Advantage estimator is already registered: grpo"):
        register_advantage_estimator("grpo")(GroupRelativeAdvantageEstimator)


def test_policy_loss_registry_rejects_unknown_and_duplicate_names() -> None:
    """Select policy losses through a MOLT-style registry."""
    assert isinstance(get_policy_loss("clipped"), ClippedPolicyObjective)

    with pytest.raises(ValueError, match="Unknown policy loss 'missing'"):
        get_policy_loss("missing")

    with pytest.raises(ValueError, match="Policy loss is already registered: clipped"):
        register_policy_loss("clipped")(ClippedPolicyObjective)
