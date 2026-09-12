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
"""CPU unit tests for HyperParallel-RL weight-sync strategy selection."""

import pytest

from rl.roles.model import ModelRegistration, resolve_vllm_model
from rl.roles.weight_sync.transfer import (
    ColocatedDirectReshardWeightTransfer,
    ColocatedFullGatherWeightTransfer,
    DirectReshardHCCLWeightTransfer,
    FallbackWeightTransfer,
    FullGatherHCCLWeightTransfer,
    build_weight_transfer,
)


def _rollout_model():
    """Return one supported Hyper Qwen3 rollout registration."""
    model = ModelRegistration(
        "qwen",
        "qwen3",
        "/model",
        "/tokenizer",
        "Qwen3ForCausalLM",
        "qwen3",
        "qwen3",
        True,
    )
    return resolve_vllm_model(model, "hyper")


@pytest.mark.parametrize(
    ("deployment", "full_type", "direct_type"),
    [
        ("colocated", ColocatedFullGatherWeightTransfer, ColocatedDirectReshardWeightTransfer),
        ("disjoint", FullGatherHCCLWeightTransfer, DirectReshardHCCLWeightTransfer),
    ],
)
def test_weight_sync_preserves_explicit_strategy_for_tp1_and_tp2(
    deployment: str,
    full_type: type,
    direct_type: type,
) -> None:
    """Explicit direct stays direct for every TP size while full gather remains selectable."""
    full = build_weight_transfer(
        deployment,
        _rollout_model(),
        tensor_parallel_size=1,
        data_parallel_size=2,
        bucket_size_bytes=64,
        strategy="full_gather",
        fallback_strategy="none",
    )
    tp1 = build_weight_transfer(
        deployment,
        _rollout_model(),
        tensor_parallel_size=1,
        data_parallel_size=2,
        bucket_size_bytes=64,
        strategy="direct_reshard",
        fallback_strategy="none",
    )
    tp2 = build_weight_transfer(
        deployment,
        _rollout_model(),
        tensor_parallel_size=2,
        data_parallel_size=2,
        bucket_size_bytes=64,
        strategy="direct_reshard",
        fallback_strategy="none",
    )
    fallback = build_weight_transfer(
        deployment,
        _rollout_model(),
        tensor_parallel_size=1,
        data_parallel_size=2,
        bucket_size_bytes=64,
        strategy="direct_reshard",
        fallback_strategy="full_gather",
    )

    assert isinstance(full, full_type)
    assert full.configured_strategy == "full_gather"
    assert isinstance(tp1, direct_type)
    assert tp1.configured_strategy == "direct_reshard"
    assert tp1._data_parallel_size == 2  # pylint: disable=protected-access
    assert isinstance(tp2, direct_type)
    assert tp2.configured_strategy == "direct_reshard"
    assert tp2._data_parallel_size == 2  # pylint: disable=protected-access
    assert tp2._tensor_parallel_size == 2  # pylint: disable=protected-access
    assert tp2._bucket_size_bytes == 64  # pylint: disable=protected-access
    assert isinstance(fallback, FallbackWeightTransfer)
    assert isinstance(fallback._primary, direct_type)  # pylint: disable=protected-access
    assert isinstance(fallback._fallback, full_type)  # pylint: disable=protected-access
