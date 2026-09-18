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
"""CPU unit tests for Hyper-RL weight-sync strategy selection."""

import pytest

import rl.roles.weight_sync.transfer as transfer_module
from rl.roles.model_setup import ModelRegistration, resolve_vllm_model
from rl.roles import weight_sync
from rl.roles.weight_sync.config import resolve_weight_sync_config
from rl.roles.weight_sync.transfer import build_weight_transfer


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


@pytest.mark.parametrize("deployment, transport_name", [("colocated", "ipc"), ("disjoint", "hccl")])
@pytest.mark.parametrize("strategy", ["direct_reshard", "full_gather"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_weight_sync_preserves_explicit_strategy_for_tp1_and_tp2(deployment, transport_name, strategy, tp_size):
    """All combinations reuse the publisher while preserving the explicit strategy."""
    result = build_weight_transfer(
        deployment, _rollout_model(), tensor_parallel_size=tp_size, data_parallel_size=2,
        bucket_size_bytes=64, strategy=strategy,
    )
    assert result.configured_strategy == strategy
    assert result.strategy.name == strategy
    assert result.transport.name == transport_name
    assert result.strategy.source.bucket_size_bytes == 64
    if strategy == "direct_reshard":
        assert result.strategy.tensor_parallel_size == tp_size
        assert result.strategy.data_parallel_size == 2


def test_only_current_publisher_interface_is_exposed() -> None:
    """Removed combination classes and publication verbs have no compatibility surface."""
    removed_names = (
        "ColocatedDirectReshardWeightTransfer",
        "ColocatedFullGatherWeightTransfer",
        "DirectReshardHCCLWeightTransfer",
        "FallbackWeightTransfer",
        "FullGatherHCCLWeightTransfer",
        "WeightTransfer",
    )
    assert all(not hasattr(weight_sync, name) for name in removed_names)
    assert all(not hasattr(transfer_module, name) for name in removed_names)
    publisher = build_weight_transfer("colocated", _rollout_model())
    assert callable(publisher.publish)
    assert not hasattr(publisher, "transfer")
    assert not hasattr(publisher, "refit")


def test_removed_fallback_configuration_is_rejected() -> None:
    """The removed fallback field cannot silently alter current behavior."""
    with pytest.raises(ValueError, match="fallback_strategy"):
        resolve_weight_sync_config(
            {"strategy": "direct_reshard", "fallback_strategy": "full_gather"},
            deployment="colocated",
            model_family="qwen3",
            rollout_tp=1,
        )
