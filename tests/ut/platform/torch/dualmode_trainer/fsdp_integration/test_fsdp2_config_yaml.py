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
"""Tests for the YAML fsdp_config section resolving into FSDP2Manager policies."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyper_models.components.distributed.config import FSDP2Config
from hyper_models.components.distributed.fsdp2 import FSDP2Manager
from hyper_models.components.distributed.infrastructure import MeshContext
from hyper_models.config.resolver import resolve_component
from tests.common.mark_utils import arg_mark

_YAML_PATH = Path(__file__).resolve().parent / "test_yamls" / "fsdp_config_watch.yaml"


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_yaml_fsdp_config_resolves_and_builds_manager_policies() -> None:
    """
    Feature: YAML fsdp_config watch.
    Description: Parse the watched YAML section into FSDP2Config, then build
        fully_shard kwargs through the manager.
    Expectation: The resolved config carries every declared value and the
        manager maps them to a mixed-precision policy plus CPU offload.
    """
    with _YAML_PATH.open(encoding="utf-8") as yaml_file:
        raw = yaml.safe_load(yaml_file)

    fsdp_config = resolve_component(
        raw["fsdp_config"],
        expected_type=FSDP2Config,
        path="$.fsdp_config",
    )

    assert fsdp_config.dp_shard_size == 2
    assert fsdp_config.edp_shard_size == 1
    assert fsdp_config.replicate_params == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    assert fsdp_config.activation_checkpointing == "selective"
    assert fsdp_config.mix_precision.param_dtype == "bfloat16"
    assert fsdp_config.mix_precision.reduce_dtype == "float32"
    assert fsdp_config.mix_precision.output_dtype == "bfloat16"
    assert fsdp_config.enable_offload
    assert not fsdp_config.reshard_after_forward
    assert fsdp_config.backward_prefetch_depth == 2
    assert fsdp_config.forward_prefetch_depth == 1
    assert fsdp_config.comm_fusion
    assert fsdp_config.comm_fusion_zero_copy

    manager = FSDP2Manager(fsdp_config, MeshContext())
    mp_policy = manager._build_mixed_precision_policy()
    offload_policy = manager._build_offload_policy()

    assert str(mp_policy.param_dtype) == "torch.bfloat16"
    assert str(mp_policy.reduce_dtype) == "torch.float32"
    assert str(mp_policy.output_dtype) == "torch.bfloat16"
    from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy

    assert isinstance(offload_policy, CPUOffloadPolicy)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_yaml_fsdp_config_all_none_mix_precision_keeps_default_policies() -> None:
    """
    Feature: YAML fsdp_config watch defaults.
    Description: Build manager policies from the default FSDP2Config.
    Expectation: Unset mix precision and offload yield the no-op policies.
    """
    fsdp_config = resolve_component(
        {"dp_shard_size": 1},
        expected_type=FSDP2Config,
        path="$.fsdp_config",
    )

    manager = FSDP2Manager(fsdp_config, MeshContext())
    mp_policy = manager._build_mixed_precision_policy()
    offload_policy = manager._build_offload_policy()

    assert mp_policy.param_dtype is None
    assert mp_policy.reduce_dtype is None
    assert mp_policy.output_dtype is None
    from hyper_parallel.core.fully_shard.utils import OffloadPolicy

    assert isinstance(offload_policy, OffloadPolicy)
