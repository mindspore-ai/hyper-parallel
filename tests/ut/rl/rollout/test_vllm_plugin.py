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
"""Regression coverage for the direct vLLM plugin entry point."""

from importlib.metadata import EntryPoint
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import rl.roles.rollout.vllm_plugin as plugin_module


def test_external_plugin_entry_point_registers_qwen3(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public entry point must import and register the supported dense model."""
    entry_point = EntryPoint(
        name="hyper_parallel",
        value="rl.roles.rollout.vllm_plugin:register_hyper_models",
        group="vllm.general_plugins",
    )
    register_models = entry_point.load()
    assert register_models is plugin_module.register_hyper_models
    registry = SimpleNamespace(get_supported_archs=lambda: (), register_model=Mock())
    versions = {"vllm": "0.22.1", "vllm-ascend": "0.22.1rc1"}
    monkeypatch.setattr(plugin_module, "package_version", versions.__getitem__)
    monkeypatch.setattr(plugin_module, "install_vllm_weight_sync_hooks", Mock())
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=registry))
    monkeypatch.delenv("HYPER_RL_CONSISTENCY_PROFILE", raising=False)
    monkeypatch.delenv("HYPER_RL_TEST_QWEN3_RMS_NORM", raising=False)

    register_models()

    registry.register_model.assert_called_once_with(
        "HyperQwen3ForCausalLM", "rl.roles.rollout.consistency_models.qwen3.model:HyperQwen3ForCausalLM",
    )
