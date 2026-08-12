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
"""Tests for the optional vLLM general plugin."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

import rl.roles.rollout.vllm_plugin as plugin
import hyper_parallel_vllm_plugin as legacy_plugin
from rl.roles.rollout.vllm_plugin import (
    HYPER_QWEN3_5_ARCHITECTURE,
    register_hyper_models,
)


def test_register_hyper_models_is_lazy_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plugin should register one lazy model class per process."""
    registrations = []

    class FakeModelRegistry:
        """Minimal vLLM registry used to validate plugin behavior."""

        supported_architectures = set()

        @classmethod
        def get_supported_archs(cls) -> set[str]:
            return cls.supported_architectures

        @classmethod
        def register_model(cls, architecture: str, model_class: str) -> None:
            registrations.append((architecture, model_class))
            cls.supported_architectures.add(architecture)

    class FakeWorkerBase:
        """Minimal worker base patched with the secure string RPC."""

    versions = {"vllm": "0.22.1+empty", "vllm-ascend": "0.22.1rc1"}
    monkeypatch.setattr(plugin, "package_version", versions.__getitem__)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(ModelRegistry=FakeModelRegistry))
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", ModuleType("vllm.v1.worker"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.worker_base",
        SimpleNamespace(WorkerBase=FakeWorkerBase),
    )

    register_hyper_models()
    register_hyper_models()

    assert registrations == [
        (
            HYPER_QWEN3_5_ARCHITECTURE,
            "rl.roles.rollout.vllm_qwen3_5:HyperQwen3_5ForCausalLM",
        )
    ]
    assert FakeWorkerBase.reload_weights is plugin._reload_weights  # pylint: disable=protected-access
    assert (  # pylint: disable=protected-access
        FakeWorkerBase.get_policy_weight_fingerprint
        is plugin._get_policy_weight_fingerprint
    )


def test_register_hyper_models_skips_unsupported_vllm(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unrelated vLLM installation should not load the private adapter APIs."""
    monkeypatch.setattr(plugin, "package_version", lambda _name: "0.18.0")
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    register_hyper_models()

    assert "supports only 0.22.1" in caplog.text


def test_register_hyper_models_skips_unsupported_vllm_ascend(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A different vLLM-Ascend release must not load private worker patches."""
    versions = {"vllm": "0.22.1", "vllm-ascend": "0.21.0"}
    monkeypatch.setattr(plugin, "package_version", versions.__getitem__)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    register_hyper_models()

    assert "vLLM-Ascend 0.21.0" in caplog.text
    assert "supports only 0.22.1rc1" in caplog.text


def test_hyper_weight_update_bypasses_layerwise_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native HCCL updates should retain Hyper load_weights without layerwise wrapping."""
    original_calls = []

    class FakeNPUWorker:
        """Minimal vLLM-Ascend worker patched by the Hyper plugin."""

        @staticmethod
        def start_weight_update(worker, is_checkpoint_format: bool = True) -> None:
            original_calls.append(("start", worker, is_checkpoint_format))

        @staticmethod
        def finish_weight_update(worker) -> None:
            original_calls.append(("finish", worker))

    ascend_module = ModuleType("vllm_ascend")
    worker_package = ModuleType("vllm_ascend.worker")
    worker_module = ModuleType("vllm_ascend.worker.worker")
    worker_module.NPUWorker = FakeNPUWorker
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker", worker_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", worker_module)
    monkeypatch.setattr(plugin, "_HYPER_LIFECYCLE_PATCHED", False)

    plugin._patch_ascend_weight_update_lifecycle()  # pylint: disable=protected-access
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[HYPER_QWEN3_5_ARCHITECTURE])
        ),
        _weight_update_active=False,
        _is_checkpoint_format=False,
        _check_weight_transfer_engine=lambda: None,
        _check_nz_disabled=lambda: None,
    )
    FakeNPUWorker.start_weight_update(worker, is_checkpoint_format=True)

    assert worker._weight_update_active is True
    assert worker._is_checkpoint_format is True
    assert original_calls == []

    FakeNPUWorker.finish_weight_update(worker)
    assert worker._weight_update_active is False
    assert original_calls == []

    native_worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"])
        )
    )
    FakeNPUWorker.start_weight_update(native_worker, is_checkpoint_format=True)
    FakeNPUWorker.finish_weight_update(native_worker)
    assert original_calls == [
        ("start", native_worker, True),
        ("finish", native_worker),
    ]


def test_legacy_plugin_forwards_to_hyper_rl_registration() -> None:
    """Pinned E2 images may still reference the historical plugin module."""
    assert legacy_plugin.HYPER_QWEN3_5_ARCHITECTURE == HYPER_QWEN3_5_ARCHITECTURE
    assert legacy_plugin.register_hyper_models is register_hyper_models
