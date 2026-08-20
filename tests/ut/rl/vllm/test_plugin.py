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
from typing import Any, Optional

import pytest
import rl.roles.rollout.vllm_plugin as plugin
from rl.roles.rollout.vllm_plugin import (
    HYPER_QWEN3_5_ARCHITECTURE,
    HYPER_QWEN3_ARCHITECTURE,
    register_hyper_models,
)
from rl.roles.weight_sync import vllm_worker

import hyper_parallel_vllm_plugin as legacy_plugin


def _install_fake_worker_base(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install the minimal import hierarchy used by stable native worker hooks."""
    class FakeWorkerBase:
        """Minimal worker base patched by the optional plugin."""

    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", ModuleType("vllm.v1.worker"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.worker_base",
        SimpleNamespace(WorkerBase=FakeWorkerBase),
    )
    _install_fake_engine_core(monkeypatch)
    return FakeWorkerBase


def _install_fake_engine_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the minimal EngineCore hierarchy patched by the plugin."""
    class FakeEngineCore:
        """Minimal wake surface for registration-only tests."""

        def wake_up(self, tags: Optional[list[str]] = None) -> None:
            """Provide the upstream method replaced by the plugin."""
            del tags

    monkeypatch.setitem(sys.modules, "vllm.v1.engine", ModuleType("vllm.v1.engine"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.core",
        SimpleNamespace(EngineCore=FakeEngineCore),
    )


def test_register_hyper_models_is_lazy_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plugin should register one lazy model class per process."""
    registrations = []

    class FakeModelRegistry:
        """Minimal vLLM registry used to validate plugin behavior."""

        supported_architectures = set()

        @classmethod
        def get_supported_archs(cls) -> set[str]:
            """Return registered fake architectures."""
            return cls.supported_architectures

        @classmethod
        def register_model(cls, architecture: str, model_class: str) -> None:
            """Record one lazy model registration."""
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
    _install_fake_engine_core(monkeypatch)

    register_hyper_models()
    register_hyper_models()

    assert registrations == [
        (
            HYPER_QWEN3_ARCHITECTURE,
            "rl.roles.rollout.vllm_qwen3:HyperQwen3ForCausalLM",
        ),
        (
            HYPER_QWEN3_5_ARCHITECTURE,
            "rl.roles.rollout.vllm_qwen3_5:HyperQwen3_5ForCausalLM",
        )
    ]
    assert getattr(FakeWorkerBase, "reload_weights") is vllm_worker.reload_weights
    assert (  # pylint: disable=protected-access
        getattr(FakeWorkerBase, "get_policy_weight_fingerprint")
        is vllm_worker.get_policy_weight_fingerprint
    )


def test_register_hyper_models_skips_unsupported_vllm(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unsupported release still receives stable native weight-sync RPCs."""
    monkeypatch.setattr(plugin, "package_version", lambda _name: "0.23.0")
    worker_base = _install_fake_worker_base(monkeypatch)

    register_hyper_models()

    assert "supports only 0.22.1" in caplog.text
    assert getattr(worker_base, "reload_weights") is vllm_worker.reload_weights
    assert (
        getattr(worker_base, "get_policy_weight_fingerprint")
        is vllm_worker.get_policy_weight_fingerprint
    )


def test_register_hyper_models_skips_unsupported_vllm_ascend(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A different vLLM-Ascend release skips only the custom model adapter."""
    versions = {"vllm": "0.22.1", "vllm-ascend": "0.21.0"}
    monkeypatch.setattr(plugin, "package_version", versions.__getitem__)
    worker_base = _install_fake_worker_base(monkeypatch)

    register_hyper_models()

    assert "vLLM-Ascend 0.21.0" in caplog.text
    assert "supports only 0.22.1rc1" in caplog.text
    assert getattr(worker_base, "reload_weights") is vllm_worker.reload_weights


def test_register_hyper_models_installs_rollout_consistency_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profiled child process installs model patches before plugin registration."""
    installed = []
    monkeypatch.setenv(
        "HYPER_RL_CONSISTENCY_PROFILE",
        "qwen3_ascend_fa3_batch_invariant_v1",
    )
    monkeypatch.setattr(
        plugin,
        "install_rollout_consistency_profile",
        installed.append,
    )
    monkeypatch.setattr(plugin, "package_version", lambda _name: "0.23.0")
    _install_fake_worker_base(monkeypatch)

    register_hyper_models()

    assert installed == ["qwen3_ascend_fa3_batch_invariant_v1"]


def test_hyper_weight_update_bypasses_layerwise_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native HCCL updates should retain Hyper load_weights without layerwise wrapping."""
    original_calls = []

    class FakeNPUWorker:
        """Minimal vLLM-Ascend worker patched by the Hyper plugin."""

        @staticmethod
        def start_weight_update(
            worker: Any,
            is_checkpoint_format: bool = True,
        ) -> None:
            """Record one native start call."""
            original_calls.append(("start", worker, is_checkpoint_format))

        @staticmethod
        def finish_weight_update(worker: Any) -> None:
            """Record one native finish call."""
            original_calls.append(("finish", worker))

        @staticmethod
        def update_weights(worker: Any, update_info: dict[str, Any]) -> None:
            """Record the version-free payload passed to the real receiver."""
            original_calls.append(("update", worker, update_info))

    ascend_module = ModuleType("vllm_ascend")
    worker_package = ModuleType("vllm_ascend.worker")
    worker_module = ModuleType("vllm_ascend.worker.worker")
    worker_module.NPUWorker = FakeNPUWorker
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker", worker_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", worker_module)
    monkeypatch.setattr(vllm_worker._patch_state, "ascend_lifecycle", False)

    vllm_worker._patch_ascend_weight_update_lifecycle()  # pylint: disable=protected-access
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
    assert not original_calls

    with pytest.raises(ValueError, match="requires a worker policy version"):
        FakeNPUWorker.update_weights(worker, {"names": ["model.norm.weight"]})
    assert not original_calls

    FakeNPUWorker.update_weights(
        worker,
        {"names": ["model.norm.weight"], "_hyper_policy_version": 1},
    )
    assert worker._hyper_pending_policy_version == 1
    assert getattr(worker, "_hyper_loaded_policy_version", 0) == 0
    assert original_calls == [
        ("update", worker, {"names": ["model.norm.weight"]})
    ]

    with pytest.raises(RuntimeError, match="already active"):
        FakeNPUWorker.start_weight_update(worker, is_checkpoint_format=True)
    assert worker._hyper_pending_policy_version == 1

    FakeNPUWorker.finish_weight_update(worker)
    assert worker._weight_update_active is False
    assert worker._hyper_loaded_policy_version == 1
    assert worker._hyper_pending_policy_version is None

    native_worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"])
        )
    )
    FakeNPUWorker.start_weight_update(native_worker, is_checkpoint_format=True)
    FakeNPUWorker.finish_weight_update(native_worker)
    assert original_calls[1:] == [
        ("start", native_worker, True),
        ("finish", native_worker),
    ]


def test_cpu_reload_commits_worker_version_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU load must not publish its identity before the controller commits it."""
    loaded = []
    model = SimpleNamespace(load_weights=loaded.append)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(get_model=lambda: model),
        _hyper_loaded_policy_version=0,
    )
    weights = [("model.norm.weight", object())]

    monkeypatch.setenv("HYPER_RL_CONSISTENCY_PROFILE", "profile")
    with pytest.raises(ValueError, match="requires a worker policy version"):
        vllm_worker.reload_weights(worker, weights_iterator=weights)
    assert not loaded

    vllm_worker.reload_weights(worker, weights_iterator=weights, policy_version=1)

    assert loaded == [weights]
    assert worker._hyper_loaded_policy_version == 0
    assert worker._hyper_pending_policy_version == 1

    vllm_worker.commit_reloaded_weights(worker, policy_version=1)
    assert worker._hyper_loaded_policy_version == 1
    assert worker._hyper_pending_policy_version is None

    with pytest.raises(ValueError, match="must increase"):
        vllm_worker.reload_weights(worker, weights_iterator=weights, policy_version=1)
    assert loaded == [weights]


def test_cpu_reload_off_profile_preserves_unversioned_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit off profile must preserve the stable unversioned reload RPC."""
    loaded = []
    weights = [("model.norm.weight", object())]
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            get_model=lambda: SimpleNamespace(load_weights=loaded.append)
        )
    )
    monkeypatch.setenv("HYPER_RL_CONSISTENCY_PROFILE", "off")

    vllm_worker.reload_weights(worker, weights_iterator=weights)

    assert loaded == [weights]


def test_engine_core_memory_wake_keeps_scheduler_paused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pinned vLLM patch must update executor state without calling scheduler wake."""
    scheduler_wakes = []
    memory_wakes = []

    class FakeEngineCore:
        """Minimal EngineCore exposing the unconditional upstream wake behavior."""

        def __init__(self) -> None:
            """Attach a fake executor that records memory wake calls."""
            def wake_memory(tags: list[str]) -> None:
                """Record one executor-level memory wake."""
                memory_wakes.append(tags)

            self.model_executor = SimpleNamespace(wake_up=wake_memory)

        def wake_up(self, tags: Optional[list[str]] = None) -> None:
            """Record the original path that would also resume scheduling."""
            scheduler_wakes.append(tags)

    engine_package = ModuleType("vllm.v1.engine")
    core_module = ModuleType("vllm.v1.engine.core")
    core_module.EngineCore = FakeEngineCore
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.v1", ModuleType("vllm.v1"))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine_package)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core_module)
    monkeypatch.setattr(vllm_worker._patch_state, "engine_core_wake", False)

    vllm_worker._patch_engine_core_wake_lifecycle()  # pylint: disable=protected-access
    engine = FakeEngineCore()
    engine.wake_up(["weights", vllm_worker.KEEP_SCHEDULER_PAUSED_TAG])

    assert memory_wakes == [["weights"]]
    assert not scheduler_wakes

    engine.wake_up(["kv_cache"])
    assert scheduler_wakes == [["kv_cache"]]


def test_legacy_plugin_forwards_to_hyper_rl_registration() -> None:
    """Pinned E2 images may still reference the historical plugin module."""
    assert legacy_plugin.HYPER_QWEN3_5_ARCHITECTURE == HYPER_QWEN3_5_ARCHITECTURE
    assert legacy_plugin.HYPER_QWEN3_ARCHITECTURE == HYPER_QWEN3_ARCHITECTURE
    assert legacy_plugin.register_hyper_models is register_hyper_models
