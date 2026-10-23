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
"""Lightweight vLLM plugin for Hyper-RL model registration."""

__all__ = [
    "ADAPTED_QWEN3_ARCHITECTURE",
    "register_hyper_models",
]

import logging
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from rl.consistency import (
    CONSISTENCY_PROFILE_OFF,
    consistency_runtime_state,
    install_rollout_consistency_profile,
)
from rl.consistency.qwen3_dense import install_qwen3_rollout_rms_norm_diagnostic
from rl.roles.model_setup import (
    ADAPTED_QWEN3_ARCHITECTURE,
)
from rl.weight_sync.vllm_worker import install_vllm_weight_sync_hooks
from rl.tool_protocol import install_tool_evidence

ADAPTED_QWEN3_MODEL_CLASS = "rl.roles.rollout.consistency_models.qwen3.model:HyperQwen3ForCausalLM"
_ADAPTED_MODELS = {
    ADAPTED_QWEN3_ARCHITECTURE: ADAPTED_QWEN3_MODEL_CLASS,
}
_SUPPORTED_RUNTIMES = {("0.22.1", "0.22.1rc1"), ("0.23.0", "0.23.0.post1")}
_LOGGER = logging.getLogger(__name__)


def register_hyper_models() -> None:
    """Register supported HyperParallel model adapters with vLLM."""
    if os.environ.get("HYPER_RL_ASYNC_TRAINING") == "1":
        # Optional server-only routes; ordinary synchronous services do not install them.
        from rl.async_training.vllm_server import install_async_routes  # pylint: disable=C0415
        install_async_routes()
    diagnostic_rms_norm = os.environ.get("HYPER_RL_TEST_QWEN3_RMS_NORM", "0")
    if diagnostic_rms_norm not in ("0", "1"):
        raise ValueError("HYPER_RL_TEST_QWEN3_RMS_NORM must be 0 or 1")
    if diagnostic_rms_norm == "1":
        install_qwen3_rollout_rms_norm_diagnostic()
    profile = os.environ.get("HYPER_RL_CONSISTENCY_PROFILE")
    if profile is not None:
        install_rollout_consistency_profile(profile)
    _LOGGER.info(
        "training-inference consistency rollout runtime: enabled=%s recipe=%s state=%s",
        profile is not None,
        profile or CONSISTENCY_PROFILE_OFF,
        consistency_runtime_state(),
    )
    try:
        installed_version = package_version("vllm").split("+", maxsplit=1)[0]
    except PackageNotFoundError:
        _LOGGER.warning(
            "Skipping adapted model registration because vLLM package metadata is unavailable."
        )
        return
    # Native rollout uses these stable worker RPCs too; private lifecycle
    # patches are installed only after both pinned versions are verified.
    install_vllm_weight_sync_hooks(private_lifecycle=False)
    try:
        installed_ascend_version = package_version("vllm-ascend").split("+", maxsplit=1)[0]
    except PackageNotFoundError:
        _LOGGER.warning(
            "Skipping adapted model registration because vLLM-Ascend package metadata is unavailable."
        )
        return
    if (installed_version, installed_ascend_version) not in _SUPPORTED_RUNTIMES:
        _LOGGER.warning(
            "Skipping adapted model registration for unsupported vLLM/vLLM-Ascend pair %s/%s.",
            installed_version,
            installed_ascend_version,
        )
        return
    install_vllm_weight_sync_hooks(private_lifecycle=True)
    install_tool_evidence()
    # vLLM is optional and imports this entry point only when installed.
    from vllm import ModelRegistry  # pylint: disable=C0415
    supported_architectures = ModelRegistry.get_supported_archs()
    for architecture, model_class in _ADAPTED_MODELS.items():
        if architecture not in supported_architectures:
            ModelRegistry.register_model(architecture, model_class)


if __name__ in ("__main__", "__mp_main__"):
    # The async service and spawned workers register hooks before loading their targets.
    os.environ["HYPER_RL_ASYNC_TRAINING"] = "1"
    register_hyper_models()
    if __name__ == "__main__":
        from vllm.entrypoints.cli.main import main

        main()
