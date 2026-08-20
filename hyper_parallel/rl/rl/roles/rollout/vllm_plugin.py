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
import logging
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from rl.consistency import install_rollout_consistency_profile
from rl.roles.model import HYPER_QWEN3_5_ARCHITECTURE, HYPER_QWEN3_ARCHITECTURE
from rl.roles.weight_sync.vllm_worker import install_vllm_weight_sync_hooks

HYPER_QWEN3_MODEL_CLASS = "rl.roles.rollout.vllm_qwen3:HyperQwen3ForCausalLM"
HYPER_QWEN3_5_MODEL_CLASS = (
    "rl.roles.rollout.vllm_qwen3_5:HyperQwen3_5ForCausalLM"
)
_HYPER_MODELS = {
    HYPER_QWEN3_ARCHITECTURE: HYPER_QWEN3_MODEL_CLASS,
    HYPER_QWEN3_5_ARCHITECTURE: HYPER_QWEN3_5_MODEL_CLASS,
}
_SUPPORTED_VLLM_VERSION = "0.22.1"
_SUPPORTED_VLLM_ASCEND_VERSION = "0.22.1rc1"
_LOGGER = logging.getLogger(__name__)


def register_hyper_models() -> None:
    """Register supported HyperParallel model adapters with vLLM."""
    profile = os.environ.get("HYPER_RL_CONSISTENCY_PROFILE")
    if profile is not None:
        install_rollout_consistency_profile(profile)
    try:
        installed_version = package_version("vllm").split("+", maxsplit=1)[0]
    except PackageNotFoundError:
        _LOGGER.warning(
            "Skipping Hyper model registration because vLLM package metadata is unavailable."
        )
        return
    # Native rollout uses these stable worker RPCs too; private lifecycle
    # patches are installed only after both pinned versions are verified.
    install_vllm_weight_sync_hooks(private_lifecycle=False)
    if installed_version != _SUPPORTED_VLLM_VERSION:
        _LOGGER.warning(
            "Skipping Hyper model registration: vLLM %s is installed, but the adapter supports only %s.",
            installed_version,
            _SUPPORTED_VLLM_VERSION,
        )
        return
    try:
        installed_ascend_version = package_version("vllm-ascend").split("+", maxsplit=1)[0]
    except PackageNotFoundError:
        _LOGGER.warning(
            "Skipping Hyper model registration because vLLM-Ascend package metadata is unavailable."
        )
        return
    if installed_ascend_version != _SUPPORTED_VLLM_ASCEND_VERSION:
        _LOGGER.warning(
            "Skipping Hyper model registration: vLLM-Ascend %s is installed, but the adapter supports only %s.",
            installed_ascend_version,
            _SUPPORTED_VLLM_ASCEND_VERSION,
        )
        return
    install_vllm_weight_sync_hooks(private_lifecycle=True)
    # vLLM is optional and imports this entry point only when installed.
    from vllm import ModelRegistry  # pylint: disable=C0415
    supported_architectures = ModelRegistry.get_supported_archs()
    for architecture, model_class in _HYPER_MODELS.items():
        if architecture not in supported_architectures:
            ModelRegistry.register_model(architecture, model_class)


__all__ = [
    "HYPER_QWEN3_5_ARCHITECTURE",
    "HYPER_QWEN3_ARCHITECTURE",
    "register_hyper_models",
]
