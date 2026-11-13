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
"""Install the native-vLLM worker hooks required by Hyper-RL."""
import logging
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from rl.weight_sync.vllm_worker import install_vllm_weight_sync_hooks
from rl.tool_protocol.protocol import install_tool_evidence
from rl.rollout.sampler import install_partial_prefill_rng_fix
from rl.unified_execution.models import model_families

_SUPPORTED_RUNTIMES = {
    ("0.22.1", "0.22.1rc1"),
    ("0.23.0", "0.23.0.post1"),
    ("0.30.0", "0.30.0"),
}
_LOGGER = logging.getLogger(__name__)


def install_runtime_hooks() -> None:
    """Install version-checked native weight, sampling and Agent hooks."""
    if os.environ.get("HYPER_RL_CONSISTENCY_PROFILE") is not None:
        raise ValueError("HYPER_RL_CONSISTENCY_PROFILE was removed; use native-vLLM")
    if os.environ.get("HYPER_RL_ASYNC_TRAINING") == "1":
        # Optional server dependency; invoked only inside the vLLM process.
        from rl.async_runtime.vllm_server import install_async_routes  # pylint: disable=C0415
        install_async_routes()
    try:
        installed = package_version("vllm").split("+", maxsplit=1)[0]
    except PackageNotFoundError:
        _LOGGER.warning("vLLM metadata is unavailable; worker hooks were not installed")
        return
    if os.environ.get("HYPER_RL_UNIFIED_EXECUTION") == "1":
        # vLLM is optional in training/configuration-only processes.
        from vllm import ModelRegistry  # pylint: disable=C0415
        for family in model_families():
            architecture = family.vllm_architecture
            ModelRegistry.register_model(
                architecture, f"rl.unified_execution.runtimes.vllm.model_entrypoints:{architecture}",
            )
    install_vllm_weight_sync_hooks(private_lifecycle=False)
    try:
        ascend = package_version("vllm-ascend").split("+", maxsplit=1)[0]
    except PackageNotFoundError:
        return
    if (installed, ascend) not in _SUPPORTED_RUNTIMES:
        raise ValueError(f"Unsupported native worker lifecycle versions: {installed}/{ascend}")
    install_vllm_weight_sync_hooks(private_lifecycle=True)
    install_tool_evidence()
    if installed == "0.23.0":
        install_partial_prefill_rng_fix()


if __name__ in ("__main__", "__mp_main__"):
    os.environ["HYPER_RL_ASYNC_TRAINING"] = "1"
    install_runtime_hooks()
    if __name__ == "__main__":
        # vLLM is an optional inference-process dependency.
        from vllm.entrypoints.cli.main import main
        main()
