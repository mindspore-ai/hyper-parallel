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
"""vLLM worker hooks used by Actor-to-rollout weight synchronization."""
from typing import Any
from rl.roles.weight_sync.sync import (
    aggregate_policy_fingerprint,
    is_policy_fingerprint_weight,
    policy_tensor_fingerprint,
)
from rl.roles.weight_sync.transfer import HYPER_QWEN3_5_ARCHITECTURE
_ASCEND_LIFECYCLE_PATCHED = False
def get_policy_weight_fingerprint(worker: Any, version: str = "") -> dict[str, Any]:
    """Hash replicated language-model norms for post-transfer verification."""
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    import torch  # pylint: disable=C0415
    tensor_digests = {}
    value_count = 0
    model = worker.model_runner.get_model()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not is_policy_fingerprint_weight(name):
            continue
        values = parameter.detach().to(device="cpu", dtype=torch.float32).contiguous()
        canonical_name, tensor_digest = policy_tensor_fingerprint(
            name,
            tuple(values.shape),
            values.view(torch.uint8).numpy().tobytes(),
        )
        if canonical_name in tensor_digests:
            raise RuntimeError(
                f"vLLM policy fingerprint has duplicate tensor {canonical_name!r}"
            )
        tensor_digests[canonical_name] = tensor_digest
        value_count += int(values.numel())
    if not tensor_digests:
        raise RuntimeError("vLLM policy fingerprint found no language-model norm tensors")
    hf_config = getattr(worker.model_config, "hf_config", None)
    architectures = tuple(getattr(hf_config, "architectures", ()) or ())
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    fingerprint = aggregate_policy_fingerprint(tensor_digests, value_count)
    fingerprint.update(
        {
            "version": str(version),
            "rank": rank,
            "architecture": architectures[0] if architectures else None,
        }
    )
    return fingerprint
def reload_weights(
    worker: Any,
    weights_iterator: Any = None,
    weights_path: str | None = None,
    is_checkpoint_format: bool = True,
) -> None:
    """Reload a Hyper checkpoint without vLLM's layerwise wrapper."""
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    if not is_checkpoint_format:
        raise ValueError("Hyper vLLM refit requires checkpoint-format weights")
    model_runner = worker.model_runner
    model = model_runner.get_model()
    if weights_iterator is not None:
        model.load_weights(weights_iterator)
        return
    if weights_path is None:
        raise ValueError("Hyper vLLM refit requires weights_iterator or weights_path")
    from vllm.model_executor.model_loader import get_model_loader  # pylint: disable=C0415
    original_model_path = model_runner.model_config.model
    try:
        model_runner.model_config.model = weights_path
        model_loader = get_model_loader(model_runner.load_config)
        model.load_weights(model_loader.get_all_weights(model_runner.model_config, model))
    finally:
        model_runner.model_config.model = original_model_path
def _is_hyper_worker(worker: Any) -> bool:
    hf_config = getattr(worker.model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", ())
    return HYPER_QWEN3_5_ARCHITECTURE in (architectures or ())
def _patch_ascend_weight_update_lifecycle() -> None:
    """Bypass vLLM's layerwise parameter wrapper for the strict Hyper loader."""
    global _ASCEND_LIFECYCLE_PATCHED
    if _ASCEND_LIFECYCLE_PATCHED:
        return
    try:
        from vllm_ascend.worker.worker import NPUWorker  # pylint: disable=C0415
    except ImportError:
        return
    original_start = NPUWorker.start_weight_update
    original_finish = NPUWorker.finish_weight_update
    def start_weight_update(worker: Any, is_checkpoint_format: bool = True) -> None:
        if not _is_hyper_worker(worker):
            original_start(worker, is_checkpoint_format=is_checkpoint_format)
            return
        if not is_checkpoint_format:
            raise ValueError("Hyper vLLM weight transfer requires checkpoint-format names")
        worker._check_weight_transfer_engine()  # pylint: disable=W0212
        if worker._weight_update_active:  # pylint: disable=W0212
            raise RuntimeError(
                "start_weight_update called while a weight update is already active"
            )
        worker._check_nz_disabled()  # pylint: disable=W0212
        worker._is_checkpoint_format = True  # pylint: disable=W0212
        worker._weight_update_active = True  # pylint: disable=W0212
    def finish_weight_update(worker: Any) -> None:
        if not _is_hyper_worker(worker):
            original_finish(worker)
            return
        worker._check_weight_transfer_engine()  # pylint: disable=W0212
        if not worker._weight_update_active:  # pylint: disable=W0212
            raise RuntimeError(
                "start_weight_update must be called before finish_weight_update"
            )
        worker._weight_update_active = False  # pylint: disable=W0212
        worker._is_checkpoint_format = True  # pylint: disable=W0212
    NPUWorker.start_weight_update = start_weight_update
    NPUWorker.finish_weight_update = finish_weight_update
    _ASCEND_LIFECYCLE_PATCHED = True
def install_vllm_weight_sync_hooks() -> None:
    """Install worker RPCs and the Hyper-specific Ascend update lifecycle."""
    from vllm.v1.worker.worker_base import WorkerBase  # pylint: disable=C0415
    if not hasattr(WorkerBase, "reload_weights"):
        setattr(WorkerBase, "reload_weights", reload_weights)
    if not hasattr(WorkerBase, "get_policy_weight_fingerprint"):
        setattr(
            WorkerBase,
            "get_policy_weight_fingerprint",
            get_policy_weight_fingerprint,
        )
    _patch_ascend_weight_update_lifecycle()
__all__ = [
    "get_policy_weight_fingerprint",
    "install_vllm_weight_sync_hooks",
    "reload_weights",
]
