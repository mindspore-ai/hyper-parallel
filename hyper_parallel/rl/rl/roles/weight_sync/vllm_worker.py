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
from dataclasses import dataclass
from functools import wraps
import os
from typing import Any, Optional
from rl.roles.model import HYPER_QWEN3_ARCHITECTURE, HYPER_QWEN3_5_ARCHITECTURE
from rl.roles.weight_sync.sync import (
    KEEP_SCHEDULER_PAUSED_TAG,
    aggregate_policy_fingerprint,
    is_policy_fingerprint_weight,
    policy_tensor_fingerprint,
)
from hyper_parallel import get_platform

platform = get_platform()
_HYPER_ARCHITECTURES = frozenset(
    (HYPER_QWEN3_ARCHITECTURE, HYPER_QWEN3_5_ARCHITECTURE)
)
_POLICY_VERSION_FIELD = "_hyper_policy_version"
_PRE_SAMPLE_GENERATOR_OFFSETS = "_hyper_rl_pre_sample_generator_offsets"


@dataclass
class _PatchState:
    """Track process-local idempotent vLLM patch installation."""

    ascend_lifecycle: bool = False
    engine_core_wake: bool = False


_patch_state = _PatchState()


def get_policy_weight_fingerprint(
    worker: Any,
    version: Optional[int] = None,
) -> dict[str, Any]:
    """Hash replicated language-model norms for post-transfer verification."""
    del version  # Retain the old RPC signature without trusting caller-owned identity.
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    tensor_digests = {}
    value_count = 0
    model = worker.model_runner.get_model()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not is_policy_fingerprint_weight(name):
            continue
        values = platform.tensor_type_cast(
            parameter.detach().to(device="cpu").contiguous(),
            "float32",
        )
        canonical_name, tensor_digest = policy_tensor_fingerprint(
            name,
            tuple(values.shape),
            platform.tensor_to_numpy(values).tobytes(),
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
    try:
        rank = platform.get_rank()
    except (RuntimeError, ValueError):
        rank = 0
    fingerprint = aggregate_policy_fingerprint(tensor_digests, value_count)
    fingerprint.update(
        {
            "version": int(getattr(worker, "_hyper_loaded_policy_version", 0)),
            "rank": rank,
            "architecture": architectures[0] if architectures else None,
        }
    )
    return fingerprint
def reload_weights(
    worker: Any,
    weights_iterator: Any = None,
    weights_path: Optional[str] = None,
    is_checkpoint_format: bool = True,
    policy_version: Optional[int] = None,
) -> None:
    """Reload a Hyper checkpoint without vLLM's layerwise wrapper."""
    if worker.model_runner is None:
        raise RuntimeError("vLLM model runner is not initialized")
    if not is_checkpoint_format:
        raise ValueError("Hyper vLLM refit requires checkpoint-format weights")
    consistency_profile = os.environ.get("HYPER_RL_CONSISTENCY_PROFILE")
    if policy_version is None and consistency_profile not in (None, "", "off"):
        raise ValueError("Consistency-profile CPU reload requires a worker policy version")
    normalized_version = None
    if policy_version is not None:
        normalized_version = int(policy_version)
        loaded_version = int(getattr(worker, "_hyper_loaded_policy_version", 0))
        if normalized_version <= loaded_version:
            raise ValueError(
                "vLLM worker policy version must increase: "
                f"loaded={loaded_version}, received={normalized_version}"
            )
        pending_version = getattr(worker, "_hyper_pending_policy_version", None)
        if pending_version is not None:
            raise RuntimeError(
                "vLLM worker has an uncommitted CPU reload: "
                f"pending={pending_version}, received={normalized_version}"
            )
    model_runner = worker.model_runner
    model = model_runner.get_model()
    if weights_iterator is not None:
        model.load_weights(weights_iterator)
        if normalized_version is not None:
            worker._hyper_pending_policy_version = normalized_version
        return
    if weights_path is None:
        raise ValueError("Hyper vLLM refit requires weights_iterator or weights_path")
    from vllm.model_executor.model_loader import get_model_loader  # pylint: disable=C0415
    original_model_path = model_runner.model_config.model
    try:
        model_runner.model_config.model = weights_path
        model_loader = get_model_loader(model_runner.load_config)
        model.load_weights(model_loader.get_all_weights(model_runner.model_config, model))
        if normalized_version is not None:
            worker._hyper_pending_policy_version = normalized_version
    finally:
        model_runner.model_config.model = original_model_path


def commit_reloaded_weights(worker: Any, policy_version: int) -> None:
    """Commit worker identity after every CPU reload RPC has completed."""
    normalized_version = int(policy_version)
    pending_version = getattr(worker, "_hyper_pending_policy_version", None)
    if pending_version != normalized_version:
        raise RuntimeError(
            "vLLM CPU reload version does not match its pending weights: "
            f"pending={pending_version}, received={normalized_version}"
        )
    worker._hyper_loaded_policy_version = normalized_version
    worker._hyper_pending_policy_version = None
def _is_hyper_worker(worker: Any) -> bool:
    hf_config = getattr(worker.model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", ())
    return bool(_HYPER_ARCHITECTURES.intersection(architectures or ()))
def _patch_ascend_weight_update_lifecycle() -> None:
    """Bypass vLLM's layerwise parameter wrapper for the strict Hyper loader."""
    if _patch_state.ascend_lifecycle:
        return
    try:
        from vllm_ascend.worker.worker import NPUWorker  # pylint: disable=C0415
    except ImportError:
        return
    original_start = NPUWorker.start_weight_update
    original_update = NPUWorker.update_weights
    original_finish = NPUWorker.finish_weight_update
    def start_weight_update(worker: Any, is_checkpoint_format: bool = True) -> None:
        """Start one Hyper checkpoint-format update transaction."""
        if not _is_hyper_worker(worker):
            original_start(worker, is_checkpoint_format=is_checkpoint_format)
            worker._hyper_pending_policy_version = None
            return
        if not is_checkpoint_format:
            raise ValueError("Hyper vLLM weight transfer requires checkpoint-format names")
        worker._check_weight_transfer_engine()  # pylint: disable=W0212
        if worker._weight_update_active:  # pylint: disable=W0212
            raise RuntimeError(
                "start_weight_update called while a weight update is already active"
            )
        worker._check_nz_disabled()  # pylint: disable=W0212
        worker._hyper_pending_policy_version = None
        worker._is_checkpoint_format = True  # pylint: disable=W0212
        worker._weight_update_active = True  # pylint: disable=W0212
    def update_weights(worker: Any, update_info: dict[str, Any]) -> None:
        """Receive weights while retaining worker-owned pending identity."""
        versioned_update = dict(update_info)
        version = versioned_update.pop(_POLICY_VERSION_FIELD, None)
        if not _is_hyper_worker(worker):
            original_update(worker, versioned_update)
            if version is not None:
                worker._hyper_pending_policy_version = int(version)
            return
        if version is None:
            raise ValueError("Hyper vLLM weight update requires a worker policy version")
        version = int(version)
        loaded_version = int(getattr(worker, "_hyper_loaded_policy_version", 0))
        pending_version = getattr(worker, "_hyper_pending_policy_version", None)
        if version <= loaded_version:
            raise ValueError(
                "vLLM worker policy version must increase: "
                f"loaded={loaded_version}, received={version}"
            )
        if pending_version is not None and version != pending_version:
            raise ValueError(
                "One vLLM weight update cannot mix policy versions: "
                f"pending={pending_version}, received={version}"
            )
        original_update(worker, versioned_update)
        worker._hyper_pending_policy_version = version
    def finish_weight_update(worker: Any) -> None:
        """Commit worker identity only after the native receiver finishes."""
        if not _is_hyper_worker(worker):
            original_finish(worker)
        else:
            worker._check_weight_transfer_engine()  # pylint: disable=W0212
            if not worker._weight_update_active:  # pylint: disable=W0212
                raise RuntimeError(
                    "start_weight_update must be called before finish_weight_update"
                )
            worker._weight_update_active = False  # pylint: disable=W0212
            worker._is_checkpoint_format = True  # pylint: disable=W0212
        pending_version = getattr(worker, "_hyper_pending_policy_version", None)
        if pending_version is not None:
            worker._hyper_loaded_policy_version = pending_version
        worker._hyper_pending_policy_version = None
    NPUWorker.start_weight_update = start_weight_update
    NPUWorker.update_weights = update_weights
    NPUWorker.finish_weight_update = finish_weight_update
    _patch_state.ascend_lifecycle = True


def _patch_engine_core_wake_lifecycle() -> None:
    """Wake executor memory while keeping the fixed vLLM scheduler paused."""
    if _patch_state.engine_core_wake:
        return
    from vllm.v1.engine.core import EngineCore  # pylint: disable=C0415
    original_wake_up = EngineCore.wake_up

    def wake_up(engine_core: Any, tags: Optional[list[str]] = None) -> Any:
        """Handle the Hyper sentinel before vLLM's unconditional scheduler resume."""
        if tags is None or KEEP_SCHEDULER_PAUSED_TAG not in tags:
            return original_wake_up(engine_core, tags)
        memory_tags = [tag for tag in tags if tag != KEEP_SCHEDULER_PAUSED_TAG]
        if memory_tags:
            engine_core.model_executor.wake_up(memory_tags)
        return None

    EngineCore.wake_up = wake_up
    _patch_state.engine_core_wake = True


def _patch_vllm_ascend_partial_prefill_rng(model_runner_cls: type[Any]) -> None:
    """Restore seeded generators after vLLM discards partial-prefill samples."""
    if getattr(model_runner_cls, "_hyper_rl_partial_prefill_rng_patched", False):
        return
    original_sample = model_runner_cls._sample
    original_bookkeeping = model_runner_cls._bookkeeping_sync

    @wraps(original_sample)
    def capture_offsets(model_runner: Any, *args: Any, **kwargs: Any) -> Any:
        """Capture seeded generator offsets before sampling mutates them."""
        if hasattr(model_runner, _PRE_SAMPLE_GENERATOR_OFFSETS):
            raise RuntimeError("Overlapping vLLM sampling calls cannot preserve generator offsets")
        generators = model_runner.input_batch.generators.values()
        offsets = {
            id(generator): (generator, int(generator.get_offset()))
            for generator in generators
        }
        setattr(model_runner, _PRE_SAMPLE_GENERATOR_OFFSETS, offsets)
        try:
            return original_sample(model_runner, *args, **kwargs)
        except Exception:
            delattr(model_runner, _PRE_SAMPLE_GENERATOR_OFFSETS)
            raise

    @wraps(original_bookkeeping)
    def restore_discarded_offsets(model_runner: Any, *args: Any, **kwargs: Any) -> Any:
        """Restore offsets consumed by discarded partial-prefill samples."""
        offsets = getattr(model_runner, _PRE_SAMPLE_GENERATOR_OFFSETS, None)
        if offsets is None:
            raise RuntimeError("vLLM bookkeeping ran without captured generator offsets")
        discarded_indices = model_runner.discard_request_indices.np[
            : model_runner.num_discarded_requests
        ]
        discarded_generators = []
        for request_index in discarded_indices:
            generator = model_runner.input_batch.generators.get(int(request_index))
            if generator is None:
                continue
            captured = offsets.get(id(generator))
            if captured is None or captured[0] is not generator:
                raise RuntimeError("vLLM changed a seeded generator before bookkeeping")
            discarded_generators.append(captured)
        try:
            result = original_bookkeeping(model_runner, *args, **kwargs)
            for generator, offset in discarded_generators:
                generator.set_offset(offset)
            return result
        finally:
            delattr(model_runner, _PRE_SAMPLE_GENERATOR_OFFSETS)

    model_runner_cls._sample = capture_offsets
    model_runner_cls._bookkeeping_sync = restore_discarded_offsets
    model_runner_cls._hyper_rl_partial_prefill_rng_patched = True


def install_vllm_ascend_partial_prefill_rng_fix() -> None:
    """Install the version-pinned vLLM-Ascend partial-prefill RNG correction."""
    try:
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner  # pylint: disable=C0415
    except ImportError as error:
        raise ValueError(f"vLLM-Ascend model runner is unavailable: {error}") from error
    _patch_vllm_ascend_partial_prefill_rng(NPUModelRunner)


def install_vllm_weight_sync_hooks(*, private_lifecycle: bool = True) -> None:
    """Install stable worker RPCs and optionally pinned private lifecycle patches."""
    from vllm.v1.worker.worker_base import WorkerBase  # pylint: disable=C0415
    if not hasattr(WorkerBase, "reload_weights"):
        setattr(WorkerBase, "reload_weights", reload_weights)
    if not hasattr(WorkerBase, "get_policy_weight_fingerprint"):
        setattr(
            WorkerBase,
            "get_policy_weight_fingerprint",
            get_policy_weight_fingerprint,
        )
    if not hasattr(WorkerBase, "commit_reloaded_weights"):
        setattr(WorkerBase, "commit_reloaded_weights", commit_reloaded_weights)
    if private_lifecycle:
        _patch_ascend_weight_update_lifecycle()
        _patch_engine_core_wake_lifecycle()
__all__ = [
    "commit_reloaded_weights",
    "get_policy_weight_fingerprint",
    "install_vllm_ascend_partial_prefill_rng_fix",
    "install_vllm_weight_sync_hooks",
    "reload_weights",
]
