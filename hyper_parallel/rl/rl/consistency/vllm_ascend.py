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
"""Consistency-only patches for the pinned vLLM-Ascend runtime."""

from functools import wraps
from typing import Any


_PRE_SAMPLE_GENERATOR_OFFSETS = "_hyper_rl_pre_sample_generator_offsets"


def patch_partial_prefill_rng(model_runner_cls: type[Any]) -> None:
    """Restore seeded generators after vLLM discards partial-prefill samples."""
    if getattr(model_runner_cls, "_hyper_rl_partial_prefill_rng_patched", False):
        return
    original_sample = model_runner_cls._sample
    original_bookkeeping = model_runner_cls._bookkeeping_sync

    @wraps(original_sample)
    def capture_offsets(model_runner: Any, *args: Any, **kwargs: Any) -> Any:
        """Capture seeded generator offsets before sampling mutates them."""
        if hasattr(model_runner, _PRE_SAMPLE_GENERATOR_OFFSETS):
            raise RuntimeError(
                "Overlapping vLLM sampling calls cannot preserve generator offsets"
            )
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
    def restore_discarded_offsets(
        model_runner: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
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


def install_partial_prefill_rng_fix() -> None:
    """Install the version-pinned vLLM-Ascend partial-prefill RNG correction."""
    try:
        from vllm_ascend.worker.model_runner_v1 import (  # pylint: disable=C0415
            NPUModelRunner,
        )
    except ImportError as error:
        raise ValueError(f"vLLM-Ascend model runner is unavailable: {error}") from error
    patch_partial_prefill_rng(NPUModelRunner)


__all__ = ["install_partial_prefill_rng_fix", "patch_partial_prefill_rng"]
