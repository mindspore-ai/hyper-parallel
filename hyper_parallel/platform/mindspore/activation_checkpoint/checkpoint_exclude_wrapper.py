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
"""MindSpore wrapper for regions that should be saved instead of recomputed."""
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict

from hyper_parallel.core.activation_checkpoint.recompute_state import get_recompute_state
from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import ActivationWrapper


class _CheckpointExcludeCache:
    """Store excluded-region outputs for one checkpoint invocation."""

    def __init__(self) -> None:
        """Initialize an empty per-checkpoint output cache."""
        self._outputs: Dict[int, Deque[Any]] = defaultdict(deque)

    def save(self, wrapper_id: int, output: Any) -> None:
        """Save one output produced by a checkpoint-excluded region."""
        self._outputs[wrapper_id].append(output)

    def pop(self, wrapper_id: int) -> Any:
        """Return the matching forward output during recomputation."""
        outputs = self._outputs.get(wrapper_id)
        if not outputs:
            raise RuntimeError("No cached forward output is available for this checkpoint exclusion wrapper")
        output = outputs.popleft()
        if not outputs:
            self._outputs.pop(wrapper_id)
        return output

    def clear(self) -> None:
        """Release outputs not consumed because recomputation stopped early."""
        self._outputs.clear()


def _pack_saved_tensor(tensor: Any) -> Any:
    """Return the tensor data without retaining the input tensor object."""
    return tensor.data


def _unpack_saved_tensor(tensor: Any) -> Any:
    """Restore the saved tensor for backward."""
    return tensor


def _saved_tensors_context() -> Any:
    """Create an inner hook that stores real tensors instead of placeholders."""
    import mindspore as ms  # pylint: disable=C0415
    return ms.saved_tensors_hooks(_pack_saved_tensor, _unpack_saved_tensor)


_CHECKPOINT_EXCLUDE_CACHE_KEY = object()


class CheckpointExcludeWrapper(ActivationWrapper):
    """Exclude a callable region from checkpoint recomputation."""

    def __init__(self, module: Callable[..., Any]) -> None:
        """Initialize a checkpoint exclusion wrapper for a MindSpore Cell or function."""
        if not callable(module):
            raise ValueError("module must be a MindSpore Cell or callable")
        super().__init__(module, track_overlaps=False)

    def construct(self, *args: Any, **kwargs: Any) -> Any:
        """Execute normally outside recompute and return the cached output in recompute."""
        state = get_recompute_state()
        if state is None:
            return self._ckpt_wrapped_module(*args, **kwargs)
        cache = state.get_resource(_CHECKPOINT_EXCLUDE_CACHE_KEY, _CheckpointExcludeCache)
        if state.is_recomputing:
            return cache.pop(id(self))

        with _saved_tensors_context():
            output = self._ckpt_wrapped_module(*args, **kwargs)
        cache.save(id(self), output)
        return output


def checkpoint_exclude_wrapper(module: Callable[..., Any]) -> CheckpointExcludeWrapper:
    """Wrap a MindSpore Cell or function so its region is not recomputed.

    Args:
        module: MindSpore Cell or callable to execute only during the original
            checkpoint forward pass.

    Returns:
        A wrapper that saves the callable's autograd tensors and reuses its
        forward output while replaying a non-reentrant checkpoint.

    Note:
        This feature requires MindSpore PyNative mode and a surrounding
        HyperParallel checkpoint configured with ``use_reentrant=False``.
    """
    return CheckpointExcludeWrapper(module)
