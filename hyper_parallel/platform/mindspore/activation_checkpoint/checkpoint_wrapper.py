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
"""Activation checkpoint wrapper implementation for MindSpore."""
from typing import Any, Callable, Union

from mindspore.nn import Cell

from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import ActivationWrapper


class CheckpointWrapper(ActivationWrapper):
    """Wrap a cell with activation checkpointing."""

    def __init__(
        self,
        mod: Union[Cell, Callable],
        **checkpoint_kwargs: Any,
    ):
        super().__init__(mod)
        self.checkpoint_kwargs = checkpoint_kwargs

    def _do_checkpoint(self, wrapped_module: Any, *args: Any, **kwargs: Any) -> Any:
        # Checkpoint may save inputs before the wrapped cell's pre-hook runs.
        group_name = getattr(self, "_swap_group_name", None)
        if group_name is not None:
            from hyper_parallel.core.activation_checkpoint.swap import SwapManager  # pylint: disable=C0415
            SwapManager().set_current_group_name(group_name)

        from hyper_parallel.core.activation_checkpoint.activation_checkpoint import checkpoint  # pylint: disable=C0415
        return checkpoint(
            wrapped_module,
            *args,
            **self.checkpoint_kwargs,
            **kwargs,
        )

    def construct(self, *args: Any, **kwargs: Any) -> Any:
        return self._do_checkpoint(self._wrapped_module, *args, **kwargs)


def ckpt_wrapper(module: Union[Cell, Callable], **checkpoint_kwargs: Any) -> CheckpointWrapper:
    """Wrap *module* with activation checkpointing."""
    return CheckpointWrapper(module, **checkpoint_kwargs)
