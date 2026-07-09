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
"""MindSpore autograd backend for the MPipe Transpose schedule.

Only the autograd-specific hooks live here; the broadcast / P2P transport and
step orchestration are inherited from
:class:`~hyper_parallel.core.pipeline_parallel.mpipe.executor_base.MPipeTransposeExecutorBase`.

Unlike torch, MindSpore's captured ``grad_fn`` is scoped to the body submodule's
own weights (see ``forward_and_gradfn`` / ``PipelineStage.forward_one_chunk``),
so the body backward only deposits the *input* gradient on the preprocess
output, never on the preprocess parameters.  Therefore **every** micro-batch —
transposed and non-transposed — needs an explicit recompute backward, and the
preprocess forward is always detached.  ``nontransposed_connected = False``
signals the schedule to emit ``MPIPE_TRANSPOSE_BWD`` for non-transposed
micro-batches too.

Note:
    This backend mirrors the established MindSpore pipeline-stage gradient
    pattern; it requires an Ascend/MindSpore environment for runtime validation.
"""
from hyper_parallel.core.pipeline_parallel.mpipe.executor_base import MPipeTransposeExecutorBase
from hyper_parallel.platform.mindspore.pipeline_parallel.backward import forward_and_gradfn


# This MindSpore backend is exercised by the MindSpore ST gate (msrun), not the
# torch/CPU coverage job (mindspore isn't importable there) — exclude from coverage.
class MPipeTransposeExecutor(MPipeTransposeExecutorBase):  # pragma: no cover
    """MindSpore runtime for the ``MPIPE_*`` steps of MPipe Transpose."""

    nontransposed_connected = False

    def _broadcast_tensors(self):
        # Only trainable params change after the optimizer step and need
        # re-syncing; frozen params are identical on every rank from init.
        for param in self._preprocess.get_parameters():
            if param.requires_grad:
                yield param

    def _detached_forward(self, args, kwargs):
        # A plain pynative Cell call builds no grad graph (it is naturally
        # detached). The base marks it grad-requiring (when T > 0) so the body's
        # grad_fn computes the input gradient on it.
        return self._preprocess(*args, **kwargs)

    def _connected_forward(self, args, kwargs):
        # Unused (nontransposed_connected is False) but required by the base API.
        return self._preprocess(*args, **kwargs)

    def _mark_requires_grad(self, tensor) -> None:
        tensor._requires_grad = True  # pylint: disable=protected-access

    def _recompute_backward(self, inputs, kwargs, grad) -> None:
        weights = tuple(self._preprocess.trainable_params())
        _, grad_fn = forward_and_gradfn(
            self._preprocess, *inputs, weights=weights, grad_position=None, **kwargs)
        grad_fn.accumulate_grad(sens=grad)
