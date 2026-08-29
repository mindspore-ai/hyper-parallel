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

MindSpore's captured ``grad_fn`` is scoped to the body submodule's own weights,
so the body backward never deposits a grad on the preprocess parameters —
``MPIPE_TRANSPOSE_BWD`` runs the stage-0 backward for the preprocess explicitly.

Note:
    Exercised by the MindSpore ST gate (msrun), not the torch/CPU coverage job
    (mindspore isn't importable there).
"""
from mindspore import ops

from hyper_parallel.core.pipeline_parallel.mpipe.executor_base import MPipeTransposeExecutorBase
from hyper_parallel.platform.mindspore.pipeline_parallel.backward import forward_and_gradfn


class MPipeTransposeExecutor(MPipeTransposeExecutorBase):  # pragma: no cover
    """MindSpore runtime for the ``MPIPE_*`` steps of MPipe Transpose."""

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

    def _mark_requires_grad(self, tensor) -> None:
        tensor._requires_grad = True  # pylint: disable=protected-access

    def _explicit_forward_before_backward(self, inputs, kwargs, grads) -> None:
        weights = tuple(self._preprocess.trainable_params())
        out, grad_fn = forward_and_gradfn(
            self._preprocess, *inputs, weights=weights, grad_position=None, **kwargs)
        # ``grad_fn`` needs a cotangent for EVERY output, so zero-fill the ones
        # that got no gradient; that matches what torch does by omitting them.
        outs = out if isinstance(out, (tuple, list)) else (out,)
        filled = [g if g is not None else ops.zeros_like(out_i)
                  for out_i, g in zip(outs, grads)]
        sens = filled[0] if len(filled) == 1 else tuple(filled)
        grad_fn(sens=sens)

    # --- owner-does-backward hooks: unreached on MindSpore --------------------
    # ScheduleMPipeTranspose forces owner_backward off on MindSpore (the captured
    # grad_fn is body-scoped, so a connected tower forward can't retain a usable
    # backward), so no MPIPE_GRAD_* step is ever emitted on this backend. These
    # satisfy the abstract base; reaching one signals a scheduling bug.
    def _connected_forward(self, args, kwargs):
        raise NotImplementedError("_connected_forward: MPipe owner-backward is unsupported on MindSpore.")

    def _detach_for_wire(self, tensor):
        raise NotImplementedError("_detach_for_wire: MPipe owner-backward is unsupported on MindSpore.")

    def _zeros_like(self, tensor):
        raise NotImplementedError("_zeros_like: MPipe owner-backward is unsupported on MindSpore.")

    def _owner_transpose_backward(self, retained_out, grads) -> None:
        raise NotImplementedError(
            "_owner_transpose_backward: MPipe owner-backward is unsupported on MindSpore.")

    def _snapshot_tower_grads(self):
        raise NotImplementedError(
            "_snapshot_tower_grads: MPipe owner-backward is unsupported on MindSpore.")

    def _reduce_grads(self, group_info, snapshot) -> None:
        raise NotImplementedError("_reduce_grads: MPipe owner-backward is unsupported on MindSpore.")
