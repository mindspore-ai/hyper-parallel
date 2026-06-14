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
"""Torch autograd backend for the MPipe Transpose schedule.

Only the autograd-specific hooks live here; the broadcast / P2P transport and
step orchestration are inherited from
:class:`~hyper_parallel.core.pipeline_parallel.mpipe.executor_base.MPipeTransposeExecutorBase`.

Torch's ``autograd.backward`` traverses the whole connected graph to all leaves,
so non-transposed micro-batches can run a graph-connected preprocess forward and
have the body backward flow into the preprocess parameters automatically (no
recompute step) — hence ``nontransposed_connected = True``.
"""
import torch

from hyper_parallel.core.pipeline_parallel.mpipe.executor_base import MPipeTransposeExecutorBase


class MPipeTransposeExecutor(MPipeTransposeExecutorBase):
    """Torch runtime for the ``MPIPE_*`` steps of MPipe Transpose."""

    nontransposed_connected = True

    def _broadcast_tensors(self):
        # Only trainable params change after the optimizer step and need
        # re-syncing; frozen params and constant buffers are identical on every
        # rank from init, so broadcasting them would be wasted bandwidth.
        for param in self._preprocess.parameters():
            if param.requires_grad:
                yield param.data

    def _detached_forward(self, args, kwargs):
        with torch.no_grad():
            out = self._preprocess(*args, **kwargs)
        return out.detach()

    def _connected_forward(self, args, kwargs):
        return self._preprocess(*args, **kwargs)

    def _mark_requires_grad(self, tensor) -> None:
        tensor.requires_grad_(True)

    def _recompute_backward(self, inputs, kwargs, grad) -> None:
        out = self._preprocess(*inputs, **kwargs)
        torch.autograd.backward(out, grad_tensors=grad)

    @staticmethod
    def _contiguous(tensor):
        return tensor.contiguous()
