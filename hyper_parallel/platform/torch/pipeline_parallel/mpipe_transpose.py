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
"""
import logging

import torch

from hyper_parallel.platform import get_platform
from hyper_parallel.core.pipeline_parallel.mpipe.executor_base import MPipeTransposeExecutorBase

platform = get_platform()
logger = logging.getLogger(__name__)


class MPipeTransposeExecutor(MPipeTransposeExecutorBase):
    """Torch runtime for the ``MPIPE_*`` steps of MPipe Transpose."""

    @staticmethod
    def _local(tensor):
        # An FSDP-sharded tower param/grad is a DTensor; operate on its local
        # shard so pp collectives skip the DTensor dispatcher. Plain: pass through.
        to_local = getattr(tensor, "to_local", None)
        return to_local() if callable(to_local) else tensor

    def _broadcast_tensors(self):
        """Yield each trainable tower param's local shard, resharded first.

        Frozen params and buffers stay identical from init, so only the
        trainable ones need re-syncing after the optimizer step.
        """
        # Reshard first: under owner-backward stage 0 holds the tower unsharded
        # while the replicas are sharded, so the sizes would disagree and hang.
        for module in self._preprocess.modules():
            reshard = getattr(module, "reshard", None)
            if callable(reshard):
                reshard()
        for param in self._preprocess.parameters():
            if param.requires_grad:
                yield self._local(param.data)

    def _detached_forward(self, args, kwargs):
        with torch.no_grad():
            out = self._preprocess(*args, **kwargs)
        # The preprocess output may be a single tensor (text body input) or a
        # tuple (e.g. a VL visual payload: image_embeds + DeepStack levels).
        if isinstance(out, (tuple, list)):
            return tuple(t.detach() for t in out)
        return out.detach()

    def _connected_forward(self, args, kwargs):
        return self._preprocess(*args, **kwargs)

    def _mark_requires_grad(self, tensor) -> None:
        tensor.requires_grad_(True)

    def _explicit_forward_before_backward(self, inputs, kwargs, grads) -> None:
        out = self._preprocess(*inputs, **kwargs)
        out = out if isinstance(out, (tuple, list)) else (out,)
        # Backprop only the outputs that received a gradient; an absent
        # dL/dfeature is a zero contribution.
        pairs = [(out_i, g) for out_i, g in zip(out, grads) if g is not None]
        if not pairs:
            return
        outs, grad_tensors = zip(*pairs)
        torch.autograd.backward(outs, grad_tensors=grad_tensors)

    @staticmethod
    def _contiguous(tensor):
        return tensor.contiguous()

    # --- owner-does-backward hooks (opt-in, trainable tower) -----------------

    def _detach_for_wire(self, tensor):
        # Ship a detached, contiguous copy so the owner keeps the autograd graph.
        return tensor.detach().contiguous()

    def _zeros_like(self, tensor):
        # Contiguous zero grad for a feature tensor that received none.
        return torch.zeros_like(tensor).contiguous()

    def _owner_transpose_backward(self, retained_out, grads) -> None:
        # Backprop dL/dfeatures through the retained connected tower graph on this
        # rank's replica; skip non-grad-requiring outputs (else autograd raises).
        pairs = [(out_i, g) for out_i, g in zip(retained_out, grads) if out_i.requires_grad]
        if pairs:
            outs, grad_tensors = zip(*pairs)
            torch.autograd.backward(outs, grad_tensors=grad_tensors)
        else:
            logger.debug("[mpipe] owner backward skipped: no retained output requires grad.")

    def _snapshot_tower_grads(self):
        # Clone each trainable tower grad's LOCAL shard (None where absent) so
        # _reduce_grads reduces only this run's contribution.
        return [None if p.grad is None else self._local(p.grad).detach().clone()
                for p in self._preprocess.parameters() if p.requires_grad]

    def _reduce_grads(self, group_info, snapshot) -> None:
        """SUM-reduce each trainable tower param's this-run contribution
        (grad - snapshot) over the pp replicas on the LOCAL shard, so an
        FSDP-sharded DTensor grad never meets a raw c10d collective. FSDP owns the
        orthogonal dp reduce (mean over dp_shard/dp_replicate); this owns only the
        pp sum. Reducing the delta -- not the total -- keeps prior grad-accumulation
        passes from being re-reduced; the write-back is in place so the DTensor grad
        wrapper the optimizer reads is preserved (reassigning param.grad drops it).
        """
        params = [p for p in self._preprocess.parameters() if p.requires_grad]
        if snapshot is None:  # first run (no prior accumulation) -> reduce full grad
            snapshot = [None] * len(params)
        for param, snap in zip(params, snapshot):
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            g_local = self._local(param.grad)
            delta = g_local if snap is None else (g_local - snap)
            reduced, _ = platform.all_reduce(delta.contiguous(), group_info)
            g_local.copy_(reduced if snap is None else (snap + reduced))
