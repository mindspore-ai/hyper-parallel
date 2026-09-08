# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""mindspore pipeline stage"""
import contextlib

from hyper_parallel.platform import get_platform
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.platform.mindspore.pipeline_parallel.backward import forward_and_gradfn

# ``get_platform()`` is called lazily inside methods, not at module scope:
# ``platform/mindspore/platform.py`` imports ``PipelineStageBase`` from this
# module during its own initialization, so a module-scope call here would
# re-enter that partial import and raise ImportError.


class PipelineStageBase:
    """
    PipelineStage represents a pipeline stage in pipeline parallelism.

    PipelineStage requires the input of a segmented model.

    PipelineStage encapsulates the forward and backward functions used in PipelineSchedule,
    as well as P2P communication.

    Args:
        submodule (Cell): Segmented model.
        stage_index (int): Stage index of current stage.
        stage_num (int): Total stage number.
        group (str): Group of p2p communication.
        has_backward (bool, optional): Specify whether this stage has backward. Default ``True``.
        recv_info(P2PInfo, optional): Specify Receive information. Default ``None``.
        send_info(P2PInfo, optional): Specify Send information. Default ``None``.
    """
    def __init__(self, submodule, stage_index, stage_num, group=None, has_backward=True):
        super().__init__()
        self.submodule = submodule
        self.pp_group = self._check_pp_group(group)
        self._backward_func = None
        self.stage_index = stage_index
        self.stage_num = stage_num
        if has_backward:
            self.submodule.set_grad(True)
            self._construct_backward_func()
        self.recompute_handles = {}
        self.fwd_outputs_cache = {}
        self.fwd_grad_fn_cache = {}
        self.bwd_cache = {}
        self.last_stage_outputs = None  # Initialized in forward_one_chunk()
        # Lazily populated after the stage is first unsharded. HSDP keeps each
        # unsharded Parameter identity stable across later shard/unshard cycles.
        self._trainable_params = None
        # Set by the schedule (``PipelineScheduleRuntime._init_stages``); called
        # as ``hook(stage_index, micro_index)`` right after a forward chunk
        # completes.  Drives the fwd-boundary P2P issue without requiring the
        # OVERLAP callback to cooperate.  ``None`` -> no-op.
        self._after_forward_chunk = None

    def clear_cache(self):
        """clear cache."""
        self.fwd_outputs_cache.clear()
        self.fwd_grad_fn_cache.clear()
        self.bwd_cache.clear()

    @staticmethod
    def _check_pp_group(group):
        """check the type of pipeline group, if it is None, perform default initialization."""
        if group is None:
            return None
        if not isinstance(group, str):
            raise TypeError("Argument 'group' must be type of str, but got type of {type(group)}.")
        return group

    @staticmethod
    def _clear_recv_buffer(recv_info, micro_index):
        """clear fwd and bwd recv buffer."""
        if micro_index not in recv_info:
            return
        for info in recv_info[micro_index]:
            info.buffer = None

    @staticmethod
    def _grad_position_from_requires_grad(composite_args):
        """Return positional Tensor indices that already require gradients.

        Pipeline stages only route gradients for positional stage inputs. Always
        returning explicit indices avoids the broader ``-1`` contract, which also
        selects keyword Tensor inputs in ``forward_and_gradfn``.
        """
        # pylint: disable=C0415
        from mindspore import Tensor
        tensor_indices = [i for i, a in enumerate(composite_args) if isinstance(a, Tensor)]
        requires_grad_indices = [
            i for i in tensor_indices
            if composite_args[i]._requires_grad  # pylint: disable=protected-access
        ]
        return tuple(requires_grad_indices)

    @property
    def is_first_stage(self):
        """return if is first stage."""
        return self.stage_index == 0

    @property
    def is_last_stage(self):
        """return if is last stage."""
        return self.stage_index == self.stage_num - 1

    def forward_one_chunk(self, micro_index, args=None, kwargs=None):
        """Execution a forward function."""
        if self.is_first_stage:
            composite_args = args
        else:
            if micro_index in self.args_recv_info:
                composite_args = [recv_info.buffer for recv_info in self.args_recv_info[micro_index]]
            else:
                raise RuntimeError(f"The exec order is wrong. The corresponding forward calculation \
                                    is executed before the Receive operation. micro is {micro_index}.")
        composite_kwargs = kwargs or {}
        if self._has_backward:
            grad_position = self._grad_position_from_requires_grad(composite_args)
            if self._trainable_params is None:
                self._trainable_params = tuple(self.submodule.trainable_params())
            platform = get_platform()
            with platform.recompute_handle_collector_ctx() as handles:
                out, grad_fn = forward_and_gradfn(
                    self.submodule,
                    *composite_args,
                    weights=self._trainable_params,
                    grad_position=grad_position,
                    **composite_kwargs,
                )
            self.fwd_grad_fn_cache[micro_index] = grad_fn
            self.recompute_handles[micro_index] = handles
        else:
            out = self.submodule(*composite_args, **composite_kwargs)
        out_tuple = out if isinstance(out, tuple) else (out,)
        self.fwd_outputs_cache[micro_index] = out_tuple
        if self.is_last_stage:
            self.last_stage_outputs = out
        if self._after_forward_chunk is not None:
            # fwd/bwd boundary signal: the chunk's kernels are enqueued and its
            # output is cached, so the schedule may issue boundary P2P (e.g.
            # this chunk's FWD_SEND) while the paired backward is still running.
            self._after_forward_chunk(self.stage_index, micro_index)
        return out

    def recompute_one_chunk(self, micro_index):
        """Re-run this chunk's checkpointed blocks ahead of its backward.

        Fires each recompute handle collected during the forward under a
        stable per-chunk session, materializing and caching the activations
        so the matching :meth:`backward_one_chunk` reuses them instead of
        re-running.

        Must be invoked on a thread with no concurrent forward: for
        ``overlap_b_f`` the caller runs this on the main thread *before*
        ``overlap.run`` submits work to the backward worker, so the forward
        re-run never races the paired chunk's forward.  A no-op when the
        chunk has no checkpointed blocks.

        Args:
            micro_index: Micro-batch index whose checkpointed blocks are
                recomputed.
        """
        if not self._has_backward:
            return
        handles = self.recompute_handles.get(micro_index)
        if not handles:
            return
        platform = get_platform()
        session_id = (self.stage_index, micro_index)
        with platform.recompute_session_ctx(session_id=session_id, retain_on_unpack=True):
            for handle in handles:
                platform.recompute_handle(handle, session_id)

    def backward_one_chunk(self, micro_index):
        """Execution a backward function."""
        if not self._has_backward:
            return
        grad_fn = self.fwd_grad_fn_cache.pop(micro_index)
        handles = self.recompute_handles.pop(micro_index, None)
        platform = get_platform()
        session_id = (self.stage_index, micro_index)
        # If recompute_one_chunk already ran (overlap_b_f, on the main thread)
        # the session cache is pre-populated and the unpack below reuses it
        # without re-running.  Otherwise (plain backward) the re-run fires
        # lazily here on this same thread, which is safe because no forward
        # runs concurrently outside overlap_b_f.
        session_ctx = (
            platform.recompute_session_ctx(session_id=session_id, retain_on_unpack=False)
            if handles else contextlib.nullcontext()
        )
        with session_ctx:
            if self.is_first_stage:
                sens = self._build_padded_sens(micro_index)
                grad_fn.accumulate_grad(sens=sens)
            else:
                if self.is_last_stage:
                    sens = self.get_last_stage_sens(self.last_stage_outputs)
                else:
                    sens = self._build_padded_sens(micro_index)
                grad_fn.accumulate_grad(sens=sens)
        if handles:
            platform.clear_recompute_session(session_id)
        if not self.is_first_stage:
            input_grads = [recv_info.buffer.grad for recv_info in self.args_recv_info[micro_index]
                           if recv_info.requires_grad]
            self.bwd_cache[micro_index] = input_grads
        self._clear_recv_buffer(self.grad_recv_info, micro_index)
        self._clear_recv_buffer(self.args_recv_info, micro_index)
        if self.is_last_stage:
            self.fwd_outputs_cache.pop(micro_index, None)

    def backward_input_one_chunk(self, micro_index):
        """dx-only backward; keeps grad_fn alive in cache for the paired dw call.

        Writes ``bwd_cache[micro_index]`` so ``exec_bwd_send_ops`` can pop and
        send the input grads while dw runs locally.  Does NOT clear recv
        buffers — dw still needs ``grad_fn._saved_intermediates`` which may
        reference activations stored in ``args_recv_info``.

        The first stage is a no-op: its input does not require grad, so there
        is no dx to compute and no gradient to send upstream.  The paired
        :meth:`backward_weight_one_chunk` runs the full backward instead, so
        ``grad_fn`` is left untouched in the cache for it to pop.
        """
        if not self._has_backward:
            return
        with get_platform().profiler_record(f"backward_input_one_chunk: stage_{self.stage_index}/mi_{micro_index}"):
            if self.is_first_stage:
                return
            # Index, NOT pop: backward_weight_one_chunk performs the terminal pop.
            grad_fn = self.fwd_grad_fn_cache[micro_index]
            handles = self.recompute_handles.get(micro_index)
            if self.is_last_stage:
                sens = self.get_last_stage_sens(self.last_stage_outputs)
            else:
                sens = self._build_padded_sens(micro_index)
            # The session registry is global but the "current session" is a
            # ContextVar, so enter it on THIS thread or the unpack misses the
            # pre-fired cache and lazily re-runs the forward here — racing the
            # paired forward and firing stray overlap hooks on the bwd thread.
            # retain_on_unpack=True: dw consumes the same session afterwards.
            session_ctx = (
                get_platform().recompute_session_ctx(
                    session_id=(self.stage_index, micro_index), retain_on_unpack=True)
                if handles else contextlib.nullcontext()
            )
            with session_ctx:
                _ = grad_fn.compute_input_grad(sens=sens)
            input_grads = [recv_info.buffer.grad for recv_info in self.args_recv_info[micro_index]
                           if recv_info.requires_grad]
            self.bwd_cache[micro_index] = input_grads

    def backward_weight_one_chunk(self, micro_index):
        """dw-only backward; pops grad_fn (terminal) and clears recv buffers.

        For the first stage there is no captured dx state — its input does not
        require grad, so :meth:`backward_input_one_chunk` was a no-op and no
        intermediate gradients were saved.  The full backward ``grad_fn(sens)``
        runs here instead, which yields only weight gradients (the stage has no
        input grad to compute).
        """
        if not self._has_backward:
            return
        with get_platform().profiler_record(f"backward_weight_one_chunk: stage_{self.stage_index}/mi_{micro_index}"):
            grad_fn = self.fwd_grad_fn_cache.pop(micro_index)
            handles = self.recompute_handles.pop(micro_index, None)
            platform = get_platform()
            session_id = (self.stage_index, micro_index)
            # Terminal consumer of the chunk's recompute session (dx retained).
            session_ctx = (
                platform.recompute_session_ctx(session_id=session_id, retain_on_unpack=False)
                if handles else contextlib.nullcontext()
            )
            with session_ctx:
                if self.is_first_stage:
                    sens = self._build_padded_sens(micro_index)
                    grad_fn.accumulate_grad(sens=sens)
                else:
                    if not grad_fn._saved_intermediates:  # pylint: disable=protected-access
                        raise RuntimeError(
                            f"stage: {self.stage_index} micro_{micro_index} dw called before dx."
                        )
                    grad_fn.compute_weight_grad()
            if handles:
                platform.clear_recompute_session(session_id)
            self._clear_recv_buffer(self.grad_recv_info, micro_index)
            self._clear_recv_buffer(self.args_recv_info, micro_index)
            if self.is_last_stage:
                self.fwd_outputs_cache.pop(micro_index, None)

    def _construct_backward_func(self):
        """construct backward func."""
        enable_mindspore_backward_compat()
        self._backward_func = None
