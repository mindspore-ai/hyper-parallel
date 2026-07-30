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
"""HSDP scheduler"""
import functools
from typing import Any, List, Optional, Tuple, Union

from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.fully_shard.hsdp_utils import (
    FSDPSchedulerState,
    HSDPConfigV2,
    get_managed_modules_parameters
)

platform = get_platform()


class HSDPSchedulerContext:
    """HSDPSchedulerContext"""

    def __init__(self) -> None:
        # Currently only record is_last_backward flag for scheduler context.
        self.is_last_backward: bool = True
        # flag to identify "root_module"
        self.root_module = None
        # all_hsdp_schedulers (for one module tree structure)
        self.all_hsdp_schedulers = []
        # Flag to avoid repeatedly initializing parameter FQNs.
        self._param_fqn_initilized = False


class HSDPSchedulerV2:
    """HSDPScheduler is used to scheduler hsdp"""
    root_bp_state = False


    def __init__(self, cell: Union[platform.Module, Tuple[platform.Module, ...]], mesh,
                 reshard_after_forward, shard_placement_fn,
                 mp_policy, offload_policy, ignored_params, replicate_params, device, comm_fusion,
                 comm_fusion_zero_copy=False):
        """init hsdp scheduler.

        Args:
            cell: A single platform.Module or tuple of platform.Module to manage as one FSDP unit.
        """
        self.modules = (cell,) if isinstance(cell, platform.Module) else tuple(cell)
        self.cell = self.modules[0]
        self.mesh: DeviceMesh = mesh
        self.reshard_after_forward = reshard_after_forward
        self.shard_placement_fn = shard_placement_fn
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.ignored_params = ignored_params
        self.replicate_params = replicate_params
        self.device = device
        self.scheduler_state = None
        self.forward_prefetch_cells = []
        self.backward_prefetch_cells = []
        self._backup_forward_fetch = None
        # Flag to identify root module.
        self._is_root = False
        # module and its all sub-modules share one same 'HSDPSchedulerContext'
        self.scheduler_ctx = HSDPSchedulerContext()
        # When ``fully_shard`` is given multiple root modules, forward pre/post hooks coordinate
        # so unshard / PostBackward / reshard run once per forward (aligned with PyTorch FSDP2).
        self._fsdp_group_post_pending: Optional[set] = set() if len(self.modules) > 1 else None
        self.config = HSDPConfigV2(
            mesh,
            reshard_after_forward,
            shard_placement_fn,
            mp_policy,
            offload_policy,
            ignored_params,
            replicate_params,
            comm_fusion=comm_fusion,
            comm_fusion_zero_copy=comm_fusion_zero_copy,
        )
        self._init_platform()
        self._new_cell_state()
        self._register_hooks()

    def _init_platform(self):
        """Initialize the platform."""
        raise NotImplementedError("HSDPScheduler subclasses must implement _init_platform")

    def _new_cell_state(self):
        """Create a new cell state."""
        raise NotImplementedError("HSDPScheduler subclasses must implement _new_cell_state")

    def _register_hooks(self):
        """Register hooks."""
        raise NotImplementedError("HSDPScheduler subclasses must implement _register_hooks.")

    def _register_forward_backward_hooks(self):
        """Register module forward and backward hook."""
        raise NotImplementedError("HSDPScheduler subclasses must implement _register_forward_backward_hooks.")

    def _get_managed_params(self):
        """Return deduplicated parameters from all managed modules."""
        return get_managed_modules_parameters(self.modules, self.ignored_params)

    def set_reshard_after_forward(self, reshard_after_forward: bool) -> None:
        """Set reshard_after_forward flag.

        Args:
            reshard_after_forward: Whether to reshard parameters after forward.
        """
        if not isinstance(reshard_after_forward, bool):
            raise ValueError(f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}")
        self.reshard_after_forward = reshard_after_forward
        self.config.reshard_after_forward = reshard_after_forward

    def set_reshard_after_backward(self, reshard_after_backward: bool) -> None:
        """Set reshard_after_backward flag.

        Args:
            reshard_after_backward: Whether to reshard after backward completes.
        """
        if not isinstance(reshard_after_backward, bool):
            raise ValueError(f"reshard_after_backward should be a bool, got {type(reshard_after_backward)}")
        if self.hsdp_state is not None:
            self.hsdp_state.reshard_after_backward = reshard_after_backward

    def set_requires_all_reduce(self, requires_all_reduce: bool) -> None:
        """Set requires_all_reduce flag.

        Args:
            requires_all_reduce: Whether this unit participates in all-reduce.
        """
        if not isinstance(requires_all_reduce, bool):
            raise ValueError(f"requires_all_reduce should be a bool, got {type(requires_all_reduce)}")
        if self.hsdp_state is not None:
            self.hsdp_state.requires_all_reduce = requires_all_reduce

    def set_requires_grad_sync(self, requires_grad_sync: bool) -> None:
        """Set flag controlling whether gradients are synchronized.

        Args:
            requires_grad_sync: When True, enable grad sync for this scheduler.
        """
        if not isinstance(requires_grad_sync, bool):
            raise ValueError(f"requires_grad_sync should be a bool, got {type(requires_grad_sync)}")
        self.hsdp_state.set_requires_grad_sync(requires_grad_sync)

    def wait_for_pending_reductions(self) -> None:
        """Wait for all asynchronous gradient reductions owned by this backend."""
        raise NotImplementedError(
            "HSDPScheduler subclasses must implement wait_for_pending_reductions."
        )

    # pylint: disable=W0613
    def _hsdp_forward_pre_hook(self, cell, args, kwargs):
        """Forward pre hook to unsharded parameter for forward process."""
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return args, kwargs
        if HSDPSchedulerV2.root_bp_state:
            self._disable_forward_prefetch_for_recompute()
        if self.scheduler_ctx.root_module is None:
            self.scheduler_ctx.root_module = self.cell
            self._is_root = True
            for module_name, module in platform.get_cells_and_names(self.scheduler_ctx.root_module):
                from hyper_parallel.core.fully_shard.api import HSDPModule  # pylint: disable=C0415
                if isinstance(module, HSDPModule):
                    submod_scheduler = module.hsdp_scheduler
                    if submod_scheduler and submod_scheduler.scheduler_ctx is not self.scheduler_ctx:
                        submod_scheduler.scheduler_ctx = self.scheduler_ctx
                        submod_scheduler.hsdp_state.module_name = module_name
                    self.scheduler_ctx.all_hsdp_schedulers.append(submod_scheduler)
        self.scheduler_state = FSDPSchedulerState.PRE_FORWARD
        self._init_params_fqn()
        self._lazy_init_all_states()
        if self.mp_policy.cast_forward_inputs and self.mp_policy.param_dtype:
            cast_fn = functools.partial(self.platform.cast_fp_tensor, self.mp_policy.param_dtype)
            args = self.platform.apply_to_tensors(cast_fn, args)
            kwargs = self.platform.apply_to_tensors(cast_fn, kwargs)
        with self.platform.profiler_record(f"pre_forward unshard:{self.hsdp_state.module_name}"):
            self.hsdp_state.unshard()
        for prefetch_cell in self.forward_prefetch_cells:
            with self.platform.profiler_record(f"pre_forward prefetch:"
                                               f"{prefetch_cell.hsdp_scheduler.hsdp_state.module_name}"):
                prefetch_cell.hsdp_scheduler.hsdp_state.prefetch()
        return args, kwargs

    def _lazy_init_all_states(self):
        if self._is_root and self.scheduler_ctx.root_module is not None:
            for submod_scheduler in self.scheduler_ctx.all_hsdp_schedulers:
                hsdp_state = submod_scheduler.hsdp_state
                if hsdp_state:
                    hsdp_state.lazy_init()

    def _init_params_fqn(self):
        """Initialize fully sharded parameter FQNs once for the module tree."""
        if not self._is_root or self.scheduler_ctx.root_module is None:
            return
        if self.scheduler_ctx._param_fqn_initilized:  # pylint: disable=protected-access
            return
        # Build a map from original (sharded) parameter tensor → hsdp_param wrapper,
        # covering both sharded hsdp_params and replicate_params.
        param_to_hsdp_param = {}
        for submod_scheduler in self.scheduler_ctx.all_hsdp_schedulers:
            hsdp_state = submod_scheduler.hsdp_state
            if hsdp_state is None:
                continue
            for hsdp_param in hsdp_state._iter_managed_params():  # pylint: disable=W0212
                orig_param = hsdp_param.sharded_param
                # Shared parameters: keep only the first mapping to preserve the
                # first-seen FQN (consistent with the deduplication in _init_hsdp_params).
                if orig_param not in param_to_hsdp_param:
                    param_to_hsdp_param[orig_param] = hsdp_param

        # Walk the full parameter tree and assign FQNs; skip params already seen
        # (shared-parameter deduplication: first name wins).
        visited_params = set()
        for param_name, parameter in platform.parameters_dict(self.scheduler_ctx.root_module):
            if parameter in visited_params:
                continue
            visited_params.add(parameter)
            hsdp_param = param_to_hsdp_param.get(parameter)
            if hsdp_param is not None:
                hsdp_param._param_fqn = param_name  # pylint: disable=W0212
        self.scheduler_ctx._param_fqn_initilized = True  # pylint: disable=protected-access

    # pylint: disable=W0613, R1710
    def _hsdp_forward_hook(self, cell, inputs, outputs):
        """Forward hook to shard parameter for saving memory."""
        if self.scheduler_state == FSDPSchedulerState.PRE_BACKWARD:
            return
        self.scheduler_state = FSDPSchedulerState.FORWARD
        if self.reshard_after_forward:
            with self.platform.profiler_record(f"forward reshard:{self.hsdp_state.module_name}"):
                self.hsdp_state.shard(shard_replicate=False)
        if self.mp_policy.output_dtype is not None:
            outputs = self.platform.apply_to_tensors(
                functools.partial(self.platform.cast_fp_tensor, self.mp_policy.output_dtype),
                outputs,
            )
        return outputs

    # pylint: disable=W0613
    def _hsdp_backward_pre_hook(self, cell, grad_outputs):
        """Backward pre hook to unsharded parameter for backward process."""
        self.scheduler_state = FSDPSchedulerState.PRE_BACKWARD
        if self.reshard_after_forward:
            with self.platform.profiler_record(f"pre_backward unshard:{self.hsdp_state.module_name}"):
                self.hsdp_state.unshard(unshard_replicate=False)
        for prefetch_cell in self.backward_prefetch_cells:
            with self.platform.profiler_record(f"pre_backward prefetch:"
                                               f"{prefetch_cell.hsdp_scheduler.hsdp_state.module_name}"):
                prefetch_cell.hsdp_scheduler.hsdp_state.prefetch(unshard_replicate=False)

    # pylint: disable=W0613
    def _hsdp_backward_hook(self, cell, grad_inputs, grad_outputs):
        """Backward hook to shard parameter for optimizer process or saving memory."""
        self.scheduler_state = FSDPSchedulerState.BACKWARD
        with self.platform.profiler_record(f"post_backward:{self.hsdp_state.module_name}"):
            self.hsdp_state.post_backward()
        if self._fsdp_group_post_pending is not None:
            self._fsdp_group_post_pending.clear()

    # pylint: disable=W0613
    @staticmethod
    def _grouped_forward_pre_hook_skip(cell, args, kwargs):
        """Return value when grouped pre-forward should not run (first module already did).

        Default matches MindSpore Cell forward pre-hooks (explicit ``(args, kwargs)``).
        ``TorchHSDPSchedulerV2`` overrides this to return ``None`` (``nn.Module`` idiom).
        """
        return args, kwargs

    @staticmethod
    def _grouped_forward_post_hook_skip(outputs):
        """Return value when grouped post-forward is deferred to a later module in the group.

        Default returns ``outputs`` (MindSpore). ``TorchHSDPSchedulerV2`` overrides to ``None``.
        """
        return outputs

    def _grouped_forward_pre_hook(self, cell, args, kwargs):
        """Run FSDP pre-forward only for the first module in the group (PyTorch FSDP2-aligned)."""
        pending = self._fsdp_group_post_pending
        if pending is None:
            return self._forward_pre_hook(cell, args, kwargs)
        if len(pending) == 0:
            pending.update(self.modules)
            return self._forward_pre_hook(cell, args, kwargs)
        return self._grouped_forward_pre_hook_skip(cell, args, kwargs)

    def _make_grouped_forward_post_hook(self, mod):
        """Build post-forward hook: last module in the group runs reshard + output backward hooks."""

        def grouped_post_hook(cell, inputs, outputs):
            pending = self._fsdp_group_post_pending
            if pending is None:
                return self._forward_hook(cell, inputs, outputs)
            pending.discard(mod)
            if len(pending) == 0:
                return self._forward_hook(cell, inputs, outputs)
            return self._grouped_forward_post_hook_skip(outputs)

        return grouped_post_hook

    def set_forward_prefetch_cells(self, hsdp_cell_list: List[Any]) -> None:
        """Set cells prefetched during forward.

        Args:
            hsdp_cell_list: HSDP cells to prefetch ahead of forward.
        """
        self.forward_prefetch_cells = hsdp_cell_list

    def set_backward_prefetch_cells(self, hsdp_cell_list: List[Any]) -> None:
        """Set cells prefetched during backward.

        Args:
            hsdp_cell_list: HSDP cells to prefetch ahead of backward.
        """
        self.backward_prefetch_cells = hsdp_cell_list

    def _disable_forward_prefetch_for_recompute(self) -> None:
        """Temporarily disable forward prefetch during activation recompute."""
        self._backup_forward_fetch = self.forward_prefetch_cells
        self.forward_prefetch_cells = []

    def _restore_forward_prefetch_after_recompute(self) -> bool:
        """Restore forward prefetch list after a recompute forward hook finishes."""
        if self._backup_forward_fetch is None:
            return False
        self.forward_prefetch_cells = self._backup_forward_fetch
        self._backup_forward_fetch = None
        return True
