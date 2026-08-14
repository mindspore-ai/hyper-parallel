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
"""MindSpore multicore handler for hyper-parallel."""

import threading
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from hyper_parallel.core.multicore.dryrun_memory import MegaKernelMemoryUsage


class MSMulticoreHandler:
    """MindSpore platform handler for MoE-FFN multicore operators."""

    def __init__(self):
        # Eagerly import platform/mindspore/__init__.py so that its module-level
        # code runs now (sets ASCEND_CUSTOM_OPP_PATH, preloads ctypes libs, adds
        # build/lib to sys.path).  This MUST happen before any `import mindspore`
        # elsewhere in the process; deferring to the first moe_fwd/bwd call
        # is too late when symmetric_memory or other modules import mindspore first.
        # Note: platform/mindspore/__init__.py itself does NOT import mindspore at
        # module level, so this import is safe to call early.
        import hyper_parallel.core.multicore.platform.mindspore  # noqa: F401  # pylint: disable=C0415,W0611

    _dryrun_lock = threading.Lock()

    @classmethod
    def measure_mega_kernel_memory(
            cls, kernel_name: str, kernel_call: Callable[[], Any]) -> "MegaKernelMemoryUsage":
        """Measure a mega kernel while skipping only its device launch."""
        # pylint: disable=C0415
        import os
        import mindspore as ms
        from hyper_parallel.core.multicore.dryrun_memory import MegaKernelMemoryUsage
        from hyper_parallel.platform.mindspore.symmetric_memory import MSSymmetricMemoryHandler

        simulation_level = os.environ.get("MS_SIMULATION_LEVEL", "")
        if simulation_level in {"0", "1", "2"}:
            raise RuntimeError(
                "kernel-level dryrun cannot run inside MindSpore compile simulation; "
                "start Python without MS_SIMULATION_LEVEL"
            )

        with cls._dryrun_lock:
            allocated_before = int(ms.runtime.memory_allocated())
            ms.runtime.reset_peak_memory_stats()
            previous_dryrun = os.environ.get("HP_MEGA_KERNEL_DRY_RUN")
            os.environ["HP_MEGA_KERNEL_DRY_RUN"] = "1"
            try:
                kernel_call()
                ms.runtime.synchronize()
            finally:
                if previous_dryrun is None:
                    os.environ.pop("HP_MEGA_KERNEL_DRY_RUN", None)
                else:
                    os.environ["HP_MEGA_KERNEL_DRY_RUN"] = previous_dryrun
            allocated_after = int(ms.runtime.memory_allocated())
            allocator_peak = max(
                allocated_before,
                allocated_after,
                int(ms.runtime.max_memory_allocated()),
            )
            external_reserved, external_logical = MSSymmetricMemoryHandler.dryrun_memory_stats()

        return MegaKernelMemoryUsage(
            kernel_name=kernel_name,
            allocated_before_bytes=allocated_before,
            allocated_after_bytes=allocated_after,
            allocator_peak_bytes=allocator_peak,
            external_reserved_bytes=external_reserved,
            external_logical_bytes=external_logical,
        )

    @staticmethod
    def mega_moe(
        dispatch_target, dispatch_target_off,
        dispatch_src, dispatch_src_off, dispatch_size,
        up_proj_weight, up_proj_glist,
        up_proj_y, swiglu_out,
        down_proj_weight, down_proj_glist, down_proj_y,
        combine_target, combine_target_off, combine_src_off, combine_size,
        gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
        runtime_config, all_event_counters,
        rank_id: int, ep: int, expert_num: int,
        hidden_size: int, seq_size: int,
    ):
        """MoE-FFN forward operator (MindSpore backend)."""
        # pylint: disable=C0415
        from hyper_parallel.core.multicore.platform.mindspore import mega_moe
        return mega_moe(
            dispatch_target, dispatch_target_off,
            dispatch_src, dispatch_src_off, dispatch_size,
            up_proj_weight, up_proj_glist,
            up_proj_y, swiglu_out,
            down_proj_weight, down_proj_glist, down_proj_y,
            combine_target, combine_target_off, combine_src_off, combine_size,
            gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
            runtime_config, all_event_counters,
            rank_id, ep, expert_num, hidden_size, seq_size,
        )

    @staticmethod
    def mega_moe_grad(
        dispatch_target, dispatch_target_off,
        dy, dispatch_src_off, dispatch_size,
        hidden, hidden_dw,
        w2, act_grad_y, gate, grad_gate, w1, gate_dx, grad_x,
        combine_target_off, combine_src_off, combine_size,
        permute_out, gate_dw, group_list,
        act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
        swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
        runtime_config, all_event_counters,
        rank_id: int, ep: int, expert_num: int,
        hidden_size: int, seq_size: int,
    ):
        """MoE-FFN backward operator (MindSpore backend)."""
        # pylint: disable=C0415
        from hyper_parallel.core.multicore.platform.mindspore import mega_moe_grad
        return mega_moe_grad(
            dispatch_target, dispatch_target_off,
            dy, dispatch_src_off, dispatch_size,
            hidden, hidden_dw,
            w2, act_grad_y, gate, grad_gate, w1, gate_dx, grad_x,
            combine_target_off, combine_src_off, combine_size,
            permute_out, gate_dw, group_list,
            act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
            swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
            runtime_config, all_event_counters,
            rank_id, ep, expert_num, hidden_size, seq_size,
        )
