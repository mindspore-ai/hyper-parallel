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
"""MoE-FFN multicore operator entry point with platform dispatch."""
from hyper_parallel.platform import get_platform

_platform = get_platform()
_multicore_handler = _platform.get_multicore_handler()


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
    """MoE-FFN forward operator (platform-dispatched)."""
    return _multicore_handler.mega_moe(
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
    """MoE-FFN backward operator (platform-dispatched)."""
    return _multicore_handler.mega_moe_grad(
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


__all__ = ["mega_moe", "mega_moe_grad"]
