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

from importlib import import_module as _import_module  # pylint: disable=invalid-name
from typing import TYPE_CHECKING, Any

from hyper_parallel.platform import get_platform

if TYPE_CHECKING:
    from .modules import MulticoreModule
    from .modules.mega_moe import MegaMoeExperts

__all__ = ["MegaMoeExperts", "MulticoreModule", "mega_moe", "mega_moe_grad"]

_platform = get_platform()
_multicore_handler = _platform.get_multicore_handler()


def mega_moe(
    dispatch_target: Any,
    dispatch_target_off: Any,
    dispatch_src: Any,
    dispatch_src_off: Any,
    dispatch_size: Any,
    up_proj_weight: Any,
    up_proj_glist: Any,
    up_proj_y: Any,
    swiglu_out: Any,
    down_proj_weight: Any,
    down_proj_glist: Any,
    down_proj_y: Any,
    combine_target: Any,
    combine_target_off: Any,
    combine_src_off: Any,
    combine_size: Any,
    gmm_workspace: Any,
    up_proj_tiling: Any,
    swiglu_tiling: Any,
    down_proj_tiling: Any,
    runtime_config: Any,
    all_event_counters: Any,
    rank_id: int,
    ep: int,
    expert_num: int,
    hidden_size: int,
    seq_size: int,
) -> Any:
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
    dispatch_target: Any,
    dispatch_target_off: Any,
    dy: Any,
    dispatch_src_off: Any,
    dispatch_size: Any,
    hidden: Any,
    hidden_dw: Any,
    w2: Any,
    act_grad_y: Any,
    gate: Any,
    grad_gate: Any,
    w1: Any,
    gate_dx: Any,
    grad_x: Any,
    combine_target_off: Any,
    combine_src_off: Any,
    combine_size: Any,
    permute_out: Any,
    gate_dw: Any,
    group_list: Any,
    act_grad_tiling: Any,
    gate_grad_tiling: Any,
    w1_grad_tiling: Any,
    w2_grad_tiling: Any,
    swiglu_grad_tiling: Any,
    gmm_workspace: Any,
    swiglu_grad_workspace: Any,
    runtime_config: Any,
    all_event_counters: Any,
    rank_id: int,
    ep: int,
    expert_num: int,
    hidden_size: int,
    seq_size: int,
) -> Any:
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


_LAZY_EXPORTS = {
    "MegaMoeExperts": ".modules.mega_moe",
    "MulticoreModule": ".modules",
}


def __getattr__(name):  # pylint: disable=invalid-name
    """Lazily import Torch-only managed multicore symbols."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():  # pylint: disable=invalid-name
    """Include managed multicore symbols in dir()."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
