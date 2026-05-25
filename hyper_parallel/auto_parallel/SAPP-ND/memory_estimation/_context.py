# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Context module for evaluator"""
from __future__ import annotations
from pprint import pformat
from typing import TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum, auto
from memory_estimation.evaluators.utils import EvalUtils

if TYPE_CHECKING:
    from typing import Self, Any


class MemType(Enum):
    """memory types"""

    MODEL_PARAM = auto()
    OPTIM_STATE = auto()
    ACCU_GRAD = auto()
    ATTN_ACTIV = auto()
    FFN_ACTIV = auto()
    NORM_ACTIV = auto()
    AG_COMM = auto()
    A2A_COMM = auto()


@dataclass
class NodeStatEval:
    """static formula pointers"""

    p: Any
    os: Any
    grad: Any

    def __repr__(self):
        return (
            f"stat.p={self.p.__qualname__}, "
            f"stat.os={self.os.__qualname__}, "
            f"stat.grad={self.grad.__qualname__}"
        )


@dataclass
class NodeCommEval:
    """comm formula pointers"""

    dp: Any
    tp: Any
    cp: Any
    ep: Any

    def __repr__(self):
        return (
            f"dyn.comm.dp={self.dp.__qualname__}, "
            f"dyn.comm.tp={self.tp.__qualname__}, "
            f"dyn.comm.cp={self.cp.__qualname__}, "
            f"dyn.comm.ep={self.ep.__qualname__}"
        )


@dataclass
class NodeDynEval:
    """dynamic formula pointers"""

    activation: Any
    comm: NodeCommEval

    def __repr__(self):
        return f"dyn.activation={self.activation.__qualname__}, " f"{str(self.comm)}"


@dataclass
class NodeEval:
    """Associate a LayerType ->
    (Num param function, static mem function, dynamic mem function)
    """

    num_p: Any
    stat: NodeStatEval
    dyn: NodeDynEval

    def __repr__(self):
        return (
            f"num_p = {self.num_p.__name__}, "
            f"{str(self.stat)}, "
            f"{str(self.dyn)}"
        )


class Context:
    """Context class"""

    def __init__(self) -> None:
        """initializing buffers"""
        # Temporary bufferes
        self.enable_node_log = True
        self.accu_mem_type = {mt: 0 for mt in list(MemType)}
        self.node_compute_log = {}

        # Map node to (static function, dynamic function)
        self.node_eval = {}
        # Variables
        self.vpp_less_mem, self.swap_os = None, None
        self.dropless_tok_factor = None
        self.attn_num_p, self.attn_qkv_activ = None, None
        self.attn_score_activ, self.attn_proj_activ = None, None
        self.ffn_num_p, self.ffn_activ, self.ffn_moe_activ = None, None, None
        self.norm_num_p, self.norm_activ = None, None
        self.pp_micro_eval = {}
        self.head_node, self.tail_node = None, None
        self.current_node = None
        self.current_stage_id, self.current_chunk_id = -1, -1
        self.current_lay_id = None
        self.real_lay_ids = []
        self.ppb, self.default_micro_factor = None, None

    def __str__(self):
        return pformat(
            dict(
                (k, v) if k != "node_eval" else (k, self.print_node_eval())
                for k, v in vars(self).items()
            )
        )

    def print_node_eval(self):
        """from all layertype"""
        return dict((k, str(v)) for k, v in self.node_eval.items())

    @property
    def eval(self):
        """shortcut"""
        return self.node_eval[self.current_node]

    def init_tmp_buff(self) -> None:
        """reset"""
        self.enable_node_log = True
        self.accu_mem_type = {mt: 0 for mt in list(MemType)}
        self.node_compute_log = {}

    def copy_tmp_buff(self, target_ctx: Self) -> None:
        """copy to target_ctx"""
        for att, val in vars(self).items():
            if att != "node_eval":
                setattr(target_ctx, att, val)

    def save2log(self, fun, val_in_bytes):
        """lay_id -> fun, val"""
        if self.enable_node_log and val_in_bytes > 0:
            name = fun
            if callable(fun):
                name = fun.__name__
            elif isinstance(fun, MemType):
                name = fun.name.lower()
            node_name = self.current_node
            if not isinstance(self.current_node, str):
                node_name = self.current_node.name[0]
            if isinstance(self.current_lay_id, int):
                real_lay_id = self.real_lay_ids[self.current_chunk_id][
                    self.current_stage_id
                ][self.current_lay_id]
            else:
                lay_id = int(self.current_lay_id.split("_")[-1])
                real_lay_id = self.real_lay_ids[self.current_chunk_id][
                    self.current_stage_id
                ][lay_id]
                real_lay_id = self.current_lay_id.replace(
                    str(lay_id), str(real_lay_id)
                )
            # Add key
            pair = (
                self.current_stage_id,
                self.current_chunk_id,
                real_lay_id,
                node_name,
            )
            if pair not in self.node_compute_log:
                self.node_compute_log[pair] = {}
            if name not in self.node_compute_log[pair]:
                self.node_compute_log[pair][name] = 0
            self.node_compute_log[pair][name] += EvalUtils.mb(val_in_bytes)
