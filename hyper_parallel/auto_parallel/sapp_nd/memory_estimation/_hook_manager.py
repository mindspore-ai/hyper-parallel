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
"""hook manager module"""
from __future__ import annotations
from typing import TYPE_CHECKING
import ast
import textwrap
import inspect
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import check_and_apply_custom_hook
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import (
    NodeEval,
    NodeStatEval,
    NodeDynEval,
    NodeCommEval,
)
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.head import EvalHead
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTail
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.body import EvalBody
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalAttn, EvalFFn
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalNorm
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._backbone import _Backbone
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._func_tracer import _FuncTracer
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import MemType

if TYPE_CHECKING:
    from typing import Callable, Any


class _HookManager(_Backbone):
    """Hook manager class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func_tracer = _FuncTracer()
        self.toggle_func_trace = kwargs.get("trace_fun", False)
        self.import_eval_yaml()
        self.fetch_hook_if_unimodal()

    def fetch_hook_if_unimodal(self):
        """Process custom model config"""
        if not self._ccfg.multimodal:
            if not self._ccfg.hooks_dict:
                logger.info(
                    "'hook_cls' not specified,"
                    "search in predefined arch_hooks"
                )
                check_and_apply_custom_hook(self)
            else:
                hook = list(self._ccfg.hooks_dict.values())[0]
                hook(self)

    def __is_valid_eval_func(self, fun: Any) -> bool:
        """check hook definition, return type"""
        if fun is None:
            return False
        if isinstance(fun, (str, int, float)):
            return True

        source = inspect.getsource(fun)
        tree = ast.parse(textwrap.dedent(source))
        for instruction in ast.walk(tree):
            if (
                isinstance(instruction, ast.Return)
                and instruction.value is None
            ):
                return False
        return True

    # Evaluation context setters

    def set_passes(
        self,
        vpp_less_mem: bool = None,
        swap_os: bool = None,
        dropless_tok_factor: float = None,
    ) -> None:
        """toggle features"""
        if isinstance(vpp_less_mem, bool):
            self._ctx.vpp_less_mem = vpp_less_mem
        if isinstance(swap_os, bool):
            self._ctx.swap_os = swap_os
        if isinstance(dropless_tok_factor, (int, float)):
            self._ctx.dropless_tok_factor = dropless_tok_factor

    def __set_node_eval_fun(self, cls_obj, target_node, *args, **kwargs):
        """overwrite given node's formulas"""
        c_stat, c_dyn = None, None
        if (args and args[-1] == 0) or kwargs.get("stat", None) == 0:
            c_stat = 0
        if (args and args[-1] == 0) or kwargs.get("dyn", None) == 0:
            c_dyn = 0
        num_p = kwargs.get("num_p", c_stat)
        stat_p = kwargs.get("stat_p", c_stat)
        stat_os = kwargs.get("stat_os", c_stat)
        stat_grad = kwargs.get("stat_grad", c_dyn)
        dyn_activ = kwargs.get("dyn_activ", c_dyn)
        if not self.__is_valid_eval_func(num_p):
            num_p = self._ctx.node_eval[target_node].num_p
        if not self.__is_valid_eval_func(stat_p):
            stat_p = self._ctx.node_eval[target_node].stat.p
        if not self.__is_valid_eval_func(stat_os):
            stat_os = self._ctx.node_eval[target_node].stat.os
        if not self.__is_valid_eval_func(stat_grad):
            stat_grad = self._ctx.node_eval[target_node].stat.grad
        if not self.__is_valid_eval_func(dyn_activ):
            dyn_activ = self._ctx.node_eval[target_node].dyn.activation
        self._ctx.node_eval[target_node] = NodeEval(
            self.__custom_getattr(cls_obj, num_p),
            NodeStatEval(
                self.__custom_getattr(cls_obj, stat_p, MemType.MODEL_PARAM),
                self.__custom_getattr(cls_obj, stat_os, MemType.OPTIM_STATE),
                self.__custom_getattr(cls_obj, stat_grad, MemType.ACCU_GRAD),
            ),
            NodeDynEval(
                self.__custom_getattr(cls_obj, dyn_activ),
                self.__set_node_eval_comm_fun(
                    cls_obj, target_node, *args, **kwargs
                ),
            ),
        )

    def __set_node_eval_comm_fun(self, cls_obj, target_node, *args, **kwargs):
        """overwrite given node's comm formulas"""
        c_comm = None
        if (
            (args and args[-1] == 0)
            or kwargs.get("dyn_comm", None) == 0
            or kwargs.get("dyn", None) == 0
        ):
            c_comm = 0
        dyn_dp_comm = kwargs.get("dyn_dp_comm", c_comm)
        dyn_tp_comm = kwargs.get("dyn_tp_comm", c_comm)
        dyn_cp_comm = kwargs.get("dyn_cp_comm", c_comm)
        dyn_ep_comm = kwargs.get("dyn_ep_comm", c_comm)
        if not self.__is_valid_eval_func(dyn_dp_comm):
            dyn_dp_comm = self._ctx.node_eval[target_node].dyn.comm.dp
        if not self.__is_valid_eval_func(dyn_tp_comm):
            dyn_tp_comm = self._ctx.node_eval[target_node].dyn.comm.tp
        if not self.__is_valid_eval_func(dyn_cp_comm):
            dyn_cp_comm = self._ctx.node_eval[target_node].dyn.comm.cp
        if not self.__is_valid_eval_func(dyn_ep_comm):
            dyn_ep_comm = self._ctx.node_eval[target_node].dyn.comm.ep
        comm_cls_obj = cls_obj
        if self.is_regular_layer(target_node):
            comm_cls_obj = EvalLayerComm
        return NodeCommEval(
            self.__custom_getattr(comm_cls_obj, dyn_dp_comm),
            self.__custom_getattr(comm_cls_obj, dyn_tp_comm),
            self.__custom_getattr(comm_cls_obj, dyn_cp_comm),
            self.__custom_getattr(comm_cls_obj, dyn_ep_comm),
        )

    def set_head_eval_fun(self, *arg, **kwarg):
        """overwrite head formulas"""
        self.__set_node_eval_fun(EvalHead, self._ctx.head_node, *arg, **kwarg)

    def set_tail_eval_fun(self, *arg, **kwarg):
        """overwrite tail formulas"""
        self.__set_node_eval_fun(EvalTail, self._ctx.tail_node, *arg, **kwarg)

    def set_body_eval_fun(self, *args, **kwargs):
        """overwrite body formulas"""
        lay_type = kwargs.get("lay_type", args[0] if args else None)
        if not lay_type:
            lt = [
                b_obj
                for b_obj in list(LayerType)
                if self.is_regular_layer(b_obj)
            ]
        else:
            if not isinstance(lay_type, LayerType):
                b_obj = self.__custom_getattr(LayerType, lay_type)
                lt = [b_obj]
            else:
                lt = [lay_type]
        for b_obj in lt:
            self.__set_node_eval_fun(EvalBody, b_obj, *args, **kwargs)

    def set_attn_eval_fun(
        self,
        num_p: Any = None,
        qkv: Any = None,
        score: Any = None,
        proj: Any = None,
    ) -> None:
        """overwrite attention formulas"""
        if self.__is_valid_eval_func(num_p):
            self._ctx.attn_num_p = self.__custom_getattr(EvalAttn, num_p)
        if self.__is_valid_eval_func(qkv):
            self._ctx.attn_qkv_activ = self.__custom_getattr(
                EvalAttn, qkv, MemType.ATTN_ACTIV
            )
        if self.__is_valid_eval_func(score):
            self._ctx.attn_score_activ = self.__custom_getattr(
                EvalAttn, score, MemType.ATTN_ACTIV
            )
        if self.__is_valid_eval_func(proj):
            self._ctx.attn_proj_activ = self.__custom_getattr(
                EvalAttn, proj, MemType.ATTN_ACTIV
            )

    def set_ffn_eval_fun(self, num_p: Any = None, activation=None, moe_activ=None):
        """overwrite feedforward formulas"""
        if self.__is_valid_eval_func(num_p):
            self._ctx.ffn_num_p = self.__custom_getattr(EvalFFn, num_p)
        if self.__is_valid_eval_func(activation):
            self._ctx.ffn_activ = self.__custom_getattr(
                EvalFFn, activation, MemType.FFN_ACTIV
            )
        if self.__is_valid_eval_func(moe_activ):
            self._ctx.ffn_moe_activ = self.__custom_getattr(
                EvalFFn, moe_activ, MemType.FFN_ACTIV
            )

    def set_norm_eval_fun(self, num_p: Any = None, activation=None):
        """overwrite norm formulas"""
        if self.__is_valid_eval_func(num_p):
            self._ctx.norm_num_p = self.__custom_getattr(EvalNorm, num_p)
        if self.__is_valid_eval_func(activation):
            self._ctx.norm_activ = self.__custom_getattr(
                EvalNorm, activation, MemType.NORM_ACTIV
            )

    def set_pp_micro_factor_eval_fun(self, sched_name, fun):
        """overwrite PP microfactor formulas"""
        if sched_name and self.__is_valid_eval_func(fun):
            self._ctx.pp_micro_eval[sched_name] = self.__custom_getattr(
                EvalUtils, fun
            )

    # Cost Model Config setter

    def set_strategy(self, **kwargs):
        """overwrite parallelism"""
        self._ccfg.set_strategy(**kwargs)
        self.fetch_hook_if_unimodal()

    def set_ccfg(self, hook):
        """overwrite cost model variable (except strategy)"""
        if hook and callable(hook):

            def custom_setter(self, name, value):
                strat_vars = ["d", "t", "ep", "p", "vp", "cp", "os_max_shard"]
                if name in strat_vars:
                    raise AttributeError(
                        f"Cannot directly modify {name}, use set_strategy()"
                    )
                self.__dict__[name] = value

            CostModelConfig.__setattr__ = custom_setter
            hook(self._ccfg)
            CostModelConfig.__setattr__ = object.__setattr__

    def __wrap_mem_counter(self, mem_type: MemType, fun: Callable) -> None:
        """Wrap formula calls to accumulate memory by type."""
        if mem_type and not hasattr(fun, "wrapped_with_counter"):

            def wrap(*args, **kwargs):
                res = fun(*args, **kwargs)
                self._ctx.accu_mem_type[mem_type] += res
                self._ctx.save2log(mem_type, res)
                return res

            wrap.__qualname__ = fun.__qualname__
            wrap.wrapped_with_counter = True
            return wrap
        return fun

    def __custom_getattr(
        self, eval_class: Any, field: Any, mem_type: MemType = None
    ) -> Callable:
        """formula retrieve/wrap"""
        # Definition priority order :
        #   1. Callable (user defined in code)
        #      OR Numeric value
        #   2. Overriding list from cost_model_preprocess (ccfg attribute)
        #   3. Function name (config_eval yaml)
        res = None
        if callable(field):
            res = field
            if self.toggle_func_trace == field.__name__:
                res = self.func_tracer.wrap(field)
        if isinstance(field, (int, float)):

            def constant(*_):
                return field

            constant.__qualname__ = str(field)
            res = constant
        if field in self._ccfg.overwrite_eval_functions:
            res = self._ccfg.overwrite_eval_functions[field]
        if isinstance(field, str):

            def zero(*_):
                return 0

            zero.__qualname__ = "0"
            res = getattr(eval_class, field, zero)
            if self.toggle_func_trace == field:
                res = self.func_tracer.wrap(getattr(eval_class, field))
        res = self.__wrap_mem_counter(mem_type, res)
        if not res:
            raise TypeError(f"In eval config yaml, non valid field: {field}")
        return res

    # Import Eval Config

    def import_eval_yaml(self) -> None:
        """import evaluator config file, init ctx (inner call only)"""
        if not self.toggle_func_trace:
            self.toggle_func_trace = self.eval_cfg.trace_fun
        # head
        h_obj = getattr(LayerType, self.eval_cfg.nodes_mem_comp.head.name)
        self._ctx.head_node = h_obj
        self.set_head_eval_fun(
            num_p=self.eval_cfg.nodes_mem_comp.head.num_param_fun,
            stat_p=self.eval_cfg.nodes_mem_comp.head.stat_fun.p,
            stat_os=self.eval_cfg.nodes_mem_comp.head.stat_fun.os,
            stat_grad=self.eval_cfg.nodes_mem_comp.head.stat_fun.grad,
            dyn_activ=self.eval_cfg.nodes_mem_comp.head.dyn_fun.activation,
            dyn_dp_comm=self.eval_cfg.nodes_mem_comp.head.dyn_fun.comm.dp,
            dyn_tp_comm=self.eval_cfg.nodes_mem_comp.head.dyn_fun.comm.tp,
            dyn_cp_comm=self.eval_cfg.nodes_mem_comp.head.dyn_fun.comm.cp,
            dyn_ep_comm=self.eval_cfg.nodes_mem_comp.head.dyn_fun.comm.ep,
        )

        # tail
        t_obj = getattr(LayerType, self.eval_cfg.nodes_mem_comp.tail.name)
        self._ctx.tail_node = t_obj
        self.set_tail_eval_fun(
            num_p=self.eval_cfg.nodes_mem_comp.tail.num_param_fun,
            stat_p=self.eval_cfg.nodes_mem_comp.tail.stat_fun.p,
            stat_os=self.eval_cfg.nodes_mem_comp.tail.stat_fun.os,
            stat_grad=self.eval_cfg.nodes_mem_comp.tail.stat_fun.grad,
            dyn_activ=self.eval_cfg.nodes_mem_comp.tail.dyn_fun.activation,
            dyn_dp_comm=self.eval_cfg.nodes_mem_comp.tail.dyn_fun.comm.dp,
            dyn_tp_comm=self.eval_cfg.nodes_mem_comp.tail.dyn_fun.comm.tp,
            dyn_cp_comm=self.eval_cfg.nodes_mem_comp.tail.dyn_fun.comm.cp,
            dyn_ep_comm=self.eval_cfg.nodes_mem_comp.tail.dyn_fun.comm.ep,
        )

        # body
        for b in self.eval_cfg.nodes_mem_comp.body:
            b_cfg = Config(b)
            self.set_body_eval_fun(
                lay_type=b_cfg.name,
                num_p=b_cfg.num_param_fun,
                stat_p=b_cfg.stat_fun.p,
                stat_os=b_cfg.stat_fun.os,
                stat_grad=b_cfg.stat_fun.grad,
                dyn_activ=b_cfg.dyn_fun.activation,
                dyn_dp_comm=b_cfg.dyn_fun.comm.dp,
                dyn_tp_comm=b_cfg.dyn_fun.comm.tp,
                dyn_cp_comm=b_cfg.dyn_fun.comm.cp,
                dyn_ep_comm=b_cfg.dyn_fun.comm.ep,
            )

        # pp micro factor
        for sc in self.eval_cfg.pp_sched:
            self.set_pp_micro_factor_eval_fun(sc["name"], sc["fun"])
        if not self._ccfg.pp_sched:
            self._ccfg.pp_sched = self.eval_cfg.default_pp_sched

        # layerblock
        self.set_attn_eval_fun(
            self.eval_cfg.base_arch_mem_comp.attention.num_param_fun,
            self.eval_cfg.base_arch_mem_comp.attention.qkv,
            self.eval_cfg.base_arch_mem_comp.attention.score,
            self.eval_cfg.base_arch_mem_comp.attention.proj,
        )
        self.set_ffn_eval_fun(
            self.eval_cfg.base_arch_mem_comp.feedforward.num_param_fun,
            self.eval_cfg.base_arch_mem_comp.feedforward.activation,
            self.eval_cfg.base_arch_mem_comp.feedforward.moe_activ,
        )
        self.set_norm_eval_fun(
            self.eval_cfg.base_arch_mem_comp.norm.num_param_fun,
            self.eval_cfg.base_arch_mem_comp.norm.activation,
        )

        # passes
        self.set_passes(
            vpp_less_mem=self.eval_cfg.passes.vpp_less_memory,
            swap_os=self.eval_cfg.passes.swap_optimizer,
            dropless_tok_factor=self.eval_cfg.passes.dropless_tok_factor,
        )

        self._ctx.comm_expr = self.eval_cfg.comm_expr

    def is_regular_layer(self, lay):
        """check if layer is not head/tail"""
        if isinstance(lay, str):
            return lay[0] not in [
                self._ctx.head_node.name[0],
                self._ctx.tail_node.name[0],
            ]
        return lay not in [self._ctx.head_node, self._ctx.tail_node]
