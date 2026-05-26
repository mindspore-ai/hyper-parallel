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
"""Utility submodule"""
from __future__ import annotations
from typing import TYPE_CHECKING
import operator
import ast
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context
    from typing import Union

OPS_MAP = {
    ast.Add: operator.add,  # x + y
    ast.Sub: operator.sub,  # x - y
    ast.Mult: operator.mul,  # x * y
    ast.Div: operator.truediv,  # x / y
    ast.FloorDiv: operator.floordiv,  # x // y
    ast.Mod: operator.mod,  # x % y
    ast.Pow: operator.pow,  # x ** y
    ast.USub: operator.neg,  # -x
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
}


class EvalUtils:
    """Utility methods class, PP Microbatch factor formulas"""

    @staticmethod
    def mb(x: Union[float, dict, tuple]) -> int:
        """Convert Byte to MB"""
        if isinstance(x, dict):
            return dict(
                (
                    (k, int(sum(v) / 1024 / 1024))
                    if isinstance(v, tuple)
                    else (k, int(v / 1024 / 1024))
                )
                for k, v in x.items()
            )
        if isinstance(x, tuple):
            return int(sum(x) / 1024 / 1024)
        return int(x / 1024 / 1024)

    # Layer Blocks
    @staticmethod
    def rec_coeff(rec_layer: bool, rec_op: bool) -> bool:
        """Masking coefficient for select recompute"""
        return int(not rec_layer) | rec_op

    @classmethod
    def eval_expr_insight(cls, **kwargs):
        """compute and categorize math expression"""
        nodes = ast.parse(kwargs.get("expr"))
        # print(ast.dump(nodes, indent=1))
        # print(kwargs["expr"])
        return cls.__eval_ast_mem(nodes.body[0].value, 0, **kwargs)[0]

    @staticmethod
    def __log_ast_mem(ctx, cat, mem):
        """Save AST memory evaluation result in the current context."""
        ctx.save2log(cat, mem)
        ctx.accu_mem_type[cat] += mem

    @classmethod
    def __eval_ast_name(cls, n: ast.Name, depth: int, wait: bool, **kwargs):
        """Evaluate a named memory term."""
        ctx = kwargs.get("ctx")
        if n.id not in kwargs.get("mem_val") or n.id not in kwargs.get(
            "mem_cat"
        ):
            raise AttributeError(
                f"Unrecognized variable '{n.id}' "
                f"from expr: '{kwargs['expr']}' "
                f"(recognized: {list(kwargs.get('mem_val').keys())})"
            )
        mem = kwargs.get("mem_val")[n.id]
        cat = kwargs.get("mem_cat")[n.id]
        if depth <= 1 and not wait:
            cls.__log_ast_mem(ctx, cat, mem)
        return mem, cat

    @classmethod
    def __eval_ast_unary(cls, n: ast.UnaryOp, depth: int, wait: bool, **kwargs):
        """Evaluate a unary expression."""
        ctx = kwargs.get("ctx")
        mem, cat = cls.__eval_ast_mem(
            n.operand,
            depth + int(not isinstance(n.operand, ast.BinOp)),
            **kwargs,
        )
        mem = OPS_MAP[type(n.op)](mem)
        if depth == 0 and not wait:
            cls.__log_ast_mem(ctx, cat, mem)
        return mem, cat

    @classmethod
    def __eval_ast_binop(cls, n: ast.BinOp, depth: int, **kwargs):
        """Evaluate a binary expression."""
        ctx = kwargs.get("ctx")
        l_is_con = isinstance(n.left, ast.Constant) or not isinstance(
            n.op, ast.Add
        )
        r_is_con = isinstance(n.right, ast.Constant) or not isinstance(
            n.op, ast.Add
        )
        l_eval = cls.__eval_ast_mem(
            n.left,
            depth + int(not isinstance(n.left, ast.BinOp)),
            wait=r_is_con,
            **kwargs,
        )
        r_eval = cls.__eval_ast_mem(
            n.right,
            depth + int(not isinstance(n.right, ast.BinOp)),
            wait=l_is_con,
            **kwargs,
        )
        mem = OPS_MAP[type(n.op)](l_eval[0], r_eval[0])
        cat = l_eval[1] if l_eval[1] else r_eval[1]
        if depth == 0 and (r_is_con or l_is_con):
            cls.__log_ast_mem(ctx, cat, mem)
        return mem, cat

    @classmethod
    def __eval_ast_call(
        cls, n: ast.Call, depth: int, wait: bool, **kwargs
    ):
        """Evaluate a supported function call."""
        ctx = kwargs.get("ctx")
        a_res = [[], []]
        for x in n.args:
            a_eval = cls.__eval_ast_mem(x, depth + 1, wait=True, **kwargs)
            a_res[0] += [a_eval[0]]
            a_res[1] += [a_eval[1]]
        mem = OPS_MAP[n.func.id](*a_res[0])
        cat = next(
            (c for c in a_res[1] if a_res[0][a_res[1].index(c)] == mem),
            a_res[1][0],
        )
        if depth <= 1 and not wait:
            cls.__log_ast_mem(ctx, cat, mem)
        return mem, cat

    @classmethod
    def __eval_ast_mem(
        cls, n: ast.AST, depth: int, wait: bool = False, **kwargs
    ):
        """compute and categorize from AST"""
        if isinstance(n, ast.Name):
            return cls.__eval_ast_name(n, depth, wait, **kwargs)
        if isinstance(n, ast.Constant):
            return n.value, None
        if isinstance(n, ast.UnaryOp):
            return cls.__eval_ast_unary(n, depth, wait, **kwargs)
        if isinstance(n, ast.BinOp):
            return cls.__eval_ast_binop(n, depth, **kwargs)
        if isinstance(n, ast.Call):
            return cls.__eval_ast_call(n, depth, wait, **kwargs)
        return 0, None

    # PP MICRO FACTOR

    @staticmethod
    def pp_1f1b_micro_factor(ccfg: CostModelConfig, ctx: Context) -> int:
        """1F1B Warm-up microbatches count"""
        stage_id, chunk_id = ctx.current_stage_id, ctx.current_chunk_id
        # Warm_up micros num compute
        micro_factor = 1
        extra = 0
        base_micro = min(ccfg.p, ccfg.m)
        if ccfg.vp == 1:
            micro_factor = base_micro - stage_id
        else:  # VPP
            if 0 < chunk_id < ccfg.vp - 1:  # Middle chunk
                micro_factor = base_micro
            else:  # First/Last chunk
                if not ctx.vpp_less_mem:  # Big memory
                    # Balance micros between last chunk and next first chunk
                    extra = base_micro - stage_id - 1
                    last_chunk_micros = base_micro - stage_id
                    if last_chunk_micros < base_micro:
                        last_chunk_micros += min(1, extra)
                        extra = max(0, extra - 1)
                    if chunk_id == 0:
                        if stage_id == 0:
                            extra -= 1
                        micro_factor = base_micro + extra
                    else:
                        micro_factor = last_chunk_micros
                else:  # Less memory
                    if chunk_id == 0:
                        micro_factor = base_micro
                    else:
                        micro_factor = base_micro - stage_id
        return micro_factor

    @staticmethod
    def pp_seq1f1b_micro_factor(ccfg: CostModelConfig, ctx: Context) -> int:
        """Seq1F1B Warm-up microbatches count"""
        stage_id, chunk_id = ctx.current_stage_id, ctx.current_chunk_id
        # Warm_up micros num compute
        micro_factor = 1
        ccfg.s /= ccfg.n_s_split  # Splitting seq length
        base_micro = min(ccfg.p, ccfg.m)
        if ccfg.vp == 1:
            micro_factor = base_micro - stage_id + ccfg.n_s_split - 1
        else:  # VPP
            if 0 < chunk_id < ccfg.vp - 1:  # Middle chunk
                micro_factor = base_micro
            else:  # First/Last chunk
                if not ctx.vpp_less_mem:  # Big memory
                    # Balance micros between last chunk and next first chunk
                    extra = base_micro - stage_id - 1
                    last_chunk_micros = base_micro - stage_id
                    last_chunk_micros += ccfg.n_s_split - 1
                    if last_chunk_micros > base_micro:
                        last_chunk_micros -= 1
                        extra += 1
                    elif last_chunk_micros < base_micro:
                        last_chunk_micros += min(1, extra)
                        extra = max(0, extra - 1)
                    if chunk_id == 0:
                        micro_factor = base_micro + extra
                    else:
                        micro_factor = last_chunk_micros
                else:  # Less memory
                    if chunk_id == 0:
                        micro_factor = base_micro
                    else:
                        micro_factor = base_micro - stage_id
                        micro_factor += ccfg.n_s_split - 1
        return micro_factor

    @staticmethod
    def pp_dualpipe_v_micro_factor(ccfg: CostModelConfig, ctx: Context) -> int:
        """DualPipeV/ZeroBubbleV Warm-up microbatches count"""
        stage_id, chunk_id = ctx.current_stage_id, ctx.current_chunk_id
        # First half layer from stage 0->PP then second half from stage PP->0
        if ccfg.vp > 2:
            logger.warning("DualPipeV with VPP>2 not handled")
            return 0
        if chunk_id == 0:
            return min(ccfg.p, ccfg.m) * 2 - 1 - stage_id
        return stage_id

    @staticmethod
    def pp_gpipe_micro_factor(ccfg: CostModelConfig, ctx: Context) -> int:
        """GPipe warm-up microbatches count."""
        _ = ctx
        return ccfg.p  # Minimum value
