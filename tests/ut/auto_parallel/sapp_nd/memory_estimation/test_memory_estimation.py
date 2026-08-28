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
"""Unit tests for SAPP-ND memory estimation.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/memory_estimation/test_memory_estimation.py
"""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import (
    Context,
    MemType,
    NodeCommEval,
    NodeDynEval,
    NodeEval,
    NodeStatEval,
)
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._bwd_overhead import _BackwardOverhead
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._func_tracer import _FuncTracer
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._ppb import _PPB
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._utils import _Utils
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hook_base import MemEvalHook, hook_runner
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hooks.template import Template
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.score import mape, r2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory, Unit
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import (
    get_layer_custom_configs,
    get_recomp_factor,
    get_table_quantity,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import (
    CustomConfig,
    P2PCommType,
    PerformanceType,
    RatioType,
    RecType,
)

WORK_PATH = os.path.dirname(os.path.abspath(__file__))
MEGABYTE = 1024 * 1024


def _eval_fun(*_args: Any, **_kwargs: Any) -> int:
    """Return a deterministic value for repr and hook tests."""
    return 1


def _micro_factor(ccfg: Any, ctx: Context) -> int:
    """Return a visible micro factor from the current stage and chunk."""
    return ccfg.m + ctx.current_stage_id + ctx.current_chunk_id


def _mark_custom_config(cfg: Any) -> None:
    """Mark a copied custom config so deepcopy behavior can be asserted."""
    cfg.marker = "custom"


def _trace_sample(value: int, holder: Any) -> int:
    """Small function used by _FuncTracer tests."""
    total = value + holder.offset
    total *= 2
    return total


def _dynamic_mem_for_ppb(**_kwargs: Any) -> tuple:
    """Return deterministic dynamic memory values."""
    return 2 * MEGABYTE, 3 * MEGABYTE


def _dynamic_mem_for_overhead(**_kwargs: Any) -> tuple:
    """Return deterministic overhead memory values."""
    return 11, 13


class _FakeTemplateEvaluator:
    """Small evaluator double that records Template hook registrations."""

    def __init__(self) -> None:
        """Initialize recorded calls."""
        self.calls = []

    def set_ccfg(self, func: Any) -> None:
        """Record a config hook."""
        self.calls.append(("ccfg", func))

    def set_passes(self, **kwargs: Any) -> None:
        """Record pass flags."""
        self.calls.append(("passes", kwargs))

    def set_head_eval_fun(self, **kwargs: Any) -> None:
        """Record head formulas."""
        self.calls.append(("head", kwargs))

    def set_tail_eval_fun(self, **kwargs: Any) -> None:
        """Record tail formulas."""
        self.calls.append(("tail", kwargs))

    def set_body_eval_fun(self, layer_name: str, **kwargs: Any) -> None:
        """Record body formulas for a layer type."""
        self.calls.append(("body", layer_name, kwargs))

    def set_attn_eval_fun(self, **kwargs: Any) -> None:
        """Record attention formulas."""
        self.calls.append(("attn", kwargs))

    def set_ffn_eval_fun(self, **kwargs: Any) -> None:
        """Record FFN formulas."""
        self.calls.append(("ffn", kwargs))

    def set_norm_eval_fun(self, **kwargs: Any) -> None:
        """Record norm formulas."""
        self.calls.append(("norm", kwargs))

    def set_pp_micro_factor_eval_fun(self, schedule: str, func: Any) -> None:
        """Record PP micro-batch formula."""
        self.calls.append(("pp_micro", schedule, func))


class _CountingRaiser:
    """Callable that raises RuntimeError after *raise_on* successful calls."""

    def __init__(self, raise_on: int) -> None:
        """Initialize with the call number that triggers the exception."""
        self.call_count = 0
        self.raise_on = raise_on

    def __call__(self, **_kwargs: Any) -> tuple:
        """Return a valid result or raise on the configured call."""
        self.call_count += 1
        if self.call_count >= self.raise_on:
            raise RuntimeError("injected failure")
        return (2 * MEGABYTE, 3 * MEGABYTE)


class TestPPBExceptionRecovery(unittest.TestCase):
    """Verify that lay_ppb and lay_ppb_new restore shared state on exception."""

    def _make_body_ppb(self, raise_on: int) -> tuple:
        """Create a _PPB with a _CountingRaiser and minimal ccfg/ctx for BODY path."""
        raiser = _CountingRaiser(raise_on)
        ppb = _PPB(SimpleNamespace(ppb_combined=[]), raiser)
        rec_op = SimpleNamespace(attBMM=10, headCast=20, dropout=30, softmax=40, normOp=50, gather=60, ffAct=70)
        ccfg = SimpleNamespace(model_name="test-model", rec_op=rec_op)
        ctx = Context()
        ctx.head_node = "head"
        ctx.tail_node = "tail"
        ctx.current_node = LayerType.NOT_REC_LAYER
        return ppb, ccfg, ctx, raiser

    def test_lay_ppb_restores_rec_op_on_exception(self) -> None:
        """
        Feature: TestPPBExceptionRecovery.
        Description: _inner_dynamic_mem raises during BODY path in lay_ppb.
        Expectation: ccfg.rec_op attributes are restored to their original values.
        """
        ppb, ccfg, ctx, _ = self._make_body_ppb(raise_on=2)
        original_vals = {k: getattr(ccfg.rec_op, k) for k in
                         ['attBMM', 'headCast', 'dropout', 'softmax', 'normOp', 'gather', 'ffAct']}
        with self.assertRaises(RuntimeError):
            ppb.lay_ppb(ccfg, ctx, 4 * MEGABYTE)
        for key, val in original_vals.items():
            self.assertEqual(getattr(ccfg.rec_op, key), val,
                             f"rec_op.{key} not restored: expected {val}, got {getattr(ccfg.rec_op, key)}")

    def test_lay_ppb_deletes_synthetic_rec_op_on_exception(self) -> None:
        """
        Feature: TestPPBExceptionRecovery.
        Description: ccfg has no rec_op; synthetic one is created, then _inner_dynamic_mem raises.
        Expectation: The synthetic rec_op is deleted (delattr) after the exception.
        """
        raiser = _CountingRaiser(raise_on=1)
        ppb = _PPB(SimpleNamespace(ppb_combined=[]), raiser)
        ccfg = SimpleNamespace(model_name="test-model")
        ctx = Context()
        ctx.head_node = "head"
        ctx.tail_node = "tail"
        ctx.current_node = LayerType.NOT_REC_LAYER
        with self.assertRaises(RuntimeError):
            ppb.lay_ppb(ccfg, ctx, 4 * MEGABYTE)
        self.assertFalse(hasattr(ccfg, 'rec_op'),
                         "synthetic rec_op should have been deleted after exception")

    def test_lay_ppb_restores_enable_node_log_on_exception(self) -> None:
        """
        Feature: TestPPBExceptionRecovery.
        Description: ctx.enable_node_log is False before call; _inner_dynamic_mem raises.
        Expectation: ctx.enable_node_log is restored to its original value (False), not hard-coded True.
        """
        ppb, ccfg, ctx, _ = self._make_body_ppb(raise_on=1)
        ctx.enable_node_log = False
        with self.assertRaises(RuntimeError):
            ppb.lay_ppb(ccfg, ctx, 4 * MEGABYTE)
        self.assertFalse(ctx.enable_node_log,
                         "enable_node_log should be restored to False, not hard-coded True")

    def test_lay_ppb_new_restores_rec_op_on_exception(self) -> None:
        """
        Feature: TestPPBExceptionRecovery.
        Description: _inner_dynamic_mem raises during BODY path in lay_ppb_new.
        Expectation: ccfg.rec_op attributes are restored to their original values.
        """
        ppb, ccfg, ctx, _ = self._make_body_ppb(raise_on=2)
        original_vals = {k: getattr(ccfg.rec_op, k) for k in
                         ['attBMM', 'headCast', 'dropout', 'softmax', 'normOp', 'gather', 'ffAct']}
        with self.assertRaises(RuntimeError):
            ppb.lay_ppb_new(ccfg, ctx, 4 * MEGABYTE)
        for key, val in original_vals.items():
            self.assertEqual(getattr(ccfg.rec_op, key), val,
                             f"rec_op.{key} not restored: expected {val}, got {getattr(ccfg.rec_op, key)}")

    def test_lay_ppb_new_restores_enable_node_log_on_exception(self) -> None:
        """
        Feature: TestPPBExceptionRecovery.
        Description: ctx.enable_node_log is False before call; _inner_dynamic_mem raises in lay_ppb_new.
        Expectation: ctx.enable_node_log is restored to its original value (False), not hard-coded True.
        """
        ppb, ccfg, ctx, _ = self._make_body_ppb(raise_on=1)
        ctx.enable_node_log = False
        with self.assertRaises(RuntimeError):
            ppb.lay_ppb_new(ccfg, ctx, 4 * MEGABYTE)
        self.assertFalse(ctx.enable_node_log,
                         "enable_node_log should be restored to False, not hard-coded True")


class TestSappNDMemoryEstimation(unittest.TestCase):
    """A test class for SAPP-ND memory estimation."""

    def test_memory_estimation_smoke(self):
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Test SAPP-ND memory estimation with a committed model config.
        Expectation: The evaluator parses config and returns valid memory results.
        """
        config_path = os.path.join(WORK_PATH, "mx_test.yaml")

        # Redirect matplotlib config dir to a temp path to avoid polluting $HOME.
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            evaluator = EvaluatorV2(config_path, log_level=0)

            self.assertEqual(
                evaluator.ccfg.model_name, "mixtral-8x7b",
                (f"unexpected model_name: got {evaluator.ccfg.model_name!r}, "
                 f"expected 'mixtral-8x7b'"),
            )
            self.assertEqual(
                set(evaluator.ctx.node_eval.keys()), set(LayerType),
                (f"node_eval keys mismatch: got {set(evaluator.ctx.node_eval.keys())}, "
                 f"expected {set(LayerType)}"),
            )

            strategy = evaluator.get_strategy()
            for key in ("dp", "tp", "pp", "ep"):
                self.assertGreater(
                    strategy[key], 0,
                    f"strategy[{key!r}] must be > 0, got {strategy[key]}",
                )

            peak_mem = evaluator.estimate_peak()
            self.assertGreater(peak_mem, 0, f"peak_mem must be > 0, got {peak_mem}")
            self.assertTrue(
                evaluator.mem_fit(peak_mem),
                f"mem_fit returned False for peak_mem={peak_mem}",
            )

            stage_static_mem = evaluator.static_mem_stage(1)
            stage_dynamic_mem = evaluator.dynamic_mem_stage(1)
            self.assertTrue(
                0 < stage_static_mem < peak_mem,
                f"static_mem_stage(1)={stage_static_mem} not in (0, peak={peak_mem})",
            )
            self.assertTrue(
                0 < stage_dynamic_mem < peak_mem,
                f"dynamic_mem_stage(1)={stage_dynamic_mem} not in (0, peak={peak_mem})",
            )

    def test_memory_score_and_perf_getters_are_stable(self) -> None:
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Exercise pure memory arithmetic, score helpers, and perf config getters.
        Expectation: Helpers return deterministic values without invoking model search or solvers.
        """
        self.assertEqual(str(Unit.from_string("gb")), "GB")
        self.assertEqual(Unit.from_string("unknown"), Unit.B)
        self.assertTrue(Unit.KB < Unit.MB)

        mem = Memory.from_string("1.5GB")
        self.assertEqual(str(mem), "1.50GB")
        self.assertEqual(mem.to_mb().size, 1536)
        self.assertEqual(Memory.from_kb(1024).to_mb().size, 1)
        self.assertEqual((Memory.from_mb(512) + Memory.from_gb(1)).to_mb().size, 1536)
        self.assertEqual((Memory.from_gb(1) - Memory.from_mb(512)).to_mb().size, 512)
        self.assertEqual(abs(Memory.from_mb(-7)).size, 7)
        self.assertTrue(Memory.from_mb(1) < Memory.from_gb(1))
        self.assertTrue(Memory.from_mb(1024) <= Memory.from_gb(1))
        self.assertEqual(Memory.zero().to_b().size, 0)

        mutable_mem = Memory.from_mb(10)
        mutable_mem.increase(Memory.from_kb(1024))
        mutable_mem.decrease(Memory.from_mb(1))
        self.assertEqual(mutable_mem.size, 10)

        with self.assertRaises(ValueError):
            Memory.from_string("")
        with self.assertRaises(ValueError):
            Memory.from_string("GB")

        self.assertEqual(mape([90, 0, -1], [100, 100, 100]), 10)
        self.assertIsNone(mape([0], [1]))
        self.assertEqual(r2([1, 2, 3], [1, 2, 3]), 1)
        self.assertIsNone(r2([1], [1]))
        self.assertIsNone(r2([1, 1], [2, 2]))

        custom = CustomConfig(
            rtype=RatioType.STATIC,
            ttype=PerformanceType.TIME,
            ptype=P2PCommType.MANUAL,
            retype=RecType.WITH,
        )
        self.assertIn("RatioType.STATIC", str(custom))

        cfg = SimpleNamespace(layer_custom_config=None, n_lay=3)
        self.assertEqual(get_layer_custom_configs(cfg), [(cfg, 3)])

        cfg_with_hooks = SimpleNamespace(layer_custom_config=[(2, _mark_custom_config)], n_lay=3)
        custom_layers = get_layer_custom_configs(cfg_with_hooks)
        self.assertEqual(custom_layers[0][1], 2)
        self.assertEqual(custom_layers[0][0].marker, "custom")
        self.assertFalse(hasattr(cfg_with_hooks, "marker"))

        lccfg = SimpleNamespace(opfoo=5, rec_op=SimpleNamespace(foo=2))
        self.assertEqual(get_recomp_factor(lccfg, LayerType.FULL_REC_LAYER, "foo"), 1)
        self.assertEqual(get_recomp_factor(lccfg, LayerType.NOT_REC_LAYER, "foo"), 0)
        self.assertEqual(get_recomp_factor(lccfg, LayerType.SEL_REC_LAYER, "foo"), 2)
        self.assertEqual(get_recomp_factor(lccfg, LayerType.OUTPUT_LAYER, "foo"), 0)
        self.assertEqual(get_table_quantity(lccfg, {"opfoo": 3}, LayerType.SEL_REC_LAYER, True), 45)

    def test_context_logging_helpers(self) -> None:
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Cover evaluator context logging and repr helpers.
        Expectation: Context records layer logs and copies temporary state.
        """
        stat_eval = NodeStatEval(_eval_fun, _eval_fun, _eval_fun)
        comm_eval = NodeCommEval(_eval_fun, _eval_fun, _eval_fun, _eval_fun)
        dyn_eval = NodeDynEval(_eval_fun, comm_eval)
        node_eval = NodeEval(_eval_fun, stat_eval, dyn_eval)
        self.assertIn("_eval_fun", repr(node_eval))

        ctx = Context()
        ctx.node_eval[LayerType.NOT_REC_LAYER] = node_eval
        ctx.current_node = LayerType.NOT_REC_LAYER
        ctx.current_stage_id = 0
        ctx.current_chunk_id = 0
        ctx.current_lay_id = 0
        ctx.real_lay_ids = [[[7]]]
        self.assertIs(ctx.eval, node_eval)
        ctx.save2log(_eval_fun, 2 * 1024 * 1024)
        self.assertEqual(ctx.node_compute_log[(0, 0, 7, "N")]["_eval_fun"], 2)

        ctx.current_lay_id = "layer_0"
        ctx.save2log(MemType.MODEL_PARAM, 3 * 1024 * 1024)
        self.assertEqual(ctx.node_compute_log[(0, 0, "layer_7", "N")]["model_param"], 3)

        target_ctx = Context()
        ctx.copy_tmp_buff(target_ctx)
        self.assertEqual(target_ctx.current_lay_id, "layer_0")
        ctx.init_tmp_buff()
        self.assertEqual(ctx.node_compute_log, {})
        self.assertIn("node_eval", str(ctx))

    def test_hook_registry_and_template_hooks(self) -> None:
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Cover hook metaclass validation and the template hook runner.
        Expectation: Hook registry filters valid hooks and rejects malformed hook classes.
        """
        old_registry = MemEvalHook.hook_registry.copy()
        try:
            MemEvalHook.hook_registry = {}

            class _UnitHook(MemEvalHook):
                """Valid hook class for registry filtering tests."""

                def marker(self) -> str:
                    """Return a test marker."""
                    return "unit"

                @staticmethod
                @hook_runner("unit_model")
                def run_hooks(e: list) -> None:
                    """Append a marker when invoked."""
                    e.append("called")

            hook = _UnitHook()
            registered_hooks = hook.get_hooks()
            calls = []
            registered_hooks["unit_model"](calls)
            self.assertEqual(calls, ["called"])

            with self.assertRaises(TypeError):
                hook_runner("")
            with self.assertRaises(TypeError):
                hook_runner("unit_model")
            with self.assertRaises(TypeError):

                class _MissingHook(MemEvalHook):  # pylint: disable=abstract-method
                    """Missing required hook implementation."""

                    def marker(self) -> str:
                        """Return a test marker."""
                        return "missing"

                    def label(self) -> str:
                        """Return a second marker for pylint friendliness."""
                        return "missing-label"
                _ = _MissingHook

            with self.assertRaises(TypeError):

                class _UndecoratedHook(MemEvalHook):
                    """Hook implementation without @hook_runner."""

                    def marker(self) -> str:
                        """Return a test marker."""
                        return "undecorated"

                    def label(self) -> str:
                        """Return a second marker for pylint friendliness."""
                        return "undecorated-label"

                    @staticmethod
                    def run_hooks(e: Any) -> None:  # pylint: disable=abstract-method
                        """No-op undecorated implementation."""
                        _ = e
                _ = _UndecoratedHook

            with self.assertRaises(TypeError):

                class _BadSignatureHook(MemEvalHook):
                    """Hook implementation with a bad signature."""

                    def marker(self) -> str:
                        """Return a test marker."""
                        return "bad-signature"

                    def label(self) -> str:
                        """Return a second marker for pylint friendliness."""
                        return "bad-signature-label"

                    @staticmethod
                    @hook_runner("bad_signature")
                    def run_hooks(e: Any, extra: Any) -> None:  # pylint: disable=arguments-differ
                        """No-op implementation with an extra argument."""
                        _ = (e, extra)
                _ = _BadSignatureHook
        finally:
            MemEvalHook.hook_registry = old_registry

        fake_evaluator = _FakeTemplateEvaluator()
        Template.run_hooks(fake_evaluator)
        self.assertEqual(fake_evaluator.calls[0][0], "ccfg")
        self.assertIn(("pp_micro", "1f1b", Template.f), fake_evaluator.calls)
        self.assertTrue(any(call[0] == "body" and call[1] == "NOT_REC_LAYER" for call in fake_evaluator.calls))
        self.assertIsNone(Template.f(None, None))
        self.assertIsNone(Template.custom_ccfg(None))

    def test_utils_accessors(self) -> None:
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Cover evaluator utility accessors with fake configs.
        Expectation: Utility methods read and update config/context state deterministically.
        """
        utils = object.__new__(_Utils)
        fake_cfg = SimpleNamespace(
            model_name="unit-model",
            device_capacity=Memory.from_gb(2),
            multimodal=False,
            n_lay=4,
            n_mtp=1,
            p=2,
            vp=2,
            m=3,
            pp_sched="1f1b",
            layer_custom_config=None,
            get_strategy=lambda: {"dp": 2, "tp": 1},
            print_stages=lambda stages, spec_stage_id=-1: (stages, spec_stage_id),
        )
        fake_ctx = Context()
        fake_ctx.pp_micro_eval["1f1b"] = _micro_factor
        setattr(utils, "_ccfg", fake_cfg)
        setattr(utils, "_ctx", fake_ctx)

        self.assertEqual(utils.get_model_name(), "unit-model")
        self.assertEqual(utils.get_strategy(), {"dp": 2, "tp": 1})
        self.assertEqual(utils.get_max_device_memory(), 2048)
        self.assertEqual(utils.get_num_layers(), 5)
        utils.set_layer_custom()
        self.assertEqual(fake_cfg.layer_custom_config, [(4, None)])
        utils.set_layer_custom([(1, _mark_custom_config)])
        self.assertEqual(fake_cfg.layer_custom_config[0][0], 1)
        utils.all_stage_micro_factors()
        self.assertEqual(utils.print_node_eval(), {})
        self.assertIsNone(utils.print_stages([[LayerType.NOT_REC_LAYER]], 0))

        multimodal_cfg = SimpleNamespace(
            model_name="multi",
            device_capacity=Memory.from_gb(1),
            multimodal=True,
            mm_order=["vision", "text"],
            mm_ccfgs={
                "vision": SimpleNamespace(n_lay=2, n_mtp=0),
                "text": SimpleNamespace(n_lay=3, n_mtp=1),
            },
            get_strategy=lambda: {},
        )
        utils.set_config(multimodal_cfg)
        self.assertEqual(utils.get_num_layers(), [2, 4])

    def test_eval_utils_expression_and_microbatch_helpers(self) -> None:
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Cover expression evaluation and PP micro-batch factor formulas.
        Expectation: Formula helpers update context accounting and return deterministic factors.
        """
        ctx = Context()
        ctx.current_node = "body"
        ctx.current_stage_id = 0
        ctx.current_chunk_id = 0
        ctx.current_lay_id = 0
        ctx.real_lay_ids = [[[0]]]

        result = EvalUtils.eval_expr_insight(
            expr="max(param, active) + aux",
            mem_val={
                "param": 2 * 1024 * 1024,
                "active": 1024 * 1024,
                "aux": 3 * 1024 * 1024,
            },
            mem_cat={
                "param": MemType.MODEL_PARAM,
                "active": MemType.ATTN_ACTIV,
                "aux": MemType.FFN_ACTIV,
            },
            ctx=ctx,
        )
        self.assertEqual(result, 5 * 1024 * 1024)
        self.assertGreater(ctx.accu_mem_type[MemType.MODEL_PARAM], 0)
        self.assertEqual(EvalUtils.mb({"x": (1024 * 1024, 2 * 1024 * 1024)})["x"], 3)
        self.assertEqual(EvalUtils.rec_coeff(False, False), 1)
        self.assertEqual(EvalUtils.rec_coeff(True, False), 0)

        with self.assertRaises(AttributeError):
            EvalUtils.eval_expr_insight(
                expr="unknown",
                mem_val={},
                mem_cat={},
                ctx=ctx,
            )

        ccfg = SimpleNamespace(p=4, m=3, vp=1, n_s_split=2, s=16)
        ctx.current_stage_id = 1
        ctx.current_chunk_id = 0
        self.assertEqual(EvalUtils.pp_1f1b_micro_factor(ccfg, ctx), 2)
        self.assertEqual(EvalUtils.pp_seq1f1b_micro_factor(ccfg, ctx), 3)

        ccfg_vpp = SimpleNamespace(p=4, m=3, vp=2, n_s_split=2, s=16)
        ctx.vpp_less_mem = False
        ctx.current_stage_id = 1
        ctx.current_chunk_id = 0
        self.assertEqual(EvalUtils.pp_1f1b_micro_factor(ccfg_vpp, ctx), 3)
        self.assertEqual(EvalUtils.pp_seq1f1b_micro_factor(ccfg_vpp, ctx), 4)
        ctx.current_chunk_id = 1
        self.assertEqual(EvalUtils.pp_dualpipe_v_micro_factor(ccfg_vpp, ctx), 1)
        self.assertEqual(EvalUtils.pp_gpipe_micro_factor(ccfg_vpp, ctx), 4)

    def test_ppb_layer_description_helpers(self) -> None:
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Cover PPB layer description builders and body combiners.
        Expectation: Descriptions are grouped and combined deterministically.
        """
        ppb = _PPB(
            SimpleNamespace(ppb_combined=[[("model-a", "body"), ("model-b", "body")]]),
            _dynamic_mem_for_ppb,
        )
        ccfg = SimpleNamespace(model_name="model-a")
        ctx = Context()
        ctx.head_node = "head"
        ctx.tail_node = "tail"

        ctx.current_node = "head"
        head_desc = ppb.lay_ppb(ccfg, ctx, 4 * MEGABYTE)
        self.assertEqual(head_desc["type"], "HEAD")
        self.assertEqual(head_desc["memory_parameter"], 9)

        ctx.current_node = "tail"
        tail_desc = ppb.lay_ppb(ccfg, ctx, 4 * MEGABYTE)
        self.assertEqual(tail_desc["type"], "TAIL")
        self.assertEqual(tail_desc["memory_parameter"], 9)

        ctx.current_node = LayerType.NOT_REC_LAYER
        body_desc = ppb.lay_ppb(ccfg, ctx, 4 * MEGABYTE)
        self.assertEqual(body_desc["type"], "BODY")
        self.assertEqual(body_desc["memory_activation"], 2)
        self.assertEqual(body_desc["memory_select_rec"], 2)
        self.assertEqual(body_desc["memory_recompute"], 2)

        descriptions = []
        ppb.add_to_ppb_list(descriptions, body_desc.copy())
        ppb.add_to_ppb_list(descriptions, body_desc.copy())
        self.assertEqual(descriptions[0]["nb_layer"], 2)
        self.assertEqual(descriptions[0]["name"], "BODY_0")
        ppb.add_to_ppb_list(descriptions, head_desc.copy())
        self.assertEqual(descriptions[-1]["name"], "HEAD")

        descriptions = [
            {"model_name": "model-a", "type": "BODY", "name": "A", "memory_parameter": 1, "nb_layer": 1},
            {"model_name": "model-b", "type": "BODY", "name": "B", "memory_parameter": 2, "nb_layer": 1},
        ]
        ppb.ppb_combine_bodies(descriptions)
        self.assertEqual(descriptions[0]["model_name"], "combined_model-a_model-b")
        self.assertEqual(descriptions[0]["memory_parameter"], 3)

        ctx.current_node = "head"
        head_new = ppb.lay_ppb_new(ccfg, ctx, 4 * MEGABYTE)
        ctx.current_node = LayerType.NOT_REC_LAYER
        body_new = ppb.lay_ppb_new(ccfg, ctx, 4 * MEGABYTE)
        self.assertEqual(head_new["options"], ["NONE", "FULL"])
        self.assertEqual(body_new["memory_activation"]["FULL"], 2)

        descriptions_new = [
            {
                "model_name": "model-a",
                "type": "BODY",
                "name": "A",
                "memory_parameter": 1,
                "memory_activation": {"NONE": 2, "FULL": 3},
                "nb_layer": 1,
            },
            {
                "model_name": "model-b",
                "type": "BODY",
                "name": "B",
                "memory_parameter": 4,
                "memory_activation": {"NONE": 5, "FULL": 6},
                "nb_layer": 1,
            },
        ]
        ppb.ppb_combine_bodies_new(descriptions_new)
        self.assertEqual(descriptions_new[0]["memory_parameter"], 5)
        self.assertEqual(descriptions_new[0]["memory_activation"]["FULL"], 9)

        disabled_ppb = _PPB(SimpleNamespace(ppb_combined=[]), _dynamic_mem_for_ppb)
        disabled_descriptions = [{"model_name": "model-a", "type": "BODY"}]
        self.assertIsNone(disabled_ppb.ppb_combine_bodies(disabled_descriptions))
        self.assertIsNone(disabled_ppb.ppb_combine_bodies_new(disabled_descriptions))

    def test_backward_overhead_helpers(self) -> None:
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Cover backward overhead estimators with tiny synthetic stages.
        Expectation: 1F1B and ZBV schedules produce deterministic overhead values.
        """
        ctx = Context()
        backbone = SimpleNamespace(apply_hook=lambda hook, ccfg=None, ctx=None: hook(backbone))
        ccfg = SimpleNamespace(pp_sched="1f1b", vp=1, m=4, p=2, n_mtp=1)
        overhead = _BackwardOverhead(backbone, ccfg, ctx, _dynamic_mem_for_overhead)
        stages = [[[LayerType.FULL_REC_LAYER, LayerType.OUTPUT_LAYER]]]
        record = {
            (0, 0, 0): (ccfg, ctx, lambda _: None),
            (0, 0, 1): (ccfg, ctx, lambda _: None),
        }

        self.assertEqual(overhead.estimate(stages, 0, record), 48)
        self.assertEqual(ctx.current_node, LayerType.NOT_REC_LAYER)
        self.assertEqual(
            overhead._fetch_node_and_switch_env(  # pylint: disable=protected-access
                stages, record, -1, -1, -1
            ),
            LayerType.OUTPUT_LAYER,
        )

        ccfg.pp_sched = "zero_bubble_v"
        ccfg.vp = 2
        stages_zbv = [[[LayerType.NOT_REC_LAYER], [LayerType.FULL_REC_LAYER]]]
        record_zbv = {
            (0, 0, 0): (ccfg, ctx, lambda _: None),
            (0, 1, 0): (ccfg, ctx, lambda _: None),
        }
        self.assertEqual(overhead.estimate(stages_zbv, 0, record_zbv), 48)

        ctx.vpp_less_mem = True
        self.assertEqual(overhead.vpp_1f1b_steady_overhead(1, [[1], [4], [2]]), 3)
        ctx.vpp_less_mem = False
        self.assertEqual(overhead.vpp_1f1b_steady_overhead(1, [[1], [4], [2]]), 3)

    def test_func_tracer_helpers(self) -> None:
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Trace a tiny local function and exercise expression substitution.
        Expectation: Tracing returns the original result and records readable expressions.
        """
        tracer = _FuncTracer()
        holder = SimpleNamespace(offset=2)
        self.assertTrue(tracer.is_constant_expr("holder.offset"))
        self.assertFalse(tracer.is_constant_expr("value + holder.offset"))
        self.assertIn("holder.offset", tracer.scrap_term_symbols("value + holder.offset"))
        self.assertEqual(tracer.substitute("value + holder.offset", {"value": 3, "holder": holder}), "3 + 2")

        previous_trace = sys.gettrace()
        try:
            result = tracer.wrap(_trace_sample)(3, holder)
        finally:
            # _FuncTracer deliberately clears tracing; restore pytest-cov's tracer.
            sys.settrace(previous_trace)

        self.assertEqual(result, 10)
        self.assertIn(_trace_sample.__code__, tracer.code_trees)
        self.assertIsNotNone(
            tracer.fetch_node_from_lineno(_trace_sample.__code__.co_firstlineno + 2, _trace_sample.__code__)
        )

    def test_evaluator_public_helpers_with_fake_backbone(self) -> None:
        """
        Feature: TestSappNDMemoryEstimation.
        Description: Cover EvaluatorV2 result wrappers without parsing or model search.
        Expectation: Config restoration, PPB caching, layer accessors and fit checks are deterministic.
        """
        evaluator = object.__new__(EvaluatorV2)
        fake_ccfg = SimpleNamespace(
            parser=SimpleNamespace(ccfg=None),
            model_name="unit-model",
            device_capacity=Memory.from_mb(100),
            multimodal=False,
            hooks_dict={},
        )
        evaluator._ccfg = fake_ccfg  # pylint: disable=protected-access
        evaluator._overhead_obj = SimpleNamespace(_ccfg=None)  # pylint: disable=protected-access
        evaluator.ppb = None

        node_key = (0, 0, 0, LayerType.NOT_REC_LAYER.name[0])
        insights = [
            {"Static": 10, "Dynamic": 20, "Node Log": {node_key: {"_param": 4, "_activ": 5, "_comm": 1}}},
            {"Static": 30, "Dynamic": 25, "Node Log": {}},
        ]
        backbone_calls = []

        def fake_backbone(*args: Any) -> tuple:
            """Return deterministic insight and PPB results."""
            backbone_calls.append(args)
            return insights, {"layers": ["unit"]}

        evaluator._estimate_backbone = fake_backbone  # pylint: disable=protected-access

        self.assertEqual(evaluator.estimate_peak(), 55)
        self.assertEqual(evaluator.estimate_peak_insight(), insights)
        self.assertEqual(evaluator.static_mem_stage(0), 10)
        self.assertEqual(evaluator.dynamic_mem_stage(0), 20)
        self.assertEqual(evaluator.logs_mem_stage(0), insights[0]["Node Log"])
        self.assertEqual(evaluator.static_mem_layer(LayerType.NOT_REC_LAYER, 0), 4)
        self.assertEqual(evaluator.dynamic_mem_layer(LayerType.NOT_REC_LAYER, 0), 6)

        self.assertEqual(evaluator.estimate_layer_memory(ppb_format=2), {"layers": ["unit"]})
        calls_after_first_ppb = len(backbone_calls)
        self.assertEqual(evaluator.estimate_layer_memory(), {"layers": ["unit"]})
        self.assertEqual(len(backbone_calls), calls_after_first_ppb)

        self.assertTrue(evaluator.mem_fit(95, tolerance=5))
        self.assertTrue(evaluator.mem_fit(90, margin=5))
        self.assertFalse(evaluator.mem_fit(101))

        reset_calls = []
        evaluator.config_path = "unit.yaml"
        evaluator.update_config = reset_calls.append
        evaluator.reset_config()
        self.assertEqual(reset_calls, ["unit.yaml"])

        hook_calls = []
        evaluator._ccfg.hooks_dict = {"unit": hook_calls.append}  # pylint: disable=protected-access
        evaluator.load_hook_cls("hook")
        self.assertEqual(evaluator.hook_cls, "hook")
        self.assertIs(hook_calls[0], evaluator)
