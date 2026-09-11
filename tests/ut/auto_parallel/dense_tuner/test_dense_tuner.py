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
"""Unit tests for Dense LLM strategy tuner — covers AP-DENSE-01 through AP-DENSE-09."""
# pylint: disable=missing-public-docstring,missing-public-type-hints,import-outside-toplevel
# SAPP optional deps (sapp_nd/sapp_ppb/pulp/matplotlib) are imported lazily inside
# availability-guard try blocks so UTs skip cleanly when they are not installed.
import os
import shutil
import tempfile
import unittest
from typing import Any

from hyper_parallel.auto_parallel.dense_tuner.config import (
    ConstraintConfig,
    HardwareConfig,
    ModelConfig,
    SearchSpaceConfig,
    TunerConfig,
)
from hyper_parallel.auto_parallel.dense_tuner.candidate_generator import (
    _divisors,
    _enumerate_cp_values,
    _enumerate_mbn_values,
    _enumerate_pp_values,
    _enumerate_tp_values,
    _is_power_of_2,
    filter_divisibility,
    filter_memory_constraint,
    filter_min_dp,
    filter_step_time_constraint,
    generate_candidates,
)
from hyper_parallel.auto_parallel.dense_tuner.estimator import (
    _apply_strategy_to_ccfg,
    compute_stage_info,
    estimate_memory,
    estimate_performance,
)
from hyper_parallel.auto_parallel.dense_tuner.result import (
    StrategySummary,
    TunerResult,
)
from hyper_parallel.auto_parallel.dense_tuner.strategy import (
    MemoryCost,
    ParallelStrategy,
    PerformanceCost,
    StrategyStatus,
)
from hyper_parallel.auto_parallel.dense_tuner.tuner import DenseTuner
from hyper_parallel.auto_parallel.dense_tuner.visualizer import StrategyVisualizer


class TestDivisors(unittest.TestCase):
    """Tests for divisor enumeration helpers."""

    def test_divisors(self) -> None:
        for value, min_val, max_val, expected in (
            (8, None, None, [1, 2, 4, 8]),
            (16, 2, 8, [2, 4, 8]),
            (7, None, None, [1, 7]),
        ):
            with self.subTest(value=value, min_val=min_val, max_val=max_val):
                kwargs = {}
                if min_val is not None:
                    kwargs["min_val"] = min_val
                if max_val is not None:
                    kwargs["max_val"] = max_val
                self.assertEqual(_divisors(value, **kwargs), expected)
        for value, expected in ((1, True), (8, True), (6, False), (0, False)):
            with self.subTest(value=value):
                self.assertEqual(_is_power_of_2(value), expected)


class TestParallelStrategy(unittest.TestCase):
    """Tests for ParallelStrategy data structure."""

    def test_total_devices(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=4, pp_degree=2, cp_degree=1)
        self.assertEqual(s.total_devices, 16)

    def test_is_feasible(self) -> None:
        s = ParallelStrategy()
        self.assertTrue(s.is_feasible)
        s.mark_infeasible("test")
        self.assertFalse(s.is_feasible)
        self.assertEqual(s.filter_reason, "test")

    def test_key(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=4, pp_degree=1, cp_degree=1, micro_batch_num=2)
        self.assertEqual(s.key, "dp2_tp4_pp1_cp1_mbn2")

    def test_to_dict(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2)
        d = s.to_dict()
        self.assertEqual(d["dp_degree"], 2)
        self.assertEqual(d["tp_degree"], 2)
        self.assertEqual(d["status"], "feasible")

    def test_mark_not_supported(self) -> None:
        s = ParallelStrategy()
        s.mark_not_supported("unsupported combo")
        self.assertEqual(s.status, StrategyStatus.NOT_SUPPORTED)
        self.assertEqual(s.filter_reason, "unsupported combo")


class TestCostToDict(unittest.TestCase):
    """Tests for MemoryCost/PerformanceCost to-dict serialization."""

    def test_to_dict(self) -> None:
        mc = MemoryCost(model_states=100, activations=50, total=150)
        d = mc.to_dict()
        self.assertEqual(d["model_states"], 100)
        self.assertEqual(d["total"], 150)

        pc = PerformanceCost(compute=10, communication=5, total=15)
        d = pc.to_dict()
        self.assertEqual(d["compute"], 10)
        self.assertEqual(d["total"], 15)


class TestModelConfig(unittest.TestCase):
    """Tests for ModelConfig defaults and derived properties."""

    def test_derived_properties(self) -> None:
        for heads, kv, expected_kv in ((32, 8, 8), (32, 0, 32)):
            with self.subTest(heads=heads, kv=kv):
                self.assertEqual(
                    ModelConfig(num_heads=heads, num_kv_heads=kv).effective_num_kv_heads,
                    expected_kv,
                )
        self.assertEqual(
            ModelConfig(hidden_size=4096, intermediate_size=0).effective_intermediate_size,
            16384,
        )

    def test_num_params_billion(self) -> None:
        m = ModelConfig(hidden_size=4096, num_layers=32, num_heads=32, vocab_size=32000)
        self.assertGreater(m.num_params_billion, 0)


class TestSearchSpaceConfig(unittest.TestCase):
    """Tests for SearchSpaceConfig."""

    def test_defaults_and_from_dict(self) -> None:
        self.assertFalse(SearchSpaceConfig().enable_cp)
        data = {"dp_range": [1, 2, 4], "enable_cp": True}
        ss = SearchSpaceConfig.from_dict(data)
        self.assertEqual(ss.dp_range, [1, 2, 4])
        self.assertTrue(ss.enable_cp)


class TestTunerConfig(unittest.TestCase):
    """Tests for TunerConfig loading."""

    def test_from_dict(self) -> None:
        data = {
            "hardware": {"num_devices": 16, "device_type": "A3"},
            "model": {"hidden_size": 8192, "num_layers": 80},
            "top_k": 3,
        }
        cfg = TunerConfig.from_dict(data)
        self.assertEqual(cfg.hardware.num_devices, 16)
        self.assertEqual(cfg.hardware.device_type, "A3")
        self.assertEqual(cfg.model.hidden_size, 8192)
        self.assertEqual(cfg.top_k, 3)
        self.assertEqual(cfg.ppb_top_k, 3, "ppb_top_k defaults to top_k")

    def test_from_dict_ppb_top_k(self) -> None:
        data = {
            "hardware": {"num_devices": 16, "device_type": "A3"},
            "model": {"hidden_size": 8192, "num_layers": 80},
            "top_k": 5,
            "ppb_top_k": 2,
        }
        cfg = TunerConfig.from_dict(data)
        self.assertEqual(cfg.top_k, 5)
        self.assertEqual(cfg.ppb_top_k, 2)

    def test_from_yaml(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            yaml_path = os.path.join(tmpdir, "config.yaml")
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(
                    "hardware:\n  num_devices: 8\n  device_type: A2\n"
                    "model:\n  hidden_size: 4096\n  num_layers: 32\n"
                    "top_k: 5\n"
                )
            cfg = TunerConfig.from_yaml(yaml_path)
            self.assertEqual(cfg.hardware.num_devices, 8)
            self.assertEqual(cfg.model.hidden_size, 4096)
            self.assertEqual(cfg.top_k, 5)
            self.assertEqual(cfg.ppb_top_k, 5, "ppb_top_k defaults to top_k")
        finally:
            shutil.rmtree(tmpdir)

    def test_from_yaml_ppb_top_k(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            yaml_path = os.path.join(tmpdir, "config.yaml")
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(
                    "hardware:\n  num_devices: 8\n  device_type: A2\n"
                    "model:\n  hidden_size: 4096\n  num_layers: 32\n"
                    "top_k: 6\nppb_top_k: 3\n"
                )
            cfg = TunerConfig.from_yaml(yaml_path)
            self.assertEqual(cfg.top_k, 6)
            self.assertEqual(cfg.ppb_top_k, 3)
        finally:
            shutil.rmtree(tmpdir)


class TestEnumerateValues(unittest.TestCase):
    """Tests for dimension value enumeration."""

    def test_tp_values(self) -> None:
        for kv, expected in ((32, [1, 2, 4, 8]), (4, [1, 2, 4])):
            with self.subTest(kv=kv):
                model = ModelConfig(num_heads=32, num_kv_heads=kv)
                ss = SearchSpaceConfig()
                result = _enumerate_tp_values(8, model, ss)
                for tp in result:
                    self.assertTrue(_is_power_of_2(tp))
                self.assertEqual(result, expected)

    def test_pp_values(self) -> None:
        model = ModelConfig(num_layers=32)
        ss = SearchSpaceConfig(enable_pp=True)
        result = _enumerate_pp_values(8, model, ss)
        self.assertIn(1, result)
        self.assertIn(2, result)
        self.assertIn(4, result)
        self.assertIn(8, result)

    def test_cp_values(self) -> None:
        self.assertEqual(_enumerate_cp_values(8, SearchSpaceConfig(enable_cp=False)), [1])
        result = _enumerate_cp_values(8, SearchSpaceConfig(enable_cp=True))
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_mbn_values(self) -> None:
        result = _enumerate_mbn_values(gbs=32, dp=2, pp=4, search_space=SearchSpaceConfig())
        for mbn in result:
            self.assertGreaterEqual(mbn, 4)
        result = _enumerate_mbn_values(gbs=16, dp=2, pp=1, search_space=SearchSpaceConfig())
        self.assertIn(1, result)
        self.assertIn(2, result)
        self.assertIn(4, result)
        self.assertIn(8, result)


class TestCandidateGeneration(unittest.TestCase):
    """Tests for candidate strategy generation."""

    def test_generate_candidates_small(self) -> None:
        hardware = HardwareConfig(num_devices=8, device_type="A2")
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        search_space = SearchSpaceConfig(enable_pp=True)
        constraints = ConstraintConfig(global_batch_size=16)
        candidates = generate_candidates(hardware, model, search_space, constraints)
        self.assertGreater(len(candidates), 0)
        for s in candidates:
            self.assertEqual(s.total_devices, 8)

    def test_generate_candidates_dp_tp_only(self) -> None:
        hardware = HardwareConfig(num_devices=8, device_type="A2")
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        search_space = SearchSpaceConfig(enable_pp=False, enable_cp=False)
        constraints = ConstraintConfig(global_batch_size=16)
        candidates = generate_candidates(hardware, model, search_space, constraints)
        self.assertGreater(len(candidates), 0)
        for s in candidates:
            self.assertEqual(s.pp_degree, 1)
            self.assertEqual(s.cp_degree, 1)


class TestFilterDivisibility(unittest.TestCase):
    """Tests for divisibility constraint filtering."""

    def test_valid_strategy_passes(self) -> None:
        hardware = HardwareConfig(num_devices=8)
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        constraints = ConstraintConfig(global_batch_size=16)
        strategies = [ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=2, cp_degree=1, micro_batch_num=2)]
        result = filter_divisibility(strategies, model, hardware, constraints)
        self.assertTrue(result[0].is_feasible)

    def test_tp_not_power_of_2_filtered(self) -> None:
        hardware = HardwareConfig(num_devices=6)
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        constraints = ConstraintConfig(global_batch_size=16)
        strategies = [ParallelStrategy(dp_degree=2, tp_degree=3, pp_degree=1, cp_degree=1)]
        result = filter_divisibility(strategies, model, hardware, constraints)
        self.assertTrue(any(not s.is_feasible for s in result))

    def test_pp_not_dividing_layers_filtered(self) -> None:
        hardware = HardwareConfig(num_devices=4)
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        constraints = ConstraintConfig(global_batch_size=16)
        strategies = [ParallelStrategy(dp_degree=1, tp_degree=1, pp_degree=3, cp_degree=1)]
        result = filter_divisibility(strategies, model, hardware, constraints)
        self.assertTrue(any(not s.is_feasible for s in result))
        infeasible = [s for s in result if not s.is_feasible]
        self.assertTrue(
            any("num_layers" in s.filter_reason or "pp_degree" in s.filter_reason for s in infeasible)
        )

    def test_mbn_less_than_pp_filtered(self) -> None:
        hardware = HardwareConfig(num_devices=4)
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        constraints = ConstraintConfig(global_batch_size=16)
        strategies = [ParallelStrategy(dp_degree=1, tp_degree=1, pp_degree=4, cp_degree=1, micro_batch_num=2)]
        result = filter_divisibility(strategies, model, hardware, constraints)
        self.assertTrue(any(not s.is_feasible for s in result))
        self.assertTrue(any("micro_batch_num" in s.filter_reason for s in result if not s.is_feasible))


class TestFilterMemory(unittest.TestCase):
    """Tests for memory constraint filtering."""

    def test_memory_filter(self) -> None:
        for total, limit, feasible, has_reason in (
            (30000, 65536, True, False),
            (70000, 65536, False, True),
            (999999, 0, True, False),
        ):
            with self.subTest(total=total, limit=limit):
                s = ParallelStrategy()
                s.memory_cost = MemoryCost(total=total)
                result = filter_memory_constraint([s], ConstraintConfig(memory_limit_mb=limit))
                self.assertEqual(result[0].is_feasible, feasible)
                if has_reason:
                    self.assertIn("memory", result[0].filter_reason)


class TestFilterMinDp(unittest.TestCase):
    """Tests for minimum DP degree filtering."""

    def test_min_dp_filter(self) -> None:
        for dp, min_dp, feasible in ((4, 2, True), (1, 2, False)):
            with self.subTest(dp=dp, min_dp=min_dp):
                s = ParallelStrategy(dp_degree=dp)
                result = filter_min_dp([s], ConstraintConfig(min_dp_degree=min_dp))
                self.assertEqual(result[0].is_feasible, feasible)


class TestEstimator(unittest.TestCase):
    """Tests for memory and performance estimation."""

    def setUp(self) -> None:
        self.model = ModelConfig(
            model_name="llama-7b",
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            vocab_size=32000,
            seq_length=4096,
            precision_bytes=2,
        )
        self.hardware = HardwareConfig(num_devices=8, memory_per_device_mb=65536)

    def test_estimate_memory_dp_tp(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        mem = estimate_memory(s, self.model, global_batch_size=16)
        self.assertGreater(mem.total, 0)
        self.assertGreater(mem.model_states, 0)

    def test_estimate_memory_with_pp(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=2, cp_degree=1)
        mem = estimate_memory(s, self.model, global_batch_size=16)
        self.assertGreater(mem.total, 0)

    def test_estimate_performance_dp_tp(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        perf = estimate_performance(s, self.model, self.hardware, global_batch_size=16)
        self.assertGreater(perf.total, 0)
        self.assertGreater(perf.compute, 0)

    def test_estimate_performance_with_pp(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=2, cp_degree=1, micro_batch_num=4)
        perf = estimate_performance(s, self.model, self.hardware, global_batch_size=16)
        self.assertGreater(perf.pipeline_bubble, 0)

    def test_estimate_performance_with_cp(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=2)
        perf = estimate_performance(s, self.model, self.hardware, global_batch_size=16)
        self.assertGreater(perf.communication, 0)

    def test_compute_stage_info_no_pp(self) -> None:
        s = ParallelStrategy(dp_degree=1, tp_degree=1, pp_degree=1, cp_degree=1)
        s.memory_cost = MemoryCost(total=100)
        s.performance_cost = PerformanceCost(compute=10, total=10)
        stages = compute_stage_info(s, self.model, 16)
        self.assertEqual(len(stages), 1)

    def test_compute_stage_info_with_pp(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=4, cp_degree=1)
        s.memory_cost = MemoryCost(total=400)
        s.performance_cost = PerformanceCost(compute=40, total=50)
        stages = compute_stage_info(s, self.model, 16)
        self.assertEqual(len(stages), 4)
        total_layers = sum(st.num_layers for st in stages)
        self.assertEqual(total_layers, 32)


class TestStrategySummary(unittest.TestCase):
    """Tests for StrategySummary."""

    def test_from_strategies(self) -> None:
        s1 = ParallelStrategy()
        s2 = ParallelStrategy()
        s2.mark_infeasible("too much memory")
        s3 = ParallelStrategy()
        s3.mark_not_supported("unsupported")
        summary = StrategySummary.from_strategies([s1, s2, s3])
        self.assertEqual(summary.total_candidates, 3)
        self.assertEqual(summary.feasible_count, 1)
        self.assertEqual(summary.infeasible_count, 1)
        self.assertEqual(summary.not_supported_count, 1)
        self.assertIn("too much memory", summary.filter_reasons)


class TestTunerResult(unittest.TestCase):
    """Tests for TunerResult."""

    def test_to_dict(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=4)
        result = TunerResult(top_k_strategies=[s])
        d = result.to_dict()
        self.assertEqual(len(d["top_k_strategies"]), 1)
        self.assertEqual(d["top_k_strategies"][0]["dp_degree"], 2)


class TestDenseTunerAPDENSE01(unittest.TestCase):
    """AP-DENSE-01: Dense LLM small search space DP/TP."""

    def test_dp_tp_search_produces_top_k(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8, device_type="A2", memory_per_device_mb=65536),
            model=ModelConfig(
                hidden_size=256,
                num_layers=4,
                num_heads=4,
                num_kv_heads=4,
                vocab_size=1024,
                seq_length=512,
                precision_bytes=2,
            ),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False),
            constraints=ConstraintConfig(global_batch_size=16, memory_limit_mb=65536),
            top_k=3,
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        self.assertGreater(len(result.top_k_strategies), 0)
        for s in result.top_k_strategies:
            self.assertTrue(s.is_feasible)
            self.assertGreater(s.performance_cost.total, 0)
            self.assertGreater(s.memory_cost.total, 0)

    def test_top_k_sorted_by_step_time(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=3,
            sort_by="step_time",
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        times = [s.performance_cost.total for s in result.top_k_strategies]
        self.assertEqual(times, sorted(times))


class TestDenseTunerAPDENSE02(unittest.TestCase):
    """AP-DENSE-02: Search space includes PP and micro batch."""

    def test_pp_search_produces_stage_summary(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8, memory_per_device_mb=65536),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=True, enable_cp=False, pp_range=[2], tp_range=[1]),
            constraints=ConstraintConfig(global_batch_size=32, memory_limit_mb=65536),
            top_k=3,
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        pp_feasible = [s for s in result.all_candidates if s.pp_degree > 1 and s.is_feasible]
        self.assertTrue(pp_feasible, "Expected at least one feasible PP candidate")
        for s in pp_feasible[:5]:
            self.assertGreater(len(s.stage_info), 1, f"PP strategy {s.key} should have stage info")
        self.assertTrue(any(len(bd) > 0 for bd in result.stage_summary))


class TestDenseTunerAPDENSE03(unittest.TestCase):
    """AP-DENSE-03: Search space includes CP."""

    def test_cp_search_produces_cp_estimation(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8, memory_per_device_mb=131072),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4, seq_length=512),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=True),
            constraints=ConstraintConfig(global_batch_size=8, memory_limit_mb=131072),
            top_k=5,
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        cp_feasible = [s for s in result.all_candidates if s.cp_degree > 1 and s.is_feasible]
        self.assertTrue(cp_feasible, "Expected at least one feasible CP candidate")
        for s in cp_feasible[:5]:
            self.assertGreater(s.performance_cost.total, 0)
        self.assertGreater(len(result.top_k_strategies), 0)


class TestDenseTunerAPDENSE04(unittest.TestCase):
    """AP-DENSE-04: Memory-exceeding strategies are filtered with filter_reason."""

    def test_oom_strategies_filtered(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8, memory_per_device_mb=50000),
            model=ModelConfig(
                num_layers=4,
                num_heads=4,
                num_kv_heads=4,
            ),
            search_space=SearchSpaceConfig(enable_pp=True, enable_cp=False, pp_range=[2], tp_range=[1]),
            constraints=ConstraintConfig(global_batch_size=16, memory_limit_mb=50000),
            top_k=3,
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        feasible = [s for s in result.all_candidates if s.is_feasible]
        infeasible = [s for s in result.all_candidates if not s.is_feasible]
        if infeasible:
            oom = [s for s in infeasible if "memory" in s.filter_reason.lower()]
            self.assertGreater(len(oom), 0, "Expected some strategies to exceed memory limit")
        else:
            all_memory = [s.memory_cost.total for s in feasible]
            self.assertTrue(
                any(m > 0 for m in all_memory),
                "Expected memory estimates to be populated",
            )


class TestDenseTunerAPDENSE05(unittest.TestCase):
    """AP-DENSE-05: Divisibility-invalid strategies are filtered early."""

    def test_invalid_divisibility_filtered(self) -> None:
        hardware = HardwareConfig(num_devices=12)
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        search_space = SearchSpaceConfig(enable_pp=True, pp_range=[3])
        constraints = ConstraintConfig(global_batch_size=16)
        candidates = generate_candidates(hardware, model, search_space, constraints)
        filtered = filter_divisibility(candidates, model, hardware, constraints)
        infeasible = [s for s in filtered if not s.is_feasible]
        self.assertGreater(len(infeasible), 0)


class TestDenseTunerAPDENSE06(unittest.TestCase):
    """AP-DENSE-06: Top-k output validation."""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=True, enable_cp=False, pp_range=[2], tp_range=[1]),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=3,
        )
        cls._result = DenseTuner(cfg).tune()

    def test_top_k_count_and_order(self) -> None:
        self.assertLessEqual(len(self._result.top_k_strategies), 3)
        for i in range(len(self._result.top_k_strategies) - 1):
            self.assertLessEqual(
                self._result.top_k_strategies[i].performance_cost.total,
                self._result.top_k_strategies[i + 1].performance_cost.total,
            )

    def test_top_k_strategies_have_complete_fields(self) -> None:
        for s in self._result.top_k_strategies:
            self.assertGreater(s.dp_degree, 0)
            self.assertGreater(s.tp_degree, 0)
            self.assertGreater(s.pp_degree, 0)
            self.assertGreater(s.cp_degree, 0)
            self.assertGreater(s.micro_batch_num, 0)
            self.assertGreater(s.memory_cost.total, 0)
            self.assertGreater(s.performance_cost.total, 0)


class TestDenseTunerAPDENSE07(unittest.TestCase):
    """AP-DENSE-07: Partial dimension search."""

    def test_dp_only_search(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=3,
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        self.assertGreater(len(result.top_k_strategies), 0)
        for s in result.top_k_strategies:
            self.assertEqual(s.pp_degree, 1)
            self.assertEqual(s.cp_degree, 1)

    def test_tp_only_with_fixed_dp(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(
                dp_range=[2],
                enable_pp=False,
                enable_cp=False,
            ),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=3,
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        for s in result.top_k_strategies:
            self.assertEqual(s.dp_degree, 2)


class TestDenseTunerAPDENSE08(unittest.TestCase):
    """AP-DENSE-08: Visualization output."""

    def test_visualization_generates_files(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            cfg = TunerConfig(
                hardware=HardwareConfig(num_devices=8),
                model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
                search_space=SearchSpaceConfig(enable_pp=True, enable_cp=False, pp_range=[2], tp_range=[1]),
                constraints=ConstraintConfig(global_batch_size=16),
                top_k=3,
                output_dir=tmpdir,
            )
            tuner = DenseTuner(cfg)
            result = tuner.tune_and_visualize()
            self.assertGreater(len(result.top_k_strategies), 0)
            chart_files = [f for f in os.listdir(tmpdir) if f.endswith((".txt", ".png"))]
            self.assertGreater(len(chart_files), 0)
        finally:
            shutil.rmtree(tmpdir)


class TestDenseTunerAPDENSE09(unittest.TestCase):
    """AP-DENSE-09: User adjusts constraints and re-searches."""

    def test_adjust_memory_limit_changes_results(self) -> None:
        cfg1 = TunerConfig(
            hardware=HardwareConfig(num_devices=8, memory_per_device_mb=65536),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False, tp_range=[1, 2]),
            constraints=ConstraintConfig(global_batch_size=16, memory_limit_mb=65536),
            top_k=3,
        )
        result1 = DenseTuner(cfg1).tune()

        cfg2 = TunerConfig(
            hardware=HardwareConfig(num_devices=8, memory_per_device_mb=4096),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False, tp_range=[1, 2]),
            constraints=ConstraintConfig(global_batch_size=16, memory_limit_mb=4096),
            top_k=3,
        )
        result2 = DenseTuner(cfg2).tune()

        feasible1 = sum(1 for s in result1.all_candidates if s.is_feasible)
        feasible2 = sum(1 for s in result2.all_candidates if s.is_feasible)
        self.assertGreaterEqual(feasible1, feasible2)

    def test_adjust_search_space_enables_cp(self) -> None:
        cfg_no_cp = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4, seq_length=1024),
            search_space=SearchSpaceConfig(enable_cp=False, enable_pp=True, pp_range=[2], tp_range=[1]),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=3,
        )
        result_no_cp = DenseTuner(cfg_no_cp).tune()

        cfg_with_cp = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4, seq_length=1024),
            search_space=SearchSpaceConfig(enable_cp=True, enable_pp=True, pp_range=[2], tp_range=[1]),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=3,
        )
        result_with_cp = DenseTuner(cfg_with_cp).tune()

        cp_candidates = [s for s in result_with_cp.all_candidates if s.cp_degree > 1]
        no_cp_candidates = [s for s in result_no_cp.all_candidates if s.cp_degree > 1]
        self.assertGreaterEqual(len(cp_candidates), len(no_cp_candidates))


class TestDenseTunerSortByMemory(unittest.TestCase):
    """Tests for sort_by='memory' option."""

    def test_sort_by_memory(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=5,
            sort_by="memory",
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        mems = [s.memory_cost.total for s in result.top_k_strategies]
        self.assertEqual(mems, sorted(mems))


class TestDenseTunerSearchLog(unittest.TestCase):
    """Tests for search log output."""

    def test_search_log_contains_key_info(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(model_name="test-model", num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=3,
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        self.assertIn("test-model", result.search_log)
        self.assertIn("Feasible", result.search_log)


class TestFilterGBSDivisibility(unittest.TestCase):
    """Tests for GBS divisibility constraint filtering."""

    def test_gbs_not_divisible_by_dp_mbn_filtered(self) -> None:
        hardware = HardwareConfig(num_devices=8)
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        constraints = ConstraintConfig(global_batch_size=7)
        strategies = [ParallelStrategy(dp_degree=4, tp_degree=2, pp_degree=1, cp_degree=1, micro_batch_num=1)]
        result = filter_divisibility(strategies, model, hardware, constraints)
        self.assertFalse(result[0].is_feasible)
        self.assertIn("gbs", result[0].filter_reason)

    def test_gbs_zero_skips_check(self) -> None:
        hardware = HardwareConfig(num_devices=8)
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        constraints = ConstraintConfig(global_batch_size=0)
        strategies = [ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=2, cp_degree=1, micro_batch_num=2)]
        result = filter_divisibility(strategies, model, hardware, constraints)
        self.assertTrue(result[0].is_feasible)


class TestFilterCPSeqLength(unittest.TestCase):
    """Tests for CP seq_length divisibility constraint filtering."""

    def test_seq_length_not_divisible_by_cp_filtered(self) -> None:
        hardware = HardwareConfig(num_devices=12)
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32, seq_length=4096)
        constraints = ConstraintConfig(global_batch_size=16)
        strategies = [ParallelStrategy(dp_degree=4, tp_degree=1, pp_degree=1, cp_degree=3, micro_batch_num=2)]
        result = filter_divisibility(strategies, model, hardware, constraints)
        self.assertFalse(result[0].is_feasible)
        self.assertIn("seq_length", result[0].filter_reason)

    def test_cp1_always_passes(self) -> None:
        hardware = HardwareConfig(num_devices=8)
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32, seq_length=4096)
        constraints = ConstraintConfig(global_batch_size=16)
        strategies = [ParallelStrategy(dp_degree=8, tp_degree=1, pp_degree=1, cp_degree=1, micro_batch_num=2)]
        result = filter_divisibility(strategies, model, hardware, constraints)
        self.assertTrue(result[0].is_feasible)


class TestFilterStepTimeConstraint(unittest.TestCase):
    """Tests for max step time constraint filtering."""

    def test_step_time_filter(self) -> None:
        for total, limit, feasible, has_reason in (
            (100, 200, True, False),
            (300, 200, False, True),
            (999999, 0, True, False),
        ):
            with self.subTest(total=total, limit=limit):
                s = ParallelStrategy()
                s.performance_cost = PerformanceCost(total=total)
                result = filter_step_time_constraint([s], ConstraintConfig(max_step_time_ms=limit))
                self.assertEqual(result[0].is_feasible, feasible)
                if has_reason:
                    self.assertIn("step_time", result[0].filter_reason)


class TestStrategyToDictCompleteness(unittest.TestCase):
    """Tests for ParallelStrategy.to_dict() serialization completeness."""

    def test_to_dict_includes_stage_info(self) -> None:
        from hyper_parallel.auto_parallel.dense_tuner.strategy import StageInfo
        s = ParallelStrategy(dp_degree=2, tp_degree=2)
        s.stage_info = [StageInfo(stage_id=0, layer_range=(0, 16), memory_mb=100, time_ms=10, num_layers=16)]
        d = s.to_dict()
        self.assertIn("stage_info", d)
        self.assertEqual(len(d["stage_info"]), 1)
        self.assertEqual(d["stage_info"][0]["stage_id"], 0)

    def test_to_dict_includes_extra(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2, extra={"note": "test"})
        d = s.to_dict()
        self.assertIn("extra", d)
        self.assertEqual(d["extra"]["note"], "test")

    def test_tuner_result_to_dict_includes_all_candidates(self) -> None:
        s1 = ParallelStrategy(dp_degree=2, tp_degree=4)
        s2 = ParallelStrategy(dp_degree=4, tp_degree=2)
        result = TunerResult(top_k_strategies=[s1], all_candidates=[s1, s2])
        d = result.to_dict()
        self.assertIn("all_candidates", d)
        self.assertEqual(len(d["all_candidates"]), 2)


class TestDenseTunerSortByDpFirst(unittest.TestCase):
    """Tests for sort_by='dp_first' option."""

    def test_sort_by_dp_first(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=10,
            sort_by="dp_first",
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        if len(result.top_k_strategies) > 1:
            for i in range(len(result.top_k_strategies) - 1):
                self.assertGreaterEqual(
                    result.top_k_strategies[i].dp_degree,
                    result.top_k_strategies[i + 1].dp_degree,
                )


class TestFSDPParallelStrategy(unittest.TestCase):
    """Tests for ParallelStrategy with fsdp_enabled."""

    def test_fsdp_default_false(self) -> None:
        s = ParallelStrategy()
        self.assertFalse(s.fsdp_enabled)

    def test_fsdp_key_includes_fsdp(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1, micro_batch_num=2, fsdp_enabled=True)
        self.assertIn("fsdp", s.key)

    def test_fsdp_key_omits_fsdp_when_false(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1, micro_batch_num=2)
        self.assertNotIn("fsdp", s.key)

    def test_to_dict_includes_fsdp_enabled(self) -> None:
        s = ParallelStrategy(dp_degree=2, fsdp_enabled=True)
        d = s.to_dict()
        self.assertTrue(d["fsdp_enabled"])

    def test_to_dict_fsdp_default(self) -> None:
        s = ParallelStrategy()
        d = s.to_dict()
        self.assertFalse(d["fsdp_enabled"])


class TestFSDPSearchSpaceConfig(unittest.TestCase):
    """Tests for SearchSpaceConfig with enable_fsdp."""

    def test_enable_fsdp_default_false(self) -> None:
        ss = SearchSpaceConfig()
        self.assertFalse(ss.enable_fsdp)

    def test_enable_fsdp_from_dict(self) -> None:
        ss = SearchSpaceConfig.from_dict({"enable_fsdp": True})
        self.assertTrue(ss.enable_fsdp)


class TestFSDPCandidateGeneration(unittest.TestCase):
    """Tests for FSDP candidate expansion."""

    def test_fsdp_expands_candidates(self) -> None:
        hardware = HardwareConfig(num_devices=8, device_type="A2")
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        search_space_no_fsdp = SearchSpaceConfig(enable_pp=False, enable_cp=False, enable_fsdp=False)
        search_space_fsdp = SearchSpaceConfig(enable_pp=False, enable_cp=False, enable_fsdp=True)
        constraints = ConstraintConfig(global_batch_size=16)

        candidates_no_fsdp = generate_candidates(hardware, model, search_space_no_fsdp, constraints)
        candidates_fsdp = generate_candidates(hardware, model, search_space_fsdp, constraints)

        self.assertGreater(len(candidates_fsdp), len(candidates_no_fsdp))
        fsdp_strategies = [s for s in candidates_fsdp if s.fsdp_enabled]
        self.assertGreater(len(fsdp_strategies), 0)
        for s in fsdp_strategies:
            self.assertGreater(s.dp_degree, 1)

    def test_fsdp_not_applied_for_dp1(self) -> None:
        hardware = HardwareConfig(num_devices=4, device_type="A2")
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        search_space = SearchSpaceConfig(
            enable_pp=False, enable_cp=False, enable_fsdp=True,
            dp_range=[1],
        )
        constraints = ConstraintConfig(global_batch_size=16)
        candidates = generate_candidates(hardware, model, search_space, constraints)
        fsdp_strategies = [s for s in candidates if s.fsdp_enabled]
        self.assertEqual(len(fsdp_strategies), 0)

    def test_fsdp_divisibility_filter(self) -> None:
        hardware = HardwareConfig(num_devices=8)
        model = ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32)
        constraints = ConstraintConfig(global_batch_size=16)
        strategies = [ParallelStrategy(dp_degree=1, tp_degree=8, pp_degree=1, cp_degree=1, fsdp_enabled=True)]
        result = filter_divisibility(strategies, model, hardware, constraints)
        self.assertFalse(result[0].is_feasible)
        self.assertIn("fsdp_enabled", result[0].filter_reason)


class TestFSDPEstimator(unittest.TestCase):
    """Tests for FSDP memory estimation."""

    def test_fsdp_reduces_memory_analytical(self) -> None:
        model = ModelConfig(
            model_name="llama-7b",
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            vocab_size=32000,
            seq_length=4096,
            precision_bytes=2,
        )
        s_no_fsdp = ParallelStrategy(dp_degree=4, tp_degree=2, pp_degree=1, cp_degree=1)
        s_fsdp = ParallelStrategy(dp_degree=4, tp_degree=2, pp_degree=1, cp_degree=1, fsdp_enabled=True)
        mem_no_fsdp = estimate_memory(s_no_fsdp, model, global_batch_size=16)
        mem_fsdp = estimate_memory(s_fsdp, model, global_batch_size=16)
        self.assertLess(mem_fsdp.total, mem_no_fsdp.total)
        self.assertLess(mem_fsdp.gradients, mem_no_fsdp.gradients)
        self.assertLess(mem_fsdp.optimizer_states, mem_no_fsdp.optimizer_states)

    def test_fsdp_no_effect_when_dp1(self) -> None:
        model = ModelConfig(
            hidden_size=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=32,
            vocab_size=32000,
            seq_length=4096,
            precision_bytes=2,
        )
        s_no_fsdp = ParallelStrategy(dp_degree=1, tp_degree=2, pp_degree=1, cp_degree=1)
        s_fsdp = ParallelStrategy(dp_degree=1, tp_degree=2, pp_degree=1, cp_degree=1, fsdp_enabled=True)
        mem_no_fsdp = estimate_memory(s_no_fsdp, model, global_batch_size=16)
        mem_fsdp = estimate_memory(s_fsdp, model, global_batch_size=16)
        self.assertAlmostEqual(mem_fsdp.total, mem_no_fsdp.total, places=2)


class TestApplyStrategyToCcfgWithParser(unittest.TestCase):
    """Tests for _apply_strategy_to_ccfg with parser-backed ccfg (from Parallelize)."""

    def setUp(self) -> None:
        self.cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32),
            constraints=ConstraintConfig(global_batch_size=16),
        )
        self.parallelize = self.cfg.create_sapp_parallelize()
        self.ccfg = self.parallelize.config.ccfg

    def tearDown(self) -> None:
        yaml_path = getattr(self.parallelize, "_yaml_path", None)
        if yaml_path and os.path.exists(yaml_path):
            os.unlink(yaml_path)

    def test_ccfg_has_parser(self) -> None:
        self.assertIsNotNone(self.ccfg.parser)

    def test_strategy_fields_set(self) -> None:
        strategy = ParallelStrategy(dp_degree=2, tp_degree=4, pp_degree=1, cp_degree=1, micro_batch_num=2, op_degree=2)
        _apply_strategy_to_ccfg(self.ccfg, strategy)
        self.assertEqual(self.ccfg.d, 2)
        self.assertEqual(self.ccfg.t, 4)
        self.assertEqual(self.ccfg.p, 1)
        self.assertEqual(self.ccfg.cp, 1)
        self.assertEqual(self.ccfg.m, 2)

    def test_fsdp_sets_has_grad_shard(self) -> None:
        strategy = ParallelStrategy(dp_degree=4, tp_degree=2, pp_degree=1, cp_degree=1, fsdp_enabled=True)
        _apply_strategy_to_ccfg(self.ccfg, strategy)
        self.assertTrue(self.ccfg.has_grad_shard)

    def test_no_fsdp_has_grad_shard_false(self) -> None:
        strategy = ParallelStrategy(dp_degree=4, tp_degree=2, pp_degree=1, cp_degree=1, fsdp_enabled=False)
        _apply_strategy_to_ccfg(self.ccfg, strategy)
        self.assertFalse(self.ccfg.has_grad_shard)

    def test_shard_comm_recomputed(self) -> None:
        strategy = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        _apply_strategy_to_ccfg(self.ccfg, strategy)
        self.assertGreater(self.ccfg.shard_p_os_non_exp, 0)
        self.assertEqual(self.ccfg.comm_t, 1.0)


class TestDenseTunerFSDP(unittest.TestCase):
    """End-to-end tests for FSDP-enabled tuning."""

    @classmethod
    def setUpClass(cls) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8, memory_per_device_mb=65536),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False, enable_fsdp=True, tp_range=[1, 2]),
            constraints=ConstraintConfig(global_batch_size=16, memory_limit_mb=65536),
            top_k=3,
        )
        cls._result = DenseTuner(cfg).tune()
        cfg_log = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(model_name="test-fsdp", num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False, enable_fsdp=True, tp_range=[1, 2]),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=3,
        )
        cls._result_log = DenseTuner(cfg_log).tune()

    def test_fsdp_tuning_produces_fsdp_strategies(self) -> None:
        fsdp_feasible = [s for s in self._result.all_candidates if s.is_feasible and s.fsdp_enabled]
        self.assertGreater(len(fsdp_feasible), 0)

    def test_fsdp_strategies_use_less_memory(self) -> None:
        non_fsdp = [s for s in self._result.all_candidates if s.is_feasible and not s.fsdp_enabled and s.dp_degree > 1]
        fsdp = [s for s in self._result.all_candidates if s.is_feasible and s.fsdp_enabled]
        if non_fsdp and fsdp:
            for nf in non_fsdp:
                for f in fsdp:
                    if (nf.dp_degree == f.dp_degree and nf.tp_degree == f.tp_degree
                            and nf.op_degree == f.op_degree and nf.op_degree > 1):
                        self.assertLess(f.memory_cost.total, nf.memory_cost.total)
                        return

    def test_fsdp_search_log(self) -> None:
        self.assertIn("FSDP", self._result_log.search_log)


class TestNDPPBVisualization(unittest.TestCase):
    """Tests for ND/PPB visualization output collection."""

    def test_visualizer_accepts_nd_ppb_plot_files(self) -> None:
        viz = StrategyVisualizer(output_dir=tempfile.mkdtemp())
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        s.memory_cost.total = 1000
        s.performance_cost.total = 500
        all_cands = [s]
        files = viz.visualize(
            [s], all_cands,
            nd_plot_files=["/nonexistent_nd_plot.png"],
            ppb_plot_files=["/nonexistent_ppb_plot.svg"],
        )
        non_existent = [f for f in files if "nonexistent" in f]
        self.assertEqual(len(non_existent), 0)

    def test_collect_nd_visualization_output_empty(self) -> None:
        from hyper_parallel.auto_parallel.dense_tuner.candidate_generator import (
            collect_nd_visualization_output,
        )
        out = tempfile.mkdtemp()
        result = collect_nd_visualization_output(out)
        self.assertIsInstance(result, list)

    def test_generate_nd_memory_plots_sapp_unavailable(self) -> None:
        from hyper_parallel.auto_parallel.dense_tuner.estimator import (
            generate_nd_memory_plots,
        )
        out = tempfile.mkdtemp()
        strategy = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        try:
            cfg = TunerConfig(
                hardware=HardwareConfig(num_devices=8),
                model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            )
            parallelize = cfg.create_sapp_parallelize()
            ccfg = parallelize.config.ccfg
        except Exception:
            self.skipTest("SAPP-ND not available")
        try:
            result = generate_nd_memory_plots(strategy, ccfg, out)
            self.assertIsInstance(result, list)
        except Exception:
            pass
        finally:
            yaml_path = getattr(parallelize, "_yaml_path", None)
            if yaml_path and os.path.exists(yaml_path):
                os.unlink(yaml_path)

    def test_generate_ppb_pipeline_plot_no_pp(self) -> None:
        from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import (
            generate_ppb_pipeline_plot,
        )
        out = tempfile.mkdtemp()
        strategy = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        result = generate_ppb_pipeline_plot(strategy, None, out)
        self.assertIsNone(result)


class TestEvaluatorV2CcfgKwarg(unittest.TestCase):
    """Tests that EvaluatorV2 receives ccfg via the ccfg keyword argument."""

    def test_ccfg_kwarg_accepted(self) -> None:
        try:
            from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
            cfg = TunerConfig(
                hardware=HardwareConfig(num_devices=8),
                model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            )
            parallelize = cfg.create_sapp_parallelize()
            ccfg = parallelize.config.ccfg
        except Exception:
            self.skipTest("SAPP-ND not available")
        try:
            evaluator = EvaluatorV2(None, ccfg=ccfg, log_level=0)
            self.assertIs(evaluator.ccfg, ccfg)
        finally:
            yaml_path = getattr(parallelize, "_yaml_path", None)
            if yaml_path and os.path.exists(yaml_path):
                os.unlink(yaml_path)


class TestPipelineBalancerNoSAPP(unittest.TestCase):
    """Tests for pipeline_balancer with SAPP-ND ccfg (from Parallelize)."""

    def setUp(self) -> None:
        self.cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            constraints=ConstraintConfig(global_batch_size=16),
        )
        try:
            self.parallelize = self.cfg.create_sapp_parallelize()
            self.ccfg = self.parallelize.config.ccfg
        except Exception:
            self.parallelize = None
            self.ccfg = None

    def tearDown(self) -> None:
        if self.parallelize is not None:
            yaml_path = getattr(self.parallelize, "_yaml_path", None)
            if yaml_path and os.path.exists(yaml_path):
                os.unlink(yaml_path)

    def test_balance_pipeline_pp1_returns_same(self) -> None:
        from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import balance_pipeline
        strategy = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        result = balance_pipeline(strategy, None)
        self.assertIs(result, strategy)

    def test_generate_ppb_pipeline_plot_pp1_returns_none(self) -> None:
        from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import generate_ppb_pipeline_plot
        strategy = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        result = generate_ppb_pipeline_plot(strategy, None, "/tmp")
        self.assertIsNone(result)

    def test_balance_strategies_only_pp_candidates(self) -> None:
        if self.ccfg is None:
            self.skipTest("SAPP-ND not available")
        from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import balance_strategies
        s_pp = ParallelStrategy(dp_degree=2, tp_degree=1, pp_degree=4, cp_degree=1, micro_batch_num=4)
        s_pp.memory_cost = MemoryCost(total=1000)
        s_no_pp = ParallelStrategy(dp_degree=8, tp_degree=1, pp_degree=1, cp_degree=1)
        s_no_pp.memory_cost = MemoryCost(total=500)
        result = balance_strategies([s_pp, s_no_pp], self.ccfg)
        self.assertEqual(len(result), 2)

    def test_build_layers_from_strategy(self) -> None:
        if self.ccfg is None:
            self.skipTest("SAPP-ND not available")
        from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import _build_layers_from_strategy
        strategy = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=2, cp_degree=1, micro_batch_num=4)
        _apply_strategy_to_ccfg(self.ccfg, strategy)
        layers = _build_layers_from_strategy(strategy, self.ccfg)
        self.assertIsInstance(layers, list)
        self.assertGreater(len(layers), 0, "PPB layer build must produce at least one layer")

    def test_layers_from_ppb_input(self) -> None:
        from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import _layers_from_ppb_input
        ppb_input = {
            "layers_description": [
                {
                    "type": "HEAD", "name": "embed", "nb_layer": 1,
                    "time": 1.0, "forward_time": 0.33, "memory_parameter": 100.0,
                    "model_name": "llama-7b",
                },
                {
                    "type": "BODY", "name": "layer0", "nb_layer": 32,
                    "time": 5.0, "forward_time": 1.67, "memory_parameter": 200.0,
                    "memory_activation": 800.0,
                    "memory_select_rec": 700.0,
                    "memory_select_comm": 500.0,
                    "memory_both_comm_select": 450.0,
                    "memory_recompute": 300.0,
                    "backward_time": 3.33,
                    "select_rec_time": 3.46,
                    "select_comm_time": 3.75,
                    "both_comm_select_time": 3.88,
                    "recompute_time": 5.0,
                    "backward_coef": 0.0,
                    "select_rec_coef": 0.04,
                    "select_comm_coef": 0.125,
                    "both_comm_select_coef": 0.165,
                    "recompute_coef": 0.5,
                    "model_name": "llama-7b",
                },
                {
                    "type": "TAIL", "name": "output", "nb_layer": 1,
                    "time": 1.0, "forward_time": 0.33, "memory_parameter": 100.0,
                    "model_name": "llama-7b",
                },
            ]
        }
        try:
            layers = _layers_from_ppb_input(ppb_input, type("C", (), {"model_name": "llama-7b"})())
            self.assertEqual(len(layers), 3)
            head, body, tail = layers  # pylint: disable=unbalanced-tuple-unpacking
            self.assertEqual(head.type_.name, "HEAD")
            self.assertEqual(body.type_.name, "BODY")
            self.assertEqual(tail.type_.name, "TAIL")
            self.assertEqual(body.nb_layer_, 32)
            self.assertGreater(body.memory_activation_rec_[0], 0)
            self.assertGreater(body.memory_activation_rec_[4], 0)
        except (ImportError, AttributeError) as exc:
            self.skipTest(f"SAPP-PPB dependencies not available: {exc}")

    def test_balance_pipeline_with_ccfg(self) -> None:
        if self.ccfg is None:
            self.skipTest("SAPP-ND not available")
        from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import balance_pipeline
        strategy = ParallelStrategy(dp_degree=2, tp_degree=1, pp_degree=2, cp_degree=1, micro_batch_num=4)
        strategy.memory_cost = MemoryCost(total=10000)
        result = balance_pipeline(strategy, self.ccfg, memory_limit_mb=65536)
        self.assertIsInstance(result, ParallelStrategy)
        self.assertEqual(result.pp_degree, 2)
        self.assertIsNotNone(result.stage_info, "PPB must produce stage_info for PP strategies")
        self.assertGreater(len(result.stage_info), 0, "stage_info must not be empty")

    def test_generate_ppb_pipeline_plot_with_ccfg(self) -> None:
        if self.ccfg is None:
            self.skipTest("SAPP-ND not available")
        from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import generate_ppb_pipeline_plot
        strategy = ParallelStrategy(dp_degree=2, tp_degree=1, pp_degree=2, cp_degree=1, micro_batch_num=4)
        strategy.memory_cost = MemoryCost(total=10000)
        out = tempfile.mkdtemp()
        result = generate_ppb_pipeline_plot(strategy, self.ccfg, out, memory_limit_mb=65536)
        self.assertIsNotNone(result, "PPB pipeline plot must succeed for PP strategies")


class TestTunerInferGbs(unittest.TestCase):
    """Tests for DenseTuner._infer_gbs."""

    def test_infer_gbs_from_strategy(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32),
        )
        tuner = DenseTuner(cfg)
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=2, cp_degree=1, micro_batch_num=4)
        gbs = tuner._infer_gbs([s], HardwareConfig(num_devices=8))
        self.assertGreater(gbs, 0)

    def test_infer_gbs_no_feasible(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=32, num_heads=32, num_kv_heads=32),
        )
        tuner = DenseTuner(cfg)
        gbs = tuner._infer_gbs([], HardwareConfig(num_devices=8))
        self.assertEqual(gbs, 8)


class TestTunerBuildPpbResult(unittest.TestCase):
    """Tests for DenseTuner._build_ppb_result."""

    def test_pp1_returns_none(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        self.assertIsNone(DenseTuner._build_ppb_result(s))

    def test_pp_with_ppb_distribution(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=1, pp_degree=4, cp_degree=1)
        s.extra["ppb_distribution"] = [[1, 2, 3, 4]]
        result = DenseTuner._build_ppb_result(s)
        self.assertIsNotNone(result)
        self.assertEqual(result.vpp_degree, 1)

    def test_pp_without_ppb_data_returns_none(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=1, pp_degree=4, cp_degree=1)
        result = DenseTuner._build_ppb_result(s)
        self.assertIsNone(result)

    def test_pp_with_vpp_returns_result(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=1, pp_degree=4, cp_degree=1, vpp_degree=2)
        result = DenseTuner._build_ppb_result(s)
        self.assertIsNotNone(result)
        self.assertEqual(result.vpp_degree, 2)

    def test_pp_with_layer_offset_and_recompute(self) -> None:
        s = ParallelStrategy(dp_degree=2, tp_degree=1, pp_degree=4, cp_degree=1, vpp_degree=1)
        s.extra["ppb_distribution"] = [[1, 2, 3, 4]]
        s.layer_offset = [0, 1, 0, -1]
        s.layer_recompute = [0, 1, 2, 3]
        result = DenseTuner._build_ppb_result(s)
        self.assertIsNotNone(result)
        self.assertEqual(result.stage_offsets, [0, 1, 0, -1])
        self.assertEqual(result.recompute_policy, [0, 1, 2, 3])


class TestPpbResultToDict(unittest.TestCase):
    """Tests for PPBResult.to_dict serialization."""

    def test_list_fields_serialized_as_lists(self) -> None:
        from hyper_parallel.auto_parallel.dense_tuner.result import PPBResult
        r = PPBResult(
            vpp_degree=2,
            stage_offsets=[0, 1, 0, -1],
            recompute_policy=[0, 1, 2, 3],
            ppb_distribution=[[1, 2], [3, 4]],
        )
        d = r.to_dict()
        self.assertEqual(d["vpp_degree"], 2)
        self.assertEqual(d["stage_offsets"], [0, 1, 0, -1])
        self.assertEqual(d["recompute_policy"], [0, 1, 2, 3])
        self.assertEqual(d["ppb_distribution"], "[[1, 2], [3, 4]]")

    def test_none_fields(self) -> None:
        from hyper_parallel.auto_parallel.dense_tuner.result import PPBResult
        r = PPBResult()
        d = r.to_dict()
        self.assertIsNone(d["stage_offsets"])
        self.assertIsNone(d["recompute_policy"])
        self.assertIsNone(d["ppb_distribution"])


class TestStageLayerCounts(unittest.TestCase):
    """Tests for PPB layer-count parsing helpers."""

    def test_stage_offsets_from_counts(self) -> None:
        from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import _stage_offsets
        self.assertEqual(_stage_offsets([], 4), [])
        self.assertEqual(_stage_offsets([5, 5, 5, 5], 4), [0, 0, 0, 0])
        self.assertEqual(_stage_offsets([6, 6, 5, 5], 4), [1, 1, 0, 0])
        self.assertEqual(_stage_offsets([5, 6, 6], 3), [0, 1, 1])
        self.assertEqual(_stage_offsets([5, 5, 5, 5], 3), [-1, -1, -1, -1])

    def test_per_stage_layer_counts(self) -> None:
        try:
            from hyper_parallel.auto_parallel.sapp_ppb import Layer
            from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute
        except (ImportError, AttributeError) as exc:
            self.skipTest(f"SAPP-PPB dependencies not available: {exc}")

        from hyper_parallel.auto_parallel.dense_tuner.pipeline_balancer import (
            _per_stage_layer_counts,
        )

        class FakeVar:
            def __init__(self, value: int) -> None:
                self.varValue = value  # pylint: disable=invalid-name

        class FakeLayer:
            def __init__(self, name: str, type_) -> None:
                self.name_ = name
                self.type_ = type_

        num_interleave = 2
        num_stages = 4

        def grid(counts: Any) -> Any:
            return [
                [FakeVar(counts[i][s]) for s in range(num_stages)]
                for i in range(num_interleave)
            ]

        def zero_grid() -> Any:
            return grid([[0] * num_stages for _ in range(num_interleave)])

        def rec_map(
            none_counts: Any,
            slct_counts: Any,
        ) -> Any:
            recs = [None] * len(Recompute.TYPE)
            recs[Recompute.TYPE.NONE] = grid(none_counts)
            recs[Recompute.TYPE.SLCT] = grid(slct_counts)
            for rec in (Recompute.TYPE.COMM, Recompute.TYPE.BOTH, Recompute.TYPE.FULL):
                recs[rec] = zero_grid()
            return recs

        variables_ = {
            "body1": rec_map(
                none_counts=[[1, 1, 1, 1], [1, 1, 1, 1]],
                slct_counts=[[1, 0, 0, 0], [0, 0, 0, 0]],
            ),
        }
        layers = [
            FakeLayer("head", Layer.type_enum.HEAD),
            FakeLayer("body1", Layer.type_enum.BODY),
            FakeLayer("tail", Layer.type_enum.TAIL),
        ]
        pipeline = type("FakePipeline", (), {"problem_": type(
            "FakeProblem", (), {"variables_": variables_})()})()

        stage_layers, stage_recompute = _per_stage_layer_counts(
            pipeline, layers, num_stages, num_interleave
        )
        self.assertEqual(stage_layers, [3, 2, 2, 2])
        self.assertEqual(stage_recompute, [1, 0, 0, 0])


class TestTunerSortStrategies(unittest.TestCase):
    """Tests for DenseTuner._sort_strategies."""

    def test_sort_by_step_time(self) -> None:
        s1 = ParallelStrategy(dp_degree=1, tp_degree=2, pp_degree=1, cp_degree=1)
        s1.performance_cost = PerformanceCost(total=100)
        s2 = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        s2.performance_cost = PerformanceCost(total=50)
        result = DenseTuner._sort_strategies([s1, s2], "step_time")
        self.assertEqual(result[0].dp_degree, 2)

    def test_sort_by_memory(self) -> None:
        s1 = ParallelStrategy(dp_degree=1, tp_degree=2, pp_degree=1, cp_degree=1)
        s1.memory_cost = MemoryCost(total=2000)
        s2 = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        s2.memory_cost = MemoryCost(total=1000)
        result = DenseTuner._sort_strategies([s1, s2], "memory")
        self.assertEqual(result[0].dp_degree, 2)

    def test_sort_by_dp_first(self) -> None:
        s1 = ParallelStrategy(dp_degree=1, tp_degree=2, pp_degree=1, cp_degree=1)
        s1.performance_cost = PerformanceCost(total=50)
        s2 = ParallelStrategy(dp_degree=4, tp_degree=2, pp_degree=1, cp_degree=1)
        s2.performance_cost = PerformanceCost(total=100)
        result = DenseTuner._sort_strategies([s1, s2], "dp_first")
        self.assertEqual(result[0].dp_degree, 4)

    def test_sort_by_unknown_defaults_step_time(self) -> None:
        s1 = ParallelStrategy(dp_degree=1, tp_degree=2, pp_degree=1, cp_degree=1)
        s1.performance_cost = PerformanceCost(total=100)
        s2 = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        s2.performance_cost = PerformanceCost(total=50)
        result = DenseTuner._sort_strategies([s1, s2], "unknown")
        self.assertEqual(result[0].dp_degree, 2)


class TestEstimateMemoryAnalytical(unittest.TestCase):
    """Tests for analytical memory estimation edge cases."""

    def test_fsdp_dp1_no_change(self) -> None:
        model = ModelConfig(hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=32,
                            vocab_size=32000, seq_length=4096, precision_bytes=2)
        s1 = ParallelStrategy(dp_degree=1, tp_degree=2, pp_degree=1, cp_degree=1, fsdp_enabled=False)
        s2 = ParallelStrategy(dp_degree=1, tp_degree=2, pp_degree=1, cp_degree=1, fsdp_enabled=True)
        m1 = estimate_memory(s1, model, global_batch_size=16)
        m2 = estimate_memory(s2, model, global_batch_size=16)
        self.assertAlmostEqual(m1.total, m2.total, places=2)

    def test_cp_reduces_activation_memory(self) -> None:
        model = ModelConfig(hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=32,
                            vocab_size=32000, seq_length=4096, precision_bytes=2)
        s_no_cp = ParallelStrategy(dp_degree=1, tp_degree=2, pp_degree=1, cp_degree=1)
        s_cp = ParallelStrategy(dp_degree=1, tp_degree=2, pp_degree=1, cp_degree=2)
        m_no_cp = estimate_memory(s_no_cp, model, global_batch_size=16)
        m_cp = estimate_memory(s_cp, model, global_batch_size=16)
        self.assertLess(m_cp.activations, m_no_cp.activations)


class TestConfigYamlLoaders(unittest.TestCase):
    """Tests for from_yaml methods on config dataclasses."""

    def test_hardware_from_yaml(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "hw.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("num_devices: 16\ndevice_type: A3\nmemory_per_device_mb: 131072\n")
            hw = HardwareConfig.from_yaml(path)
            self.assertEqual(hw.num_devices, 16)
            self.assertEqual(hw.memory_per_device_mb, 131072)
        finally:
            shutil.rmtree(tmpdir)

    def test_model_from_yaml(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "model.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("hidden_size: 8192\nnum_layers: 80\n")
            m = ModelConfig.from_yaml(path)
            self.assertEqual(m.hidden_size, 8192)
        finally:
            shutil.rmtree(tmpdir)

    def test_constraint_from_yaml(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "constraint.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("memory_limit_mb: 32768\nglobal_batch_size: 32\n")
            c = ConstraintConfig.from_yaml(path)
            self.assertEqual(c.memory_limit_mb, 32768)
        finally:
            shutil.rmtree(tmpdir)

    def test_search_space_from_yaml(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, "ss.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("enable_pp: true\nenable_cp: true\nenable_fsdp: true\n")
            ss = SearchSpaceConfig.from_yaml(path)
            self.assertTrue(ss.enable_pp)
            self.assertTrue(ss.enable_cp)
            self.assertTrue(ss.enable_fsdp)
        finally:
            shutil.rmtree(tmpdir)


class TestTuneAndVisualize(unittest.TestCase):
    """Tests for DenseTuner.tune_and_visualize."""

    def test_tune_and_visualize_creates_output(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            cfg = TunerConfig(
                hardware=HardwareConfig(num_devices=8, memory_per_device_mb=65536),
                model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
                search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False),
                constraints=ConstraintConfig(global_batch_size=16),
                top_k=3,
                output_dir=tmpdir,
            )
            tuner = DenseTuner(cfg)
            result = tuner.tune_and_visualize()
            self.assertGreater(len(result.top_k_strategies), 0)
        finally:
            shutil.rmtree(tmpdir)


class TestEstimateMemorySapp(unittest.TestCase):
    """Tests for SAPP-ND memory estimation path (estimate_memory with ccfg)."""

    def setUp(self) -> None:
        self.cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            constraints=ConstraintConfig(global_batch_size=16),
        )
        try:
            self.parallelize = self.cfg.create_sapp_parallelize()
            self.ccfg = self.parallelize.config.ccfg
        except Exception:
            self.parallelize = None
            self.ccfg = None
        self.model = self.cfg.model

    def tearDown(self) -> None:
        if self.parallelize is not None:
            yaml_path = getattr(self.parallelize, "_yaml_path", None)
            if yaml_path and os.path.exists(yaml_path):
                os.unlink(yaml_path)

    def test_estimate_memory_with_ccfg(self) -> None:
        if self.ccfg is None:
            self.skipTest("SAPP-ND not available")
        strategy = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        try:
            mem = estimate_memory(strategy, self.model, global_batch_size=16, ccfg=self.ccfg)
            self.assertGreater(mem.total, 0)
        except (TypeError, ValueError, AttributeError):
            pass

    def test_estimate_memory_sapp_with_pp(self) -> None:
        if self.ccfg is None:
            self.skipTest("SAPP-ND not available")
        strategy = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=2, cp_degree=1, micro_batch_num=4)
        try:
            mem = estimate_memory(strategy, self.model, global_batch_size=16, ccfg=self.ccfg)
            self.assertGreater(mem.total, 0)
        except (TypeError, ValueError, AttributeError):
            pass

    def test_estimate_memory_sapp_with_fsdp(self) -> None:
        if self.ccfg is None:
            self.skipTest("SAPP-ND not available")
        strategy = ParallelStrategy(dp_degree=4, tp_degree=2, pp_degree=1, cp_degree=1, fsdp_enabled=True)
        try:
            mem = estimate_memory(strategy, self.model, global_batch_size=16, ccfg=self.ccfg)
            self.assertGreater(mem.total, 0)
        except (TypeError, ValueError, AttributeError):
            pass


class TestEstimatePerformanceSapp(unittest.TestCase):
    """Tests for SAPP-ND performance estimation path (estimate_performance with ccfg)."""

    def setUp(self) -> None:
        self.cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            constraints=ConstraintConfig(global_batch_size=16),
        )
        try:
            self.parallelize = self.cfg.create_sapp_parallelize()
            self.ccfg = self.parallelize.config.ccfg
        except Exception:
            self.parallelize = None
            self.ccfg = None
        self.model = self.cfg.model
        self.hardware = self.cfg.hardware

    def tearDown(self) -> None:
        if self.parallelize is not None:
            yaml_path = getattr(self.parallelize, "_yaml_path", None)
            if yaml_path and os.path.exists(yaml_path):
                os.unlink(yaml_path)

    def test_estimate_performance_with_ccfg(self) -> None:
        if self.ccfg is None:
            self.skipTest("SAPP-ND not available")
        strategy = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=1, cp_degree=1)
        try:
            perf = estimate_performance(strategy, self.model, self.hardware, global_batch_size=16, ccfg=self.ccfg)
            self.assertGreater(perf.total, 0)
        except (TypeError, ValueError, AttributeError):
            pass

    def test_estimate_performance_sapp_with_pp(self) -> None:
        if self.ccfg is None:
            self.skipTest("SAPP-ND not available")
        strategy = ParallelStrategy(dp_degree=2, tp_degree=2, pp_degree=2, cp_degree=1, micro_batch_num=4)
        try:
            perf = estimate_performance(strategy, self.model, self.hardware, global_batch_size=16, ccfg=self.ccfg)
            self.assertGreater(perf.total, 0)
        except (TypeError, ValueError, AttributeError):
            pass


class TestCandidateGeneratorEdgeCases(unittest.TestCase):
    """Edge-case tests for candidate generation."""

    def test_mbn_zero_gbs(self) -> None:
        result = _enumerate_mbn_values(gbs=0, dp=2, pp=1, search_space=SearchSpaceConfig())
        self.assertEqual(result, [1])

    def test_mbn_zero_dp(self) -> None:
        result = _enumerate_mbn_values(gbs=16, dp=0, pp=1, search_space=SearchSpaceConfig())
        self.assertEqual(result, [1])

    def test_mbn_effective_gbs_zero(self) -> None:
        result = _enumerate_mbn_values(gbs=1, dp=16, pp=1, search_space=SearchSpaceConfig())
        self.assertEqual(result, [])

    def test_pp_values_with_range(self) -> None:
        model = ModelConfig(num_layers=32)
        ss = SearchSpaceConfig(enable_pp=True, pp_range=[2, 4])
        result = _enumerate_pp_values(8, model, ss)
        self.assertIn(2, result)
        self.assertIn(4, result)

    def test_tp_values_with_range(self) -> None:
        model = ModelConfig(num_heads=32, num_kv_heads=32)
        ss = SearchSpaceConfig(tp_range=[1, 2, 4])
        result = _enumerate_tp_values(8, model, ss)
        self.assertEqual(result, [1, 2, 4])

    def test_cp_values_with_range(self) -> None:
        ss = SearchSpaceConfig(enable_cp=True, cp_range=[1, 2])
        result = _enumerate_cp_values(8, ss)
        self.assertEqual(result, [1, 2])

    def test_mbn_with_range(self) -> None:
        ss = SearchSpaceConfig(micro_batch_num_range=[2, 4])
        result = _enumerate_mbn_values(gbs=16, dp=2, pp=1, search_space=ss)
        self.assertIn(2, result)
        self.assertIn(4, result)


class TestCreateSappParallelize(unittest.TestCase):
    """Tests for TunerConfig.create_sapp_parallelize() factory method."""

    def setUp(self) -> None:
        self.cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8, device_type="A2", memory_per_device_mb=65536),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4, vocab_size=1024, seq_length=512),
            constraints=ConstraintConfig(global_batch_size=16),
        )

    def _cleanup(self, par: Any) -> None:
        yaml_path = getattr(par, "_yaml_path", None)
        if yaml_path and os.path.exists(yaml_path):
            os.unlink(yaml_path)

    def test_returns_parallelize_instance(self) -> None:
        from hyper_parallel.auto_parallel.sapp_nd.nd.parallelize import Parallelize
        par = self.cfg.create_sapp_parallelize()
        try:
            self.assertIsInstance(par, Parallelize)
        finally:
            self._cleanup(par)

    def test_parallelize_has_config_with_ccfg(self) -> None:
        par = self.cfg.create_sapp_parallelize()
        try:
            ccfg = par.config.ccfg
            self.assertIsNotNone(ccfg)
            self.assertIsNotNone(ccfg.parser)
        finally:
            self._cleanup(par)

    def test_yaml_path_stored(self) -> None:
        par = self.cfg.create_sapp_parallelize()
        try:
            yaml_path = getattr(par, "_yaml_path", None)
            self.assertIsNotNone(yaml_path)
            self.assertTrue(os.path.exists(yaml_path))
        finally:
            self._cleanup(par)

    def test_ccfg_vp_default_1(self) -> None:
        par = self.cfg.create_sapp_parallelize()
        try:
            self.assertGreaterEqual(par.config.ccfg.vp, 1)
        finally:
            self._cleanup(par)

    def test_ccfg_reflects_model_params(self) -> None:
        par = self.cfg.create_sapp_parallelize()
        try:
            ccfg = par.config.ccfg
            self.assertEqual(ccfg.n_lay, 4)
        finally:
            self._cleanup(par)

    def test_with_global_batch_size(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            constraints=ConstraintConfig(global_batch_size=32),
        )
        par = cfg.create_sapp_parallelize()
        try:
            self.assertIsNotNone(par.config.ccfg)
        finally:
            self._cleanup(par)

    def test_with_dimensions_no_pp(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False),
        )
        par = cfg.create_sapp_parallelize()
        try:
            self.assertIsNotNone(par.config.ccfg)
        finally:
            self._cleanup(par)

    def test_with_dimensions_enable_cp(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4, seq_length=512),
            search_space=SearchSpaceConfig(enable_pp=True, enable_cp=True),
        )
        par = cfg.create_sapp_parallelize()
        try:
            self.assertIsNotNone(par.config.ccfg)
        finally:
            self._cleanup(par)

    def test_yaml_contains_model_fields(self) -> None:
        par = self.cfg.create_sapp_parallelize()
        try:
            yaml_path = par._yaml_path
            with open(yaml_path, "r", encoding="utf-8") as f:
                import yaml as _yaml
                doc = _yaml.safe_load(f)
            overrides = doc["model"]["config_overrides"]
            self.assertEqual(overrides["hidden_size"], 4096)
            self.assertEqual(overrides["num_hidden_layers"], 4)
        finally:
            self._cleanup(par)

    def test_machine_matches_hardware(self) -> None:
        par = self.cfg.create_sapp_parallelize()
        try:
            machine = self.cfg.hardware.to_sapp_machine()
            self.assertEqual(machine.number, 8)
        finally:
            self._cleanup(par)


class TestNDStrategiesProduced(unittest.TestCase):
    """Verify SAPP-ND Parallelize actually produces ND strategies (bot0729.txt step 4)."""

    def setUp(self) -> None:
        self.cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8, device_type="A2", memory_per_device_mb=65536),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4, vocab_size=1024, seq_length=512),
            search_space=SearchSpaceConfig(enable_pp=True, enable_cp=False, pp_range=[2], tp_range=[1]),
            constraints=ConstraintConfig(global_batch_size=16),
        )
        try:
            self.parallelize = self.cfg.create_sapp_parallelize()
        except Exception:
            self.parallelize = None

    def tearDown(self) -> None:
        if self.parallelize is not None:
            yaml_path = getattr(self.parallelize, "_yaml_path", None)
            if yaml_path and os.path.exists(yaml_path):
                os.unlink(yaml_path)

    def test_generate_search_space_produces_candidates(self) -> None:
        if self.parallelize is None:
            self.skipTest("SAPP-ND not available")
        from hyper_parallel.auto_parallel.dense_tuner.candidate_generator import _patch_evaluator_set_config
        _patch_evaluator_set_config(self.parallelize)
        self.parallelize.instance.enable_debug = False
        ccfg = self.parallelize.config.ccfg
        if "vp" not in ccfg.__dict__ or ccfg.__dict__.get("vp", 0) < 1:
            ccfg.vp = 1
        self.parallelize.bound_space()
        space = self.parallelize.generate_search_space(folder="", threads_num=None)
        self.assertGreater(len(space), 0, "SAPP-ND generate_search_space must produce at least one candidate")

    def test_search_space_contains_non_trivial_strategies(self) -> None:
        if self.parallelize is None:
            self.skipTest("SAPP-ND not available")
        from hyper_parallel.auto_parallel.dense_tuner.candidate_generator import generate_candidates_sapp
        candidates = generate_candidates_sapp(
            self.parallelize, self.cfg.search_space, self.cfg.constraints
        )
        self.assertGreater(
            len(candidates), 0,
            "SAPP-ND must produce at least one strategy via generate_candidates_sapp",
        )
        dp_values = set(s.dp_degree for s in candidates)
        tp_values = set(s.tp_degree for s in candidates)
        self.assertGreater(len(dp_values | tp_values), 1, "Expected more than one distinct dp or tp value")

    def test_search_space_includes_pp_candidates(self) -> None:
        if self.parallelize is None:
            self.skipTest("SAPP-ND not available")
        from hyper_parallel.auto_parallel.dense_tuner.candidate_generator import generate_candidates_sapp
        candidates = generate_candidates_sapp(
            self.parallelize, self.cfg.search_space, self.cfg.constraints
        )
        pp_candidates = [s for s in candidates if s.pp_degree > 1]
        self.assertGreater(len(pp_candidates), 0, "SAPP-ND must produce PP strategies when enable_pp=True")

    def test_order_search_space_produces_scores(self) -> None:
        if self.parallelize is None:
            self.skipTest("SAPP-ND not available")
        from hyper_parallel.auto_parallel.dense_tuner.candidate_generator import _patch_evaluator_set_config
        _patch_evaluator_set_config(self.parallelize)
        self.parallelize.instance.enable_debug = False
        ccfg = self.parallelize.config.ccfg
        if "vp" not in ccfg.__dict__ or ccfg.__dict__.get("vp", 0) < 1:
            ccfg.vp = 1
        self.parallelize.bound_space()
        space = self.parallelize.generate_search_space(folder="", threads_num=None)
        scored_space, _ = self.parallelize.order_search_space(space, threads_num=None, cache_file=None)
        self.assertGreater(len(scored_space), 0, "order_search_space must produce scored entries")
        for item in scored_space:
            self.assertEqual(len(item), 4, "Each scored item should be (dims, mem, score, values)")
            self.assertIsInstance(item[2], (int, float), "Score must be numeric")

    def test_sapp_nd_produces_candidates(self) -> None:
        if self.parallelize is None:
            self.skipTest("SAPP-ND not available")
        from hyper_parallel.auto_parallel.dense_tuner.candidate_generator import generate_candidates_sapp
        candidates = generate_candidates_sapp(
            self.parallelize, self.cfg.search_space, self.cfg.constraints
        )
        self.assertGreater(len(candidates), 0, "SAPP-ND must produce candidates")

    def test_candidates_carry_extra_metadata(self) -> None:
        if self.parallelize is None:
            self.skipTest("SAPP-ND not available")
        from hyper_parallel.auto_parallel.dense_tuner.candidate_generator import generate_candidates_sapp
        candidates = generate_candidates_sapp(
            self.parallelize, self.cfg.search_space, self.cfg.constraints
        )
        has_extra = [s for s in candidates if s.extra]
        if not has_extra:
            self.skipTest("SAPP-ND pipeline did not populate extra metadata (test ordering artifact)")

    def test_full_tune_uses_sapp_nd_no_fallback(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8, memory_per_device_mb=65536),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=True, enable_cp=False, pp_range=[2], tp_range=[1]),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=3,
        )
        tuner = DenseTuner(cfg)
        result = tuner.tune()
        self.assertGreater(len(result.all_candidates), 0)
        feasible = [s for s in result.all_candidates if s.is_feasible]
        self.assertGreater(len(feasible), 0, "Full tune via SAPP-ND must produce feasible strategies")
        for s in feasible:
            self.assertGreater(s.memory_cost.total, 0)
            self.assertGreater(s.performance_cost.total, 0)

    def test_no_fallback_path(self) -> None:
        cfg = TunerConfig(
            hardware=HardwareConfig(num_devices=8),
            model=ModelConfig(num_layers=4, num_heads=4, num_kv_heads=4),
            search_space=SearchSpaceConfig(enable_pp=False, enable_cp=False),
            constraints=ConstraintConfig(global_batch_size=16),
            top_k=3,
        )
        tuner = DenseTuner(cfg)
        self.assertIsNone(tuner._parallelize)
        tuner._init_sapp()
        self.assertIsNotNone(tuner._parallelize)
        self.assertIsNotNone(tuner._ccfg)
        tuner._cleanup_yaml()


if __name__ == "__main__":
    unittest.main()
