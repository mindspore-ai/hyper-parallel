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
"""Unit tests for SAPP-ND `run_nd` end-to-end pipeline.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_run_nd.py
"""
import copy
import json
import os
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd import debug as Debug
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd import balancing_adapter as BA
from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd import global_config as GC
from hyper_parallel.auto_parallel.sapp_nd.nd.common import arch_hooks as ArchHooks
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config, YamlObject
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import (
    _CostModelParser,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_hyperparallel import (
    CostModelParserHyperparallel,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_mindspeed import (
    CostModelParserMindspeed,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.generate_partitions import PartitionGenerator
from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation import comm_time as CommTime
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation import estimate as PerfEstimate
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import (
    CustomConfig,
    P2PCommType,
    PerformanceType,
    RatioType,
    RecType,
    NetworkLevel,
)

WORK_PATH = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(WORK_PATH, "deepseek.yaml")


def _make_partition_generator(**kwargs: Any) -> PartitionGenerator:
    """Build a partition generator without invoking cost-model preprocessing."""
    generator = object.__new__(PartitionGenerator)
    defaults = {
        "p": 2,
        "vp": 1,
        "n_lay": 4,
        "n_mtp": 1,
        "is_mtp_in_offset": False,
        "emb_out_in_offset": False,
        "config_format": "yaml",
        "offset": None,
        "full_rec": None,
        "sel_rec": None,
        "pp_sched": "1f1b",
        "multimodal": False,
        "mm_order": [],
        "mm_ccfgs": {},
    }
    defaults.update(kwargs)
    for key, value in defaults.items():
        setattr(generator, key, value)
    return generator


class _FakeDumpConfig:
    """Config double that records yaml dump requests."""

    def __init__(self) -> None:
        """Initialize recorded dump calls."""
        self.dumps = []

    def dump(self, file_name: str, folder: str) -> None:
        """Record dump arguments."""
        self.dumps.append((file_name, folder))

    def last_dump(self) -> tuple:
        """Return the latest dump request."""
        return self.dumps[-1]


class _FakeCostModelConfig:
    """Cost-model config double for GlobalConfig tests."""

    def __init__(self) -> None:
        """Initialize a tiny valid cost-model configuration."""
        self.model_name = "unit-transformer"
        self.d = 2
        self.t = 2
        self.cp = 1
        self.p = 2
        self.vp = 1
        self.m = 2
        self.b = 4
        self.ep = 1
        self.os_max_shard = 2
        self.sp = False
        self.n_lay = 4
        self.n_mtp = 1
        self.emb_out_in_offset = False
        self.is_mtp_in_offset = False
        self.n_exp = 4
        self.optimizer = "adam"
        self.dc_kv = 0
        self.dhr = 0
        self.h = 8
        self.has_op = False
        self.offset = [0, 0]
        self.full_rec = [0, 0]
        self.multimodal = True
        self.hooks_dict = {}
        self.config = _FakeDumpConfig()
        self.strategy_calls = []

    def set_strategy(self, **kwargs: Any) -> None:
        """Record and apply strategy updates."""
        self.strategy_calls.append(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_strategy(self) -> dict:
        """Return a compact strategy snapshot."""
        return {"dp": self.d, "tp": self.t, "pp": self.p, "ep": self.ep}


class _FakeBalancing:
    """Balancing double with deterministic adapted configs."""

    def __init__(self) -> None:
        """Initialize recorded balancing calls."""
        self.calls = []

    def treat_recompute(self, new_pp: int, new_vpp: int) -> list:
        """Return a deterministic recompute layout."""
        self.calls.append(("recompute", new_pp, new_vpp))
        return [0 for _ in range(new_pp)]

    def treat_offset(self, new_pp: int, new_vpp: int) -> list:
        """Return a deterministic offset layout."""
        self.calls.append(("offset", new_pp, new_vpp))
        if new_vpp > 1:
            return [[0 for _ in range(new_pp)] for _ in range(new_vpp)]
        return [0 for _ in range(new_pp)]

    def offset_checker(self, new_pp: int, new_vpp: int, offset: list) -> bool:
        """Accept all fake offsets and record the validation call."""
        self.calls.append(("check", new_pp, new_vpp, copy.deepcopy(offset)))
        return True


class _FakeParallelize:
    """Parallelize double used by run_nd CLI tests."""

    instances = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Record constructor arguments for CLI assertions."""
        self.args = args
        self.kwargs = kwargs
        self.run_args = None
        self.run_kwargs = None
        self.__class__.instances.append(self)

    def run_generation_to_ordering(self, *args: Any, **kwargs: Any) -> list:
        """Return a small sorted search result without invoking ND search."""
        self.run_args = args
        self.run_kwargs = kwargs
        return [("parallel-config", 128.0, 1.0, {})]

    def last_run_kwargs(self) -> dict:
        """Return keyword arguments from the latest fake run call."""
        return self.run_kwargs


class _ParserCostModelConfig:
    """Minimal cost-model object for framework parser unit tests."""

    def __init__(self, input_config: Any = None) -> None:
        """Initialize parser target state."""
        self.config = Config(input_config or {})
        self.hooks_dict = {}
        self.source_code = None

    def __getattr__(self, attr: str) -> int:
        """Match CostModelConfig's permissive missing-attribute behavior."""
        _ = attr
        return 0

    @staticmethod
    def fp_bytes(precision: str) -> int:
        """Return bytes encoded in a precision string."""
        if "16" in precision:
            return 2
        if "32" in precision:
            return 4
        return 0


def _make_perf_cfg(**kwargs: Any) -> SimpleNamespace:
    """Build a tiny performance-estimation config."""
    defaults = {
        "n_kv": 2,
        "a": 2,
        "b": 1,
        "s": 8,
        "h": 16,
        "hff": 32,
        "t": 2,
        "sp": 1,
        "cp": 1,
        "p": 2,
        "m": 4,
        "vp": 1,
        "d": 2,
        "ep": 2,
        "os_max_shard": 2,
        "bytes_p": 2,
        "bytes_compute": 2,
        "bytes_softmax": 4,
        "bytes_norm": 4,
        "dc_kv": 0,
        "dc_q": 0,
        "dh": 8,
        "dhr": 0,
        "v": 64,
        "hff_exp": 64,
        "n_chosen_exp": 2,
        "n_shared_exp": 1,
        "n_exp": 1,
        "cap_fact": 1,
        "has_op": True,
        "has_grad_shard": True,
        "comm_d_non_exp": 3,
        "comm_t": 1.0,
        "comm_ep": 1.0,
        "comm_dp_overlap": 0.9,
        "comm_tp_overlap": 0.5,
        "layer_custom_config": [(2, None)],
        "n_lay": 2,
        "n_mtp": 1,
        "gbs": 8,
        "pp_sched": "1f1b",
        "shard_embed": 2,
        "shard_output_activ": 1,
        "shard_recompute_input": 1,
        "n_attMM": 1,
        "n_ffMM": 1,
        "n_attBMM": 1,
        "n_ffBMM": 1,
        "n_softmax": 1,
        "n_headCast": 1,
        "n_gather": 1,
        "n_ffAct": 1,
        "n_normOp": 1,
        "n_dropout": 1,
        "rec_op": SimpleNamespace(
            attMM=1,
            ffMM=1,
            attBMM=1,
            ffBMM=1,
            softmax=1,
            headCast=1,
            gather=1,
            ffAct=1,
            normOp=1,
            dropout=1,
        ),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_arch_cfg(**kwargs: Any) -> SimpleNamespace:
    """Build a tiny config for arch hook functions."""
    defaults = {
        "model_name": "deepseek-unit",
        "has_op": False,
        "p": 2,
        "h": 16,
        "t": 2,
        "dh": 4,
        "hff": 32,
        "hff_exp": 64,
        "n_lay": 4,
        "n_mtp": 1,
        "n_chosen_exp": 2,
        "n_exp": 4,
        "n_shared_exp": 1,
        "ep": 2,
        "k_1st_dense": 2,
        "config_format": "json",
        "ffn_hidden_size": 32,
        "shard_p_os_exp_partial": 2,
        "shard_p_os_non_exp": 2,
        "shard_embed": 1,
        "overwrite_eval_functions": {},
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestSappNDRunND(unittest.TestCase):
    """A test class for the SAPP-ND ``run_nd`` end-to-end pipeline."""

    def test_run_nd_deepseek_smoke(self):
        """
        Feature: TestSappNDRunND.
        Description: Run the ND search-space generation and performance-ordering
                     pipeline on the shipped DeepSeek yaml, mirroring what
                     ``run_nd.py`` does at the CLI entry point.
        Expectation: ``run_generation_to_ordering`` returns a non-empty scored
                     space whose entries have the documented
                     ``(parallel_config, mem_mb, perf_score, debug_parts)`` shape,
                     and the entries are monotonically sorted by score.
        """
        self.assertTrue(
            os.path.exists(config_path),
            f"shipped DeepSeek yaml missing: path={config_path!r}",
        )

        top_k = 3

        # Redirect matplotlib config dir to a temp path to avoid polluting $HOME
        # and silence the ND logger so that no debug plot/CSV is emitted into
        # the source tree (see `enable_debug` gating in parallelize.py).
        memory_call_count = {"count": 0}
        perf_call_count = {"count": 0}
        original_memory_estim = Par.ParallelizeLayer.memory_estim
        original_estimate_performance = Par.estimate_performance

        def mostly_fast_memory_estim(runner: Par.ParallelizeLayer, debugger: Any = None) -> float:
            """Run one real memory estimate, then use a deterministic fitting value."""
            memory_call_count["count"] += 1
            if memory_call_count["count"] == 1:
                return original_memory_estim(runner, debugger)
            return 1.0

        def mostly_fast_estimate_performance(*args: Any, **kwargs: Any) -> float:
            """Run one real performance estimate, then avoid repeating expensive formulas."""
            perf_call_count["count"] += 1
            if perf_call_count["count"] == 1:
                return original_estimate_performance(*args, **kwargs)
            return 2.0

        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}), \
                patch.object(Par.ParallelizeLayer, "memory_estim", mostly_fast_memory_estim), \
                patch.object(Par, "estimate_performance", mostly_fast_estimate_performance):
            set_verbose_level(1)

            machine = Hard.Machine(16, "A2")
            dims = Dim.get_dims(["DP", "MP", "PP", "EP", "MB"])

            runner = Par.Parallelize(
                "mindformers",
                config_path,
                machine,
                global_batch_size=None,
                dimensions=dims,
                swap_os=False,
                mppb=False,
                model=None,
                max_mem=None,
                mem_for_ppb=Memory.from_string("0GB"),
            )

            scored_space = runner.run_generation_to_ordering(
                None,
                threads_num=None,
                top_num=top_k,
                cache_file=None,
            )

        self.assertGreater(memory_call_count["count"], 1)
        self.assertGreater(perf_call_count["count"], 1)

        self.assertIsInstance(
            scored_space, list,
            f"scored_space must be list, got type={type(scored_space).__name__}",
        )
        self.assertGreater(
            len(scored_space), 0,
            f"scored_space must be non-empty for DeepSeek yaml, got len={len(scored_space)}",
        )

        top = scored_space[:top_k]
        self.assertGreaterEqual(
            len(top), 1,
            f"expected at least 1 top configuration, got len={len(top)}",
        )

        for idx, entry in enumerate(top):
            self.assertGreaterEqual(
                len(entry), 3,
                (f"scored entry must be a tuple of at least "
                 f"(config, mem, score, ...): idx={idx}, entry={entry!r}"),
            )
            mem_mb = entry[1]
            perf_score = entry[2]
            self.assertGreater(
                mem_mb, 0,
                f"entry[{idx}].mem_mb must be > 0, got mem_mb={mem_mb}",
            )
            self.assertGreater(
                perf_score, 0,
                f"entry[{idx}].perf_score must be > 0, got perf_score={perf_score}",
            )

        scores = [entry[2] for entry in top]
        self.assertTrue(
            scores == sorted(scores) or scores == sorted(scores, reverse=True),
            (f"top-{top_k} scored configurations are not monotonically "
             f"sorted by score: scores={scores}"),
        )

    def test_dimensions_and_hardware_helpers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Exercise pure dimension and hardware helpers.
        Expectation: Helpers return stable values without distributed setup.
        """
        dims = Dim.Dimensions(
            [
                (Dim.DP, 2),
                (Dim.EP, 1),
                (Dim.TP, 2),
                (Dim.CP, 1),
                (Dim.PP, 2),
                (Dim.VPP, 1),
                (Dim.MBN, 2),
                (Dim.MBS, 4),
                (Dim.SP, False),
                (Dim.OP, 2),
            ],
            all_dims=Dim.ALL_DIMS.copy(),
        )
        self.assertEqual(dims.global_batch_size(), 16)
        self.assertEqual(dims.values()[0], "2")
        self.assertIn("_", dims.unique_name())
        self.assertTrue(dims.has_dim(Dim.PP))
        self.assertTrue(dims.is_valid())
        dims.steal(2, Dim.DP, Dim.EP)
        self.assertEqual(dims.val(Dim.DP), 1)
        self.assertEqual(dims.val(Dim.EP), 2)
        dims.set(Dim.TP, 3)
        self.assertFalse(dims.is_valid())
        dims.set(Dim.TP, 2)

        Dim.DP.set_bound(2)
        self.assertTrue(Dim.DP.is_valid(2))
        self.assertFalse(Dim.DP.is_valid(3))
        self.assertEqual(Dim.DP.get_bound(), 2)
        Dim.DP.reset_bound()
        self.assertEqual(Dim.DP.from_config(SimpleNamespace(d=4)), 4)
        with self.assertRaises(TypeError):
            Dim.Dimensions({"dp": 1})
        with self.assertRaises(ValueError):
            Dim.get_dim("missing")
        self.assertEqual(Dim.get_dims(["DP", "MP"]), [Dim.DP, Dim.TP])

        self.assertEqual(Hard.Device_A2.intra_node_num(), 8)
        self.assertEqual(Hard.Device_A2.levels_used(9), 1)
        assignment = Hard.Device_A2.level_assign(dp=2, tp=2, cp=1, pp=2)
        self.assertEqual(assignment[Dim.TP][0], 2)
        machine = Hard.Machine(None, "A2")
        machine.update_num_if_none(8)
        self.assertEqual(machine.number, 8)
        self.assertGreaterEqual(machine.pipeline_bound(), 1)
        self.assertEqual(Hard.Machine(4, 2).device, Hard.Device_A2)
        self.assertEqual(Hard.Machine(4, 3).device, Hard.Device_A3)
        with self.assertRaises(ValueError):
            Hard.Machine(4, 7)
        with self.assertRaises(ValueError):
            Hard.Machine(4, "unknown")
        self.assertEqual(Hard.prime_factors(12), [2, 2, 3])
        self.assertEqual(Hard.from_prime_factors([2, 3, 5]), 30)
        self.assertEqual(Hard.all_divisors(12, min_bound=3, max_bound=6), [3, 4, 6])
        self.assertEqual(Hard.split_node(16, Hard.Device_A2), [[2, 2, 2], [2]])
        self.assertEqual(Hard.unique_factors([2, 2, 3, 3]), [2, 3])
        self.assertEqual(Hard.highest_power_of_2_divisor(24), 8)

    def test_config_and_debug_helpers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Exercise config loading and debug csv helpers.
        Expectation: File helpers read/write only temporary files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = os.path.join(tmp_dir, "config.yaml")
            json_path = os.path.join(tmp_dir, "config.json")
            toml_path = os.path.join(tmp_dir, "config.toml")
            with open(yaml_path, "w", encoding="utf-8") as yaml_file:
                yaml_file.write("root:\n  value: 3\nname: yaml\n")
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump({"root": {"value": 4}, "name": "json"}, json_file)
            with open(toml_path, "w", encoding="utf-8") as toml_file:
                toml_file.write("name = 'toml'\n[root]\nvalue = 5\n")

            yaml_cfg = Config(yaml_path)
            json_cfg = Config(json_path)
            toml_cfg = Config(toml_path)
            self.assertEqual(yaml_cfg.root.value, 3)
            self.assertEqual(json_cfg.root.value, 4)
            self.assertEqual(toml_cfg.root.value, 5)
            self.assertEqual(yaml_cfg.missing_attribute, 0)
            copied_cfg = copy.copy(yaml_cfg)
            deep_cfg = copy.deepcopy(yaml_cfg)
            self.assertEqual(copied_cfg.root.value, 3)
            self.assertEqual(deep_cfg.root.value, 3)
            self.assertEqual(Config({"nested": {"value": 6}}).nested.value, 6)
            self.assertEqual(Config(yaml_cfg).root.value, 3)
            with self.assertRaises(TypeError):
                Config(3)

            yaml_obj = YamlObject({"nested": {"value": 7}})
            state = yaml_obj.__getstate__()
            restored = YamlObject({})
            restored.__setstate__(state)
            self.assertEqual(restored.nested.value, 7)
            yaml_obj.dump("unit", tmp_dir)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "config_unit.yaml")))

            debug_dims = Dim.Dimensions([(Dim.DP, 2), (Dim.MBS, 4)], all_dims=[Dim.DP, Dim.MBS])
            debug = Debug.Debug(debug_dims, [Debug.PerfParts.TOTAL], output_file="unit_debug.csv")
            debug.output_file = os.path.join(tmp_dir, "unit_debug.csv")
            debug.info[Debug.PerfParts.TOTAL] = 9
            self.assertTrue(debug.is_enabled())
            self.assertIn("DP", debug.column_titles())
            self.assertIn("9", debug.values())
            debug.write()
            self.assertTrue(os.path.exists(debug.output_file))

        self.assertEqual(Debug.PerfParts.FW_COMPUTE.short_name(), "FW")
        self.assertEqual(str(Debug.RealParts.COMP), "comp")
        self.assertEqual(str(Debug.MemParts.TOTAL), "TOTAL")
        self.assertEqual(Debug.dim_color("DP"), "orange")
        self.assertEqual(Debug.dim_color("unknown", default="fallback"), "fallback")
        self.assertEqual(Debug.pastel("white"), (1.0, 1.0, 1.0))
        self.assertEqual(Debug.pastel("black"), (0.5, 0.5, 0.5))
        self.assertEqual(len(Debug.gen_colors(["FW_COMPUTE", "DP_COMM"])), 2)
        self.assertEqual(Debug.PerfParts.BW_COMPUTE.short_name(), "BW")
        self.assertEqual(Debug.PerfParts.RECOMPUTE.short_name(), "Rec")
        self.assertEqual(Debug.PerfParts.MP_COMM.short_name(), "MP")
        self.assertEqual(Debug.PerfParts.EP_COMM.short_name(), "EP")
        self.assertEqual(Debug.PerfParts.CP_COMM.short_name(), "CP")
        self.assertEqual(Debug.PerfParts.PP_COMM.short_name(), "P2P")
        self.assertEqual(Debug.PerfParts.BUBBLE.short_name(), "BBL")
        self.assertEqual(Debug.PerfParts.MEMORY.short_name(), "MEM")
        self.assertEqual(Debug.PerfParts.TOTAL.short_name(), "Perf")
        self.assertEqual(Debug.dim_color(Dim.MBS, default="fallback"), "fallback")
        self.assertEqual(len(Debug.pastel("#123456", lbl=0.5, sat=0.4)), 3)
        self.assertEqual(len(Debug.near_white("red", 0.25)), 3)

    def test_balancing_adapter_helpers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Exercise PP/VPP balancing adapter branches with tiny layouts.
        Expectation: Offset and recompute adapters produce valid deterministic layouts.
        """
        self.assertEqual(str(BA.Pipeline(2, 3)), "PP: 2, VPP: 3")
        self.assertEqual(BA.Pipeline(2, 3).chunk_stage(), 6)
        self.assertEqual(BA.infer_pp_and_vpp(0).chunk_stage(), 1)
        self.assertEqual(BA.infer_pp_and_vpp([0, 1]).pp, 2)
        self.assertEqual(BA.infer_pp_and_vpp([[0, 0], [1, 0]]).vpp, 2)
        self.assertTrue(BA.is_zero_d(0))
        self.assertTrue(BA.is_one_d([0, 1]))
        self.assertTrue(BA.is_two_d([[0, 1]]))
        self.assertEqual(BA.copy_offset([0, 1]), [0, 1])
        self.assertEqual(BA.make_one_d([[0, 1], [2, 3]]), [0, 1, 2, 3])
        self.assertEqual(BA.make_two_d([0, 1]), [[0, 1]])
        with self.assertRaises(TypeError):
            BA.infer_pp_and_vpp("bad")
        with self.assertRaises(TypeError):
            BA.make_one_d("bad")
        with self.assertRaises(TypeError):
            BA.make_two_d("bad")

        adapter = BA.BalancingAdapter(5, [1, 0], [1, 0], True)
        self.assertEqual(len(adapter.treat_pp_list(BA.Pipeline(4, 1), [1, 0])), 4)
        self.assertEqual(len(adapter.treat_pp_list(BA.Pipeline(1, 1), [1, 0])), 1)
        self.assertEqual(len(adapter.treat_vpp_list(BA.Pipeline(2, 2), [[1, 0]])), 2)
        self.assertEqual(len(adapter.treat_recompute(4, 1)), 4)
        self.assertEqual(len(adapter.treat_offset(4, 1)), 4)
        self.assertTrue(adapter.offset_checker(4, 1, adapter.treat_offset(4, 1)))
        default_adapter = BA.BalancingAdapter(5, 0, True, False)
        self.assertTrue(default_adapter.treat_recompute(2, 2))
        self.assertTrue(default_adapter.offset_checker(2, 2, default_adapter.treat_offset(2, 2)))

    def test_partition_generator_helpers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover partition generation branches.
        Expectation: Layer partitions are deterministic for tiny fake model configs.
        """
        base_generator = _make_partition_generator()
        partitions = base_generator.generate_partitions_vpp()
        self.assertEqual(partitions[0][0][0], LayerType.EMBEDDING_LAYER)
        self.assertEqual(partitions[-1][-1][-1], LayerType.OUTPUT_LAYER)
        self.assertIn(LayerType.NOT_REC_LAYER, partitions[0][0])

        full_rec_generator = _make_partition_generator(
            vp=2,
            n_lay=8,
            n_mtp=0,
            is_mtp_in_offset=True,
            emb_out_in_offset=True,
            offset=[0, 0],
            full_rec=[2, 1],
        )
        full_rec_partitions = full_rec_generator.generate_partitions_vpp_unimodal()
        flat_full_rec = [layer for stage in full_rec_partitions for chunk in stage for layer in chunk]
        self.assertIn(LayerType.FULL_REC_LAYER, flat_full_rec)
        self.assertEqual(flat_full_rec[0], LayerType.EMBEDDING_LAYER)
        self.assertEqual(flat_full_rec[-1], LayerType.OUTPUT_LAYER)

        sel_rec_generator = _make_partition_generator(
            n_mtp=0,
            sel_rec=["attention"],
        )
        sel_rec_partitions = sel_rec_generator.generate_partitions_vpp_unimodal()
        flat_sel_rec = [layer for stage in sel_rec_partitions for chunk in stage for layer in chunk]
        self.assertIn(LayerType.SEL_REC_LAYER, flat_sel_rec)

        zbv_generator = _make_partition_generator(
            pp_sched="zero_bubble_v",
            vp=2,
            n_mtp=0,
            emb_out_in_offset=True,
            is_mtp_in_offset=True,
        )
        zbv_partitions = zbv_generator.generate_partitions_vpp_unimodal()
        self.assertEqual(zbv_partitions[0][0][0], LayerType.EMBEDDING_LAYER)
        self.assertEqual(zbv_partitions[-1][-1][-1], LayerType.OUTPUT_LAYER)
        self.assertEqual(
            zbv_generator.first_and_last_non_empty_stage([[[], []], [[], []]]),
            ((0, 0), (1, 1)),
        )

        multimodal_generator = _make_partition_generator(
            multimodal=True,
            mm_order=["vision", "text"],
            mm_ccfgs={
                "vision": SimpleNamespace(
                    generate_partitions_vpp_unimodal=lambda: [
                        [[LayerType.EMBEDDING_LAYER]],
                        [[LayerType.NOT_REC_LAYER]],
                    ]
                ),
                "text": SimpleNamespace(
                    generate_partitions_vpp_unimodal=lambda: [
                        [[LayerType.NOT_REC_LAYER]],
                        [[LayerType.OUTPUT_LAYER]],
                    ]
                ),
            },
        )
        multimodal_parts = multimodal_generator.generate_partitions_vpp()
        combined_parts = multimodal_generator.combine_partition_multimodal(multimodal_parts)
        self.assertEqual(combined_parts[0][0], [LayerType.EMBEDDING_LAYER, LayerType.NOT_REC_LAYER])
        self.assertEqual(combined_parts[1][0], [LayerType.NOT_REC_LAYER, LayerType.OUTPUT_LAYER])

    def test_global_config_arch_hooks_and_cli(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover GlobalConfig helpers, arch hook wrappers, and run_nd CLI branches.
        Expectation: Branches run with small fakes and do not invoke the real ND solver.
        """
        fake_ccfg = _FakeCostModelConfig()
        global_config = object.__new__(GC.GlobalConfig)
        global_config.ccfg = fake_ccfg
        global_config.dimensions = Dim.ALL_DIMS.copy()
        global_config.balancing = _FakeBalancing()
        parallel_config = global_config.make_parallel_config(
            (2, 2, 2, 1),
            (4, 2),
            (1, 1, 2, False),
        )
        self.assertEqual(global_config.dim_val(Dim.DP, parallel_config), 2)
        self.assertEqual(global_config.global_batch_size(parallel_config), 16)
        self.assertEqual(global_config.layer_num_for_offset(), 4)
        self.assertEqual(global_config.total_layer_num(), 5)
        self.assertEqual(global_config.adapt_config(2, 1), ([0, 0], [0, 0]))
        self.assertTrue(global_config.moe_valid(parallel_config))
        self.assertTrue(global_config.set_parallel_config(parallel_config))
        self.assertIn("offset", fake_ccfg.strategy_calls[-1])
        self.assertEqual(global_config.range_space(Dim.PP, 3), range(1, 4))
        self.assertEqual(global_config.bool_space(Dim.SP), [False, True])
        self.assertEqual(global_config.max_op(dp=4, tp=2, ep=1), 4)

        Dim.TP.set_bound(2)
        try:
            self.assertEqual(global_config.space(Dim.TP, 8), [1, 2])
        finally:
            Dim.TP.reset_bound()

        fake_ccfg.optimizer = "muon"
        fake_ccfg.dc_kv = 2
        fake_ccfg.dhr = 2
        self.assertEqual(global_config.max_op(dp=4, tp=2, ep=1), 4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            global_config.write(tmp_dir, parallel_config)
            self.assertEqual(fake_ccfg.config.last_dump()[1], tmp_dir)

        wrapped_cfg = _FakeCostModelConfig()
        wrapped_cfg.multimodal = False
        wrapper = ArchHooks.CWrap(wrapped_cfg)
        self.assertEqual(wrapper.get_model_name(), "unit-transformer")
        wrapper.set_ccfg(lambda cfg: setattr(cfg, "custom_value", 11))
        self.assertEqual(getattr(wrapped_cfg, "custom_value"), 11)
        wrapper.set_strategy(d=3)
        self.assertEqual(wrapped_cfg.d, 3)
        self.assertEqual(wrapper.get_strategy()["dp"], 3)
        self.assertIsNone(wrapper.unknown_method())
        ArchHooks.check_and_apply_custom_hook(wrapper)
        self.assertEqual(getattr(wrapped_cfg, "n_attMM"), 4)

    def test_run_nd_cli_uses_fake_parallelize(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover run_nd CLI parsing with a fake Parallelize implementation.
        Expectation: CLI branches pass normalized arguments without running the real solver.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            _FakeParallelize.instances = []
            missing_cache = os.path.join(tmp_dir, "missing_cache.csv")
            argv = [
                "run_nd.py",
                "-y",
                config_path,
                "-d",
                "8",
                "-l",
                "DP",
                "MP",
                "-v",
                "0",
                "-t",
                "2",
                "-M",
                "1GB",
                "-mem",
                "512MB",
                "-c",
                missing_cache,
            ]
            with patch.object(sys, "argv", argv):
                result = runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertEqual(result["space"], [("parallel-config", 128.0, 1.0, {})])
            first_instance = _FakeParallelize.instances[-1]
            self.assertEqual(first_instance.args[0], "mindformers")
            self.assertEqual(first_instance.args[1], config_path)
            self.assertEqual(first_instance.args[2].number, 8)
            self.assertEqual(first_instance.kwargs["max_mem"].to_mb().size, 1024)
            self.assertEqual(first_instance.kwargs["mem_for_ppb"].to_mb().size, 512)
            self.assertIsNone(first_instance.last_run_kwargs()["cache_file"])
            self.assertEqual(first_instance.last_run_kwargs()["top_num"], 2)

            argv = [
                "run_nd.py",
                "-f",
                "torchtitan",
                "-y",
                "module.name:tiny_config",
                "-d",
                "4",
                "-v",
                "0",
            ]
            with patch.object(sys, "argv", argv):
                runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            torchtitan_instance = _FakeParallelize.instances[-1]
            self.assertEqual(torchtitan_instance.args[0], "torchtitan")
            self.assertEqual(torchtitan_instance.args[1]["module"], "module.name")
            self.assertEqual(torchtitan_instance.args[1]["config"], "tiny_config")

            argv = [
                "run_nd.py",
                "-f",
                "hyperparallel2",
                "-y",
                os.path.join(tmp_dir, "model.yaml"),
                "--train-yaml",
                os.path.join(tmp_dir, "train.yaml"),
                "--accelerate-yaml",
                os.path.join(tmp_dir, "accelerate.yaml"),
                "-d",
                "4",
                "-v",
                "0",
            ]
            with patch.object(sys, "argv", argv):
                runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            hyperparallel_instance = _FakeParallelize.instances[-1]
            self.assertEqual(hyperparallel_instance.args[0], "hyperparallel2")
            self.assertEqual(hyperparallel_instance.args[1]["machine"], 4)

    def test_debug_csv_and_correlation_helpers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover debug CSV readers and correlation helpers with tiny CSV files.
        Expectation: Helpers parse dimensions, split components, and compute stable metrics.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "profile.csv")
            with open(csv_path, "w", encoding="utf-8") as csv_file:
                csv_file.write("DP,MP,time,comp,dp_wait,mp_wait,ep_wait,cp_wait,pp_wait,op_wait,sp_wait\n")
                csv_file.write("1,1,10,5,1,1,1,1,1,0.5,0.25\n")
                csv_file.write("2,1,20,8,2,2,2,2,2,0.5,0.25\n")

            configs, row_num = Debug.get_real_data(csv_path)
            self.assertEqual(row_num, 2)
            self.assertEqual(configs[0][0].val(Dim.DP), 1)
            self.assertEqual(Debug.get_diff_dims(csv_path), [Dim.DP])

            classified = Debug.get_comm_classified_data(csv_path, plot_idle=True)
            self.assertEqual(classified[0][2]["IDLE"], -0.75)
            self.assertEqual(classified[0][2]["BUBBLE"], 1)

        estimations = [1.0] * (max(part.value for part in Debug.PerfParts) - 1)
        real_parts = {part: [] for part in Debug.RealParts}
        real_parts = Debug.estimation_in_real_parts(real_parts, estimations, 12.0)
        self.assertEqual(real_parts[Debug.RealParts.TOTAL], [12.0])
        real_parts = Debug.real_in_parts(real_parts, classified[0][2], 10.0)
        self.assertGreater(real_parts[Debug.RealParts.DP_WAIT][-1], 1)

        configs_estimated = [
            (configs[0][0], 1, 10.0, 9.0, estimations, classified[0][2]),
            (configs[1][0], 2, 20.0, 18.0, [value * 2 for value in estimations], classified[1][2]),
        ]
        correls, distances, topk, total = Debug.correlation_with_classified_comms(configs_estimated)
        self.assertEqual((topk, total), (2, 2))
        self.assertIn(Debug.RealParts.TOTAL, correls)
        self.assertIn(Debug.RealParts.TOTAL, distances)

        correl, top_k = Debug.correlation_topk(
            [(configs[0][0], 1, 10.0, 9.0, []), (configs[1][0], 2, 20.0, 18.0, [])],
            "profile.csv",
        )
        self.assertGreater(correl, 0)
        self.assertEqual(top_k, 2)
        self.assertIn("improved", Debug.color_diff(1))
        self.assertIn("worsened", Debug.color_diff(-1))
        self.assertIn("%", Debug.color_correl(0.95))
        self.assertTrue(Debug.is_constant([3, 3]))
        self.assertFalse(Debug.is_constant([3, 4]))

        metric_data = (correls, distances, topk, total)
        self.assertIn("total", Debug.print_part_x_file([metric_data], Debug.get_distance_i))
        self.assertIsNone(Debug.print_correlations_classified([metric_data]))

    def test_debug_plot_data_helpers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover Plot data parsing and plotting wrappers with patched rendering.
        Expectation: Plot helpers build data frames without writing real figures.
        """
        dims = Dim.Dimensions([(Dim.DP, 1), (Dim.MBS, 2)], all_dims=[Dim.DP, Dim.MBS])
        debug_parts = [Debug.PerfParts.FW_COMPUTE, Debug.PerfParts.DP_COMM]
        configs_estimated = [
            (dims, 128, 10.0, [6.0, 4.0]),
            (dims, 256, 20.0, [12.0, 8.0]),
        ]

        plot = Debug.Plot("unit", dims.keys(), debug_parts, top=1)
        plot.parse_data(configs_estimated)
        self.assertEqual(len(plot.data), 1)
        self.assertEqual(plot.row_title[-1], "MEM")

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Debug.Plot, "make_table", lambda self: None), \
                patch.object(Debug.Plot, "close", lambda self, output_path, filename: None), \
                patch.object(Debug, "set_twin_handles", lambda ax1, data_frame, dbg_cols: None):
            Debug.plot_nd(configs_estimated, tmp_dir, debug_parts, max_num=1)
            real_wait = {"comp": 5, "dp_wait": 1, "mp_wait": 1, "ep_wait": 1, "BUBBLE": 1, "IDLE": 1}
            real_configs = [(dims, 128, 15.0, 10.0, [6.0, 4.0], real_wait)]
            Debug.plot_vs_real(real_configs, "profile.csv", tmp_dir, debug_parts)
            Debug.plot_vs_real_comm_classified(
                real_configs,
                "profile.csv",
                tmp_dir,
                debug_parts,
                plot_idle=True,
            )

    def test_arch_hook_variants(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover predefined architecture hooks using tiny fake configs.
        Expectation: Hooks update model-specific attributes and layer custom hooks.
        """
        cfg = _make_arch_cfg(model_name="llama2")
        ArchHooks.custom_default_transformer(cfg)
        self.assertEqual(cfg.n_attMM, 4)
        ArchHooks.custom_llama2(cfg)
        self.assertEqual(cfg.bytes_grad, 2)
        ArchHooks.custom_mixtral(cfg)
        self.assertEqual(cfg.hff, cfg.hff_exp)
        ArchHooks.custom_pangualpha(cfg)
        self.assertEqual(cfg.n_normOp, 4)
        ArchHooks.custom_qwen(cfg)
        self.assertEqual(cfg.shard_recompute_input, cfg.t)

        t5_cfg = _make_arch_cfg(model_name="t5", n_lay=4, n_mtp=0)
        ArchHooks.custom_t5(t5_cfg)
        self.assertEqual(len(t5_cfg.layer_custom_config), 2)
        t5_wrap = ArchHooks.CWrap(t5_cfg)
        t5_cfg.layer_custom_config[0][1](t5_wrap)
        self.assertEqual(t5_cfg.n_attBMM, 1)
        t5_cfg.layer_custom_config[1][1](t5_wrap)
        self.assertEqual(t5_cfg.n_attMM, 8)

        deepseek_cfg = _make_arch_cfg(model_name="deepseek")
        ArchHooks.custom_deepseek3(deepseek_cfg)
        self.assertEqual(len(deepseek_cfg.layer_custom_config), 3)
        deepseek_wrap = ArchHooks.CWrap(deepseek_cfg)
        deepseek_cfg.layer_custom_config[0][1](deepseek_wrap)
        self.assertEqual(deepseek_cfg.n_exp, 1)
        deepseek_cfg.layer_custom_config[1][1](deepseek_wrap)
        self.assertEqual(deepseek_cfg.n_exp, 4)

        cm_cfg = _make_arch_cfg(model_name="cm")
        ArchHooks.custom_cm(cm_cfg)
        self.assertIn("num_params_norm", cm_cfg.overwrite_eval_functions)
        self.assertGreater(cm_cfg.overwrite_eval_functions["num_params_norm"](cm_cfg, None), 0)

    def test_performance_formula_helpers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover performance estimation formulas with tiny fake stages.
        Expectation: Formula helpers return positive deterministic values.
        """
        cfg = _make_perf_cfg()
        stages = [
            [[LayerType.EMBEDDING_LAYER, LayerType.NOT_REC_LAYER]],
            [[LayerType.FULL_REC_LAYER, LayerType.OUTPUT_LAYER]],
        ]
        ccfg = CustomConfig(rtype=RatioType.DYNAMIC, ptype=P2PCommType.MANUAL, retype=RecType.WITH)
        self.assertGreater(PerfEstimate.op_table(cfg)["n_attMM"], 0)
        self.assertGreater(sum(PerfEstimate.estimate_op_bulk_comp(cfg, ccfg, stages, with_recomp=True)), 0)
        self.assertGreater(PerfEstimate.estimate_comp_flop_time(cfg, 10**9), 0)
        self.assertGreater(PerfEstimate.throughput(2, 10**9), 0)
        self.assertGreater(PerfEstimate.get_dynamic_ratio(cfg), 0)

        debugger = Debug.Debug(
            Dim.Dimensions([(Dim.DP, 2), (Dim.MBS, 2)], all_dims=[Dim.DP, Dim.MBS]),
            Debug.PerfParts,
        )
        for part in (Debug.PerfParts.DP_COMM, Debug.PerfParts.MP_COMM,
                     Debug.PerfParts.EP_COMM, Debug.PerfParts.CP_COMM):
            debugger.info[part] = [1.0, 2.0]
        stage_perf = PerfEstimate.estimate_stage(
            cfg,
            ccfg,
            [10.0, 20.0],
            [1.0, 2.0],
            [12.0, 25.0],
            [2.0, 3.0],
            debugger=debugger,
        )
        self.assertEqual(len(stage_perf), 2)
        self.assertGreater(PerfEstimate.estimate_pipeline(cfg, stage_perf, debugger=debugger), 0)
        self.assertGreater(PerfEstimate.estimate_p2p_comm(cfg, max(stage_perf), debugger=debugger), 0)
        self.assertGreater(PerfEstimate.estimate_p2p(cfg, ccfg, stage_perf, debugger=debugger), 0)

        cfg_single = _make_perf_cfg(p=1, vp=1, m=2)
        self.assertEqual(PerfEstimate.estimate_pipeline(cfg_single, [5.0]), 10.0)
        cfg_vpp = _make_perf_cfg(p=2, vp=2, m=4)
        self.assertGreater(PerfEstimate.estimate_pipeline(cfg_vpp, [5.0, 7.0]), 0)

        coeffs = {
            "COMPUTE": 1.0,
            "DP_COMM": 1.0,
            "MP_COMM": 1.0,
            "EP_COMM": 1.0,
            "CP_COMM": 1.0,
            "PP_COMM": 1.0,
            "BUBBLE": 1.0,
        }
        debugger.info.update({part: 1.0 for part in Debug.PerfParts})
        self.assertGreater(PerfEstimate.apply_regression_coefficients(coeffs, debugger, 10.0), 0)

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(PerfEstimate, "estimate_comp", return_value=[10.0, 20.0]), \
                patch.object(PerfEstimate, "estimate_comm", return_value=[1.0, 2.0]):
            cfg_for_perf = object.__new__(PerfEstimate.CostModelConfig)
            cfg_for_perf.__dict__.update(vars(cfg))
            perf_debugger = Debug.Debug(
                Dim.Dimensions([(Dim.DP, 2), (Dim.MBS, 2)], all_dims=[Dim.DP, Dim.MBS]),
                Debug.PerfParts,
            )
            for part in (Debug.PerfParts.DP_COMM, Debug.PerfParts.MP_COMM,
                         Debug.PerfParts.EP_COMM, Debug.PerfParts.CP_COMM):
                perf_debugger.info[part] = [1.0, 2.0]
            cache_file = os.path.join(tmp_dir, "cache.json")
            with open(cache_file, "w", encoding="utf-8") as cache:
                json.dump(coeffs, cache)
            perf = PerfEstimate.estimate_performance(
                cfg_for_perf,
                stages=stages,
                extra_custom_func=lambda cost_cfg: None,
                ccfg=ccfg,
                debugger=perf_debugger,
                device_type=Hard.Device_A2,
                memory=123,
                cache_file=cache_file,
            )
            self.assertGreater(perf, 0)
            self.assertEqual(perf_debugger.info[Debug.PerfParts.MEMORY], 123)

    def test_comm_time_helpers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover communication-time helper formulas with patched layer comm evaluators.
        Expectation: Communication helpers return deterministic positive values.
        """
        cfg = _make_perf_cfg(n_exp=2)
        tables = {}
        CommTime.fill_dp_table(cfg, tables)
        CommTime.fill_tp_table(cfg, tables)
        CommTime.fill_ep_table(cfg, tables, Hard.Device_A2)
        self.assertIn(Dim.DP, tables)
        self.assertIn("tp", tables)
        self.assertGreater(CommTime.dp_ratio(cfg, Hard.Device_A2), 0)
        self.assertGreater(sum(CommTime.comm_embed_ouput(cfg)), 0)
        self.assertIsNotNone(CommTime.prepare_context().node_eval[LayerType.NOT_REC_LAYER])

        self.assertEqual(CommTime.level_efficiency(NetworkLevel.NODE), 0.7)
        self.assertEqual(CommTime.level_bandwidth(NetworkLevel.CLUSTER), 25)
        self.assertEqual(CommTime.level_latency(NetworkLevel.NODE), 0.00001)
        self.assertGreater(CommTime.comm_throughput(NetworkLevel.NODE), 0)
        self.assertGreater(CommTime.estimate_comm_size_time(None, 10, NetworkLevel.NODE), 0)
        self.assertGreater(CommTime.estimate_comm_score(cfg, 10, Dim.TP, device=Hard.Device_A2), 0)
        with self.assertRaises(ValueError):
            CommTime.level_efficiency("bad")
        with self.assertRaises(ValueError):
            CommTime.level_bandwidth("bad")
        with self.assertRaises(ValueError):
            CommTime.level_latency("bad")

        stages = [[[LayerType.NOT_REC_LAYER, LayerType.OUTPUT_LAYER]], [[LayerType.EMBEDDING_LAYER]]]
        debugger = Debug.Debug(
            Dim.Dimensions([(Dim.DP, 2), (Dim.MBS, 2)], all_dims=[Dim.DP, Dim.MBS]),
            Debug.PerfParts,
        )
        with patch.object(CommTime.EvalLayerComm, "dp_comm_layer", return_value=3), \
                patch.object(CommTime.EvalLayerComm, "tp_comm_layer", return_value=5), \
                patch.object(CommTime.EvalLayerComm, "ep_comm_layer", return_value=7), \
                patch.object(CommTime.EvalLayerComm, "cp_comm_layer", return_value=11):
            comm = CommTime.estimate_comm(
                cfg,
                CustomConfig(ttype=PerformanceType.FLOP),
                stages,
                Hard.Device_A3,
                debugger=debugger,
            )
        self.assertEqual(len(comm), 2)
        self.assertGreater(comm[0], 0)
        self.assertIn(Debug.PerfParts.CP_COMM, debugger.info)

        bulk_cfg = _make_perf_cfg(dc_kv=1, n_exp=2)
        bulk_debugger = Debug.Debug(
            Dim.Dimensions([(Dim.DP, 2), (Dim.MBS, 2)], all_dims=[Dim.DP, Dim.MBS]),
            Debug.PerfParts,
        )

        def fake_bulk_layer(param, lccfgs, **kwargs: Any) -> tuple:
            """Advance the bulk comm layer cursor without doing layer math."""
            del param, lccfgs
            return kwargs["layer_count"] + 1, kwargs["idx_lccfg"]

        with patch.object(CommTime, "estimate_op_bulk_comm_layer", side_effect=fake_bulk_layer):
            bulk_comm = CommTime.estimate_op_bulk_comm(
                bulk_cfg,
                CustomConfig(ttype=PerformanceType.TIME),
                stages,
                Hard.Device_A3,
                debugger=bulk_debugger,
            )
        self.assertEqual(len(bulk_comm), 2)
        self.assertIn(Debug.PerfParts.EP_COMM, bulk_debugger.info)

    def test_comm_overlap_fields_in_parsers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Verify the transitional comm overlap fields
            (comm_dp_overlap, comm_tp_overlap) are populated by the real
            framework parsers via config_comm_flag — the mechanism that
            estimate_from_mem_comm reads on the search (FLOP) path.
        Expectation: After parsing, both overlap fields hold the documented
            defaults (0.9 and 0.5).  CostModelParserHyperparallel and
            CostModelParserMindformers both call the base config_comm_flag,
            so the hyperparallel path covers both.  CostModelParserMindspeed
            has an inline copy (see cost_model_parser_mindspeed.py:293-294)
            but is not tested here per project priority.
        """
        # --- Hyperparallel path: calls base config_comm_flag ---
        # (same base method is also called by CostModelParserMindformers)
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "__init__.py")
            with open(source_path, "w", encoding="utf-8") as source_file:
                source_file.write(
                    "def get_train_spec():\n"
                    "    return TrainSpec(model_args=model_args)\n"
                    "model_args = {\n"
                    "    'tiny': ModelArgs(dim=16, inter_dim=32, hidden_dim=0, vocab_size=64,\n"
                    "                      n_heads=2, n_layers=2, n_kv_heads=0, kv_lora_rank=0,\n"
                    "                      q_lora_rank=0, qk_rope_head_dim=0, n_dense_layers=0,\n"
                    "                      moe_inter_dim=0, moe_enabled=False, moe_args=None,\n"
                    "                      enable_weight_tying=False, multiple_of=1,\n"
                    "                      ffn_dim_multiplier=1)\n"
                    "}\n"
                )
            hp_config = Config(
                {
                    "model": {"name": "llama-unit", "flavor": "tiny"},
                    "parallelism": {
                        "data_parallel_replicate_degree": 1,
                        "data_parallel_shard_degree": 2,
                        "tensor_parallel_degree": 2,
                        "pipeline_parallel_degree": 2,
                        "context_parallel_degree": 1,
                        "expert_parallel_degree": 1,
                        "expert_tensor_parallel_degree": 0,
                        "pipeline_parallel_schedule": "Interleaved1F1B",
                    },
                    "activation_checkpoint": {"mode": "full"},
                    "training": {"seq_len": 8, "local_batch_size": 1},
                }
            )
            hp_ccfg = _ParserCostModelConfig()
            hp_ccfg.config = hp_config
            hp_ccfg.source_code = source_path
            CostModelParserHyperparallel(hp_ccfg).parse()
            self.assertEqual(hp_ccfg.comm_dp_overlap, 0.9)
            self.assertEqual(hp_ccfg.comm_tp_overlap, 0.5)

        # --- Direct test of base _CostModelParser.config_comm_flag ---
        # Covers the shared method used by mindformers and hyper parsers.
        ccfg_direct = _ParserCostModelConfig()
        ccfg_direct.d = 2
        ccfg_direct.t = 2
        ccfg_direct.ep = 1
        ccfg_direct.cp = 1
        ccfg_direct.has_op = True
        ccfg_direct.has_grad_shard = True
        ccfg_direct.n_exp = 1

        class _ConcreteParser(_CostModelParser):
            """Concrete subclass for testing the abstract base."""

            def parse(self) -> None:
                """No-op parse; only config_comm_flag is under test."""
                return None
        _ConcreteParser(ccfg_direct).config_comm_flag(ccfg_direct)
        self.assertEqual(ccfg_direct.comm_dp_overlap, 0.9)
        self.assertEqual(ccfg_direct.comm_tp_overlap, 0.5)

    def test_cp_resolve_topology_cross_node_single_device(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover the ``intra_ranks == 1`` branch of
            ``_cp_resolve_topology`` — cross-node CP when cp exceeds
            devices-per-node but each node has only one device.
        Expectation: topology is "cross-node" and bandwidth equals the
            inter-node value (no intra-node mixing possible).
        """
        topology, bw = CommTime._cp_resolve_topology(
            cp=2, device_per_node=1, bw_intra=10.0, bw_inter=2.0)
        self.assertEqual(topology, "cross-node")
        self.assertEqual(bw, 2.0)

    def test_cp_comm_layer_detailed_rejects_zero_heads(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover the ``ccfg.a <= 0`` validation guard in
            ``cp_comm_layer_detailed``.
        Expectation: A ValueError is raised when attention heads is
            non-positive, regardless of cp degree.
        """
        cfg = _make_perf_cfg(cp=2, a=0)
        with self.assertRaises(ValueError):
            CommTime.cp_comm_layer_detailed(cfg, CommTime.prepare_context())

    def test_estimate_comm_time_path_and_cp_debugger(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover the TIME performance-type branch and the
            ``cp > 1`` CP debugger info block in ``estimate_from_mem_comm``.
        Expectation: With ``ttype=PerformanceType.TIME`` and ``cp > 1``,
            the debugger receives both the standard comm keys and the
            CP-specific detail keys (CP_KV_VOLUME, CP_EXPOSED_TIME,
            CP_TOPOLOGY, CP_BANDWIDTH).
        """
        cfg = _make_perf_cfg(cp=2, n_exp=2)
        stages = [[[LayerType.NOT_REC_LAYER, LayerType.OUTPUT_LAYER]],
                  [[LayerType.EMBEDDING_LAYER]]]
        debugger = Debug.Debug(
            Dim.Dimensions([(Dim.DP, 2), (Dim.MBS, 2)], all_dims=[Dim.DP, Dim.MBS]),
            Debug.PerfParts,
        )
        with patch.object(CommTime.EvalLayerComm, "dp_comm_layer", return_value=3), \
                patch.object(CommTime.EvalLayerComm, "tp_comm_layer", return_value=5), \
                patch.object(CommTime.EvalLayerComm, "ep_comm_layer", return_value=7), \
                patch.object(CommTime, "cp_comm_layer_detailed",
                             return_value=CommTime._cp_comm_zero(cfg)):
            comm = CommTime.estimate_comm(
                cfg,
                CustomConfig(ttype=PerformanceType.TIME),
                stages,
                Hard.Device_A3,
                debugger=debugger,
            )
        self.assertEqual(len(comm), 2)
        self.assertGreater(comm[0], 0)
        self.assertIn("CP_KV_VOLUME", debugger.info)
        self.assertIn("CP_EXPOSED_TIME", debugger.info)
        self.assertIn("CP_TOPOLOGY", debugger.info)
        self.assertIn("CP_BANDWIDTH", debugger.info)

    def test_framework_parsers_with_synthetic_configs(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover HyperParallel and MindSpeed parsers with synthetic configs.
        Expectation: Parsers populate cost-model fields without importing real training packages.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "__init__.py")
            with open(source_path, "w", encoding="utf-8") as source_file:
                source_file.write(
                    "def get_train_spec():\n"
                    "    return TrainSpec(model_args=model_args)\n"
                    "model_args = {\n"
                    "    'tiny': ModelArgs(dim=16, inter_dim=32, hidden_dim=0, vocab_size=64,\n"
                    "                      n_heads=2, n_layers=2, n_kv_heads=0, kv_lora_rank=0,\n"
                    "                      q_lora_rank=0, qk_rope_head_dim=0, n_dense_layers=0,\n"
                    "                      moe_inter_dim=0, moe_enabled=False, moe_args=None,\n"
                    "                      enable_weight_tying=False, multiple_of=1,\n"
                    "                      ffn_dim_multiplier=1)\n"
                    "}\n"
                )
            hp_config = Config(
                {
                    "model": {"name": "llama-unit", "flavor": "tiny"},
                    "parallelism": {
                        "data_parallel_replicate_degree": 1,
                        "data_parallel_shard_degree": 2,
                        "tensor_parallel_degree": 2,
                        "pipeline_parallel_degree": 2,
                        "context_parallel_degree": 1,
                        "expert_parallel_degree": 1,
                        "expert_tensor_parallel_degree": 0,
                        "pipeline_parallel_schedule": "Interleaved1F1B",
                    },
                    "activation_checkpoint": {"mode": "full"},
                    "training": {"seq_len": 8, "local_batch_size": 1},
                }
            )
            hp_ccfg = _ParserCostModelConfig()
            hp_ccfg.config = hp_config
            hp_ccfg.source_code = source_path
            CostModelParserHyperparallel(hp_ccfg).parse()
            self.assertEqual(hp_ccfg.model_name, "llama-unit")
            self.assertEqual(hp_ccfg.vp, 2)
            self.assertEqual(hp_ccfg.layer_custom_config, [(2, None)])

        ms_mod = {
            "model_id": "vision",
            "freeze": False,
            "moe_grouped_gemm": False,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "sequence_parallel": False,
            "pipeline_num_layers": [1, 0],
            "num_layers": 2,
            "hidden_size": 16,
            "ffn_hidden_size": 32,
            "vocab_size": 64,
            "num_attention_heads": 2,
            "num_query_groups": 0,
            "kv_channels": 0,
            "k_lora_rank": 0,
            "q_lora_rank": 0,
            "qk_rope_head_dim": 0,
            "num_moe_experts": 1,
            "moe_router_topk": 1,
            "n_shared_exp": 0,
            "moe_intermediate_size": 0,
            "first_k_dense_replace": 0,
            "recompute_num_layers": 1,
            "params_dtype": "float16",
            "attention_softmax_in_fp32": True,
            "mtp_num_layers": 0,
        }
        ms_config = Config(
            {
                "model_id": "multi-unit",
                "tmp": {"pp": 2, "mbs": 1, "dp": 2, "tp": 1, "cp": 1, "vpp": 1, "ep": 1, "seqlen": 8, "etp": 0},
                "module": ms_mod,
            }
        )
        ms_ccfg = _ParserCostModelConfig()
        ms_ccfg.config = ms_config
        CostModelParserMindspeed(ms_ccfg).parse()
        self.assertEqual(ms_ccfg.model_name, "multi-unit")
        self.assertFalse(ms_ccfg.multimodal)
        self.assertEqual(ms_ccfg.n_lay, 0)

    def test_cost_model_config_strategy_helpers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover cost-model copying, validation and strategy mutation with fake parser hooks.
        Expectation: Strategy fields update consistently without parsing a model config.
        """
        parser_calls = []
        parser = SimpleNamespace(
            config_shard_emb=lambda: parser_calls.append("embed"),
            config_dp_tp_exp=lambda cfg: parser_calls.append(("dp_tp", cfg.d, cfg.t)),
            config_optimizer_shard=lambda cfg: parser_calls.append(("optimizer", cfg.os_max_shard)),
            config_comm_flag=lambda cfg: parser_calls.append(("comm", cfg.sp)),
        )
        cost_cfg = object.__new__(CostModelConfig)
        cost_cfg.__dict__.update(
            model_name="unit",
            multimodal=False,
            parser=parser,
            d=2,
            t=2,
            p=2,
            ep=1,
            cp=1,
            vp=1,
            etp=1,
            m=2,
            b=2,
            gbs=8,
            sp=2,
            os_max_shard=1,
            offset=[0, 0],
            full_rec=[0, 0],
            sel_rec=[0, 0],
            pp_sched="1f1b",
            d_exp=1,
            t_exp=1,
            shard_grad_exp=1,
            shard_grad_non_exp=1,
            shard_p_os_exp=1,
            shard_p_os_non_exp=1,
            shard_embed=1,
            shard_output_activ=1,
            shard_recompute_input=1,
            layer_custom_config=[],
        )

        self.assertIn("model_name", str(cost_cfg))
        self.assertEqual(cost_cfg.missing_value, 0)
        self.assertEqual(cost_cfg.fp_bytes("float16"), 2)
        self.assertEqual(cost_cfg.fp_bytes("bf32"), 4)
        self.assertEqual(cost_cfg.fp_bytes(None), 0)
        self.assertEqual(cost_cfg.strategy_num_devices(), 8)
        self.assertTrue(cost_cfg.is_consistent_pp_config())
        self.assertEqual(cost_cfg.count_layers([[[1, 2]], [[3, 4]]]), 2)
        cost_cfg.print_stages([[[LayerType.EMBEDDING_LAYER]], [[LayerType.OUTPUT_LAYER]]])
        cost_cfg.print_stages([[[LayerType.EMBEDDING_LAYER]]], spec_stage_id=4)
        cost_cfg.print_parallelism()

        shallow = copy.copy(cost_cfg)
        deep = copy.deepcopy(cost_cfg)
        self.assertIs(shallow.parser, cost_cfg.parser)
        self.assertIsNot(deep, cost_cfg)

        cost_cfg.set_strategy(
            dp=4,
            mp=2,
            cp=1,
            ep=2,
            op=2,
            etp=1,
            pp=2,
            vpp=2,
            mb=2,
            mbs=1,
            offset=[[0, 0], [0, 0]],
            full_rec=[[0, 0], [0, 0]],
            sel_rec=[[0, 0], [0, 0]],
        )
        self.assertEqual(cost_cfg.get_strategy()["dp"], 4)
        self.assertEqual(cost_cfg.gbs, 8)
        self.assertIn("embed", parser_calls)

        cost_cfg.offset = []
        with self.assertRaises(AttributeError):
            cost_cfg.set_strategy(dp=2)
        cost_cfg.offset = [[0, 0], [0, 0]]

        child = copy.copy(cost_cfg)
        child.model_name = "child"
        child.multimodal = False
        multimodal = object.__new__(CostModelConfig)
        multimodal.__dict__.update(
            model_name="multi",
            multimodal=True,
            mm_ccfgs={"child": child},
        )
        multimodal.set_strategy(model_name="child", dp=3)
        self.assertEqual(multimodal.get_strategy()["child"]["dp"], 3)
        multimodal.print_parallelism()
        with self.assertRaises(TypeError):
            multimodal.set_strategy(model_name="missing", dp=1)

        hook_calls = []

        def original_hook(target: Any) -> None:
            """Record execution of the original layer hook."""
            hook_calls.append(("original", target))

        def custom_hook(target: Any) -> None:
            """Record execution of the injected cost-model hook."""
            hook_calls.append(("custom", target))

        cost_cfg.layer_custom_config = [(1, original_hook)]
        cost_cfg.layer_custom_config_callback(custom_hook)
        wrapped_hook = cost_cfg.layer_custom_config[0][1]
        wrapped_hook(cost_cfg)
        evaluator = SimpleNamespace(set_ccfg=lambda hook: hook_calls.append(("set_ccfg", hook)))
        wrapped_hook(evaluator)
        self.assertEqual(wrapped_hook.__name__, "original_hook_custom_hook")
        self.assertTrue(any(call[0] == "set_ccfg" for call in hook_calls))

    def test_parallelize_layer_control_flow_without_estimators(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover ND filtering, generation and ordering with deterministic fakes.
        Expectation: No multiprocessing, model parsing, memory estimation or performance solver is invoked.
        """
        runner = object.__new__(Par.ParallelizeLayer)
        writes = []
        config_state = SimpleNamespace(
            ccfg=SimpleNamespace(),
            balancing=SimpleNamespace(from_config=True),
            dimensions=[Dim.DP],
            set_parallel_config=lambda config: True,
            write=lambda folder, config: writes.append((folder, config)),
        )
        runner.config = config_state
        runner.machine = SimpleNamespace(number=16, device=Hard.Device_A2)
        runner.global_batch_size = 8
        runner.model_name = "unit"
        runner.enable_debug = False
        runner.mem_eval = SimpleNamespace(
            mem_fit=lambda peak: peak < 100,
            get_strategy=lambda: {},
        )

        class _ParallelConfig:
            """Small validity-controlled parallel configuration."""

            def __init__(self, valid: bool = True) -> None:
                self.valid = valid
                self.all_dims = [Dim.DP]

            def is_valid(self) -> bool:
                """Return the configured validity."""
                return self.valid

            def values(self) -> list:
                """Return printable dimension values."""
                return ["2"]

        parallel_config = _ParallelConfig()
        config_state.moe_valid = lambda config: config.valid
        config_state.global_batch_size = lambda config: 8
        runner.filtered_out = lambda config: False
        self.assertTrue(runner.is_valid(parallel_config))
        parallel_config.valid = False
        self.assertFalse(runner.is_valid(parallel_config))
        parallel_config.valid = True
        config_state.global_batch_size = lambda config: 4
        self.assertFalse(runner.is_valid(parallel_config))
        config_state.global_batch_size = lambda config: 8

        runner.device_loops = lambda space, pool: ({"fit": 10, "large": 200}, 2)
        self.assertEqual(runner.generate_search_space("out", threads_num=None), [("fit", 10)])
        self.assertEqual(writes, [("out", "fit")])

        runner.memory_estim = lambda debugger=None: 12
        config_state.make_parallel_config = lambda *dims: "inside"
        runner.is_valid = lambda config: True
        configs, size = runner.inside_loop_nest(({}, 0), None, ((1, 2, 2, 1), (1, 4), (1, 1, 1, True)))
        self.assertEqual(configs, {"inside": 12})
        self.assertEqual(size, 1)

        with patch.object(Par, "estimate_performance", side_effect=[3.0, 1.0]):
            scored, debug_parts = runner.order_search_space(
                [(_ParallelConfig(), 10), (_ParallelConfig(), 20)],
                threads_num=None,
                cache_file=None,
            )
        self.assertEqual([entry[2] for entry in scored], [1.0, 3.0])
        self.assertEqual(debug_parts, [])
        self.assertEqual(runner.order_search_space([], None, None), ([], []))

        runner.generate_search_space = lambda folder, threads_num: [(_ParallelConfig(), 10)]
        runner.order_search_space = (
            lambda space, threads_num, cache_file: ([(space[0][0], 10, 2.0, [])], [])
        )
        with patch.object(Par.Debug, "plot_nd") as plot_nd:
            result = runner.run_generation_to_ordering(None, top_num=1)
        self.assertEqual(result[0][2], 2.0)
        plot_nd.assert_not_called()
        self.assertIn("Top 1 configurations", Par.space_to_string(result, max_num=1, debug_parts=[]))

        wrapper = object.__new__(Par.Parallelize)
        wrapper.instance = SimpleNamespace(marker="delegated")
        self.assertEqual(wrapper.marker, "delegated")

        with patch.object(Par, "EvaluatorV2") as evaluator_cls:
            evaluator_cls.return_value.estimate_peak.return_value = 7
            self.assertEqual(Par.pool_estimate_memory("config"), 7)
        with patch.object(Par, "estimate_performance", return_value=9):
            self.assertEqual(Par.pool_estimate_performance("config", Hard.Device_A2), 9)

    def test_parallelize_profile_ordering_without_estimators(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover profiling-order and CSV wrappers with fake measurements.
        Expectation: Results are sorted and plotting dispatches without running estimation formulas.
        """
        runner = object.__new__(Par.ParallelizeLayer)
        runner.enable_debug = False
        runner.machine = SimpleNamespace(device=Hard.Device_A2, number=8)
        runner.model_name = "unit"
        runner.global_batch_size = 8
        runner.config = SimpleNamespace(
            ccfg=SimpleNamespace(),
            dimensions=[Dim.DP],
            set_parallel_config=lambda config: True,
        )
        runner.mem_eval = SimpleNamespace(get_strategy=lambda: {"dp": 2})
        runner.memory_estim = lambda: 32

        class _FakeDebug:
            """Debug facade with stable accounting entries."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                """Initialize deterministic debug values."""
                self.info = {"compute": 1.0, "total": 2.0, "memory": 3.0}

            def write(self) -> None:
                """Accept debug CSV writes without touching disk."""

        with patch.object(Par.Debug, "Debug", _FakeDebug), \
                patch.object(Par, "estimate_performance", return_value=5.0):
            ordered, parts = runner.order_space_test([("cfg-b", 20), ("cfg-a", 10)], order_by=2)
            classified, classified_parts = runner.order_space_test_comm_classified(
                [("cfg-b", 20, 4), ("cfg-a", 10, 2)],
                order_by=2,
            )

        self.assertEqual([entry[2] for entry in ordered], [10, 20])
        self.assertEqual([entry[2] for entry in classified], [10, 20])
        self.assertEqual(parts, ["compute"])
        self.assertEqual(classified_parts, ["compute"])

        runner.order_space_test = lambda configs, order_by=2: (configs, ["compute"])
        runner.order_space_test_comm_classified = (
            lambda configs, order_by=2: (configs, ["compute"])
        )
        with patch.object(Par.Debug, "get_real_data", return_value=([("cfg", 10)], 1)), \
                patch.object(Par.Debug, "plot_vs_real") as plot_vs_real, \
                patch.object(Par.Debug, "correlation_topk", return_value=(0.9, 1)), \
                patch.object(Par.Debug, "get_comm_classified_data", return_value=[("cfg", 10, 2)]), \
                patch.object(Par.Debug, "plot_vs_real_comm_classified") as plot_vs_real_comm, \
                patch.object(Par.Debug, "correlation_with_classified_comms", return_value=0.8):
            self.assertEqual(runner.test_from_csv("profile.csv", "out"), (0.9, 1, 1))
            self.assertEqual(
                runner.test_from_csv_comm_classified("profile.csv", "out", plot_idle=True),
                0.8,
            )

        plot_vs_real.assert_called_once()
        plot_vs_real_comm.assert_called_once()
