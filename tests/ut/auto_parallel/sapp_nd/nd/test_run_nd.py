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
import csv
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import yaml

from hyper_parallel.auto_parallel._layer_stack import derive_layers, resolve_layers
from hyper_parallel.auto_parallel._op_profiles import load_op_profile
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hook_base import MemEvalHook, hook_runner
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd import debug as Debug
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd import balancing_adapter as BA
from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd import global_config as GC
from hyper_parallel.auto_parallel.sapp_nd.nd import run_nd as RunND
from hyper_parallel.auto_parallel.sapp_nd.nd.common import arch_hooks as ArchHooks
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config, YamlObject
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common import cost_model_preprocess as PreProcess
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import HYPER_SELECTIVE_REC_OP, derive_comm_flags
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_hyperparallel import (
    CostModelParserHyperparallel,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_mindspeed import (
    CostModelParserMindspeed,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.generate_partitions import PartitionGenerator
from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger as nd_logger, set_verbose_level
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
        self.to_ppb_calls = []
        self.compare_args = None
        self.__class__.instances.append(self)

    def run_generation_to_ordering(self, *args: Any, **kwargs: Any) -> list:
        """Return a small sorted search result without invoking ND search."""
        self.run_args = args
        self.run_kwargs = kwargs
        return [("parallel-config", 128.0, 1.0, {})]

    def to_ppb(self, *args: Any) -> None:
        """Record which configuration the CLI hands to pipeline balancing."""
        self.to_ppb_calls.append(args)
    def compare_with_csv(self, csv_f: str, output_path: Any = None, plot_idle: bool = False) -> tuple:
        """Return one measured configuration and its metrics without estimating."""
        self.compare_args = (csv_f, output_path, plot_idle)
        real = {"comp": 6.0, "dp_wait": 4.0}
        # One value per part a score splits into: every PerfParts but TOTAL and MEMORY.
        parts = [1.0] * (len(Debug.PerfParts) - 2)
        configs = [(Dim.Dimensions([(Dim.DP, 8)], all_dims=[Dim.DP]), 1, 10.0, 10.0, parts, real)]
        return configs, Debug.correlation_with_classified_comms(configs)

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
    """Build a tiny config to apply a family to."""
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


def _torchtitan_flavor(name: str, args: str, **parallelism: Any) -> CostModelConfig:
    """Parse a TorchTitan flavor of model *name* stating *args*, as the TOML parser reads one."""
    toml = Config({
        "model": {"name": name, "flavor": "tiny"},
        "parallelism": {
            "data_parallel_replicate_degree": 1, "data_parallel_shard_degree": 2,
            "tensor_parallel_degree": 1, "pipeline_parallel_degree": 1, "context_parallel_degree": 1,
            "expert_parallel_degree": 1, "expert_tensor_parallel_degree": 0,
            "pipeline_parallel_schedule": "1F1B", **parallelism,
        },
        "activation_checkpoint": {"mode": "full"},
        "training": {"seq_len": 128, "local_batch_size": 1},
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        source_path = os.path.join(tmp_dir, "__init__.py")
        with open(source_path, "w", encoding="utf-8") as source_file:
            source_file.write(
                "def get_train_spec():\n    return TrainSpec(model_args=model_args)\n"
                f"model_args = {{'tiny': ModelArgs({args})}}\n"
            )
        return CostModelConfig(toml, framework="hyperparallel", source_code=source_path)


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
        global_config.siblings = []
        parallel_config = global_config.make_parallel_config(
            (2, 2, 2, 1),
            (4, 2),
            (1, 1, 2, False),
        )
        self.assertEqual(global_config.dim_val(Dim.DP, parallel_config), 2)
        self.assertEqual(global_config.global_batch_size(parallel_config), 16)
        no_pipeline = global_config.make_parallel_config((2, 2, 1, 1), (4, 2), (1, 1, 2, False))
        self.assertEqual(global_config.global_batch_size(no_pipeline), 8)
        self.assertFalse(no_pipeline.is_valid())
        fake_ccfg.accumulates_grads = True
        accumulating = global_config.make_parallel_config((2, 2, 1, 1), (4, 2), (1, 1, 2, False))
        self.assertEqual(global_config.global_batch_size(accumulating), 16)
        self.assertTrue(accumulating.is_valid())
        fake_ccfg.accumulates_grads = False
        self.assertEqual(global_config.layer_num_for_offset(), 4)
        self.assertEqual(global_config.total_layer_num(), 5)
        self.assertEqual(global_config.adapt_config(2, 1), ([0, 0], [0, 0]))
        self.assertTrue(global_config.moe_valid(parallel_config))
        self.assertTrue(global_config.set_parallel_config(parallel_config))
        self.assertIn("offset", fake_ccfg.strategy_calls[-1])
        self.assertEqual(global_config.range_space(Dim.PP, 3), range(1, 4))
        self.assertEqual(global_config.bool_space(Dim.SP), [False, True])
        self.assertEqual(global_config.max_op(dp=4, tp=2, ep=1), 4)
        # HyperParallel's shard spans DP x CP, so under CP it reaches that
        # domain; a legacy schema's stays within DP.
        got = []
        for spans in (True, False):
            fake_ccfg.shard_spans_cp = spans
            got.append(global_config.max_op(dp=4, tp=2, ep=1, cp=2))
        fake_ccfg.shard_spans_cp = False
        self.assertEqual(got, [8, 4])

        Dim.TP.set_bound(2)
        try:
            self.assertEqual(global_config.space(Dim.TP, 8), [1, 2])
        finally:
            Dim.TP.reset_bound()

        # Muon does not narrow the optimizer shard: with an expert count and
        # MLA widths sharing only a factor of 2 with DP, the cap that came in
        # with the original search import returned 2 of the 4 available ranks.
        fake_ccfg.optimizer = "muon"
        fake_ccfg.dc_kv = 2
        fake_ccfg.dhr = 2
        experts = fake_ccfg.n_exp
        fake_ccfg.n_exp = 6
        self.assertEqual(global_config.max_op(dp=4, tp=2, ep=1), 4)
        fake_ccfg.n_exp = experts

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

    def test_hyperparallel_spreads_experts_over_dp_cp_and_tp(self) -> None:
        """
        Feature: GlobalConfig.max_ep and moe_valid (F3 of Nelson's review).
        Description: 64 experts at DP 4, TP 2 and CP 2 with EP 16, wider than
            DP x TP: under HyperParallel, whose expert mesh spans DP x CP x TP,
            then under a legacy schema.
        Expectation: HyperParallel's bound is that domain's 16 ranks, so EP 16
            is generated and valid, where the search stopped at DP x TP's 8
            and refused it; a legacy schema keeps DP x TP.
        """
        ccfg = _FakeCostModelConfig()
        ccfg.n_exp = 64
        global_config = object.__new__(GC.GlobalConfig)
        global_config.ccfg = ccfg
        global_config.dimensions = Dim.ALL_DIMS.copy()
        parallel_config = global_config.make_parallel_config((4, 2, 1, 2), (1, 1), (16, 1, 1, False))
        got = []
        for spans in (True, False):
            ccfg.shard_spans_cp = spans
            got.append((global_config.max_ep(dp=4, tp=2, cp=2), global_config.moe_valid(parallel_config)))
        self.assertEqual(got, [(16, True), (8, False)])

    def test_ep_constraints_valid_in_global_config(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover GlobalConfig.ep_constraints_valid for C1 (expert
            divisibility) and C2 (expert-hidden divisibility), plus the dense
            fast-path and the is_valid() wiring in ParallelizeLayer.
        Expectation: MoE configs that violate C1 or C2 are rejected; dense
            models and valid MoE configs are accepted.
        """

        def _make_gc(n_exp=4, ep=1, hff_exp=0, etp=0, tp=2, dp=2, pp=1):
            """Build a minimal GlobalConfig for EP-constraint tests."""
            ccfg = _FakeCostModelConfig()
            ccfg.n_exp = n_exp
            ccfg.ep = ep
            ccfg.hff_exp = hff_exp
            ccfg.etp = etp
            ccfg.t = tp
            ccfg.d = dp
            ccfg.p = pp
            gc = object.__new__(GC.GlobalConfig)
            gc.ccfg = ccfg
            gc.dimensions = Dim.ALL_DIMS.copy()
            gc.balancing = _FakeBalancing()
            gc.siblings = []
            return gc

        # Dense model (n_exp=1) — fast-path, always True regardless of ep.
        gc_dense = _make_gc(n_exp=1, ep=2, hff_exp=0)
        pc_dense = gc_dense.make_parallel_config(
            (2, 2, 1, 1), (4, 1), (2, 1, 2, False)
        )
        self.assertTrue(gc_dense.ep_constraints_valid(pc_dense))

        # MoE model, valid C1 (8 % 4 == 0) and C2 (14336 % 2 == 0).
        gc_ok = _make_gc(n_exp=8, ep=4, hff_exp=14336, etp=0, tp=2)
        pc_ok = gc_ok.make_parallel_config(
            (2, 2, 1, 1), (4, 1), (4, 1, 2, False)
        )
        self.assertTrue(gc_ok.ep_constraints_valid(pc_ok))

        # C4: DP 2 x TP 2 over EP 4 leaves one rank per expert group, which
        # an expert shard of 2 cannot divide; a shard of 1 always does.
        gc_ok.ccfg.expert_shard = 2
        self.assertFalse(gc_ok.ep_constraints_valid(pc_ok))
        gc_ok.ccfg.expert_shard = 1
        self.assertTrue(gc_ok.ep_constraints_valid(pc_ok))
        # A run that shards over each strategy's whole group fits any group.
        gc_ok.ccfg.expert_shard = 2
        gc_ok.ccfg.expert_shard_group = True
        self.assertTrue(gc_ok.ep_constraints_valid(pc_ok))
        gc_ok.ccfg.expert_shard_group = False

        # MoE model, C1 fail: n_exp=8, ep=3 (8 % 3 != 0).
        gc_c1 = _make_gc(n_exp=8, ep=3, hff_exp=14336, etp=0, tp=2)
        pc_c1 = gc_c1.make_parallel_config(
            (6, 2, 1, 1), (4, 1), (3, 1, 2, False)
        )
        self.assertFalse(gc_c1.ep_constraints_valid(pc_c1))

        # MoE model, C2 fail: hff_exp=14336, tp=5 (14336 % 5 != 0), etp=0.
        gc_c2 = _make_gc(n_exp=8, ep=4, hff_exp=14336, etp=0, tp=5)
        pc_c2 = gc_c2.make_parallel_config(
            (2, 5, 1, 1), (4, 1), (4, 1, 2, False)
        )
        self.assertFalse(gc_c2.ep_constraints_valid(pc_c2))

        # MoE model, etp overrides tp for C2: hff_exp=14336, etp=4, tp=5.
        # C2 uses t_exp=etp=4, 14336 % 4 == 0 → pass despite tp=5.
        gc_etp = _make_gc(n_exp=8, ep=4, hff_exp=14336, etp=4, tp=5)
        pc_etp = gc_etp.make_parallel_config(
            (2, 5, 1, 1), (4, 1), (4, 1, 2, False)
        )
        self.assertTrue(gc_etp.ep_constraints_valid(pc_etp))

    def test_run_nd_cli_writes_the_ppb_description_of_rank_k(self) -> None:
        """
        Feature: run_nd -k/--ppb_k.
        Description: Rank the space, then ask for the pipeline balancer's
            description of rank 0, then of a rank the space does not have.
        Expectation: Rank 0 is handed to to_ppb under the yaml's name; the
            missing rank is refused before anything is written.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            _FakeParallelize.instances = []
            argv = ["run_nd.py", "-y", config_path, "-d", "8", "-l", "DP", "MP", "-v", "0", "-k", "0"]
            with patch.object(sys, "argv", argv):
                runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertEqual(_FakeParallelize.instances[-1].to_ppb_calls,
                             [([("parallel-config", 128.0, 1.0, {})], 0, "deepseek")])

            argv[-1] = "1"
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertEqual(_FakeParallelize.instances[-1].to_ppb_calls, [])

    def test_run_nd_cli_asks_for_auto_recompute(self) -> None:
        """
        Feature: run_nd -ar/--auto_recompute.
        Description: Ask for auto recompute alone, then with the yaml's
            recompute, then with a search config.
        Expectation: Alone, the search is built with it; with the yaml's
            recompute, the command line is refused before any search is
            built; a search config is told to choose its recompute.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            _FakeParallelize.instances = []
            argv = ["run_nd.py", "-y", config_path, "-d", "8", "-l", "DP", "MP", "-v", "0", "-ar"]
            with patch.object(sys, "argv", argv):
                runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertTrue(_FakeParallelize.instances[-1].kwargs["auto_recompute"])
            _FakeParallelize.instances = []
            with patch.object(sys, "argv", argv + ["-mppb"]), self.assertRaises(SystemExit):
                runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertEqual(_FakeParallelize.instances, [])
        run_nd = runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="run_nd")
        search_cfg = SimpleNamespace(constraint={}, cluster_spec={}, estimator={"recompute_strategy": "full"})
        cli = SimpleNamespace(global_batch_size=None, max_mem=None, devices=None, device_type=None, auto_recompute=True)
        run_nd["_apply_cli_overrides"](search_cfg, cli)
        self.assertEqual(search_cfg.estimator["recompute_strategy"], "auto")

    def test_run_nd_cli_asks_for_auto_offload(self) -> None:
        """
        Feature: run_nd -ao/--auto_offload.
        Description: Ask for offload with auto recompute, then with the
            link's figures stated, then without auto recompute, then the
            figures without offload, then with a search config.
        Expectation: The search gets the device's link, then one with the
            stated figures; the last three command lines are refused before
            any search is built.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            _FakeParallelize.instances = []
            argv = ["run_nd.py", "-y", config_path, "-d", "8", "-l", "DP", "MP", "-v", "0", "-A", "A3"]
            with patch.object(sys, "argv", argv + ["-ar", "-ao"]):
                runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            kwargs = _FakeParallelize.instances[-1].kwargs
            self.assertTrue(kwargs["auto_offload"])
            self.assertEqual(kwargs["host_link"], Hard.Device_A3.host_link)
            with patch.object(sys, "argv", argv + ["-ar", "-ao", "--host_link_gibps", "32",
                                                   "--sustained_tflops", "200"]):
                runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertEqual(_FakeParallelize.instances[-1].kwargs["host_link"],
                             Hard.HostLink.of(Hard.Device_A3.host_link, {"gib_per_s": 32.0, "sustained_tflops": 200.0}))
            _FakeParallelize.instances = []
            for extra in (["-ao"], ["-ar", "--host_link_gibps", "32"], ["-ar", "-ao", "-s", config_path]):
                with patch.object(sys, "argv", argv + extra), self.assertRaises(SystemExit):
                    runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertEqual(_FakeParallelize.instances, [])

    def test_run_nd_cli_states_the_overlap_and_calibrates_the_link(self) -> None:
        """
        Feature: run_nd --host_link_overlap, and -c with -ao.
        Description: Ask for offload with an overlap stated and a ratios
            file, then with a ratios file that also states a FORWARD ratio
            and a copy cost of 0; then the overlap without offload, a copy
            cost without offload, offload with a ratios file that states no
            COMPUTE ratio, and one whose FORWARD ratio is 0.
        Expectation: The search gets the device's link with the overlap and
            the file's FORWARD ratio replacing its own where it states one,
            else its COMPUTE ratio, the COMPUTE ratio for the copies' cost,
            and A3's measured cost of 2.8 ms a GiB unless one is stated; the
            last four command lines are refused before any search is built.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            files = {}
            for name, ratios in (("ratios", {"COMPUTE": 6.4e-11}),
                                 ("forward", {"COMPUTE": 6.4e-11, "FORWARD": 2.3e-11}),
                                 ("other", {"DP_COMM": 1e-11}),
                                 ("zero", {"COMPUTE": 6.4e-11, "FORWARD": 0.0})):
                files[name] = os.path.join(tmp_dir, f"{name}.json")
                with open(files[name], "w", encoding="utf-8") as fh:
                    json.dump(ratios, fh)
            argv = ["run_nd.py", "-y", config_path, "-d", "8", "-l", "DP", "MP", "-v", "0", "-A", "A3"]
            for name, ratio, cost in (("ratios", 6.4e-11, []), ("forward", 2.3e-11, ["--host_link_cost", "0"])):
                _FakeParallelize.instances = []
                with patch.object(sys, "argv",
                                  argv + ["-ar", "-ao", "--host_link_overlap", "1", "-c", files[name]] + cost):
                    runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
                figures = {"overlap": 1.0, "ms_per_unit": ratio, "score_ms_per_unit": 6.4e-11}
                if cost:
                    figures["copy_cost_ms_per_gib"] = 0.0
                self.assertEqual(_FakeParallelize.instances[-1].kwargs["host_link"],
                                 Hard.HostLink.of(Hard.Device_A3.host_link, figures))
            self.assertEqual(Hard.Device_A3.host_link.copy_cost_ms_per_gib, 2.8)
            _FakeParallelize.instances = []
            for extra in (["-ar", "--host_link_overlap", "1"], ["-ar", "--host_link_cost", "1"],
                          ["-ar", "-ao", "-c", files["other"]], ["-ar", "-ao", "-c", files["zero"]]):
                with patch.object(sys, "argv", argv + extra), self.assertRaises(SystemExit):
                    runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertEqual(_FakeParallelize.instances, [])

    def test_run_nd_cli_hyper_v2_offloads_in_a_mode_per_layer(self) -> None:
        """
        Feature: run_nd -ao with a search config.
        Description: A search config choosing a recompute mode per layer, run
            with -ao, an overlap and a ratios file, whose best configuration
            offloads part of its first layer; then one choosing one mode for
            every layer, run with -ao.
        Expectation: The first search is handed the figures the CLI states,
            None for those left to the device, and the log states what the
            offloaded layer moves; the second is refused before it searches.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            train_yaml, search_yaml, ratios = (os.path.join(tmp_dir, name)
                                               for name in ("train.yaml", "search.yaml", "ratios.json"))
            with open(train_yaml, "w", encoding="utf-8") as fh:
                fh.write("model:\n  name: test\n")
            with open(search_yaml, "w", encoding="utf-8") as fh:
                fh.write("parallelism:\n  dp: [1, 2]\n")
            with open(ratios, "w", encoding="utf-8") as fh:
                json.dump({"COMPUTE": 6.4e-11}, fh)
            fake_result = {
                "dp": 2, "tp": 1, "pp": 1, "cp": 1, "ep": 1,
                "micro_batch_num": 1, "memory_estimate_mb": 100.0, "score": 1.0,
                "activation_checkpoint": "full", "activation_checkpoint_layers": {"0": "off"},
                "offloaded_layers": ["0"], "offloaded_gib": {"0": 3.92}, "offloaded_share": {"0": 0.71},
            }
            outcomes, lines = [], []
            for strategy in ("per_layer", "auto"):
                config = SimpleNamespace(resolved_strategy=None, estimator={"recompute_strategy": strategy},
                                         cluster_spec={}, constraint={})
                with patch("hyper_parallel.auto_parallel.config_adapter.read_search_config", return_value=config), \
                        patch("hyper_parallel.auto_parallel.config_adapter.validate", return_value=[]), \
                        patch("hyper_parallel.auto_parallel.config_adapter.search_strategies",
                              return_value=fake_result) as mock_search, \
                        patch("hyper_parallel.auto_parallel.config_adapter.write_resolved_yaml"):
                    argv = ["run_nd.py", "-f", "hyper_v2", "-y", train_yaml, "-s", search_yaml, "-o", tmp_dir,
                            "-v", "0", "-ao", "--host_link_overlap", "1", "-c", ratios]
                    with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as exc_info, \
                            patch.object(nd_logger, "output") as output:
                        runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
                outcomes.append((exc_info.exception.code, mock_search.call_args))
                lines += [call.args[0] % call.args[1:] for call in output.call_args_list
                          if str(call.args[0]).startswith("Offloaded")]
        self.assertEqual(outcomes[0][0], 0)
        self.assertEqual(outcomes[0][1].kwargs["offload"],
                         {"gib_per_s": None, "sustained_tflops": None, "overlap": 1.0, "ms_per_unit": 6.4e-11,
                          "score_ms_per_unit": 6.4e-11, "copy_cost_ms_per_gib": None})
        self.assertEqual(outcomes[1], (2, None))
        self.assertIn("layers 0 (3.92 GiB a micro-batch, 71% of what it keeps). The trainer runs no offload yet",
                      "\n".join(lines))

    def test_a_plan_per_layer_is_logged_as_the_trainer_runs_it(self) -> None:
        """
        Feature: run_nd's log of the activation checkpointing a search chose.
        Description: A result of full recompute with layers 3 to 7 off, for a
            trainer whose activation_checkpoint takes layers, then for one
            whose does not; then a result of one mode for every layer.
        Expectation: The mode and the plan where the trainer runs it; else
            the mode for every layer, as the resolved yaml states it, and the
            plan as advice the trainer cannot run; one mode alone as before,
            the trainer not asked.
        """
        plan = {"activation_checkpoint": "full", "activation_checkpoint_layers": {"3-7": "off"}}
        lines = {}
        for runs_plans in (True, False):
            with patch.object(RunND, "trainer_runs_layer_plans", return_value=runs_plans), \
                    patch.object(nd_logger, "output") as output:
                RunND._log_activation_checkpoint(plan)  # pylint: disable=protected-access
            lines[runs_plans] = [call.args[0] % call.args[1:] for call in output.call_args_list]
        self.assertEqual(lines[True], ["Activation checkpoint mode full, and per layer: {'3-7': 'off'}"])
        self.assertEqual(lines[False], ["Activation checkpoint mode for every layer: full, as the resolved yaml "
                                        "states it. The plan per layer {'3-7': 'off'} is advice: the installed "
                                        "trainer's activation_checkpoint has no layers field"])
        with patch.object(RunND, "trainer_runs_layer_plans") as asked, patch.object(nd_logger, "output") as output:
            RunND._log_activation_checkpoint({"activation_checkpoint": "full"})  # pylint: disable=protected-access
        asked.assert_not_called()
        self.assertEqual(output.call_args.args, ("Activation checkpoint mode for every layer: %s", "full"))

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

    def test_run_nd_cli_hyper_v2_basic(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover run_nd CLI with ``-f hyper_v2`` and no search config,
                     exercising the standard Parallelize path.
        Expectation: ``Parallelize`` is invoked with framework ``hyper_v2`` and
                     the yaml path as ``input_config``.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            _FakeParallelize.instances = []
            argv = [
                "run_nd.py",
                "-f", "hyper_v2",
                "-y", config_path,
                "-d", "128",
                "-l", "DP", "MP", "PP",
                "-v", "0",
            ]
            with patch.object(sys, "argv", argv):
                runpy.run_module(
                    "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd",
                    run_name="__main__",
                )
            instance = _FakeParallelize.instances[-1]
            self.assertEqual(instance.args[0], "hyper_v2")
            self.assertEqual(instance.args[1], config_path)
            self.assertEqual(instance.args[2].number, 128)

    def test_run_nd_cli_hyper_v2_requires_devices(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: ``-f hyper_v2`` without ``-d`` and without a search
                     config. An AutoModels train.yaml carries no world size,
                     so ND would otherwise infer the cluster as d*t*cp*p.
        Expectation: The CLI exits rather than searching a wrong machine.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            _FakeParallelize.instances = []
            argv = ["run_nd.py", "-f", "hyper_v2", "-y", config_path, "-v", "0"]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    runpy.run_module(
                        "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd",
                        run_name="__main__",
                    )
            self.assertEqual(_FakeParallelize.instances, [])

    def test_run_nd_cli_hyper_v2_with_search_config(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover run_nd CLI with ``-f hyper_v2 -s search.yaml``,
                     exercising the config_adapter branch that writes
                     ``resolved.yaml``.
        Expectation: ``read_search_config``, ``validate``,
                     ``search_strategies``, and ``write_resolved_yaml`` are
                     called; the standard ``Parallelize`` path is bypassed.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            train_yaml = os.path.join(tmp_dir, "train.yaml")
            search_yaml = os.path.join(tmp_dir, "search.yaml")
            with open(train_yaml, "w", encoding="utf-8") as fh:
                fh.write("model:\n  name: test\n")
            with open(search_yaml, "w", encoding="utf-8") as fh:
                fh.write("parallelism:\n  dp: [1, 2]\n")

            fake_config = SimpleNamespace(resolved_strategy=None)
            fake_result = {
                "dp": 2, "tp": 1, "pp": 1, "cp": 1, "ep": 1,
                "micro_batch_num": 1, "memory_estimate_mb": 100.0,
                "score": 1.0,
            }

            with patch(
                "hyper_parallel.auto_parallel.config_adapter.read_search_config",
                return_value=fake_config,
            ) as mock_read, \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.validate",
                        return_value=[],
                    ) as mock_validate, \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.search_strategies",
                        return_value=fake_result,
                    ) as mock_search, \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.write_resolved_yaml",
                    ) as mock_write, \
                    patch.object(Par, "Parallelize", _FakeParallelize):
                _FakeParallelize.instances = []
                argv = [
                    "run_nd.py",
                    "-f", "hyper_v2",
                    "-y", train_yaml,
                    "-s", search_yaml,
                    "-o", tmp_dir,
                    "-v", "0",
                ]
                with patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit) as exc_info:
                        runpy.run_module(
                            "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd",
                            run_name="__main__",
                        )
                self.assertEqual(exc_info.exception.code, 0)

            mock_read.assert_called_once_with(search_yaml)
            mock_validate.assert_called_once_with(fake_config)
            mock_search.assert_called_once_with(fake_config)
            mock_write.assert_called_once_with(
                fake_config, train_yaml, os.path.join(tmp_dir, "resolved.yaml"),
            )
            self.assertEqual(fake_config.resolved_strategy, fake_result)
            self.assertEqual(len(_FakeParallelize.instances), 0)

    def test_run_nd_cli_hyper_v2_refuses_another_train_yaml(self) -> None:
        """
        Feature: run_nd's -y against the search config's train_yaml.
        Description: A search config naming one train.yaml, run with -y
            naming another, and with -y naming the same file by another
            path.
        Expectation: Two files are refused before the search reads its
            config, since the resolved yaml would state one run's strategy
            over another's run; one file under two spellings runs.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            train_yaml = os.path.join(tmp_dir, "train.yaml")
            other_yaml = os.path.join(tmp_dir, "other.yaml")
            search_yaml = os.path.join(tmp_dir, "search.yaml")
            for path in (train_yaml, other_yaml):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("model:\n  name: test\n")
            with open(search_yaml, "w", encoding="utf-8") as fh:
                fh.write(f"train_yaml: {train_yaml}\nparallelism:\n  dp: [1, 2]\n")
            fake_result = {
                "dp": 2, "tp": 1, "pp": 1, "cp": 1, "ep": 1,
                "micro_batch_num": 1, "memory_estimate_mb": 100.0, "score": 1.0,
            }
            codes = []
            for given in (other_yaml, os.path.join(tmp_dir, ".", "train.yaml")):
                with patch(
                    "hyper_parallel.auto_parallel.config_adapter.read_search_config",
                    return_value=SimpleNamespace(resolved_strategy=None),
                ) as mock_read, \
                        patch("hyper_parallel.auto_parallel.config_adapter.validate", return_value=[]), \
                        patch("hyper_parallel.auto_parallel.config_adapter.search_strategies",
                              return_value=fake_result), \
                        patch("hyper_parallel.auto_parallel.config_adapter.write_resolved_yaml"):
                    argv = ["run_nd.py", "-f", "hyper_v2", "-y", given, "-s", search_yaml, "-o", tmp_dir, "-v", "0"]
                    with patch.object(sys, "argv", argv):
                        with self.assertRaises(SystemExit) as exc_info:
                            runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
                codes.append((exc_info.exception.code, mock_read.called))
        self.assertEqual(codes, [(2, False), (0, True)])

    def test_run_nd_cli_hyper_v2_search_config_validation_error(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover run_nd CLI with ``-f hyper_v2 -s search.yaml`` when
                     validation reports hard errors — ``parser.error`` should
                     exit and ``search_strategies`` must not be called.
        Expectation: ``SystemExit`` is raised and no resolved.yaml is written.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            train_yaml = os.path.join(tmp_dir, "train.yaml")
            search_yaml = os.path.join(tmp_dir, "search.yaml")
            with open(train_yaml, "w", encoding="utf-8") as fh:
                fh.write("model:\n  name: test\n")
            with open(search_yaml, "w", encoding="utf-8") as fh:
                fh.write("parallelism:\n  dp: [1, 2]\n")

            fake_config = SimpleNamespace(resolved_strategy=None)
            fake_error = SimpleNamespace(
                severity="error", field_path="cluster", message="missing",
            )

            with patch(
                "hyper_parallel.auto_parallel.config_adapter.read_search_config",
                return_value=fake_config,
            ), \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.validate",
                        return_value=[fake_error],
                    ), \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.search_strategies",
                    ) as mock_search, \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.write_resolved_yaml",
                    ) as mock_write:
                argv = [
                    "run_nd.py",
                    "-f", "hyper_v2",
                    "-y", train_yaml,
                    "-s", search_yaml,
                    "-o", tmp_dir,
                    "-v", "0",
                ]
                with patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit):
                        runpy.run_module(
                            "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd",
                            run_name="__main__",
                        )

            mock_search.assert_not_called()
            mock_write.assert_not_called()

    def test_run_nd_cli_hyper_v2_b_override_search_config(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover run_nd CLI with ``-f hyper_v2 -s search.yaml -b 512``,
                     verifying that ``-b/--global-batch-size`` overrides search
                     config ``constraint.global_batch_size`` (Req 5).
        Expectation: ``search_cfg.constraint["global_batch_size"]`` is set to
                     the CLI value before ``validate`` and ``search_strategies``
                     are invoked.
        """
        constraint = {"global_batch_size": 2048}
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            train_yaml = os.path.join(tmp_dir, "train.yaml")
            search_yaml = os.path.join(tmp_dir, "search.yaml")
            with open(train_yaml, "w", encoding="utf-8") as fh:
                fh.write("model:\n  name: test\n")
            with open(search_yaml, "w", encoding="utf-8") as fh:
                fh.write("parallelism:\n  dp: [1, 2]\n")

            fake_config = SimpleNamespace(
                resolved_strategy=None,
                constraint=constraint,
                cluster_spec={},
            )
            fake_result = {"dp": 1, "tp": 1, "pp": 1, "cp": 1, "ep": 1,
                           "micro_batch_num": 1, "memory_estimate_mb": 100.0,
                           "score": 1.0}

            with patch(
                "hyper_parallel.auto_parallel.config_adapter.read_search_config",
                return_value=fake_config,
            ), \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.validate",
                        return_value=[],
                    ), \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.search_strategies",
                        return_value=fake_result,
                    ) as mock_search, \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.write_resolved_yaml",
                    ):
                argv = [
                    "run_nd.py",
                    "-f", "hyper_v2",
                    "-y", train_yaml,
                    "-s", search_yaml,
                    "-b", "512",
                    "-v", "0",
                ]
                with patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit) as exc_info:
                        runpy.run_module(
                            "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd",
                            run_name="__main__",
                        )
                self.assertEqual(exc_info.exception.code, 0)
            self.assertEqual(constraint["global_batch_size"], 512)
            mock_search.assert_called_once()

    def test_run_nd_cli_hyper_v2_d_override_search_config(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover run_nd CLI with ``-f hyper_v2 -s search.yaml -d 64``,
                     verifying that ``-d/--devices`` overrides search config
                     ``cluster_spec.num_nodes``.
        Expectation: ``search_cfg.cluster_spec["num_nodes"]`` is recomputed to
                     ``max(1, 64 // cards_per_node)`` before the search starts.
        """
        cluster_spec = {"num_nodes": 16, "cards_per_node": 8, "device_type": "A2"}
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            train_yaml = os.path.join(tmp_dir, "train.yaml")
            search_yaml = os.path.join(tmp_dir, "search.yaml")
            with open(train_yaml, "w", encoding="utf-8") as fh:
                fh.write("model:\n  name: test\n")
            with open(search_yaml, "w", encoding="utf-8") as fh:
                fh.write("parallelism:\n  dp: [1, 2]\n")

            fake_config = SimpleNamespace(
                resolved_strategy=None,
                constraint={},
                cluster_spec=cluster_spec,
            )
            fake_result = {"dp": 1, "tp": 1, "pp": 1, "cp": 1, "ep": 1,
                           "micro_batch_num": 1, "memory_estimate_mb": 100.0,
                           "score": 1.0}

            with patch(
                "hyper_parallel.auto_parallel.config_adapter.read_search_config",
                return_value=fake_config,
            ), \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.validate",
                        return_value=[],
                    ), \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.search_strategies",
                        return_value=fake_result,
                    ) as mock_search, \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.write_resolved_yaml",
                    ):
                argv = [
                    "run_nd.py",
                    "-f", "hyper_v2",
                    "-y", train_yaml,
                    "-s", search_yaml,
                    "-d", "64",
                    "-v", "0",
                ]
                with patch.object(sys, "argv", argv):
                    with self.assertRaises(SystemExit) as exc_info:
                        runpy.run_module(
                            "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd",
                            run_name="__main__",
                        )
                self.assertEqual(exc_info.exception.code, 0)
                self.assertEqual(cluster_spec["device_type"], "A2")
                with patch.object(sys, "argv", argv + ["-A", "A3"]):
                    with self.assertRaises(SystemExit) as exc_info:
                        runpy.run_module(
                            "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd",
                            run_name="__main__",
                        )
                self.assertEqual(exc_info.exception.code, 0)
            self.assertEqual(cluster_spec["num_nodes"], 8)
            self.assertEqual(cluster_spec["device_type"], "A3")
            self.assertEqual(mock_search.call_count, 2)

    def test_run_nd_cli_hyper_v2_c_reaches_the_search_config(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: ``-f hyper_v2 -s search.yaml -c ratios.json``.
        Expectation: The ratios file reaches the search as ``estimator["ratios_file"]``, where it
                     was dropped before.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            train_yaml = os.path.join(tmp_dir, "train.yaml")
            search_yaml = os.path.join(tmp_dir, "search.yaml")
            ratios = os.path.join(tmp_dir, "ratios.json")
            with open(train_yaml, "w", encoding="utf-8") as fh:
                fh.write("model:\n  name: test\n")
            with open(search_yaml, "w", encoding="utf-8") as fh:
                fh.write("parallelism:\n  dp: [1, 2]\n")
            with open(ratios, "w", encoding="utf-8") as fh:
                json.dump({"COMPUTE": 1.0}, fh)
            fake_config = SimpleNamespace(
                resolved_strategy=None, constraint={}, estimator={},
                cluster_spec={"num_nodes": 2, "cards_per_node": 8, "device_type": "A2"},
            )
            fake_result = {"dp": 1, "tp": 1, "pp": 1, "cp": 1, "ep": 1,
                           "micro_batch_num": 1, "memory_estimate_mb": 100.0, "score": 1.0}
            argv = ["run_nd.py", "-f", "hyper_v2", "-y", train_yaml, "-s", search_yaml, "-c", ratios, "-v", "0"]
            with patch(
                "hyper_parallel.auto_parallel.config_adapter.read_search_config",
                return_value=fake_config,
            ), \
                    patch("hyper_parallel.auto_parallel.config_adapter.validate", return_value=[]), \
                    patch(
                        "hyper_parallel.auto_parallel.config_adapter.search_strategies",
                        return_value=fake_result,
                    ) as mock_search, \
                    patch("hyper_parallel.auto_parallel.config_adapter.write_resolved_yaml"), \
                    patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit) as exc_info:
                    runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertEqual(exc_info.exception.code, 0)
            self.assertEqual(mock_search.call_count, 1)
            self.assertEqual(fake_config.estimator.get("ratios_file"), ratios,
                             f"estimator after -c: {fake_config.estimator}")

    def test_run_nd_cli_hyper_v2_d_refuses_part_of_a_node(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: ``-f hyper_v2 -s search.yaml`` with a ``-d`` that is not a whole number of
                     nodes: 8 cards on A3, whose node holds 16, and 12 where the config states 8.
        Expectation: The run stops with a usage error before any search: the search config counts
                     whole nodes and would price 16 devices for either.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            train_yaml = os.path.join(tmp_dir, "train.yaml")
            search_yaml = os.path.join(tmp_dir, "search.yaml")
            with open(train_yaml, "w", encoding="utf-8") as fh:
                fh.write("model:\n  name: test\n")
            with open(search_yaml, "w", encoding="utf-8") as fh:
                fh.write("parallelism:\n  dp: [1, 2]\n")

            for flags, cluster_spec, node in (
                    (["-A", "A3", "-d", "8"], {"num_nodes": 4, "device_type": "A2"}, 16),
                    (["-d", "12"], {"num_nodes": 4, "cards_per_node": 8, "device_type": "A2"}, 8),
            ):
                fake_config = SimpleNamespace(resolved_strategy=None, constraint={}, cluster_spec=cluster_spec)
                argv = ["run_nd.py", "-f", "hyper_v2", "-y", train_yaml, "-s", search_yaml, "-v", "0"] + flags
                with patch(
                    "hyper_parallel.auto_parallel.config_adapter.read_search_config",
                    return_value=fake_config,
                ), \
                        patch("hyper_parallel.auto_parallel.config_adapter.validate", return_value=[]), \
                        patch(
                            "hyper_parallel.auto_parallel.config_adapter.search_strategies",
                        ) as mock_search, \
                        patch("hyper_parallel.auto_parallel.config_adapter.write_resolved_yaml"), \
                        patch.object(sys, "argv", argv), \
                        patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    with self.assertRaises(SystemExit) as exc_info:
                        runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
                self.assertEqual(exc_info.exception.code, 2, f"{flags}: exit code {exc_info.exception.code}")
                self.assertIn(f"not a whole number of nodes of {node} cards", stderr.getvalue(),
                              f"{flags}: stderr was {stderr.getvalue()!r}")
                mock_search.assert_not_called()
                self.assertEqual(cluster_spec["num_nodes"], 4, f"{flags}: num_nodes {cluster_spec['num_nodes']}")

    def test_debug_rows_held_open_reach_the_file_when_the_block_ends(self) -> None:
        """
        Feature: Debug.holding_open.
        Description: Write two rows of one CSV inside the block, as a search writes a row a
            candidate, then one row of another CSV outside any block.
        Expectation: The held CSV is opened once and holds its titles and both rows once the block
            ends; the row outside a block is on disk at once; and the folder can be deleted, so no
            file is left open.
        """
        dims = Dim.Dimensions([(Dim.DP, 2), (Dim.MBS, 4)], all_dims=[Dim.DP, Dim.MBS])
        with tempfile.TemporaryDirectory() as tmp_dir:
            def row(name: str, total: int) -> Any:
                """A row of *total* for the CSV *name* in the folder."""
                debug = Debug.Debug(dims, [Debug.PerfParts.TOTAL])
                debug.output_file = os.path.join(tmp_dir, name)
                debug.info[Debug.PerfParts.TOTAL] = total
                return debug

            first, second, single = row("held.csv", 9), row("held.csv", 10), row("single.csv", 11)
            with patch("builtins.open", wraps=open) as opened:
                with Debug.holding_open():
                    first.write()
                    second.write()
            self.assertEqual(opened.call_count, 1)
            single.write()
            for name, rows in (("held.csv", (first, second)), ("single.csv", (single,))):
                with open(os.path.join(tmp_dir, name), encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), first.column_titles() + "".join(r.values() for r in rows))

    def test_a_run_starts_its_debug_csvs_afresh_and_its_workers_append(self) -> None:
        """
        Feature: Debug.set_output_dir.
        Description: An earlier run left debug.csv and debug_mem.csv in the output folder, beside
            another file. A new run sets the folder with fresh=True and writes a row, then a pool
            worker sets the same folder without it.
        Expectation: The run removes the two CSVs and nothing else, so its row sits under new
            column titles, and the worker's call keeps what the run wrote.
        """
        dims = Dim.Dimensions([(Dim.DP, 2), (Dim.MBS, 4)], all_dims=[Dim.DP, Dim.MBS])
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name in (*Debug.DEBUG_CSVS, "resolved.yaml"):
                with open(os.path.join(tmp_dir, name), "w", encoding="utf-8") as handle:
                    handle.write("from an earlier run\n")
            try:
                Debug.set_output_dir(tmp_dir, fresh=True)
                left = sorted(os.listdir(tmp_dir))
                debug = Debug.Debug(dims, [Debug.PerfParts.TOTAL])
                debug.info[Debug.PerfParts.TOTAL] = 9
                debug.write()
                Debug.set_output_dir(tmp_dir)
                with open(os.path.join(tmp_dir, "debug.csv"), encoding="utf-8") as handle:
                    written = handle.read()
            finally:
                Debug.set_output_dir(None)
        self.assertEqual(left, ["resolved.yaml"], f"files left by a fresh start: {left}")
        self.assertEqual(written, debug.column_titles() + debug.values(), f"debug.csv after the run: {written!r}")

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

    def test_classified_comparison_report_and_distances(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Format measured versus estimated parts and compute per-configuration distances.
        Expectation: Shares are over each side's total, idle is the measured remainder,
            and TOTAL keeps one distance per configuration.
        """
        dims = Dim.Dimensions([(Dim.DP, 4), (Dim.OP, 4)], all_dims=[Dim.DP, Dim.OP])
        # FW, BW, RECOMPUTE, DP, DP_REDUCE, MP, EP, CP, PP, BUBBLE: ND compute 6, DP 2 of
        # which 0.5 all-reduced, EP 1, PP + bubble 1.
        estimations = [2.0, 3.0, 1.0, 1.5, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0]
        real = {"comp": 5.0, "dp_wait": 2.0, "mp_wait": 0.0, "ep_wait": 1.0, "cp_wait": 0.0,
                "pp_wait": 1.0, "op_wait": 1.0, "sp_wait": 0.0}
        doubled = {part: 2 * value for part, value in real.items()}
        configs_estimated = [(dims, 1, 12.0, 10.0, estimations, real),
                             (dims, 1, 24.0, 10.0, estimations, doubled)]

        report = Debug.format_classified_comparison(configs_estimated[:1])
        rows = {line.split()[0]: line.split()[1:] for line in report.splitlines()[2:]}
        self.assertEqual(rows["comp"], ["5", "41.7%", "60.0%", "-18.3%"])
        self.assertEqual(rows["dp_wait"], ["3", "25.0%", "20.0%", "+5.0%"])
        self.assertEqual(rows["idle"], ["2", "16.7%", "-"])

        _, distances, _, total = Debug.correlation_with_classified_comms(configs_estimated)
        self.assertEqual(total, 2)
        self.assertEqual(len(distances[Debug.RealParts.TOTAL]), 2)
        self.assertAlmostEqual(Debug.get_distance_i(Debug.RealParts.TOTAL, ({}, distances, 0, total)),
                               sum(distances[Debug.RealParts.TOTAL]) / 2)

    def test_compare_with_csv_without_debug_logging(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Estimate the configurations of a classified CSV with debug logging off.
        Expectation: The per-part split is still collected, debug.csv is not written,
            and both the estimates and the metrics are returned.
        """
        runner = object.__new__(Par.ParallelizeLayer)
        runner.enable_debug = False
        runner.machine = SimpleNamespace(device=Hard.Device_A2, number=8)
        runner.config = SimpleNamespace(ccfg=SimpleNamespace(), set_parallel_config=lambda config: True)
        runner.mem_eval = SimpleNamespace(ccfg=runner.config.ccfg)
        runner.memory_estim = lambda debugger=None: 32

        def fake_estimate(ccfg: Any, debugger: Any = None, **_kwargs: Any) -> float:
            """Fill two parts the way estimate_performance does and return the score."""
            debugger.info[Debug.PerfParts.FW_COMPUTE] = 6.0
            debugger.info[Debug.PerfParts.DP_COMM] = 4.0
            return 10.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "real.csv")
            with open(csv_path, "w", encoding="utf-8") as csv_file:
                csv_file.write("DP,OP,time,comp,dp_wait,mp_wait,ep_wait,cp_wait,pp_wait\n4,4,12,5,6,0,0,0,0\n")
            with patch.object(Par, "estimate_performance", side_effect=fake_estimate), \
                    patch.object(Debug.Debug, "write") as write:
                configs_estimated, metrics = runner.compare_with_csv(csv_path)

        write.assert_not_called()
        self.assertEqual(len(configs_estimated), 1)
        _, _, real_time, score, values, _ = configs_estimated[0]
        self.assertEqual((real_time, score), (12.0, 10.0))
        self.assertEqual(values[Debug.PerfParts.DP_COMM.value - 1], 4.0)
        self.assertEqual(metrics[3], 1)

    def test_run_nd_cli_real_csv_compares_instead_of_searching(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Run the run_nd CLI with --real_csv on a fake Parallelize.
        Expectation: The runner compares with the CSV and exits without searching;
            a missing CSV is rejected before any runner is built.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            _FakeParallelize.instances = []
            real_csv = os.path.join(tmp_dir, "real.csv")
            with open(real_csv, "w", encoding="utf-8") as csv_file:
                csv_file.write("DP,time,comp,dp_wait\n8,10,6,4\n")
            out_dir = os.path.join(tmp_dir, "cmp")
            argv = ["run_nd.py", "-y", config_path, "-d", "8", "-v", "0", "--real_csv", real_csv, "-o", out_dir]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as done:
                runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertEqual(done.exception.code, 0)
            instance = _FakeParallelize.instances[-1]
            self.assertEqual(instance.compare_args, (real_csv, out_dir, True))
            self.assertIsNone(instance.last_run_kwargs())
            self.assertTrue(os.path.isdir(out_dir))

            argv[argv.index(real_csv)] = os.path.join(tmp_dir, "missing.csv")
            _FakeParallelize.instances = []
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as missing:
                runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            self.assertEqual(missing.exception.code, 2)
            self.assertEqual(_FakeParallelize.instances, [])

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

    def test_top_ranks_match_cli_and_plot_with_exhaustive_additions(self) -> None:
        """Test CLI and plot TOP positions retain gaps from the full ND ranking."""
        dimensions = [Dim.DP, Dim.EP]
        degrees = [(1, 1), (2, 1), (4, 1), (2, 2), (8, 1), (16, 1), (4, 4)]
        scored_space = [
            (
                Dim.Dimensions([(Dim.DP, dp), (Dim.EP, ep)], all_dims=dimensions),
                128,
                float(rank),
                [float(rank)],
            )
            for rank, (dp, ep) in enumerate(degrees, start=1)
        ]
        runner = object.__new__(Par.ParallelizeLayer)
        runner.enable_debug = True
        runner.model_name = "unit"
        runner.global_batch_size = 16
        runner.machine = SimpleNamespace(device=Hard.Device_A2, number=16)
        runner.config = SimpleNamespace(
            balancing=SimpleNamespace(from_config=False), dimensions=dimensions,
        )
        runner.generate_search_space = lambda folder, threads_num: [entry[:2] for entry in scored_space]
        runner.order_search_space = lambda space, threads_num, cache_file: (
            scored_space, [Debug.PerfParts.FW_COMPUTE],
        )
        # The search first checks that a batch is reachable, and notes what the
        # priced config left unstated, neither of which this runner's config holds.
        runner.batch_reachable = lambda: True
        runner.priced = SimpleNamespace
        cases = [
            ({}, ["1", "2"], True),
            ({"exhaustive": 2}, ["1", "2", "4", "7"], True),
            ({"force_exhaustive": 2}, ["1", "2", "4", "7"], True),
            ({"fforce_exhaustive": 2}, ["1", "2", "4", "7"], True),
            ({"fforce_exhaustive": 2}, ["1", "2", "4", "7"], False),
        ]

        for options, expected_ranks, enable_debug in cases:
            runner.enable_debug = enable_debug
            with self.subTest(options=options, enable_debug=enable_debug), \
                    tempfile.TemporaryDirectory() as tmp_dir, \
                    patch.object(Debug, "output_dir", return_value=tmp_dir), \
                    patch.object(Debug.Plot, "close"), \
                    patch.object(Par.logger, "output") as cli_output:
                try:
                    result = runner.run_generation_to_ordering(None, top_num=2, dimensions=[Dim.EP], **options)
                    self.assertEqual(result, scored_space)
                    self.assertEqual([entry[0].rank for entry in result], list(range(1, 8)))
                    cli_tables = [
                        call.args[0] for call in cli_output.call_args_list
                        if "Performance score" in call.args[0]
                    ]
                    for text in cli_tables:
                        self.assertEqual(text.splitlines()[1].split()[0], "TOP")
                    self.assertEqual(
                        [line.split()[0] for text in cli_tables for line in text.splitlines()[2:] if line.strip()],
                        expected_ranks,
                    )
                    if enable_debug:
                        table = plt.gca().tables[0]
                        self.assertEqual(table[1, -1].get_text().get_text(), "TOP")
                        self.assertEqual(
                            [table[1, column].get_text().get_text() for column in range(len(expected_ranks))],
                            expected_ranks,
                        )
                finally:
                    plt.close("all")

    def test_exhaustive_plot_space_precedence_and_top_up(self) -> None:
        """Test exhaustive counts and numeric force-exhaustive precedence."""
        values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        scored_space = [
            (
                Dim.Dimensions([(Dim.DP, value)], all_dims=[Dim.DP]),
                128,
                float(index),
                [],
            )
            for index, value in enumerate(values, start=1)
        ]
        cases = [
            # (top M, exhaustive, force exhaustive, additions, total qualifying)
            (3, 5, 2, 3, 5),
            (3, 2, 2, 2, 4),
            (3, 2, 4, 4, 6),
            (8, 5, 2, 2, 9),
            (3, 4, 0, 2, 4),
        ]

        for top_num, exhaustive, force_exhaustive, expected_additions, expected_total in cases:
            with self.subTest(
                top_num=top_num,
                exhaustive=exhaustive,
                force_exhaustive=force_exhaustive,
            ):
                normal_plot = Debug.top_plot_configs(scored_space, top_num)
                plot_space, additions = Par._exhaustive_plot_space(
                    scored_space,
                    [Dim.DP],
                    top_num,
                    exhaustive,
                    force_exhaustive,
                )
                qualifying_total = sum(
                    1
                    for config in plot_space
                    if config[0].has_dim(Dim.DP) and config[0].val(Dim.DP) > 1
                )

                self.assertEqual(len(additions), expected_additions)
                self.assertEqual(qualifying_total, expected_total)
                self.assertLessEqual(len(normal_plot), top_num)

        multi_dim_space = [
            (
                Dim.Dimensions(
                    [(Dim.DP, dp), (Dim.OP, op)],
                    all_dims=[Dim.DP, Dim.OP],
                ),
                128,
                float(index),
                [],
            )
            for index, (dp, op) in enumerate(
                [(1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (1, 4)],
                start=1,
            )
        ]
        plot_space, additions = Par._exhaustive_plot_space(
            multi_dim_space,
            [Dim.DP, Dim.OP],
            top_num=2,
            exhaustive=2,
            force_exhaustive=0,
        )
        for dimension in (Dim.DP, Dim.OP):
            qualifying_total = sum(
                1
                for config in plot_space
                if config[0].has_dim(dimension) and config[0].val(dimension) > 1
            )
            self.assertGreaterEqual(qualifying_total, 2)
        self.assertEqual(len(additions), 2)

    def test_exhaustive_plot_includes_candidates_past_standard_score_cutoff(self) -> None:
        """Test exhaustive candidates survive the normal 20x score cutoff."""
        scores = [1.0, 2.0, 3.0, 21.0]
        values = [1, 2, 4, 8]
        scored_space = [
            (
                Dim.Dimensions([(Dim.DP, value)], all_dims=[Dim.DP]),
                128,
                score,
                [],
            )
            for value, score in zip(values, scores)
        ]
        plot_space, additions = Par._exhaustive_plot_space(
            scored_space,
            [Dim.DP],
            top_num=10,
            exhaustive=3,
            force_exhaustive=0,
        )
        debug_parts = [Debug.PerfParts.FW_COMPUTE]

        standard_plot = Debug.Plot("unit", [Dim.DP], debug_parts, top=len(plot_space))
        standard_plot.parse_data(plot_space)
        exhaustive_plot = Debug.Plot("unit", [Dim.DP], debug_parts, top=len(plot_space))
        exhaustive_plot.parse_data(plot_space, include_all=True)

        self.assertEqual(len(additions), 1)
        self.assertEqual(len(standard_plot.data), 3)
        self.assertEqual(len(exhaustive_plot.data), 4)

    def test_fforce_exhaustive_adds_distinct_results_per_dimension(self) -> None:
        """Test each requested dimension receives distinct additional configs."""
        dimensions = [Dim.DP, Dim.EP, Dim.TP, Dim.CP, Dim.OP]
        candidate_values = [
            {Dim.DP: 2, Dim.EP: 2},
            {Dim.DP: 4, Dim.EP: 4},
            {Dim.EP: 8},
            {Dim.EP: 16},
            {Dim.TP: 2},
            {Dim.TP: 4},
            {Dim.CP: 2},
            {Dim.CP: 4},
            {Dim.OP: 2},
            {Dim.OP: 4},
        ]
        scored_space = [
            (
                Dim.Dimensions([(dimension, 1) for dimension in dimensions], all_dims=dimensions),
                128,
                1.0,
                [],
            )
        ]
        scored_space.extend(
            (
                Dim.Dimensions(
                    [
                        (dimension, values.get(dimension, 1))
                        for dimension in dimensions
                    ],
                    all_dims=dimensions,
                ),
                128,
                float(index),
                [],
            )
            for index, values in enumerate(candidate_values, start=2)
        )

        plot_space, additions = Par._fforce_exhaustive_plot_space(
            scored_space,
            dimensions,
            top_num=1,
            fforce_exhaustive=2,
        )
        addition_ids = {entry[0].unique_name() for entry in additions}

        self.assertEqual(len(additions), 10)
        self.assertEqual(len(addition_ids), 10)
        self.assertEqual(len(plot_space), 11)
        for dimension in dimensions:
            qualifying = sum(
                1
                for entry in additions
                if entry[0].has_dim(dimension) and entry[0].val(dimension) > 1
            )
            self.assertGreaterEqual(qualifying, 2)

    def test_a_comparison_writes_a_plot_without_idle_and_its_estimates(self) -> None:
        """
        Feature: the files run_nd --real_csv writes beside its plot.
        Description: Two measured configurations whose order by step and by
            step less idle differ, compared with idle.
        Expectation: A second plot without idle, ordered by step less idle, and a
            CSV of ND's estimate of each configuration with its memory.
        """
        runner = object.__new__(Par.ParallelizeLayer)
        runner.enable_debug = False
        runner.model_name = "unit"
        runner.global_batch_size = 8
        runner.machine = SimpleNamespace(device=Hard.Device_A2, number=8)
        runner.config = SimpleNamespace(ccfg=SimpleNamespace(), set_parallel_config=lambda config: True)
        runner.mem_eval = SimpleNamespace(ccfg=runner.config.ccfg)
        runner.memory_estim = lambda debugger=None: 2048

        def fake_estimate(ccfg: Any, debugger: Any = None, **_kwargs: Any) -> float:
            """Fill one part the way estimate_performance does and return the score."""
            debugger.info[Debug.PerfParts.DP_COMM] = 4.0
            return 10.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "real.csv")
            with open(csv_path, "w", encoding="utf-8") as csv_file:
                # OP 2: step 12, idle 10. OP 4: step 10, idle 1.
                csv_file.write("DP,OP,time,comp,dp_wait\n8,2,12,1,1\n8,4,10,5,4\n")
            with patch.object(Par, "estimate_performance", side_effect=fake_estimate), \
                    patch.object(Debug, "plot_vs_real_comm_classified") as plot:
                configs_estimated, _ = runner.compare_with_csv(csv_path, output_path=tmp_dir, plot_idle=True)
            with open(os.path.join(tmp_dir, "real_estimates.csv"), newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual([entry[0].val(Dim.OP) for entry in configs_estimated], [4, 2])
        (with_idle, first_kwargs), (without_idle, second_kwargs) = plot.call_args_list
        self.assertEqual((first_kwargs["plot_idle"], first_kwargs.get("suffix", "")), (True, ""))
        self.assertEqual((second_kwargs["plot_idle"], second_kwargs["suffix"]), (False, "_no_idle"))
        self.assertEqual([entry[0].val(Dim.OP) for entry in with_idle[0]], [4, 2])
        self.assertEqual([entry[0].val(Dim.OP) for entry in without_idle[0]], [2, 4])
        self.assertEqual(rows[0][:5], ["DP", "OP", "time", "memory_mb", "score"])
        self.assertEqual([row[:5] for row in rows[1:]], [["8", "4", "10.0", "2048", "10.0"],
                                                         ["8", "2", "12.0", "2048", "10.0"]])
        self.assertEqual(rows[1][5 + Debug.PerfParts.DP_COMM.value - 1], "4.0")

    def test_a_comparison_leaves_out_what_nd_cannot_cost(self) -> None:
        """
        Feature: run_nd --real_csv on a strategy the cost model cannot represent.
        Description: Two measured configurations, one refused the way the MoE parser
            refuses expert parallelism wider than its stage; then both refused.
        Expectation: The other is still estimated and the refused one is named; a
            CSV none of whose configurations ND can cost is an error.
        """
        refused = {2}

        def set_parallel_config(config: Any) -> bool:
            """Refuse the configurations whose OP is in *refused*."""
            if config.val(Dim.OP) in refused:
                raise TypeError("MoE parsing error: d_exp(0)/t_exp(1)")
            return True

        runner = object.__new__(Par.ParallelizeLayer)
        runner.enable_debug = False
        runner.machine = SimpleNamespace(device=Hard.Device_A2, number=8)
        runner.config = SimpleNamespace(ccfg=SimpleNamespace(), set_parallel_config=set_parallel_config)
        runner.mem_eval = SimpleNamespace(ccfg=runner.config.ccfg)
        runner.memory_estim = lambda debugger=None: 32
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "real.csv")
            with open(csv_path, "w", encoding="utf-8") as csv_file:
                csv_file.write("DP,OP,time,comp,dp_wait\n8,2,12,1,1\n8,4,10,5,4\n")
            with patch.object(Par, "estimate_performance", return_value=10.0), \
                    patch.object(Par.logger, "output") as output:
                configs_estimated, _ = runner.compare_with_csv(csv_path)
                self.assertEqual([entry[0].val(Dim.OP) for entry in configs_estimated], [4])
                self.assertIn("MoE parsing error", str(output.call_args_list))
                refused.add(4)
                with self.assertRaises(ValueError):
                    runner.compare_with_csv(csv_path)

    def test_the_measured_stack_adds_up_to_the_step(self) -> None:
        """
        Feature: the measured bars of ND's comparison plot.
        Description: A classified row with FSDP, sequence-parallel, CP and pipeline waits.
        Expectation: FSDP counts as DP and sequence parallel as MP, CP has a bar of
            its own, and with idle the stack adds up to the measured step.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "real.csv")
            with open(csv_path, "w", encoding="utf-8") as csv_file:
                csv_file.write("DP,EP,time,comp,dp_wait,mp_wait,ep_wait,cp_wait,pp_wait,op_wait,sp_wait\n")
                csv_file.write("8,2,100,30,5,4,6,7,3,20,2\n")
            dims, time, waits = Debug.get_comm_classified_data(csv_path, plot_idle=True)[0]
        bars = Debug.measured_bars(waits, plot_idle=True)
        self.assertEqual(bars, [30.0, 25.0, 6.0, 6.0, 7.0, 3.0, 23.0])
        self.assertAlmostEqual(sum(bars), time)
        self.assertEqual(len(Debug.measured_bars(waits)), len(Debug.MEASURED_BARS))
        plot = Debug.Plot("unit", dims.keys(), [Debug.PerfParts.FW_COMPUTE])
        real_data = []
        plot.parse_data([(dims, 1, time, 90.0, [90.0], waits)], real_data=real_data)
        self.assertEqual(real_data, [tuple(bars[:-1])])

    def test_the_degree_table_reads_a_boolean_dimension(self) -> None:
        """
        Feature: the degree table under ND's plot of a search.
        Description: A search that varied SP, whose degrees print as True or False.
        Expectation: The table is drawn instead of failing on the boolean cell.
        """
        dims = Dim.Dimensions([(Dim.DP, 8), (Dim.SP, True)], all_dims=[Dim.DP, Dim.SP])
        plot = Debug.Plot("unit", dims.keys(), [Debug.PerfParts.FW_COMPUTE])
        plot.parse_data([(dims, 128, 10.0, [10.0])])
        figure = plt.figure()
        try:
            plot.make_table()
        finally:
            plt.close(figure)
        self.assertEqual(plot.cell_text, [["8"], ["True"], [128]])

    def test_the_plot_states_the_recompute_mode_and_the_plan_per_layer(self) -> None:
        """
        Feature: ND's plot of a search that chose a recompute mode per layer.
        Description: One configuration whose plan fully recomputes every
            layer but its last five, then the same plot without a plan, then
            a configuration of a search with a recompute dimension.
        Expectation: The table gains a mode row and a plan row, the mode in
            capitals and the plan as the trainer's yaml states it; without a
            plan neither row is drawn; a configuration that states a mode of
            its own shows the mode row alone.
        """
        dims = Dim.Dimensions([(Dim.DP, 8), (Dim.PP, 2)], all_dims=[Dim.DP, Dim.PP])
        configs = [(dims, 128, 10.0, [10.0])]
        drawn = []
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Debug.Plot, "make_table",
                             lambda self: drawn.append((list(self.row_title), list(self.cell_text)))), \
                patch.object(Debug.Plot, "close", lambda self, output_path, filename: None):
            Debug.plot_nd(configs, tmp_dir, [Debug.PerfParts.FW_COMPUTE],
                          recompute_plans={dims: ("full", {"3-7": "off"})})
            Debug.plot_nd(configs, tmp_dir, [Debug.PerfParts.FW_COMPUTE])
            searched = Dim.Dimensions([(Dim.DP, 8), (Dim.PP, 2)], all_dims=[Dim.DP, Dim.PP],
                                      recompute="selective")
            Debug.plot_nd([(searched, 128, 10.0, [10.0])], tmp_dir, [Debug.PerfParts.FW_COMPUTE])
        plt.close("all")
        self.assertEqual([str(row) for row in drawn[0][0]],
                         ["TOP", "DP", "PP", Debug.MODE_ROW, Debug.LAYER_ROW, "MEM"])
        self.assertEqual(drawn[0][1], [[1, "8", "2", "FULL", "3-7: 'off'", 128]])
        self.assertEqual([str(row) for row in drawn[1][0]], ["TOP", "DP", "PP", "MEM"])
        self.assertEqual(drawn[1][1], [[1, "8", "2", 128]])
        self.assertEqual([str(row) for row in drawn[2][0]], ["TOP", "DP", "PP", Debug.MODE_ROW, "MEM"])
        self.assertEqual(drawn[2][1], [[1, "8", "2", "SELECTIVE", 128]])

    def test_the_degree_table_is_drawn_with_a_plan_of_several_lines(self) -> None:
        """
        Feature: the degree table under ND's plot of a search.
        Description: A plan naming two ranges, which the plan row states on
            one line each.
        Expectation: The table is drawn, the plan row holding both lines.
        """
        dims = Dim.Dimensions([(Dim.DP, 8), (Dim.PP, 2)], all_dims=[Dim.DP, Dim.PP])
        plot = Debug.Plot("unit", dims.keys(), [Debug.PerfParts.FW_COMPUTE])
        plot.row_title.insert(-1, Debug.MODE_ROW)
        plot.row_title.insert(-1, Debug.LAYER_ROW)
        plot.parse_data([(dims, 128, 10.0, [10.0])],
                        recompute_plans={dims: ("full", {"0": "selective", "3-7": "off"})})
        figure = plt.figure()
        try:
            plot.make_table()
        finally:
            plt.close(figure)
        self.assertEqual(plot.cell_text[-3], ["FULL"])
        self.assertEqual(plot.cell_text[-2], ["'0': selective\n3-7: 'off'"])

    def test_the_recompute_plot_is_written_for_every_configuration(self) -> None:
        """
        Feature: Debug.plot_recompute.
        Description: Two configurations, the first with two plans and the
            second with none, then an empty selection, a non-positive page
            size and a record missing a score component.
        Expectation: A PDF is written for the configurations; nothing is
            written for an empty selection, and the other two are refused.
        """
        parts = [Debug.PerfParts.FW_COMPUTE, Debug.PerfParts.DP_COMM]
        first = Dim.Dimensions([(Dim.DP, 8), (Dim.PP, 2)], all_dims=[Dim.DP, Dim.PP])
        second = Dim.Dimensions([(Dim.DP, 4), (Dim.PP, 4)], all_dims=[Dim.DP, Dim.PP])
        first.rank, second.rank = 1, 2
        records = [
            {"score": 10.0, "parts": (6.0, 4.0), "memory": 1024.0, "mode": "full",
             "layers": {"3-7": "off"}, "full_layers": 3},
            {"score": 11.0, "parts": (7.0, 4.0), "memory": 1000.0, "mode": "full",
             "layers": {}, "full_layers": 8},
        ]
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                Debug.plot_recompute([(first, records), (second, [])], tmp_dir, parts, "recompute",
                                     title="unit", columns_per_page=1)
                written = os.path.join(tmp_dir, "recompute.pdf")
                self.assertTrue(os.path.isfile(written))
                self.assertGreater(os.path.getsize(written), 1000)
                Debug.plot_recompute([], tmp_dir, parts, "empty")
                self.assertFalse(os.path.exists(os.path.join(tmp_dir, "empty.pdf")))
                with self.assertRaises(ValueError):
                    Debug.plot_recompute([(first, records)], tmp_dir, parts, "bad", columns_per_page=0)
                with self.assertRaises(ValueError):
                    Debug.plot_recompute([(first, [dict(records[0], parts=(6.0,))])], tmp_dir, parts, "short")
        finally:
            plt.close("all")

    def test_the_ranking_keeps_nd_order_and_every_digit(self) -> None:
        """
        Feature: the CSV of a search's configurations in ND's order.
        Description: Two configurations, one with its score split into parts and
            one without, with and without ranks from the full search.
        Expectation: Rows keep stored global ranks or fall back to their input
            positions, scores keep full precision, and missing parts are blank.
        """
        first = Dim.Dimensions([(Dim.DP, 8), (Dim.SP, False), (Dim.OP, 4)],
                               all_dims=[Dim.DP, Dim.SP, Dim.OP])
        second = Dim.Dimensions([(Dim.DP, 4), (Dim.SP, True), (Dim.OP, 2)],
                                all_dims=[Dim.DP, Dim.SP, Dim.OP])
        parts = [str(part) for part in Debug.PerfParts][:-2]
        for ranks, expected in (((None, None), ("1", "2")), ((4, 7), ("4", "7"))):
            first.rank, second.rank = ranks
            with self.subTest(ranks=ranks), tempfile.TemporaryDirectory() as tmp_dir:
                path = os.path.join(tmp_dir, "nested", "ranking.csv")
                Debug.write_ranking_csv(
                    [(first, 100, 91599458344632.31, [1.5] * len(parts)), (second, 120, 1e14, [])], path)
                with open(path, newline="", encoding="utf-8") as handle:
                    rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["rank", "DP", "SP", "OP", "memory_mb", "score"] + parts)
            self.assertEqual(rows[1][:6], [expected[0], "8", "False", "4", "100", "91599458344632.31"])
            self.assertEqual(float(rows[1][5]), 91599458344632.31)
            self.assertEqual(rows[1][6:], ["1.5"] * len(parts))
            self.assertEqual(rows[2][:6], [expected[1], "4", "True", "2", "120", "100000000000000.0"])
            self.assertEqual(rows[2][6:], [""] * len(parts))

    def test_the_ranking_is_written_before_the_plot(self) -> None:
        """
        Feature: a search asked to write its ranking.
        Description: The search orders one configuration and its plot then fails.
        Expectation: The ranking is on disk all the same.
        """
        dims = Dim.Dimensions([(Dim.DP, 8), (Dim.OP, 4)], all_dims=[Dim.DP, Dim.OP])
        runner = object.__new__(Par.ParallelizeLayer)
        runner.enable_debug = True
        runner.auto_recompute = False
        runner.model_name = "unit"
        runner.global_batch_size = 8
        runner.machine = SimpleNamespace(device=Hard.Device_A2, number=8)
        runner.config = SimpleNamespace(balancing=SimpleNamespace(from_config=False),
                                        dimensions=[Dim.DP, Dim.OP])
        runner.batch_reachable = lambda: True
        runner.priced = SimpleNamespace
        runner.generate_search_space = lambda folder, threads_num: [(dims, 100)]
        runner.order_search_space = lambda space, threads_num, cache_file: ([(dims, 100, 2.5, [])], [])
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Debug, "plot_nd", side_effect=ValueError("plot")), \
                patch.object(Debug, "output_dir", return_value=tmp_dir):
            path = os.path.join(tmp_dir, "ranking.csv")
            with self.assertRaises(ValueError):
                runner.run_generation_to_ordering(None, ranking_csv=path)
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[1][:5], ["1", "8", "4", "100", "2.5"])

    def test_run_nd_cli_passes_exhaustive_options_to_the_search(self) -> None:
        """Test short and long exhaustive CLI options reach the search runner."""
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            option_forms = [
                (["-e", "5", "-ee", "2"], 5, 2, 0),
                (["--exhaustive", "5", "--force_exhaustive", "2"], 5, 2, 0),
                (["-eee", "2"], 0, 0, 2),
                (["--fforce-exhaustive", "2"], 0, 0, 2),
            ]
            for options, exhaustive, force_exhaustive, fforce_exhaustive in option_forms:
                _FakeParallelize.instances = []
                argv = ["run_nd.py", "-y", config_path, "-d", "8", "-v", "0"] + options
                with patch.object(sys, "argv", argv):
                    runpy.run_module(
                        "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd",
                        run_name="__main__",
                    )

                run_kwargs = _FakeParallelize.instances[-1].last_run_kwargs()
                self.assertEqual(run_kwargs["exhaustive"], exhaustive)
                self.assertEqual(run_kwargs["force_exhaustive"], force_exhaustive)
                self.assertEqual(run_kwargs["fforce_exhaustive"], fforce_exhaustive)

    def test_run_nd_cli_asks_for_the_recompute_potentials_only_when_told_to(self) -> None:
        """
        Feature: run_nd -rp/--recompute_potentials.
        Description: The CLI search on a fake Parallelize, with the short and
            the long form of the flag and without it.
        Expectation: The search is asked for the report only with the flag,
            so a plain search does none of that work.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            argv = ["run_nd.py", "-y", config_path, "-d", "8", "-v", "2", "-ar", "--recompute", "off", "full"]
            for extra, expected in (([], False), (["-rp"], True), (["--recompute_potentials"], True)):
                with patch.object(sys, "argv", argv + extra):
                    runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
                run_kwargs = _FakeParallelize.instances[-1].last_run_kwargs()
                self.assertEqual(run_kwargs["recompute_potentials"], expected)

    def test_run_nd_cli_refuses_the_recompute_potentials_without_a_plan_per_layer(self) -> None:
        """
        Feature: run_nd -rp/--recompute_potentials.
        Description: The flag without -ar, without the modes each layer
            chooses among, below the verbosity that writes files, and with
            --real_csv.
        Expectation: Each is refused, rather than running a longer search
            that writes nothing or extends a plan that does not exist.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            real_csv = os.path.join(tmp_dir, "real.csv")
            with open(real_csv, "w", encoding="utf-8") as handle:
                handle.write("DP,time\n8,100\n")
            refused = [
                ["-rp", "--recompute", "off", "full"],
                ["-rp", "-ar"],
                ["-rp", "-ar", "--recompute", "off", "full", "-v", "1"],
                ["-rp", "-ar", "--recompute", "off", "full", "--real_csv", real_csv],
            ]
            for extra in refused:
                argv = ["run_nd.py", "-y", config_path, "-d", "8", "-v", "2"] + extra
                with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as refusal:
                    runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
                self.assertEqual(refusal.exception.code, 2, extra)

    def test_run_nd_cli_passes_the_ranking_path_to_the_search(self) -> None:
        """
        Feature: run_nd --ranking_csv.
        Description: Run the CLI search on a fake Parallelize with and without the flag.
        Expectation: The search receives the path, and None when the flag is absent.
        """
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(Par, "Parallelize", _FakeParallelize), \
                patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}):
            ranking = os.path.join(tmp_dir, "ranking.csv")
            argv = ["run_nd.py", "-y", config_path, "-d", "8", "-v", "0"]
            for extra, expected in (([], None), (["--ranking_csv", ranking], ranking)):
                with patch.object(sys, "argv", argv + extra):
                    runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
                self.assertEqual(_FakeParallelize.instances[-1].last_run_kwargs()["ranking_csv"], expected)

    def test_a_field_no_parser_set_is_reported_or_refused(self) -> None:
        """
        Feature: CostModelConfig.__getattr__, unset_reads_report and run_nd --strict.
        Description: A config no parser filled, read directly for an op count
            twice and for its hooks, then through getattr with a default, with
            and without the strict mode run_nd --strict sets.
        Expectation: By default each read is 0 and the op count is recorded
            with its reader and reported once, in one line, the hooks being
            exempt; under --strict the direct read is refused by name and
            reader, while getattr gets its stated default; run_nd sets the
            mode from its flag.
        """
        ccfg = object.__new__(CostModelConfig)
        with patch.object(PreProcess, "UNSET_READS", PreProcess.defaultdict(PreProcess.Counter)), \
                patch.dict(PreProcess._STRICT, {"on": False}):  # pylint: disable=protected-access
            self.assertEqual((ccfg.n_gather, ccfg.n_gather, ccfg.hooks_dict), (0, 0, 0))
            lines = PreProcess.unset_reads_report()
            self.assertEqual(len(lines), 1)
            self.assertIn("1 config field(s)", lines[0])
            self.assertIn("n_gather by test_run_nd.py:test_a_field_no_parser_set_is_reported_or_refused", lines[0])
            self.assertEqual(PreProcess.unset_reads_report(), [])
            PreProcess.set_strict(True)
            with self.assertRaisesRegex(AttributeError, "reads n_gather, which no parser set"):
                _ = ccfg.n_gather
            self.assertEqual(getattr(ccfg, "n_gather", 5), 5)
            for flag, strict in (([], False), (["--strict"], True)):
                with tempfile.TemporaryDirectory() as tmp_dir, patch.object(Par, "Parallelize", _FakeParallelize), \
                        patch.dict(os.environ, {"MPLCONFIGDIR": tmp_dir}), \
                        patch.object(sys, "argv", ["run_nd.py", "-y", config_path, "-d", "8", "-v", "0"] + flag):
                    runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
                self.assertEqual(PreProcess._STRICT["on"], strict)  # pylint: disable=protected-access

    def test_arch_hook_variants(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Apply each family to tiny fake configs.
        Expectation: Each family's profile sets its fields, and each stack kind applies per layer.
        """
        cfg = _make_arch_cfg(arch="default")
        ArchHooks.apply_family(cfg)
        self.assertEqual(cfg.n_attMM, 4)
        cfg.arch = "llama2"
        ArchHooks.apply_family(cfg)
        self.assertEqual(cfg.bytes_grad, 2)
        cfg.arch = "mixtral"
        ArchHooks.apply_family(cfg)
        self.assertEqual((cfg.bytes_grad, cfg.hff), (2, 32))
        cfg.arch = "pangualpha"
        ArchHooks.apply_family(cfg)
        self.assertEqual((cfg.n_normOp, cfg.bytes_dropout), (4, 1))
        cfg.arch = "qwen"
        ArchHooks.apply_family(cfg)
        self.assertEqual((cfg.shard_recompute_input, cfg.shard_output_activ), (cfg.t, cfg.t))

        t5_cfg = _make_arch_cfg(arch="t5", n_lay=4, n_mtp=0)
        t5_cfg.layer_stack = resolve_layers("t5", derive_layers(load_op_profile("t5"), 4))
        ArchHooks.apply_family(t5_cfg)
        ArchHooks.bind_layer_stack(t5_cfg)
        t5_groups = ArchHooks.layer_groups(t5_cfg)
        self.assertEqual(len(t5_groups), 2)
        t5_wrap = ArchHooks.CWrap(t5_cfg)
        ArchHooks.apply_layer_kind(t5_wrap, t5_groups[0][0])
        self.assertEqual(t5_cfg.n_attBMM, 1)
        ArchHooks.apply_layer_kind(t5_wrap, t5_groups[1][0])
        self.assertEqual(t5_cfg.n_attMM, 8)

        deepseek_cfg = _make_arch_cfg(arch="deepseek")
        deepseek_cfg.layer_stack = resolve_layers(
            "deepseek", derive_layers(load_op_profile("deepseek"), 4, 1, first_k_dense=2)
        )
        ArchHooks.apply_family(deepseek_cfg)
        self.assertEqual(deepseek_cfg.dh, 128)
        ArchHooks.bind_layer_stack(deepseek_cfg)
        deepseek_groups = ArchHooks.layer_groups(deepseek_cfg)
        self.assertEqual(len(deepseek_groups), 3)
        deepseek_wrap = ArchHooks.CWrap(deepseek_cfg)
        ArchHooks.apply_layer_kind(deepseek_wrap, deepseek_groups[0][0])
        self.assertEqual(deepseek_cfg.n_exp, 1)
        ArchHooks.apply_layer_kind(deepseek_wrap, deepseek_groups[1][0])
        self.assertEqual(deepseek_cfg.n_exp, 4)

    def test_t5_states_the_tp_gathers_its_embedding_and_output_run(self) -> None:
        """
        Feature: apply_family on t5, the model's own config.
        Description: A T5 config through its family, then each kind of its
            stack applied in turn.
        Expectation: The model states the 4 TP gathers of its default kind,
            the encoder, which the communication walk prices the embedding
            and the output layer with, where it stated none and they read 0;
            the encoder's and the decoder's layers keep their 4 and 6.
        """
        t5_cfg = _make_arch_cfg(arch="t5", n_lay=4, n_mtp=0)
        t5_cfg.layer_stack = resolve_layers("t5", derive_layers(load_op_profile("t5"), 4))
        ArchHooks.apply_family(t5_cfg)
        ArchHooks.bind_layer_stack(t5_cfg)
        self.assertEqual(vars(t5_cfg).get("n_gather"), 4)
        wrap = ArchHooks.CWrap(t5_cfg)
        gathers = []
        for kind, _ in ArchHooks.layer_groups(t5_cfg):
            ArchHooks.apply_layer_kind(wrap, kind)
            gathers.append(t5_cfg.n_gather)
        self.assertEqual(gathers, [4, 6])

    def test_the_pipeline_takes_the_all_reduce_out_of_dp(self) -> None:
        """
        Feature: estimate_pipeline, PerfParts.DP_REDUCE.
        Description: Two stages whose DP times of 8 and 10 hold all-reduces of
            2 and 5, priced without and with the all-reduce recorded.
        Expectation: The same time and bubble either way; the all-reduce is the
            straggler stage's times the two micro-batches, and it and the split
            run's DP_COMM add up to the plain run's DP_COMM.
        """
        cfg = _make_perf_cfg(p=2, vp=1, m=2)
        runs = []
        for reduces in (None, [2.0, 5.0]):
            debugger = Debug.Debug(Dim.Dimensions([(Dim.DP, 2)], all_dims=[Dim.DP]), Debug.PerfParts)
            for part in (Debug.PerfParts.FW_COMPUTE, Debug.PerfParts.BW_COMPUTE, Debug.PerfParts.RECOMPUTE,
                         Debug.PerfParts.MP_COMM, Debug.PerfParts.EP_COMM, Debug.PerfParts.CP_COMM):
                debugger.info[part] = [1.0, 2.0]
            debugger.info[Debug.PerfParts.DP_COMM] = [8.0, 10.0]
            if reduces is not None:
                debugger.info[Debug.PerfParts.DP_REDUCE] = reduces
            time = PerfEstimate.estimate_pipeline(cfg, [12.0, 18.0], debugger=debugger)
            runs.append((time, dict(debugger.info)))
        plain_time, plain = runs[0]
        split_time, split = runs[1]
        self.assertEqual(split_time, plain_time)
        self.assertEqual(split[Debug.PerfParts.BUBBLE], plain[Debug.PerfParts.BUBBLE])
        self.assertEqual(plain[Debug.PerfParts.DP_REDUCE], 0)
        self.assertEqual(split[Debug.PerfParts.DP_REDUCE], 10.0)
        self.assertEqual(split[Debug.PerfParts.DP_COMM] + split[Debug.PerfParts.DP_REDUCE],
                         plain[Debug.PerfParts.DP_COMM])

    def test_the_straggler_check_speaks_when_its_parts_miss_its_time(self) -> None:
        """
        Feature: estimate_pipeline, the check of the straggler stage's parts.
        Description: Two stages of 12 and 18 whose recorded parts add up to
            their times, then the same with the straggler's DP part 3 short.
        Expectation: Nothing said when they agree; one error naming the
            straggler stage when they do not, and either way the bubble is the
            pipeline's time less the parts, so the parts and the bubble still
            add up to the time.
        """
        cfg = _make_perf_cfg(p=2, vp=1, m=2)
        timed = (Debug.PerfParts.FW_COMPUTE, Debug.PerfParts.BW_COMPUTE, Debug.PerfParts.RECOMPUTE,
                 Debug.PerfParts.MP_COMM, Debug.PerfParts.EP_COMM, Debug.PerfParts.CP_COMM)
        for short, errors in ((0.0, 0), (3.0, 1)):
            with self.subTest(short=short):
                debugger = Debug.Debug(Dim.Dimensions([(Dim.DP, 2)], all_dims=[Dim.DP]), Debug.PerfParts)
                for part in timed:
                    debugger.info[part] = [1.0, 2.0]
                debugger.info[Debug.PerfParts.DP_COMM] = [6.0, 6.0 - short]
                with patch.object(PerfEstimate, "nd_logger") as perf_logger:
                    time = PerfEstimate.estimate_pipeline(cfg, [12.0, 18.0], debugger=debugger)
                self.assertEqual(perf_logger.error.call_count, errors)
                if errors:
                    self.assertEqual(perf_logger.error.call_args[0][1], 1)
                parts = sum(debugger.info[part] for part in timed + (Debug.PerfParts.DP_COMM,))
                self.assertEqual(parts, 2 * (18.0 - short))
                self.assertEqual(debugger.info[Debug.PerfParts.BUBBLE] + parts, time)

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

        def walk(per_stage: list) -> Any:
            """A walk giving *per_stage*, plain and recomputed alike; both from one walk with ``both``."""
            def priced(*_: Any, both: bool = False, **__: Any) -> Any:
                """The walk's sums, plain then recomputed with ``both``."""
                return (list(per_stage), list(per_stage)) if both else list(per_stage)
            return priced

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(PerfEstimate, "estimate_comp", side_effect=walk([10.0, 20.0])), \
                patch.object(PerfEstimate, "estimate_comm", side_effect=walk([1.0, 2.0])):
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

        def fake_bulk_layer(param: Any, lccfgs: Any, **kwargs: Any) -> tuple:
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

    def test_time_path_counts_a_moe_layers_experts(self) -> None:
        """
        Feature: comm_time.prepare_context.
        Description: Count a MoE layer's parameters through the time path's
            context, on the DeepSeek yaml, whose experts run on two
            data-parallel ranks with optimizer sharding.
        Expectation: The routed and shared experts are counted as the memory
            path counts them, and the layer's DP term carries them.
        """
        ccfg = CostModelConfig(config_path)
        moe = next(kind for kind in ArchHooks.layer_kinds(ccfg) if kind is not None and kind.name == "moe")
        ArchHooks.apply_layer_kind(ccfg, moe)
        ctx = CommTime.prepare_context()
        ctx.current_node = LayerType.NOT_REC_LAYER
        _, routed, shared = CommTime.EvalBody.num_params_layer(ccfg, ctx)
        self.assertGreater(routed, 0)
        self.assertEqual(routed, CommTime.EvalFFn.num_params_routed_expert(ccfg, ctx))
        self.assertEqual(shared, CommTime.EvalFFn.num_params_shared_expert(ccfg, ctx))
        self.assertGreater(
            CommTime.EvalLayerComm.dp_comm_layer(ccfg, ctx), CommTime.EvalLayerComm.dp_comm_non_exp(ccfg, ctx)
        )

    def test_comm_walk_leaves_the_model_as_it_found_it(self) -> None:
        """
        Feature: comm_time.estimate_from_mem_comm.
        Description: Walk the DeepSeek yaml's stages, dense and MoE layers,
            twice on one config, as an estimate does for its communication and
            again for its recompute's.
        Expectation: The config leaves the walk as it entered it, and both
            walks price every stage alike, its embedding and output layers
            included.
        """
        ccfg = CostModelConfig(config_path)
        ArchHooks.check_and_apply_custom_hook(ccfg)
        stages = ccfg.generate_partitions_vpp()
        before = {k: v for k, v in vars(ccfg).items() if isinstance(v, (bool, int, float, str))}
        first = CommTime.estimate_from_mem_comm(ccfg, CustomConfig(), stages, Hard.Device_A2)
        after = {k: v for k, v in vars(ccfg).items() if isinstance(v, (bool, int, float, str))}
        second = CommTime.estimate_from_mem_comm(ccfg, CustomConfig(), stages, Hard.Device_A2)
        self.assertEqual(after, before)
        self.assertEqual(first, second)

    def test_toml_flavors_as_torchtitan_states_them(self) -> None:
        """
        Feature: the TOML parser, on the fields a TorchTitan flavor may leave out.
        Description: A DeepSeek-style flavor that states its experts and no
            moe_enabled, the same flavor with moe_enabled False, and one that
            states ffn_dim_multiplier as None, TorchTitan's default.
        Expectation: The first runs its experts, the second is dense, and the
            third parses with no multiplier.
        """
        experts = ("dim=64, inter_dim=128, vocab_size=256, n_heads=4, n_layers=4, n_kv_heads=0, "
                   "kv_lora_rank=0, q_lora_rank=0, qk_rope_head_dim=0, n_dense_layers=1, moe_inter_dim=32, "
                   "moe_args=MoEArgs(num_experts=8, top_k=2, num_shared_experts=1), multiple_of=1")
        cases = (
            (experts + ", ffn_dim_multiplier=1", (8, 2, 1, 1)),
            (experts + ", moe_enabled=False, ffn_dim_multiplier=1", (1, 1, 0, 1)),
            (experts + ", ffn_dim_multiplier=None", (8, 2, 1, 1)),
        )
        for args, want in cases:
            ccfg = _torchtitan_flavor("deepseek_v3", args)
            got = (ccfg.n_exp, ccfg.n_chosen_exp, ccfg.n_shared_exp, ccfg.fdm)
            self.assertEqual(got, want, args)

    def test_each_parser_states_whether_queries_and_keys_are_normalized(self) -> None:
        """
        Feature: the QK-norm, as the MindFormers and TOML parsers read it.
        Description: MindFormers' Qwen3 yaml, which states qk_layernorm, and
            the DeepSeek yaml, which does not; TorchTitan flavors of Qwen3
            without qk_norm and with it False, and of Llama.
        Expectation: Qwen3 runs one QK-norm per layer wherever its config
            does not say otherwise, and the others none.
        """
        qwen3 = os.path.join(os.path.dirname(Par.__file__), "yamls", "pretrain_qwen3_72b.yaml")
        got = [(ccfg.qk_norm, ccfg.n_qknorm) for ccfg in (CostModelConfig(qwen3), CostModelConfig(config_path))]
        flavor = "dim=64, inter_dim=128, vocab_size=256, n_heads=4, n_layers=2, n_kv_heads=2"
        for name, args in (("qwen3", flavor), ("qwen3", flavor + ", qk_norm=False"), ("llama3", flavor)):
            ccfg = _torchtitan_flavor(name, args)
            got.append((ccfg.qk_norm, ccfg.n_qknorm))
        self.assertEqual(got, [(True, 1), (False, 0), (True, 1), (False, 0), (False, 0)])

    def test_torchtitan_reshards_as_its_policy_says(self) -> None:
        """
        Feature: the TOML parser's FSDP resharding.
        Description: TorchTitan's default policy without pipelining and at
            PP 2, and its always and never policies at PP 2.
        Expectation: The default reshards only without pipelining; always
            and never do as they say.
        """
        flavor = "dim=64, inter_dim=128, vocab_size=256, n_heads=4, n_layers=4, n_kv_heads=4"
        cases = ({}, {"pipeline_parallel_degree": 2},
                 {"pipeline_parallel_degree": 2, "fsdp_reshard_after_forward": "always"},
                 {"fsdp_reshard_after_forward": "never"})
        got = [_torchtitan_flavor("llama3", flavor, **case).reshards for case in cases]
        self.assertEqual(got, [True, False, True, False])

    def test_a_hook_class_prices_time_as_it_prices_memory(self) -> None:
        """
        Feature: estimate_performance, on a config priced through a hook class.
        Description: A hook class whose hook applies the DeepSeek yaml's family
            and then doubles its sequence, priced through the hook and through
            the family alone.
        Expectation: The time estimate applies the config's own hook, as the
            memory estimate does, so the two differ.
        """
        old_registry = MemEvalHook.hook_registry.copy()
        try:
            MemEvalHook.hook_registry = {}

            class _LongerSequence(MemEvalHook):
                """The family, then twice the sequence."""

                @staticmethod
                @hook_runner("deepseek_longer")
                def run_hooks(e: Any) -> None:
                    """Apply the family, then double the sequence."""
                    ArchHooks.check_and_apply_custom_hook(e)
                    e.set_ccfg(lambda c: setattr(c, "s", 2 * c.s))

            ccfg = CostModelConfig(config_path, hook_cls=_LongerSequence())
        finally:
            MemEvalHook.hook_registry = old_registry
        hooked = PerfEstimate.estimate_performance(ccfg, device_type=Hard.Device_A2)
        family = PerfEstimate.estimate_performance(
            ccfg, device_type=Hard.Device_A2, extra_custom_func=ArchHooks.check_and_apply_custom_hook
        )
        self.assertGreater(hooked, family)

    def test_comm_overlap_fields_in_parsers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Verify the transitional comm overlap fields
            (comm_dp_overlap, comm_tp_overlap) are populated by the real
            framework parsers through derive, which is what
            estimate_from_mem_comm reads on the search (FLOP) path.
        Expectation: After parsing, both overlap fields hold the documented
            defaults (0.9 and 0.5).  Every parser calls derive, so the
            hyperparallel path covers them all.
        """
        # --- Hyperparallel path: calls derive ---
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

        # --- Direct test of derive_comm_flags ---
        ccfg_direct = _ParserCostModelConfig()
        ccfg_direct.d = 2
        ccfg_direct.t = 2
        ccfg_direct.ep = 1
        ccfg_direct.cp = 1
        ccfg_direct.has_op = True
        ccfg_direct.has_grad_shard = True
        ccfg_direct.n_exp = 1
        derive_comm_flags(ccfg_direct)
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
            the debugger receives the standard comm keys and nothing but
            ``PerfParts`` keys, so the debug CSV row stays numeric.
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
        self.assertIn(Debug.PerfParts.CP_COMM, debugger.info)
        self.assertTrue(all(isinstance(part, Debug.PerfParts) for part in debugger.info))

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
            self.assertEqual(ArchHooks.layer_groups(hp_ccfg), [(None, 2)])
            self.assertEqual(vars(hp_ccfg.rec_op), dict.fromkeys(HYPER_SELECTIVE_REC_OP, 1))

            hp_config.activation_checkpoint.mode = "selective"
            sel_ccfg = _ParserCostModelConfig()
            sel_ccfg.config = hp_config
            sel_ccfg.source_code = source_path
            CostModelParserHyperparallel(sel_ccfg).parse()
            self.assertTrue(sel_ccfg.sel_rec)
            self.assertEqual(vars(sel_ccfg.rec_op), HYPER_SELECTIVE_REC_OP)

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
        # A config of one module is that module.
        self.assertEqual(ms_ccfg.model_name, "vision")
        self.assertFalse(ms_ccfg.multimodal)
        self.assertEqual(ms_ccfg.n_lay, 2)

    def test_a_mindspeed_model_counts_its_mtp_layers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: A MindSpeed text module of 4 layers and one MTP layer, beside a vision tower.
        Expectation: The text module's layer group covers its MTP layer too, as MindFormers' and
            hyper_v2's groups do.
        """
        def module(model_id: str, layers: int, mtp: int) -> dict:
            """One MindSpeed submodule."""
            return {
                "model_id": model_id, "freeze": False, "moe_grouped_gemm": False,
                "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
                "expert_model_parallel_size": 1, "sequence_parallel": False,
                "num_layers": layers, "hidden_size": 16, "ffn_hidden_size": 32, "vocab_size": 64,
                "num_attention_heads": 2, "num_query_groups": 0, "kv_channels": 0, "k_lora_rank": 0,
                "q_lora_rank": 0, "qk_rope_head_dim": 0, "num_moe_experts": 1, "moe_router_topk": 1,
                "n_shared_exp": 0, "moe_intermediate_size": 0, "first_k_dense_replace": 0,
                "recompute_num_layers": 1, "params_dtype": "bfloat16", "attention_softmax_in_fp32": True,
                "mtp_num_layers": mtp,
            }

        ms_ccfg = _ParserCostModelConfig()
        ms_ccfg.config = Config({
            "model_id": "multi-unit",
            "tmp": {"pp": 2, "mbs": 1, "dp": 2, "tp": 1, "cp": 1, "vpp": 1, "ep": 1, "seqlen": 8, "etp": 0},
            "image_encoder": module("vit", 2, 0),
            "text_decoder": module("qwen3", 4, 1),
        })
        ms_ccfg.hooks_dict = {"vit": None, "qwen3": None}
        CostModelParserMindspeed(ms_ccfg).parse()
        self.assertEqual(ArchHooks.layer_groups(ms_ccfg.mm_ccfgs["qwen3"]), [(None, 5)])
        self.assertEqual(ArchHooks.layer_groups(ms_ccfg.mm_ccfgs["vit"]), [(None, 2)])

    def test_cost_model_config_strategy_helpers(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: Cover cost-model copying, validation and strategy mutation without a parser.
        Expectation: Strategy fields update consistently without parsing a model config, and the
            fields derived from them follow.
        """
        parser = SimpleNamespace()
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
            n_exp=1,
            hff_exp=8,
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
        self.assertEqual(cost_cfg.shard_embed, 8, f"shard_embed={cost_cfg.shard_embed}, want d * t = 8")

        cost_cfg.offset = []
        with self.assertRaises(AttributeError):
            cost_cfg.set_strategy(dp=2)
        cost_cfg.offset = [[0, 0], [0, 0]]

        self._test_multimodal_strategy(cost_cfg)

    def test_strategy_change_derives_mindformers_fields(self) -> None:
        """
        Feature: TestSappNDRunND.
        Description: The DeepSeek MindFormers yaml keeps its recompute input sliced at TP 4
            (recompute_slice_activation), and MindFormers' selective recompute depends on
            sequence parallelism, which a strategy change sets to TP.
        Expectation: Both follow each strategy change, where they kept their parse-time values.
        """
        cfg = CostModelConfig(config_path)
        self.assertEqual(cfg.shard_recompute_input, 4, f"parsed shard_recompute_input={cfg.shard_recompute_input}")
        cfg.set_strategy(mp=1)
        self.assertEqual(cfg.shard_recompute_input, 1, f"at TP 1 shard_recompute_input={cfg.shard_recompute_input}")

        cfg.set_strategy(mp=2, sel_rec=True)
        dropped = sorted(op for op, keep in vars(cfg.rec_op).items() if not keep)
        self.assertEqual(dropped, ["ffAct", "headCast", "normOp"], f"recomputed at TP 2: {dropped}")
        cfg.set_strategy(mp=1, sel_rec=True)
        dropped = sorted(op for op, keep in vars(cfg.rec_op).items() if not keep)
        self.assertEqual(dropped, ["headCast"], f"recomputed at TP 1: {dropped}")

    def test_the_search_states_its_machines_links(self) -> None:
        """
        Feature: state_machine_links.
        Description: A config and a multimodal one with a submodule, given
            the A3 machine, then a config given no machine.
        Expectation: Each config takes A3's 16 ranks a node and its 200 and
            25 GB/s links, where every config kept 8, 400 and 25 whatever
            the machine; without a machine nothing is stated.
        """
        plain, child = SimpleNamespace(), SimpleNamespace()
        multimodal = SimpleNamespace(mm_ccfgs={"vision": child})
        for cfg in (plain, multimodal):
            Par.state_machine_links(cfg, Hard.Device_A3)
        got = [(cfg.device_per_node, cfg.bw_intra, cfg.bw_inter) for cfg in (plain, multimodal, child)]
        self.assertEqual(got, [(16, 200, 25)] * 3)
        untouched = SimpleNamespace()
        Par.state_machine_links(untouched, None)
        self.assertEqual(vars(untouched), {})

    def test_a_strategy_keeps_the_runs_sequence_parallelism(self) -> None:
        """
        Feature: CostModelConfig.set_strategy, the sequence-parallel divisor.
        Description: The DeepSeek MindFormers yaml at TP 4 without sequence
            parallelism and with it, each moved to TP 2; then a searched
            candidate at TP 2 stating SP off and on.
        Expectation: The divisor follows the run's choice at the new TP, 1
            without and 2 with, where the strategy setter set it to TP for
            every candidate; a candidate that states SP decides it.
        """
        with open(config_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        got = []
        for use_sp in (False, True):
            data["parallel_config"]["use_seq_parallel"] = use_sp
            cfg = CostModelConfig(copy.deepcopy(data))
            cfg.set_strategy(mp=2)
            got.append(cfg.sp)
            for stated in (False, True):
                cfg.set_strategy(mp=2, sp=stated)
                got.append(cfg.sp)
        self.assertEqual(got, [1, 1, 2, 2, 1, 2])

    def _test_multimodal_strategy(self, cost_cfg: CostModelConfig) -> None:
        """Exercise set_strategy via model_name routing and error handling."""
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
    def test_a_quiet_search_hands_its_ratios_file_to_every_estimate(self) -> None:
        """
        Feature: Parallelize.order_search_space.
        Description: Order two configurations with a ratios file at a verbosity that writes no debug output.
        Expectation: Every estimate receives the ratios file, as it does when debugging, so the
                     verbosity cannot change what is calibrated.
        """
        runner = object.__new__(Par.ParallelizeLayer)
        runner.config = SimpleNamespace(set_parallel_config=lambda config: True)
        runner.priced = SimpleNamespace
        runner.machine = SimpleNamespace(device=Hard.Device_A2)
        runner.enable_debug = False
        space = [(SimpleNamespace(name="a"), 10), (SimpleNamespace(name="b"), 20)]
        with patch.object(Par, "estimate_performance", side_effect=[3.0, 1.0]) as estimate:
            scored, _ = runner.order_search_space(space, threads_num=None, cache_file="ratios.json")
        self.assertEqual([entry[2] for entry in scored], [1.0, 3.0])
        files = [call.kwargs.get("cache_file") for call in estimate.call_args_list]
        self.assertEqual(files, ["ratios.json", "ratios.json"], f"the ratios file each estimate got: {files}")

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
        runner.priced = lambda: config_state.ccfg
        runner.machine = SimpleNamespace(number=16, device=Hard.Device_A2)
        runner.global_batch_size = 8
        runner.model_name = "unit"
        runner.enable_debug = False
        runner.auto_recompute = False
        runner.mem_eval = SimpleNamespace(
            mem_fit=lambda peak: peak < 100,
            get_strategy=lambda: {},
            get_max_device_memory=lambda: 100.0,
        )

        class _ParallelConfig:
            """Small validity-controlled parallel configuration."""

            def __init__(self, valid: bool = True) -> None:
                """Initialize with configured validity."""
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
        with patch.object(Par.logger, "output") as output:
            self.assertEqual(runner.generate_search_space("out", threads_num=None), [("fit", 10)])
        self.assertEqual(writes, [("out", "fit")])
        said = [call.args for call in output.call_args_list]
        self.assertEqual([args[1:] for args in said if "dropped:" in args[0]], [("large", 200, 100.0, 100.0)])
        self.assertIn((2, 0, "", 2), [args[1:] for args in said])
        self.assertIn((1, 1), [args[1:] for args in said])

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

        runner.batch_reachable = lambda: True
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

    def test_a_candidate_carries_the_searchs_micro_batch_count(self) -> None:
        """
        Feature: GlobalConfig.make_parallel_config, the micro-batch count.
        Description: A config at DP 2, PP 2, micro-batches of 4 and two of
            them, searched over EP and OP alone, so -l names neither MB nor PP,
            given a candidate whose loop counted four micro-batches.
        Expectation: The candidate carries the loop's four and its batch is
            DP x MB x MBS with them, 32; kept from the config, the two made
            another batch, 16.
        """
        fake_ccfg = _FakeCostModelConfig()
        global_config = object.__new__(GC.GlobalConfig)
        global_config.ccfg = fake_ccfg
        global_config.dimensions = [Dim.EP, Dim.OP]
        candidate = global_config.make_parallel_config((2, 2, 2, 1), (4, 4), (1, 1, 2, False))
        self.assertEqual(candidate.val(Dim.MBN), 4)
        self.assertEqual(global_config.global_batch_size(candidate), 32)
        self.assertIn(Dim.MBN, global_config.dimensions)

    def test_a_search_no_candidate_of_which_makes_the_batch_stops_with_one_line(self) -> None:
        """
        Feature: ParallelizeLayer.batch_reachable.
        Description: 64 devices searched over EP and OP alone, so every
            candidate is DP 64 with micro-batches of 1, for a batch of 16, then
            of 128.
        Expectation: No candidate makes 16, so the search stops before
            building one, with one error naming the batch, the yaml's and DP x
            MBS, 64; 128 is two micro-batches of it, and nothing is said.
        """
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
        fake_ccfg = _FakeCostModelConfig()
        fake_ccfg.d, fake_ccfg.t, fake_ccfg.p, fake_ccfg.b, fake_ccfg.m = 64, 1, 1, 1, 1
        fake_ccfg.accumulates_grads = True
        fake_ccfg.gbs = 16
        global_config = object.__new__(GC.GlobalConfig)
        global_config.ccfg = fake_ccfg
        global_config.dimensions = [Dim.EP, Dim.OP]
        runner = object.__new__(Par.ParallelizeLayer)
        runner.config = global_config
        runner.machine = SimpleNamespace(number=64)
        for batch, reachable in ((16, False), (128, True)):
            with self.subTest(batch=batch), patch.object(Par.logger, "error") as error:
                runner.global_batch_size = batch
                self.assertEqual(runner.batch_reachable(), reachable)
                self.assertEqual(error.call_count, 0 if reachable else 1)
                if not reachable:
                    self.assertEqual(error.call_args.args[1:4], (16, 16, "64"))

    def test_a_ranking_priced_at_a_length_nobody_stated_says_so_under_its_table(self) -> None:
        """
        Feature: ParallelizeLayer.run_generation_to_ordering, the sequence length.
        Description: A search over one configuration of a model whose config
            states no training length, so the parser costed its context
            limit of 32768; then the same model with a stated length.
        Expectation: The line under the table names the 32768 tokens and why;
            with a stated length nothing is said.
        """
        dims = Dim.Dimensions([(Dim.DP, 8)], all_dims=[Dim.DP])
        runner = object.__new__(Par.ParallelizeLayer)
        runner.enable_debug = False
        runner.model_name = "unit"
        runner.global_batch_size = 8
        runner.machine = SimpleNamespace(device=Hard.Device_A2, number=8)
        runner.config = SimpleNamespace(balancing=SimpleNamespace(from_config=False), dimensions=[Dim.DP])
        runner.batch_reachable = lambda: True
        runner.generate_search_space = lambda folder, threads_num: [(dims, 100)]
        runner.order_search_space = lambda space, threads_num, cache_file: ([(dims, 100, 2.5, [])], [])
        for stated, said in ((False, 1), (True, 0)):
            with self.subTest(stated=stated), patch.object(Par.logger, "output") as output:
                runner.priced = lambda stated=stated: SimpleNamespace(seq_len_stated=stated, s=32768)
                runner.run_generation_to_ordering(None)
                lines = [call.args[0] % call.args[1:] for call in output.call_args_list
                         if "context limit" in call.args[0]]
                self.assertEqual(len(lines), said)
                if said:
                    self.assertIn("priced at 32768 tokens", lines[0])
        # A field the estimate read unset is named under the table too.
        reads = PreProcess.defaultdict(PreProcess.Counter, {"n_gather": PreProcess.Counter({"comm.py:tp": 2})})
        with patch.object(PreProcess, "UNSET_READS", reads), patch.object(Par.logger, "output") as output:
            runner.run_generation_to_ordering(None)
        self.assertTrue(any("n_gather by comm.py:tp" in call.args[0] for call in output.call_args_list))

    def test_a_ranking_says_when_it_prices_another_recompute_than_the_runs(self) -> None:
        """
        Feature: ParallelizeLayer.run_generation_to_ordering, the run's recompute.
        Description: A search whose balancing gives every candidate full
            recompute, for a run training selective, off and full; then the
            selective run searched with -mppb, over a recompute dimension,
            and with -ar.
        Expectation: The line under the table names selective and off, the
            two modes the search did not price; it says nothing for full,
            under -mppb, or over a recompute dimension, which price the
            run's own, nor under -ar, which chooses each candidate's.
        """
        dims = Dim.Dimensions([(Dim.DP, 8)], all_dims=[Dim.DP])
        runner = object.__new__(Par.ParallelizeLayer)
        runner.enable_debug = False
        runner.model_name = "unit"
        runner.global_batch_size = 8
        runner.machine = SimpleNamespace(device=Hard.Device_A2, number=8)
        runner.batch_reachable = lambda: True
        runner.set_recompute_mode = lambda mode: None
        runner.generate_search_space = lambda folder, threads_num: [(dims, 100)]
        runner.order_search_space = lambda space, threads_num, cache_file: (
            [(Dim.Dimensions([(Dim.DP, 8)], all_dims=[Dim.DP]), 100, 2.5, [])], [])
        said = []
        runner._log_recompute = lambda space: None
        for mode, from_config, modes, auto in (
                ("selective", False, None, False), ("off", False, None, False), ("full", False, None, False),
                ("selective", True, None, False), ("selective", False, ("off", "full"), False),
                ("selective", False, None, True)):
            runner.config = SimpleNamespace(balancing=SimpleNamespace(from_config=from_config), dimensions=[Dim.DP])
            runner.recompute_dimension = modes
            runner.auto_recompute = auto
            runner.priced = lambda mode=mode: SimpleNamespace(stated_ac_mode=mode)
            with patch.object(Par.logger, "output") as output:
                runner.run_generation_to_ordering(None)
            said.append([call.args[1] for call in output.call_args_list if "checkpoint mode is" in call.args[0]])
        self.assertEqual(said, [["selective"], ["off"], [], [], [], []])

    def test_a_ranking_names_the_mtp_layers_it_does_not_price(self) -> None:
        """
        Feature: ParallelizeLayer.run_generation_to_ordering, a dropped MTP depth.
        Description: A search over one configuration of a model stating 3 MTP
            layers the AutoModels trainer does not build, then of one stating
            none.
        Expectation: The line under the table names the 3 layers; with none,
            nothing is said.
        """
        dims = Dim.Dimensions([(Dim.DP, 8)], all_dims=[Dim.DP])
        runner = object.__new__(Par.ParallelizeLayer)
        runner.enable_debug = False
        runner.model_name = "unit"
        runner.global_batch_size = 8
        runner.machine = SimpleNamespace(device=Hard.Device_A2, number=8)
        runner.config = SimpleNamespace(balancing=SimpleNamespace(from_config=False), dimensions=[Dim.DP])
        runner.batch_reachable = lambda: True
        runner.generate_search_space = lambda folder, threads_num: [(dims, 100)]
        runner.order_search_space = lambda space, threads_num, cache_file: ([(dims, 100, 2.5, [])], [])
        for dropped, said in ((3, 1), (0, 0)):
            with self.subTest(dropped=dropped), patch.object(Par.logger, "output") as output:
                runner.priced = lambda dropped=dropped: SimpleNamespace(mtp_unpriced=dropped)
                runner.run_generation_to_ordering(None)
                lines = [call.args[0] % call.args[1:] for call in output.call_args_list
                         if "MTP layer(s)" in call.args[0]]
                self.assertEqual(len(lines), said)
                if said:
                    self.assertIn("states 3 MTP layer(s)", lines[0])

    def test_the_search_counts_what_its_checks_refuse_apart_from_what_memory_drops(self) -> None:
        """
        Feature: ParallelizeLayer.generate_search_space, what it says it left out.
        Description: A search whose loops meet a candidate with a degree out of
            bounds and one of the wrong global batch, and price two, one of
            them 50 MB over a 100 MB budget.
        Expectation: The candidate over the budget is named with its peak; the
            first summary line counts four tested, two refused, each check by
            name, and two priced; the second, one fitting and one dropped.
        """
        runner = object.__new__(Par.ParallelizeLayer)
        runner.config = SimpleNamespace(moe_valid=lambda config: True, global_batch_size=lambda config: config.batch)
        runner.global_batch_size = 8
        runner.filtered_out = lambda config: False
        runner.mem_eval = SimpleNamespace(mem_fit=lambda peak: peak < 100, get_max_device_memory=lambda: 100.0)

        def loops(*_: Any) -> Any:
            """Refuse two candidates the way inside_loop_nest checks them, then price two."""
            for config in (SimpleNamespace(is_valid=lambda: False, batch=8),
                           SimpleNamespace(is_valid=lambda: True, batch=4)):
                self.assertFalse(runner.is_valid(config))
            return {"fit": 10, "large": 150}, 4

        runner.device_loops = loops
        with patch.object(Par.logger, "output") as output:
            self.assertEqual(runner.generate_search_space(None, threads_num=None), [("fit", 10)])
        self.assertEqual([call.args[0] % call.args[1:] for call in output.call_args_list], [
            "large dropped: its peak of 150 MB is over the 100 MB budget by 50 MB",
            "4 configurations tested: 2 refused (a degree out of bounds 1, global batch 1), 2 priced for memory",
            "1 configuration fitting memory to order, 1 dropped over the memory budget",
        ])

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
        runner.priced = lambda: runner.config.ccfg
        runner.memory_estim = lambda debugger=None: 32

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
                patch.object(Par.Debug, "get_comm_classified_data",
                             return_value=[("cfg", 1, 10, 2.0, [], {"IDLE": 3.0})]), \
                patch.object(Par.Debug, "plot_vs_real_comm_classified") as plot_vs_real_comm, \
                patch.object(Par.Debug, "write_estimates_csv") as write_estimates, \
                patch.object(Par.Debug, "correlation_with_classified_comms", return_value=0.8):
            self.assertEqual(runner.test_from_csv("profile.csv", "out"), (0.9, 1, 1))
            self.assertEqual(
                runner.test_from_csv_comm_classified("profile.csv", "out", plot_idle=True),
                0.8,
            )

        plot_vs_real.assert_called_once()
        # With idle, once as measured and once without the idle remainder.
        self.assertEqual(plot_vs_real_comm.call_count, 2)
        write_estimates.assert_called_once()
