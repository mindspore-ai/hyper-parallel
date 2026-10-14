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
"""Unit tests for SAPP-ND's recompute dimension: the activation checkpoint modes a search ranks.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_recompute_dimension.py
"""
import csv
import os
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import yaml

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.nd import debug as Debug
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd import run_nd as RunND
from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import HYPER_SELECTIVE_REC_OP
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level
from hyper_parallel.auto_parallel.sapp_nd.nd.recompute_dimension import (
    RECOMPUTE_MODES,
    parsed_recompute,
    read_activation_checkpoint_mode,
    read_recompute_modes,
    restore_recompute,
    state_recompute_mode,
)


def _dense_train_yaml(folder: str, mode: str = "full") -> str:
    """Write a small dense model the hyper_v2 parser prices, recomputed as *mode* states."""
    config = {
        "model": {
            "name": "nonexistent",
            "config_overrides": {
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "intermediate_size": 4096,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
            },
        },
        "data": {"max_seq_len": 4096},
        "train": {
            "accelerator": {"dp_shard": 1, "dp_replicate": 1, "tp_degree": 1, "pipeline_parallel_degree": 1},
            "micro_batch_size": 1,
            "micro_batch_num": 1,
            "gradient_checkpointing": {"activation_checkpoint": mode},
            "optimizer": {"max_grad_norm": 1.0},
        },
        "context": {"max_device_memory": "64GB"},
    }
    path = os.path.join(folder, "train.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return path


def _search(path: str, **extra: Any) -> Tuple[Any, list]:
    """Rank the small model's DP x TP strategies on 8 devices, as run_nd does."""
    runner = Par.Parallelize("hyper_v2", path, Hard.Machine(8, "A2"), dimensions=Dim.get_dims(["DP", "MP"]),
                             **extra)
    return runner, runner.run_generation_to_ordering(None)


def _rows(scored: list) -> List[Tuple[Any, ...]]:
    """Each entry's degrees, mode, memory and score."""
    return [(tuple(entry[0].values()), entry[0].recompute, entry[1], entry[2]) for entry in scored]


class TestReadRecomputeModes(unittest.TestCase):
    """The dimension is stated as a degree is: one value, a list, or auto."""

    def test_stated_forms(self):
        """None, auto, one mode, a list with repeats, YAML's False and the old none all read one way."""
        cases = (
            (None, None),
            ("auto", RECOMPUTE_MODES),
            (["AUTO"], RECOMPUTE_MODES),
            ("full", ("full",)),
            (["off", "full", "off"], ("off", "full")),
            ([False, "Selective"], ("off", "selective")),
            ("none", ("off",)),
        )
        for stated, expected in cases:
            with self.subTest(stated=stated):
                self.assertEqual(read_recompute_modes(stated, "here"), expected)

    def test_refusals_name_where_and_what(self):
        """A value that is no mode, auto among modes, and an empty list are refused."""
        cases = (("swap", "'swap' is not a recompute mode"), (["off", "auto"], "'auto' is not a recompute mode"),
                 ([], "an empty list allows no recompute mode"))
        for stated, message in cases:
            with self.subTest(stated=stated):
                with self.assertRaisesRegex(ValueError, f"parallelism.recompute: {message}"):
                    read_recompute_modes(stated, "parallelism.recompute")


class TestReadActivationCheckpointMode(unittest.TestCase):
    """A train.yaml's one mode, read as the trainer reads it."""

    def test_the_modes_the_trainer_runs(self):
        """
        Feature: read_activation_checkpoint_mode.
        Description: Nothing stated, YAML's False for an unquoted off, the
            older none, and each mode.
        Expectation: Off for the first three, as the trainer's default and
            its config resolver read them; each mode as itself.
        """
        for stated, expected in ((None, "off"), (False, "off"), ("none", "off"), ("off", "off"),
                                 ("full", "full"), ("selective", "selective")):
            with self.subTest(stated=stated):
                self.assertEqual(read_activation_checkpoint_mode(stated, "here"), expected)

    def test_what_the_trainer_would_not_run_is_refused(self):
        """
        Feature: read_activation_checkpoint_mode.
        Description: A typo of full, True, auto and a list.
        Expectation: Each refused by name and place, where ND used to price
            the first as no recompute.
        """
        for stated in ("ful", True, "auto", ["full"]):
            with self.subTest(stated=stated):
                with self.assertRaisesRegex(ValueError, "activation_checkpoint.mode"):
                    read_activation_checkpoint_mode(stated, "activation_checkpoint.mode")

    def test_the_parser_prices_the_mode_it_is_given(self):
        """
        Feature: the hyper_v2 parser's recompute, on the legacy and the AutoModels key.
        Description: The small dense model stated full, a typo of it, and an
            AutoModels mode that YAML reads as False above a legacy full.
        Expectation: Full recompute; a refusal naming the key; off, where
            the False used to fall through to the legacy key and price full.
        """
        with tempfile.TemporaryDirectory() as folder:
            full = EvaluatorV2(_dense_train_yaml(folder, "full"), framework="hyper_v2", log_level=0).ccfg
            self.assertEqual((full.full_rec, full.sel_rec), (True, False))
            with self.assertRaisesRegex(ValueError, "gradient_checkpointing.activation_checkpoint: 'ful'"):
                EvaluatorV2(_dense_train_yaml(folder, "ful"), framework="hyper_v2", log_level=0)
            with open(_dense_train_yaml(folder, "full"), encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
            raw["activation_checkpoint"] = yaml.safe_load("mode: off")
            off = EvaluatorV2(raw, framework="hyper_v2", log_level=0).ccfg
            self.assertEqual((off.full_rec, off.sel_rec), (False, False))


class TestStateRecomputeMode(unittest.TestCase):
    """A mode sets the switches the HyperParallel parsers set, on every config of the model."""

    def test_each_mode_sets_the_parser_switches(self):
        """Full sets full_rec, selective sets sel_rec and HyperParallel's policy, off neither."""
        for mode in RECOMPUTE_MODES:
            with self.subTest(mode=mode):
                ccfg = SimpleNamespace(full_rec=True, sel_rec=False, rec_op=None)
                state_recompute_mode(ccfg, mode)
                self.assertEqual((ccfg.full_rec, ccfg.sel_rec), (mode == "full", mode == "selective"))
                expected = HYPER_SELECTIVE_REC_OP if mode == "selective" else dict.fromkeys(HYPER_SELECTIVE_REC_OP, 1)
                self.assertEqual(vars(ccfg.rec_op), expected)

    def test_multimodal_submodules_take_it_and_give_it_back(self):
        """Every submodule takes the mode, and restoring gives each its parsed recompute again."""
        vision = SimpleNamespace(full_rec=False, sel_rec=True, rec_op=Config({"attBMM": 1}))
        text = SimpleNamespace(full_rec=True, sel_rec=False, rec_op=None)
        whole = SimpleNamespace(full_rec=True, sel_rec=False, rec_op=None, multimodal=True,
                                mm_ccfgs={"vision": vision, "text": text}, mm_order=["vision", "text"])
        parsed = parsed_recompute(whole)

        state_recompute_mode(whole, "off")
        self.assertEqual([(c.full_rec, c.sel_rec) for c in (whole, vision, text)], [(False, False)] * 3)

        restore_recompute(parsed)
        self.assertEqual([(c.full_rec, c.sel_rec) for c in (whole, vision, text)],
                         [(True, False), (False, True), (True, False)])
        self.assertEqual(vars(vision.rec_op), {"attBMM": 1})


class TestSearchWithRecomputeDimension(unittest.TestCase):
    """A search prices each candidate under each allowed mode and ranks the pairs together."""

    def setUp(self) -> None:
        """Keep the search quiet and its output out of the tree, and its dimension bounds out of later tests."""
        set_verbose_level(1)
        # A search bounds the module's dimensions, which every later test shares.
        self.addCleanup(lambda: [dim.reset_bound() for dim in Dim.ALL_DIMS])
        self.folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.folder.cleanup)
        Debug.set_output_dir(self.folder.name)
        self.path = _dense_train_yaml(self.folder.name)

    def test_full_pass_is_the_search_without_a_dimension(self):
        """Its full entries are, in order, exactly what the search ranks without a recompute dimension."""
        _, plain = _search(self.path)
        _, every = _search(self.path, recompute_dimension=RECOMPUTE_MODES)

        self.assertTrue(plain, "the small model must fit somewhere")
        self.assertEqual([row[1] for row in _rows(plain)], [None] * len(plain))
        full = [(row[0], row[2], row[3]) for row in _rows(every) if row[1] == "full"]
        self.assertEqual(full, [(row[0], row[2], row[3]) for row in _rows(plain)])
        self.assertEqual([entry[2] for entry in every], sorted(entry[2] for entry in every))

    def test_modes_order_memory_one_way_and_time_the_other(self):
        """For one strategy: off keeps most and runs fastest, full keeps least and runs slowest."""
        _, every = _search(self.path, recompute_dimension=RECOMPUTE_MODES)
        by_strategy: Dict[tuple, Dict[str, Tuple[float, float]]] = {}
        for degrees, mode, memory, score in _rows(every):
            by_strategy.setdefault(degrees, {})[mode] = (memory, score)
        complete = {degrees: modes for degrees, modes in by_strategy.items() if len(modes) == 3}

        self.assertTrue(complete, f"no strategy fits in all three modes: {by_strategy}")
        for degrees, modes in complete.items():
            with self.subTest(degrees=degrees):
                self.assertGreater(modes["off"][0], modes["selective"][0])
                self.assertGreater(modes["selective"][0], modes["full"][0])
                self.assertLess(modes["off"][1], modes["selective"][1])
                self.assertLess(modes["selective"][1], modes["full"][1])

    def test_a_mode_left_out_is_never_proposed(self):
        """Without selective, the other pairs are the same as with it, and no selective pair is left."""
        _, every = _search(self.path, recompute_dimension=RECOMPUTE_MODES)
        runner, runnable = _search(self.path, recompute_dimension=("off", "full"))

        self.assertNotIn("selective", {row[1] for row in _rows(runnable)})
        self.assertEqual(_rows(runnable), [row for row in _rows(every) if row[1] != "selective"])
        ccfg = runner.mem_eval.ccfg
        self.assertEqual((ccfg.full_rec, ccfg.sel_rec), (True, False), "the parsed recompute comes back")
        self.assertIsNone(runner.config.balancing.stated_recompute)

    def test_the_dimension_needs_the_hyperparallel_parsers_and_no_mppb(self):
        """Another framework's recompute, or recompute taken from the config, cannot take the dimension."""
        with self.assertRaisesRegex(ValueError, "the mindformers parser states its own recompute"):
            Par.Parallelize("mindformers", "unused.yaml", Hard.Machine(8, "A2"), recompute_dimension=("full",))
        with self.assertRaisesRegex(ValueError, "mppb takes the recompute from the config"):
            _search(self.path, recompute_dimension=("full",), mppb=True)


class TestRecomputeColumns(unittest.TestCase):
    """The mode goes in its own column, written only with a dimension and read from measured CSVs."""

    def test_ranking_states_the_mode_only_with_a_dimension(self):
        """A tagged ranking gains the recompute column after the degrees; an untagged one is as before."""
        degrees = [(Dim.DP, 4), (Dim.TP, 2)]
        cases = ((Dim.Dimensions(degrees, recompute="off"), ["DP", "MP", "recompute", "memory_mb"]),
                 (Dim.Dimensions(degrees), ["DP", "MP", "memory_mb"]))
        for config, header in cases:
            with self.subTest(recompute=config.recompute), tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "ranking.csv")
                Debug.write_ranking_csv([(config, 1024, 1.5, [])], path)
                with open(path, newline="", encoding="utf-8") as handle:
                    rows = list(csv.reader(handle))
                self.assertEqual(rows[0][1:len(header) + 1], header)
                self.assertEqual(rows[1][1:len(header)], ["4", "2", "off"][:len(header) - 1])

    def test_measured_rows_carry_their_mode(self):
        """A measured CSV's recompute column tags each row, and a blank cell leaves it untagged."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "real.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("DP,MP,recompute,time,comp,dp_wait\n8,1,off,12,1,1\n8,1,,10,5,4\n")
            configs = Debug.get_comm_classified_data(path)

        self.assertEqual([entry[0].recompute for entry in configs], ["off", None])
        self.assertEqual(configs[0][2]["comp"], 1.0, "recompute is no compute column")
        self.assertIn("recompute=off", str(configs[0][0]))
        self.assertTrue(configs[0][0].unique_name().endswith("_off"))

    def test_a_measured_row_runs_one_mode(self):
        """A mode the trainer does not run, or more than one, is refused with the column named."""
        for cell, message in (("swap", "'swap' is not a recompute mode"), ("auto", "ran one mode, not 'auto'")):
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "real.csv")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(f"DP,recompute,time,comp\n8,{cell},12,1\n")
                with self.assertRaisesRegex(ValueError, f"the recompute column.*{message}"):
                    Debug.get_comm_classified_data(path)


class _RecordingParallelize:
    """Parallelize double that records its keyword arguments."""

    calls: List[Dict[str, Any]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Record the keyword arguments of the call."""
        del args
        self.__class__.calls.append(kwargs)

    def run_generation_to_ordering(self, *args: Any, **kwargs: Any) -> list:
        """Return no configuration."""
        del args, kwargs
        return []


class TestRunNdRecompute(unittest.TestCase):
    """run_nd states the dimension from --recompute, else from a hyper_v2 yaml's context."""

    def test_cli_over_the_yaml_context(self):
        """--recompute wins; without it a hyper_v2 yaml's context.recompute states the dimension."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "train.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump({"context": {"recompute": ["off", "full"]}}, handle)
            args = SimpleNamespace(recompute=None, framework="hyper_v2", search_config=None, yaml_config=path)
            self.assertEqual(RunND._recompute_modes(args), ("off", "full"))  # pylint: disable=protected-access
            args.recompute = ["full"]
            self.assertEqual(RunND._recompute_modes(args), ("full",))  # pylint: disable=protected-access
            args.recompute, args.framework = None, "mindformers"
            self.assertIsNone(RunND._recompute_modes(args))  # pylint: disable=protected-access

    def test_cli_states_it_over_a_search_config(self):
        """With a search config, --recompute lands where parallelism.recompute does."""
        search_cfg = SimpleNamespace(estimator={}, cluster_spec={}, constraint={})
        args = SimpleNamespace(device_type=None, global_batch_size=None, max_mem=None, devices=None,
                               recompute=["off", "full"])
        RunND._apply_cli_overrides(search_cfg, args)  # pylint: disable=protected-access
        self.assertEqual(search_cfg.estimator["recompute_modes"], ("off", "full"))

    def test_cli_passes_the_modes_to_the_search_only_when_stated(self):
        """The search gets recompute_dimension from --recompute, and no such argument without it."""
        with tempfile.TemporaryDirectory() as folder:
            path = _dense_train_yaml(folder)
            for extra, expected in (([], None), (["--recompute", "off", "full"], ("off", "full"))):
                with self.subTest(extra=extra), \
                        patch.object(Par, "Parallelize", _RecordingParallelize), \
                        patch.dict(os.environ, {"MPLCONFIGDIR": folder}):
                    _RecordingParallelize.calls = []
                    argv = ["run_nd.py", "-f", "hyper_v2", "-y", path, "-d", "8", "-v", "0", "-o", folder] + extra
                    with patch.object(sys, "argv", argv):
                        runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
                    self.assertEqual(_RecordingParallelize.calls[-1].get("recompute_dimension"), expected)

    def test_with_ar_the_modes_are_chosen_per_layer(self):
        """With -ar, --recompute names the modes each layer chooses among, the plan the trainer runs."""
        with tempfile.TemporaryDirectory() as folder:
            path = _dense_train_yaml(folder)
            with patch.object(Par, "Parallelize", _RecordingParallelize), \
                    patch.dict(os.environ, {"MPLCONFIGDIR": folder}):
                _RecordingParallelize.calls = []
                argv = ["run_nd.py", "-f", "hyper_v2", "-y", path, "-d", "8", "-v", "0", "-o", folder,
                        "-ar", "--recompute", "off", "full"]
                with patch.object(sys, "argv", argv):
                    runpy.run_module("hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", run_name="__main__")
            call = _RecordingParallelize.calls[-1]
            self.assertEqual((call.get("recompute_modes"), call.get("recompute_mode_per_layer")),
                             (("off", "full"), True))
            self.assertNotIn("recompute_dimension", call)


if __name__ == "__main__":
    unittest.main()
