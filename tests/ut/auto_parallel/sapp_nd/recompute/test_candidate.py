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
"""Tests for choosing every layer's recompute option for a candidate of the search.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/recompute/test_candidate.py -v
"""
import copy
import dataclasses
import itertools
import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

import yaml

import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.debug as Debug
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel import _hf_model_spec
from hyper_parallel.auto_parallel._exec_spec import RECOMPUTE_OPS, ExecSpec, RecomputeRange
from hyper_parallel.auto_parallel._model_spec import KindActivations
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd.common.apply_exec import apply_exec
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import layer_kinds
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import HYPER_SELECTIVE_REC_OP
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level
from hyper_parallel.auto_parallel.sapp_nd.recompute import candidate as Candidate
from hyper_parallel.auto_parallel.sapp_nd.recompute.candidate import (
    MODES,
    LayerRange,
    RecomputeChoice,
    choose_recompute,
    describe,
    micro_batches_in_flight,
    mode_ranges,
    mode_recompute,
    option_label,
    to_records,
    trainer_plan,
    whole_modes,
)
from hyper_parallel.auto_parallel.sapp_nd.recompute.front import LayerOption, build_front, layer_profiles
from hyper_parallel.auto_parallel.sapp_nd.recompute.knapsack import MEGABYTE, Stage
from hyper_parallel.auto_parallel.sapp_nd.recompute.profile import SWITCHES

DEEPSEEK_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nd", "deepseek.yaml"
)

# The parallel degrees a fixed analysis reports on each of its alternatives.
_DEGREES = (("DP", 4), ("TP", 2), ("PP", 2), ("EP", 2))

# A dense model, from its config overrides alone, at TP 4 and PP 2.
_DENSE = {
    "model": {"name": "nonexistent", "config_overrides": {
        "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 8,
        "intermediate_size": 5632, "vocab_size": 32000, "max_position_embeddings": 4096}},
    "data": {"max_seq_len": 4096},
    "train": {"accelerator": {"dp_shard": 1, "dp_replicate": 1, "tp_degree": 4, "pipeline_parallel_degree": 2},
              "micro_batch_size": 1, "micro_batch_num": 4,
              "gradient_checkpointing": {"activation_checkpoint": "full"}, "optimizer": {"max_grad_norm": 1.0}},
    "context": {"max_device_memory": "64GB"},
}


# A vision-language model: a small tower in front of a four-layer MoE text model.
_VL = SimpleNamespace(
    model_type="qwen3_vl_moe",
    text_config=SimpleNamespace(
        hidden_size=2048, num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=8,
        intermediate_size=5632, vocab_size=32000, max_position_embeddings=8192, head_dim=128,
        num_experts=16, num_experts_per_tok=4, moe_intermediate_size=768),
    vision_config=SimpleNamespace(
        hidden_size=1152, depth=2, num_heads=16, intermediate_size=4304, out_hidden_size=2048,
        patch_size=16, spatial_merge_size=2, num_position_embeddings=2304),
)
_VL_TRAINING = {
    "model": {"pretrained_model_name_or_path": "local/vl", "torch_dtype": "bfloat16"},
    "training": {"global_batch_size": 16, "micro_batch_size": 1, "max_grad_norm": 1.0},
    "accelerator": {"tp_size": 1, "pp_size": 2, "cp_size": 1, "ep_size": 1},
    "fsdp_config": {"dp_shard_size": 4},
    "activation_checkpoint": {"mode": "full"},
    "dataset": {"data_transform": {"max_seq_len": 4096}},
    "context": {"max_device_memory": "64GB", "device_num": 8},
}


def _small_deepseek(folder: str, interleave: int, **parallel: int) -> str:
    """A seven-layer DeepSeek, fully recomputed, with *interleave* chunks per stage.

    At DP 4, TP 2, PP 2, EP 2 and 4 micro-batches, but for what *parallel*
    states in MindFormers' ``parallel_config``.
    """
    with open(DEEPSEEK_YAML, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config["model"]["model_config"]
    model.update(num_layers=7, offset=0, pp_interleave_num=interleave)
    config["parallel_config"].update(data_parallel=4, model_parallel=2, pipeline_stage=2, expert_parallel=2,
                                     micro_batch_num=4)
    config["parallel_config"].update(parallel)
    config["moe_config"]["expert_num"] = 16
    config["recompute_config"]["recompute"] = True
    name = "_".join([f"deepseek_{interleave}"] + [f"{key}{value}" for key, value in sorted(parallel.items())])
    path = os.path.join(folder, f"{name}.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return path


def _stage_peaks(evaluator: EvaluatorV2, full_rec: bool) -> List[float]:
    """The memory model's peak of each stage, every layer fully recomputed or plain."""
    config = copy.deepcopy(evaluator.ccfg)
    config.full_rec, config.sel_rec = full_rec, False
    own = evaluator.ccfg
    evaluator.set_config(config)
    try:
        return [insight["Static"] + insight["Dynamic"] for insight in evaluator.estimate_peak_insight()]
    finally:
        evaluator.set_config(own)


def _stage_peaks_of(evaluator: EvaluatorV2, choice: RecomputeChoice) -> List[float]:
    """The memory model's peak of each stage with *choice* stated as recompute ranges."""
    ranges = []
    for item in choice.ranges:
        if item.option.recompute is None:
            ranges.append(RecomputeRange(first=item.first, count=item.count, option="full"))
        elif not item.option.recompute:
            ranges.append(RecomputeRange(first=item.first, count=item.count, option="none"))
        else:
            ranges.append(RecomputeRange(first=item.first, count=item.count, option="selective",
                                         ops=dict.fromkeys(item.option.recompute, "recompute")))
    config = copy.deepcopy(evaluator.ccfg)
    apply_exec(config, ExecSpec(recompute=tuple(ranges)))
    own = evaluator.ccfg
    evaluator.set_config(config)
    try:
        return [insight["Static"] + insight["Dynamic"] for insight in evaluator.estimate_peak_insight()]
    finally:
        evaluator.set_config(own)


def _stage_points_of(evaluator: EvaluatorV2, choice: Any) -> List[float]:
    """The memory model's stage memory, in MB, as a micro-batch's backward ends, with *choice* stated as ranges."""
    ranges = []
    for item in choice.ranges:
        if item.option.recompute is None:
            ranges.append(RecomputeRange(first=item.first, count=item.count, option="full"))
        elif not item.option.recompute:
            ranges.append(RecomputeRange(first=item.first, count=item.count, option="none"))
        else:
            ranges.append(RecomputeRange(first=item.first, count=item.count, option="selective",
                                         ops=dict.fromkeys(item.option.recompute, "recompute")))
    config = copy.deepcopy(evaluator.ccfg)
    apply_exec(config, ExecSpec(recompute=tuple(ranges)))
    own = evaluator.ccfg
    evaluator.set_config(config)
    try:
        insights = evaluator.estimate_peak_insight()
        return [insight["Static"] + backward for insight, (_, backward) in zip(insights, evaluator.peak_points)]
    finally:
        evaluator.set_config(own)


def _with_capacity(evaluator: EvaluatorV2, megabytes: float) -> EvaluatorV2:
    """*evaluator*, its device holding *megabytes*."""
    evaluator.ccfg.device_capacity.set(Memory.from_mb(megabytes))
    return evaluator


def _plain_values(config: Any) -> Dict[str, Any]:
    """The config's plain attributes."""
    return {name: value for name, value in vars(config).items() if isinstance(value, (bool, int, float, str))}


def _is_plain(option: LayerOption) -> bool:
    """Whether the option recomputes nothing."""
    return option.recompute == frozenset()


def _option(recompute: Any) -> LayerOption:
    """An option recomputing *recompute*, costs aside."""
    return LayerOption(recompute=recompute, memory_per_micro_batch=0.0, memory_once=0.0, forward_time=1.0,
                       backward_time=2.0)


def _per_layer(*spans: Tuple[int, int, Optional[str], Optional[str]]) -> RecomputeChoice:
    """A choice of a mode per layer, each span ``(first, count, kind's name, mode)`` running its mode's option."""
    ranges = tuple(
        LayerRange(first, count, None if kind is None else SimpleNamespace(name=kind),
                   _option(mode_recompute(mode, HYPER_SELECTIVE_REC_OP) if mode else frozenset({"ffAct"})), mode)
        for first, count, kind, mode in spans
    )
    return RecomputeChoice(ranges=ranges, stage_memory=(1.0,), stage_savings=(0.0,))


def _plan_modes(mode: str, layers: Dict[str, str], count: int) -> List[str]:
    """Each of *count* layers' mode, as the trainer reads a plan."""
    modes = [mode] * count
    for key, value in layers.items():
        first, _, last = key.partition("-")
        for index in range(int(first), int(last or first) + 1):
            modes[index] = value
    return modes


class TestOnDeepSeek(unittest.TestCase):
    """DeepSeek-V3 at its own strategy: 61 layers over 16 stages, 32 micro-batches, 58 GB."""

    @classmethod
    def setUpClass(cls) -> None:
        """The fixture's evaluator and its choice."""
        cls.evaluator = EvaluatorV2(DEEPSEEK_YAML, framework="mindformers", log_level=0)
        cls.choice = choose_recompute(cls.evaluator, Hard.Device_A2)

    def test_every_layer_gets_one_option_in_model_order(self):
        """
        Feature: choose_recompute.
        Description: The ranges of the choice.
        Expectation: They cover the 61 layers once, in order, each range
            one kind as the model gives it.
        """
        kinds = layer_kinds(self.evaluator.ccfg)
        start = 0
        for item in self.choice.ranges:
            self.assertEqual(item.first, start)
            self.assertEqual(set(kinds[item.first:item.first + item.count]), {item.kind})
            start += item.count
        self.assertEqual(start, len(kinds))

    def test_the_choice_fits_and_saves_time(self):
        """
        Feature: choose_recompute.
        Description: The memory and savings of the choice, every layer fully
            recomputed as the config says.
        Expectation: Every stage fits the device, none is slower, and some
            are faster.
        """
        capacity = self.evaluator.ccfg.device_capacity.to_mb().size
        self.assertEqual(len(self.choice.stage_memory), 16)
        self.assertLessEqual(self.choice.memory, capacity)
        self.assertTrue(all(saved >= 0 for saved in self.choice.stage_savings))
        self.assertGreater(sum(self.choice.stage_savings), 0)
        self.assertTrue(any(not _is_plain(item.option) for item in self.choice.ranges))

    def test_the_micro_batches_in_flight_are_the_memory_models(self):
        """
        Feature: micro_batches_in_flight.
        Description: 1F1B over 16 stages with 32 micro-batches.
        Expectation: Stage s keeps 16 - s in flight.
        """
        self.assertEqual(micro_batches_in_flight(self.evaluator), [[16 - stage] for stage in range(16)])

    def test_asking_twice_gives_the_same_choice_and_leaves_the_config_alone(self):
        """
        Feature: choose_recompute.
        Description: Choose again.
        Expectation: The same choice, and the evaluator's config and peak as
            they were.
        """
        before = _plain_values(self.evaluator.ccfg)
        peak = self.evaluator.estimate_peak()
        self.assertEqual(choose_recompute(self.evaluator, Hard.Device_A2), self.choice)
        self.assertEqual(_plain_values(self.evaluator.ccfg), before)
        self.assertEqual(self.evaluator.estimate_peak(), peak)


class TestStageMemory(unittest.TestCase):
    """The memory the choice says each stage keeps is what the memory model says."""

    @classmethod
    def setUpClass(cls) -> None:
        """Seven DeepSeek layers, without and with two chunks per stage."""
        cls.folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.paths = {interleave: _small_deepseek(cls.folder.name, interleave) for interleave in (1, 2)}

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the configs."""
        cls.folder.cleanup()

    def _evaluator(self, interleave: int) -> EvaluatorV2:
        """A fresh evaluator of the small DeepSeek."""
        return EvaluatorV2(self.paths[interleave], framework="mindformers", log_level=0)

    def test_a_roomy_device_runs_every_layer_plain(self):
        """
        Feature: choose_recompute.
        Description: A 1 TB device, one and two chunks per stage.
        Expectation: Every layer runs plain, and each stage keeps what the
            memory model says it keeps with every layer plain, to its MB.
        """
        for interleave in (1, 2):
            evaluator = _with_capacity(self._evaluator(interleave), 1024 * 1024)
            choice = choose_recompute(evaluator, Hard.Device_A2)
            self.assertTrue(all(_is_plain(item.option) for item in choice.ranges), interleave)
            for mine, model in zip(choice.stage_memory, _stage_peaks(evaluator, full_rec=False)):
                self.assertLessEqual(abs(mine - model), 1.0, interleave)

    def test_a_device_the_fully_recomputed_stages_fill_keeps_them_so(self):
        """
        Feature: choose_recompute.
        Description: A device a little larger than the heaviest stage fully
            recomputed, one and two chunks per stage.
        Expectation: The heaviest stage stays fully recomputed, and every
            stage fits.
        """
        for interleave in (1, 2):
            evaluator = self._evaluator(interleave)
            full = _stage_peaks(evaluator, full_rec=True)
            evaluator = _with_capacity(evaluator, max(full) + 16)
            choice = choose_recompute(evaluator, Hard.Device_A2)
            self.assertLessEqual(choice.memory, max(full) + 16)
            heaviest = full.index(max(full))
            self.assertEqual(choice.stage_savings[heaviest], 0.0, interleave)

    def test_a_smaller_device_never_runs_faster(self):
        """
        Feature: choose_recompute.
        Description: Devices from all plain down to all fully recomputed.
        Expectation: Every choice fits, and the time saved never grows as the
            device shrinks.
        """
        evaluator = self._evaluator(1)
        plain, full = max(_stage_peaks(evaluator, full_rec=False)), max(_stage_peaks(evaluator, full_rec=True))
        saved = []
        for share in (1.0, 0.75, 0.5, 0.25, 0.0):
            capacity = full + 16 + share * (plain - full)
            choice = choose_recompute(_with_capacity(evaluator, capacity), Hard.Device_A2)
            self.assertLessEqual(choice.memory, capacity)
            saved.append(sum(choice.stage_savings))
        self.assertEqual(saved, sorted(saved, reverse=True))
        self.assertGreater(saved[0], saved[-1])

    def test_a_mix_keeps_what_the_config_priced_whole_keeps(self):
        """
        Feature: choose_recompute.
        Description: Devices between all plain and all fully recomputed, one
            and two chunks per stage. Each choice is stated as recompute
            ranges, several selective settings included, and the whole
            config priced with them.
        Expectation: Each stage keeps what the choice says it keeps, to its
            MB, a choice whose last layer ends warm-up on a selective option
            and one that mixes selective settings included.
        """
        selective_ends = mixes = 0
        for interleave in (1, 2):
            evaluator = self._evaluator(interleave)
            plain, full = max(_stage_peaks(evaluator, full_rec=False)), max(_stage_peaks(evaluator, full_rec=True))
            for share in (0.7, 0.4, 0.35, 0.2, 0.175):
                choice = choose_recompute(_with_capacity(evaluator, full + 16 + share * (plain - full)),
                                          Hard.Device_A2)
                settings = {item.option.recompute for item in choice.ranges if item.option.recompute}
                mixes += len(settings) > 1
                selective_ends += bool(choice.ranges[-1].option.recompute)
                for mine, model in zip(choice.stage_memory, _stage_peaks_of(evaluator, choice)):
                    self.assertLessEqual(abs(mine - model), 1.0, (interleave, share, describe(choice)))
        self.assertGreater(selective_ends, 0)
        self.assertGreater(mixes, 0)

    def test_a_stage_between_the_first_and_the_last_keeps_what_the_config_priced_whole_keeps(self):
        """
        Feature: choose_recompute.
        Description: The small DeepSeek at DP 2, TP 4, PP 4 with 8
            micro-batches, whose stages keep 4, 3, 2 and 1 in flight, and
            devices between all plain and all fully recomputed. On the second
            stage a DP buffer hides the dense layer's gathers of the first
            micro-batches. Each choice is stated as recompute ranges and the
            whole config priced with them.
        Expectation: Each stage keeps what the choice says it keeps, to its
            MB.
        """
        path = _small_deepseek(self.folder.name, 1, data_parallel=2, model_parallel=4, pipeline_stage=4,
                               micro_batch_num=8)
        evaluator = EvaluatorV2(path, framework="mindformers", log_level=0)
        self.assertEqual(micro_batches_in_flight(evaluator), [[4], [3], [2], [1]])
        plain, full = max(_stage_peaks(evaluator, full_rec=False)), max(_stage_peaks(evaluator, full_rec=True))
        for share in (0.1, 0.5, 0.9):
            choice = choose_recompute(_with_capacity(evaluator, full + 16 + share * (plain - full)), Hard.Device_A2)
            for mine, model in zip(choice.stage_memory, _stage_peaks_of(evaluator, choice)):
                self.assertLessEqual(abs(mine - model), 1.0, (share, describe(choice)))


# The dense model at TP 1 on 1024 tokens, with a small vocabulary: under
# HyperParallel's FSDP, which holds each layer's gradient output until the
# backward ends, its first stage peaks as a later micro-batch's backward ends.
_DEFERRING = copy.deepcopy(_DENSE)
_DEFERRING["model"]["config_overrides"]["vocab_size"] = 4000
_DEFERRING["train"]["accelerator"]["tp_degree"] = 1
_DEFERRING["data"]["max_seq_len"] = 1024


class TestTwoPeaks(unittest.TestCase):
    """A stage whose peak can come as warm-up ends or as a later micro-batch's backward ends."""

    @classmethod
    def setUpClass(cls) -> None:
        """The dense model at TP 1, PP 2 and 4 micro-batches, fully recomputed."""
        cls.folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.path = os.path.join(cls.folder.name, "train.yaml")
        with open(cls.path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(_DEFERRING, handle)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the config."""
        cls.folder.cleanup()

    def _evaluator(self) -> EvaluatorV2:
        """A fresh evaluator of the dense model."""
        return EvaluatorV2(self.path, framework="hyper_v2", log_level=0)

    def test_the_memory_model_states_both_points(self):
        """
        Feature: EvaluatorV2.peak_points.
        Description: The dense model's stages, fully recomputed.
        Expectation: Each stage's dynamic memory is the higher of its two
            points, and the first stage's comes as a backward ends.
        """
        evaluator = self._evaluator()
        insights = evaluator.estimate_peak_insight()
        self.assertEqual([insight["Dynamic"] for insight in insights],
                         [max(points) for points in evaluator.peak_points])
        warm_up, backward = evaluator.peak_points[0]
        self.assertGreater(backward, warm_up)

    def test_a_roomy_device_runs_every_layer_plain(self):
        """
        Feature: choose_recompute.
        Description: A 1 TB device.
        Expectation: Every layer runs plain, and each stage keeps what the
            memory model says it keeps with every layer plain, to its MB,
            though the first stage peaks as a backward ends.
        """
        evaluator = _with_capacity(self._evaluator(), 1024 * 1024)
        choice = choose_recompute(evaluator, Hard.Device_A2)
        self.assertTrue(all(_is_plain(item.option) for item in choice.ranges))
        for mine, model in zip(choice.stage_memory, _stage_peaks(evaluator, full_rec=False)):
            self.assertLessEqual(abs(mine - model), 1.0)

    def test_a_mix_keeps_what_the_config_priced_whole_keeps(self):
        """
        Feature: choose_recompute.
        Description: Devices between all plain and all fully recomputed. Each
            choice is stated as recompute ranges, several selective settings
            included, and the whole config priced with them.
        Expectation: Every choice fits, and each stage keeps what the choice
            says it keeps, to its MB, one mode for every layer too.
        """
        evaluator = self._evaluator()
        plain, full = max(_stage_peaks(evaluator, full_rec=False)), max(_stage_peaks(evaluator, full_rec=True))
        for share in (0.7, 0.5, 0.3, 0.1, 0.0):
            capacity = full + 16 + share * (plain - full)
            for modes in (None, ("off", "full")):
                choice = choose_recompute(_with_capacity(evaluator, capacity), Hard.Device_A2, modes=modes)
                self.assertLessEqual(choice.memory, capacity)
                for mine, model in zip(choice.stage_memory, _stage_peaks_of(evaluator, choice)):
                    self.assertLessEqual(abs(mine - model), 1.0, (share, modes, describe(choice)))


# The dense model at DP shard 2: HyperParallel's FSDP reshards, so a layer's
# backward holds its own and the next layer's gathered parameters, in the
# buffers its tensor-parallel gathers take.
_RESHARDING = copy.deepcopy(_DENSE)
_RESHARDING["train"]["accelerator"]["dp_shard"] = 2


class TestWorkingSet(unittest.TestCase):
    """The working set of the backward of the layer that ends a stage's warm-up."""

    def test_each_option_of_the_layer_that_ends_warm_up_keeps_what_the_config_priced_whole_keeps(self):
        """
        Feature: the working set of the layer that ends warm-up.
        Description: The dense model at DP shard 2, TP 4 and PP 2 under
            HyperParallel's FSDP, which reshards: every layer plain but the
            first stage's last, which runs each option of its kind's front
            in turn, stated as recompute ranges and the whole config priced
            with them.
        Expectation: The first stage keeps what the search says it keeps, to
            its MB, whether the option keeps its gathers or recomputes them,
            though recomputing them leaves its backward's working set as it
            was.
        """
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "train.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(_RESHARDING, handle)
            evaluator = EvaluatorV2(path, framework="hyper_v2", log_level=0)
        counts = micro_batches_in_flight(evaluator)
        profiles = layer_profiles(evaluator, Hard.Device_A2, most_in_flight=max(max(row) for row in counts),
                                  in_flight=[count for row in counts for count in row])
        fronts = {key: build_front(profile) for key, profile in profiles.items()}
        layers, ends = Candidate._body_layers(evaluator, fronts, counts)
        layers, fronts, _, own = Candidate._charge_working_sets(layers, ends, fronts, None)
        _, peaks = Candidate._stages(evaluator, layers, fronts, own)
        ending = next(layer for layer in layers[0] if layer.index == ends[0][0].index)
        plain = {layer.index: Candidate._plain(fronts[layer.key]) for stage in layers for layer in stage}
        total = sum(len(stage) for stage in layers)
        kept_plain = own.get(plain[ending.index], plain[ending.index])
        recomputing_gathers = 0
        for option in fronts[ending.key]:
            original = own.get(option, option)
            recomputing_gathers += original.recompute is None or "gather" in original.recompute
            mine = Candidate._stage_memory(layers[0], peaks[0], {**plain, ending.index: option}, fronts, own)
            whole = _stage_peaks_of(evaluator, SimpleNamespace(ranges=(
                LayerRange(0, ending.index, None, kept_plain),
                LayerRange(ending.index, 1, None, original),
                LayerRange(ending.index + 1, total - ending.index - 1, None, kept_plain),
            )))
            self.assertLessEqual(abs(mine / MEGABYTE - whole[0]), 1.0, option_label(original))
        self.assertGreater(recomputing_gathers, 1)

    def test_each_option_of_the_first_layer_keeps_what_the_backward_end_priced_whole_keeps(self):
        """
        Feature: the working set of the backward a stage runs last.
        Description: The same model, every layer plain but the first
            stage's first, which runs each option of its kind's front in
            turn, stated as recompute ranges and the whole config priced
            with them.
        Expectation: As a micro-batch's backward ends, the first stage
            keeps what the search says it keeps, to its MB: the first
            layer's backward holds its own gathered parameters alone, with
            none left to prefetch.
        """
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "train.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(_RESHARDING, handle)
            evaluator = EvaluatorV2(path, framework="hyper_v2", log_level=0)
        counts = micro_batches_in_flight(evaluator)
        profiles = layer_profiles(evaluator, Hard.Device_A2, most_in_flight=max(max(row) for row in counts),
                                  in_flight=[count for row in counts for count in row])
        fronts = {key: build_front(profile) for key, profile in profiles.items()}
        layers, ends = Candidate._body_layers(evaluator, fronts, counts)
        layers, fronts, _, own = Candidate._charge_working_sets(layers, ends, fronts, None)
        _, peaks = Candidate._stages(evaluator, layers, fronts, own)
        self.assertIsNotNone(peaks[0].backward)
        first = layers[0][0]
        plain = {layer.index: Candidate._plain(fronts[layer.key]) for stage in layers for layer in stage}
        total = sum(len(stage) for stage in layers)
        kept_plain = own.get(plain[1], plain[1])
        recomputing_gathers = 0
        for option in fronts[first.key]:
            recomputing_gathers += option.recompute is None or "gather" in option.recompute
            chosen = {**plain, first.index: option}
            mine = peaks[0].backward + Candidate._backward_kept(layers[0], chosen, fronts, own)
            whole = _stage_points_of(evaluator, SimpleNamespace(ranges=(
                LayerRange(0, 1, None, option), LayerRange(1, total - 1, None, kept_plain),
            )))
            self.assertLessEqual(abs(mine / MEGABYTE - whole[0]), 1.0, option_label(option))
        self.assertGreater(recomputing_gathers, 1)


# The resharding dense model with a census of its layers' kind, which states
# what HyperParallel's selective policy keeps and the shares of the matmuls it
# recomputes: the census prices the plain layer and the policy, and the
# formulas every other setting.
_CENSUS = copy.deepcopy(_RESHARDING)
_CENSUS["model"]["config_overrides"]["activations"] = {"decoder": {
    "saved": 30000.0, "saved_tp": 60000.0, "working": 40000.0, "working_tp": 90000.0, "seq_length": 4096,
    "selective": 8000.0, "selective_tp": 12000.0, "selective_attention_mm": 0.3, "selective_ffn_mm": 0.5}}
_POLICY = frozenset(name for name, state in HYPER_SELECTIVE_REC_OP.items() if not state)
# The same census stating what the layer keeps for each op, the ops' bytes
# summing to what it keeps plain, and a backward that holds less than that.
_CENSUS_BY_OP = copy.deepcopy(_CENSUS)
_CENSUS_BY_OP["model"]["config_overrides"]["activations"]["decoder"].update({
    "working": 24000.0, "working_tp": 36000.0,
    "ops": {"attMM": 6000.0, "ffMM": 12000.0, "normOp": 9000.0, "other": 3000.0},
    "ops_tp": {"attMM": 12000.0, "attBMM": 16000.0, "softmax": 4000.0, "ffMM": 12000.0, "normOp": 8000.0,
               "ffAct": 8000.0}})


class TestCensus(unittest.TestCase):
    """The options of a layer kind a census prices."""

    CONFIG = _CENSUS

    @classmethod
    def setUpClass(cls) -> None:
        """The model's profile and fronts, and its first stage with its peaks at both points."""
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "train.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(cls.CONFIG, handle)
            cls.evaluator = EvaluatorV2(path, framework="hyper_v2", log_level=0)
        counts = micro_batches_in_flight(cls.evaluator)
        profiles = layer_profiles(cls.evaluator, Hard.Device_A2, most_in_flight=max(max(row) for row in counts),
                                  in_flight=[count for row in counts for count in row])
        cls.kind = next(iter(profiles))
        cls.profile = profiles[cls.kind]
        fronts = {key: build_front(profile) for key, profile in profiles.items()}
        layers, ends = Candidate._body_layers(cls.evaluator, fronts, counts)
        layers, cls.fronts, _, cls.own = Candidate._charge_working_sets(layers, ends, fronts, None)
        _, cls.peaks = Candidate._stages(cls.evaluator, layers, cls.fronts, cls.own)
        cls.stage = layers[0]
        cls.ending = next(layer for layer in cls.stage if layer.index == ends[0][0].index)
        cls.plain = {layer.index: Candidate._plain(cls.fronts[layer.key]) for stage in layers for layer in stage}
        cls.total = sum(len(stage) for stage in layers)

    def test_the_policy_is_measured_whole(self):
        """
        Feature: the profile of a kind a census prices.
        Description: The setting of HyperParallel's selective policy, as
            measured and as its switches would add up.
        Expectation: Measured whole: it keeps what the census says, less
            than the formulas' switches add up to, and its backward is
            slower, recomputing the matmuls no switch covers. It is on the
            kind's front.
        """
        measured = self.profile.selective(_POLICY)
        added = dataclasses.replace(self.profile, whole={}).selective(_POLICY)
        self.assertLess(measured.memory_per_micro_batch, added.memory_per_micro_batch)
        self.assertGreater(measured.backward_time, added.backward_time)
        self.assertIn(_POLICY, [option.recompute for option in self.fronts[self.kind]])

    def test_each_option_of_the_layer_that_ends_warm_up_keeps_what_the_config_priced_whole_keeps(self):
        """
        Feature: the working set of the layer that ends warm-up, for a kind
            a census prices.
        Description: The model at DP shard 2, TP 4 and PP 2, every layer
            plain but the first stage's last, which runs each option of its
            kind's front in turn, stated as recompute ranges and the whole
            config priced with them.
        Expectation: The first stage keeps what the search says it keeps, to
            its MB: the plain layer and the policy as the census prices
            them, the plain layer's working set beyond what the stage keeps
            for it already, and every other setting as the census's records
            per op, or else the formulas, price it.
        """
        # pylint: disable=protected-access
        kept_plain = self.own.get(self.plain[self.ending.index], self.plain[self.ending.index])
        index = self.ending.index
        labels = []
        for option in self.fronts[self.ending.key]:
            original = self.own.get(option, option)
            labels.append(original.recompute)
            mine = Candidate._stage_memory(self.stage, self.peaks[0], {**self.plain, index: option}, self.fronts,
                                           self.own)
            whole = _stage_peaks_of(self.evaluator, SimpleNamespace(ranges=(
                LayerRange(0, index, None, kept_plain),
                LayerRange(index, 1, None, original),
                LayerRange(index + 1, self.total - index - 1, None, kept_plain),
            )))
            self.assertLessEqual(abs(mine / MEGABYTE - whole[0]), 1.0, option_label(original))
        self.assertIn(_POLICY, labels)

    def test_each_option_of_the_first_layer_keeps_what_the_backward_end_priced_whole_keeps(self):
        """
        Feature: the working set of the backward a stage runs last, for a
            kind a census prices.
        Description: The same model, every layer plain but the first
            stage's first, which runs each option of its kind's front in
            turn, stated as recompute ranges and the whole config priced
            with them.
        Expectation: As a micro-batch's backward ends, the first stage
            keeps what the search says it keeps, to its MB.
        """
        # pylint: disable=protected-access
        first = self.stage[0]
        kept_plain = self.own.get(self.plain[1], self.plain[1])
        self.assertIsNotNone(self.peaks[0].backward)
        for option in self.fronts[first.key]:
            chosen = {**self.plain, first.index: option}
            mine = self.peaks[0].backward + Candidate._backward_kept(self.stage, chosen, self.fronts, self.own)
            whole = _stage_points_of(self.evaluator, SimpleNamespace(ranges=(
                LayerRange(0, 1, None, option), LayerRange(1, self.total - 1, None, kept_plain),
            )))
            self.assertLessEqual(abs(mine / MEGABYTE - whole[0]), 1.0, option_label(option))

    def test_the_runtimes_selective_mode_runs_the_policy_the_census_prices(self):
        """
        Feature: choose_recompute modes, with the runtime's own selective
            switches.
        Description: The model on a device between what it keeps with
            every layer running HyperParallel's selective policy and what it
            keeps plain, choosing among off, selective and full as the
            trainer runs them, its selective mode the policy whatever the
            config's switches.
        Expectation: Selective, every layer running the policy, faster than
            full recompute, and each stage keeps what the config priced
            whole with the policy keeps, to its MB.
        """
        evaluator = self.evaluator
        policy = SimpleNamespace(ranges=(LayerRange(0, self.total, None, _option(_POLICY)),))
        selective = max(_stage_peaks_of(evaluator, policy))
        plain = max(_stage_peaks(evaluator, full_rec=False))
        self.assertLess(selective + 16, plain)
        own = evaluator.ccfg.device_capacity.to_mb().size
        try:
            _with_capacity(evaluator, selective + 16)
            choice = choose_recompute(evaluator, Hard.Device_A2, modes=MODES,
                                      selective=dict(HYPER_SELECTIVE_REC_OP))
            full = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"))
        finally:
            _with_capacity(evaluator, own)
        self.assertEqual(choice.mode, "selective")
        self.assertEqual({item.option.recompute for item in choice.ranges}, {_POLICY})
        self.assertEqual(full.mode, "full")
        self.assertGreater(sum(choice.stage_savings), sum(full.stage_savings))
        for mine, model in zip(choice.stage_memory, _stage_peaks_of(evaluator, choice)):
            self.assertLessEqual(abs(mine - model), 1.0)


class TestCensusByOp(TestCensus):
    """The options of a layer kind a census prices, its records per op pricing every setting but the policy."""

    CONFIG = _CENSUS_BY_OP

    def test_a_selective_layer_keeping_every_op_keeps_what_the_plain_one_keeps(self):
        """
        Feature: the profile of a kind a census's records per op price.
        Description: The layer selective with every op kept, which the
            settings add up from, beside the plain layer; and recomputing
            the norms.
        Expectation: The two keep the same, where the formulas priced the
            base below the census's plain layer; the norms free what the
            census states for them. The profile states what the backward
            holds, for the working sets to be clamped.
        """
        base, plain = self.profile.selective_base, self.profile.plain
        self.assertEqual(base.values()[:2], plain.values()[:2])
        freed = base.memory_per_micro_batch - self.profile.selective(["normOp"]).memory_per_micro_batch
        self.assertGreater(freed, 0)
        self.assertIsNotNone(self.profile.census_held)


class TestRuntimeSelective(unittest.TestCase):
    """A runtime's own selective mode, where no census prices it."""

    def test_without_a_census_the_mode_is_not_offered(self):
        """
        Feature: choose_recompute modes, with the runtime's own selective
            switches.
        Description: The resharding dense model, no census stating its
            kind, on a device every mode fits, choosing among off,
            selective and full with HyperParallel's policy as the selective
            mode.
        Expectation: Off, and the selective mode is left out: the formulas
            do not price the matmuls the policy recomputes.
        """
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "train.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(_RESHARDING, handle)
            evaluator = _with_capacity(EvaluatorV2(path, framework="hyper_v2", log_level=0), 1024 * 1024)
        choice = choose_recompute(evaluator, Hard.Device_A2, modes=MODES, selective=dict(HYPER_SELECTIVE_REC_OP))
        self.assertEqual(choice.mode, "off")
        # pylint: disable=protected-access
        profiles = layer_profiles(evaluator, Hard.Device_A2, each_switch=False)
        self.assertEqual(Candidate._priced_modes(MODES, profiles, HYPER_SELECTIVE_REC_OP), ("off", "full"))


def _kind(per_micro_batch: int, once: int, forward: float) -> Tuple[LayerOption, ...]:
    """A kind's options, from its plain layer's MB per micro-batch and once and its forward time."""
    options = []
    for recompute, kept, slower in ((frozenset(), per_micro_batch, 0.0), (frozenset({"ffAct"}), per_micro_batch // 2,
                                                                          0.2), (None, 2, 1.0)):
        options.append(LayerOption(recompute=recompute, memory_per_micro_batch=kept * MEGABYTE,
                                   memory_once=once * MEGABYTE, forward_time=forward,
                                   backward_time=(2.0 + slower) * forward))
    return tuple(options)


class TestTwoPeaksChoice(unittest.TestCase):
    """A stage's choice when its peak can also come as a later micro-batch's backward ends."""

    def test_the_choice_is_the_fastest_that_fits_at_both_points(self):
        """
        Feature: the choice of a stage that peaks at two points.
        Description: Random stages of one to four layers of two kinds, the
            last ending warm-up, kept at one to three micro-batches in
            flight, what the stage keeps outside its layers at each point,
            and devices from too small for any choice to roomy.
        Expectation: Every choice fits at both points and none that fits is
            faster, trying every one; where none fits, there is no choice.
        """
        rng = random.Random(11)
        # pylint: disable=protected-access
        for _ in range(80):
            fronts = {("unit", kind): _kind(rng.randint(20, 200), rng.randint(0, 50), rng.uniform(1.0, 5.0))
                      for kind in "ab"}
            in_flight = rng.randint(1, 3)
            layers = []
            for index in range(rng.randint(1, 4)):
                key = ("unit", rng.choice("ab"))
                layers.append(Candidate._Layer(index, key, in_flight, fronts[key][2]))
            (stage_layers,), fronts, _, own = Candidate._charge_working_sets([layers], [[layers[-1]]], fronts, None)
            peaks = Candidate._Peaks(rng.randint(0, 500) * MEGABYTE, rng.randint(0, 900) * MEGABYTE)
            capacity = rng.randint(200, 2500) * MEGABYTE
            stage = Stage(groups=Candidate._groups(stage_layers), budget=capacity - peaks.warm_up)
            chosen = Candidate._choose_stage(stage_layers, stage, peaks, fronts, own, MEGABYTE, capacity)
            best = None
            for picks in itertools.product(*(fronts[layer.key] for layer in stage_layers)):
                trial = {layer.index: option for layer, option in zip(stage_layers, picks)}
                if Candidate._stage_memory(stage_layers, peaks, trial, fronts, own) <= capacity:
                    time = sum(option.forward_time + option.backward_time for option in picks)
                    best = time if best is None else min(best, time)
            with self.subTest(layers=[layer.key for layer in stage_layers], in_flight=in_flight, peaks=peaks,
                              capacity=capacity):
                if best is None:
                    self.assertIsNone(chosen)
                    continue
                self.assertLessEqual(Candidate._stage_memory(stage_layers, peaks, chosen, fronts, own), capacity)
                self.assertAlmostEqual(sum(option.forward_time + option.backward_time for option in chosen.values()),
                                       best, places=6)


# A link so fast that a copy takes no time, and one too slow for any layer's.
_FAST_LINK = Hard.HostLink(gib_per_s=10.0 ** 6, sustained_tflops=140.0)
_SLOW_LINK = Hard.HostLink(gib_per_s=10.0 ** -3, sustained_tflops=140.0)


def _offloaded_indices(choice: RecomputeChoice) -> List[int]:
    """The layers the choice offloads, in model order."""
    return [index for item in choice.ranges if item.option.link_bandwidth
            for index in range(item.first, item.first + item.count)]


class TestOffloadWindow(unittest.TestCase):
    """A stage's first layers offload what their copies carry while the forward runs."""

    @staticmethod
    def _shares(copies: Sequence[float], ending: Optional[int] = None, per_byte: float = 1.0) -> List[float]:
        """What each of four layers with a forward of 1 moves to the host, keeping *copies* bytes each."""
        stage_layers = []
        for position in range(len(copies)):
            key = ("unit", None) + (("ends warm-up",) if position == ending else ())
            stage_layers.append(Candidate._Layer(position, key, 1, _option(frozenset())))  # pylint: disable=protected-access
        plain = [LayerOption(recompute=frozenset(), memory_per_micro_batch=copy, memory_once=0.0, forward_time=1.0,
                             backward_time=2.0) for copy in copies]
        link = Candidate._Link(per_byte=per_byte, overlap=1.0)  # pylint: disable=protected-access
        return Candidate._offload_shares(stage_layers, plain, link)  # pylint: disable=protected-access

    def test_a_layer_moves_what_the_forward_left_after_it_carries(self):
        """
        Feature: the offload window.
        Description: Four layers with a forward of 1 each, whose copies take
            half a forward, one forward, one and a half forwards, or four.
        Expectation: One stream copies each layer's activations once its
            forward ends and is done when the forward is: three layers move
            all they keep at half and at one forward, two at one and a half,
            and at four the first moves what the three forwards after it
            carry, three quarters of what it keeps; the last layer, with no
            forward left after it, nothing.
        """
        self.assertEqual(self._shares([0.5] * 4), [0.5, 0.5, 0.5, 0.0])
        self.assertEqual(self._shares([1.0] * 4), [1.0, 1.0, 1.0, 0.0])
        self.assertEqual(self._shares([1.5] * 4), [1.5, 1.5, 0.0, 0.0])
        self.assertEqual(self._shares([4.0] * 4), [3.0, 0.0, 0.0, 0.0])

    def test_the_layer_that_ends_warm_up_keeps_its_activations(self):
        """
        Feature: the offload window.
        Description: Copies that take no time, the third layer ending warm-up.
        Expectation: Only the two layers before it move anything, all they
            keep.
        """
        self.assertEqual(self._shares([1.0] * 4, ending=2, per_byte=0.0), [1.0, 1.0])

    @staticmethod
    def _unit_layers(count: int) -> Tuple[Dict[Any, Tuple[LayerOption, ...]], List[Any]]:
        """*count* layers of one kind that keep 4 MB plain and 1 MB fully recomputed, with a forward of 1."""
        # pylint: disable=protected-access
        plain = LayerOption(recompute=frozenset(), memory_per_micro_batch=4 * MEGABYTE, memory_once=0.0,
                            forward_time=1.0, backward_time=2.0)
        full = LayerOption(recompute=None, memory_per_micro_batch=MEGABYTE, memory_once=0.0, forward_time=1.0,
                           backward_time=3.0)
        fronts = {("unit", None): (plain, full)}
        return fronts, [Candidate._Layer(index, ("unit", None), 1, full) for index in range(count)]

    def _offload_stage(self, count: int, budget: float, per_byte: float,
                       cost: float = 0.0) -> Tuple[Dict[int, LayerOption], float]:
        """The choice of *count* unit layers in *budget* MB, with a link that takes *per_byte* forwards a MB.

        Each MB moved costs the step *cost*.
        """
        # pylint: disable=protected-access
        fronts, layers = self._unit_layers(count)
        stage = Stage(groups=Candidate._groups(layers), budget=budget * MEGABYTE)
        link = Candidate._Link(per_byte=per_byte / MEGABYTE, overlap=1.0, cost_per_byte=cost / MEGABYTE)
        return Candidate._offload_stage(layers, stage, Candidate._Peaks(0.0), fronts, {}, link, MEGABYTE,
                                        budget * MEGABYTE)

    def test_the_copies_cost_time_and_only_what_the_stage_needs_moves(self):
        """
        Feature: the choice of a stage whose first layers may offload, the copies costing time.
        Description: The three layers in 8 MB, a link that carries 2 MB in
            the two forwards after the first, and copies that cost 0.1 a MB
            moved; then 0.3 a MB.
        Expectation: At 0.1 the first layer runs plain and offloads only the
            1 MB the stage needs to fit, its time 3 and 0.1 for each MB out
            and back, 3.2, and the stage fills its 8 MB. At 0.3 the 2 MB
            the third layer's plain run needs cost 1.2, more than the 1 it
            saves, and nothing offloads.
        """
        chosen, transit = self._offload_stage(3, 8, 1.0, cost=0.1)
        self.assertEqual(transit, 0.0)
        self.assertEqual(chosen[0].link_bandwidth, 1 * MEGABYTE)
        self.assertAlmostEqual(chosen[0].forward_time + chosen[0].backward_time, 3.2)
        self.assertEqual([chosen[index].recompute for index in range(3)], [frozenset(), None, frozenset()])
        self.assertEqual(sum(chosen[index].memory(1) for index in range(3)), 8 * MEGABYTE)
        chosen, _ = self._offload_stage(3, 8, 1.0, cost=0.3)
        self.assertFalse(any(option.link_bandwidth for option in chosen.values()))

    def test_a_budget_of_a_fraction_of_a_megabyte_is_filled_to_its_whole_megabytes(self):
        """
        Feature: the choice of a stage whose first layers may offload, the copies costing time.
        Description: The three layers in 7.5 MB, with copies that cost 0.1 a
            MB moved, so the stage gives back what it does not need.
        Expectation: The stage keeps 7 MB, the whole megabytes of its budget,
            and offloads the 2 MB that takes: filled to the byte, it would
            keep 7.5 MB, which a result reported in whole megabytes reads as
            8, over the budget its own fit check accepted.
        """
        chosen, _ = self._offload_stage(3, 7.5, 1.0, cost=0.1)
        kept = sum(chosen[index].memory(1) for index in range(3))
        self.assertEqual(kept, 7 * MEGABYTE, f"kept {kept / MEGABYTE} MB in a 7.5 MB budget")
        self.assertEqual(chosen[0].link_bandwidth, 2 * MEGABYTE,
                         f"offloaded {chosen[0].link_bandwidth / MEGABYTE} MB, expected 2")

    def test_a_layer_offloads_the_part_its_window_carries(self):
        """
        Feature: the choice of a stage whose first layers may offload.
        Description: Three layers that keep 4 MB plain and 1 MB fully
            recomputed, in 7 MB, which hold one plain layer, and a link that
            carries 2 MB in the two forwards after the first.
        Expectation: The first layer runs plain and moves half of what it
            keeps to the host, and the third runs plain too: its backward
            releases room for those 2 MB to come back, so nothing is held in
            transit and the stage keeps 7 MB.
        """
        chosen, transit = self._offload_stage(3, 7, 1.0)
        self.assertEqual(transit, 0.0)
        self.assertEqual([chosen[index].link_bandwidth for index in range(3)], [2 * MEGABYTE, 0.0, 0.0])
        self.assertEqual([chosen[index].recompute for index in range(3)], [frozenset(), None, frozenset()])
        self.assertEqual(sum(chosen[index].memory(1) for index in range(3)), 7 * MEGABYTE)
        self.assertEqual(option_label(chosen[0]), "no recompute, 50% offloaded to the host")

    def test_a_copy_back_with_no_room_to_land_is_not_offloaded(self):
        """
        Feature: the choice of a stage whose first layers may offload.
        Description: Two such layers in 6 MB, which hold one plain layer,
            and a link that carries 2 MB in the forward of the second.
        Expectation: Nothing offloads: the first layer's 2 MB would come
            back while the second's backward holds all it keeps, and held in
            transit they leave the second no more room than it has without
            the link.
        """
        chosen, transit = self._offload_stage(2, 6, 0.5)
        self.assertEqual(transit, 0.0)
        self.assertFalse(any(option.link_bandwidth for option in chosen.values()))
        self.assertEqual(sorted(chosen[index].recompute is None for index in range(2)), [False, True])

    def test_no_layer_offloads_to_save_only_rounding(self):
        """
        Feature: the choice of a stage whose first layers may offload.
        Description: Three layers of 0.1, 0.2 and 0.3 that all fit plain,
            and copies that take no time. Offloading the first sums the
            same times in another order, 0.1 + (0.3 + 0.2) = 0.6 against
            (0.1 + 0.2) + 0.3 = 0.6000000000000001.
        Expectation: No layer offloads.
        """
        # pylint: disable=protected-access
        fronts, layers = {}, []
        for index, forward in enumerate((0.1, 0.2, 0.3)):
            plain = LayerOption(recompute=frozenset(), memory_per_micro_batch=MEGABYTE, memory_once=0.0,
                                forward_time=forward, backward_time=0.0)
            fronts["unit", index] = (plain,)
            layers.append(Candidate._Layer(index, ("unit", index), 1, plain))
        capacity = 64 * MEGABYTE
        stage = Stage(groups=Candidate._groups(layers), budget=capacity)
        link = Candidate._Link(per_byte=0.0, overlap=1.0)
        chosen, transit = Candidate._offload_stage(layers, stage, Candidate._Peaks(0.0), fronts, {}, link, MEGABYTE,
                                                   capacity)
        self.assertFalse(any(option.link_bandwidth for option in chosen.values()))
        self.assertEqual(transit, 0.0)


class TestOffload(unittest.TestCase):
    """With a host link, a choice per layer may offload each stage's first layers."""

    @classmethod
    def setUpClass(cls) -> None:
        """The small DeepSeek at PP 1 and at PP 2, and at two chunks per stage."""
        cls.folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.paths = {
            "pp1": _small_deepseek(cls.folder.name, 1, pipeline_stage=1, data_parallel=8),
            "pp2": _small_deepseek(cls.folder.name, 1),
            "vpp": _small_deepseek(cls.folder.name, 2),
        }

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the configs."""
        cls.folder.cleanup()

    def _evaluator(self, name: str, share: float) -> EvaluatorV2:
        """The small DeepSeek *name*, its device *share* of the way from all fully recomputed to all plain."""
        evaluator = EvaluatorV2(self.paths[name], framework="mindformers", log_level=0)
        plain, full = max(_stage_peaks(evaluator, full_rec=False)), max(_stage_peaks(evaluator, full_rec=True))
        return _with_capacity(evaluator, full + 16 + share * (plain - full))

    def test_early_layers_offload_and_the_last_cannot(self):
        """
        Feature: choose_recompute with a host link.
        Description: PP 1, a device a fifth of the way from all fully
            recomputed to all plain, A2's placeholder link and one whose
            copies take no time.
        Expectation: Each offloads a run of first layers, which run plain,
            the faster link more of them, and never the last layer.
        """
        evaluator = self._evaluator("pp1", 0.2)
        last = len(layer_kinds(evaluator.ccfg)) - 1
        offloaded = {}
        for name, link in (("A2", Hard.Device_A2.host_link), ("fast", _FAST_LINK)):
            choice = choose_recompute(evaluator, Hard.Device_A2, link=link)
            offloaded[name] = _offloaded_indices(choice)
            self.assertEqual(offloaded[name], list(range(len(offloaded[name]))), name)
            self.assertNotIn(last, offloaded[name], name)
            self.assertTrue(all(item.option.recompute == frozenset() for item in choice.ranges
                                if item.option.link_bandwidth), name)
        self.assertGreater(len(offloaded["A2"]), 0)
        self.assertGreater(len(offloaded["fast"]), len(offloaded["A2"]))

    def test_an_offloading_choice_fits_and_is_no_slower(self):
        """
        Feature: choose_recompute with a host link.
        Description: PP 1 and PP 2, devices a fifth and three fifths of the
            way from all fully recomputed to all plain, with a link whose
            copies take no time.
        Expectation: Every stage fits the device, and the layers save at
            least as much time as they do without the link, more in some.
        """
        gained = 0
        for name in ("pp1", "pp2"):
            for share in (0.2, 0.6):
                evaluator = self._evaluator(name, share)
                capacity = evaluator.ccfg.device_capacity.to_mb().size
                without = choose_recompute(evaluator, Hard.Device_A2)
                choice = choose_recompute(evaluator, Hard.Device_A2, link=_FAST_LINK)
                self.assertLessEqual(choice.memory, capacity, (name, share))
                self.assertGreaterEqual(sum(choice.stage_savings), sum(without.stage_savings), (name, share))
                gained += sum(choice.stage_savings) > sum(without.stage_savings)
        self.assertGreater(gained, 0)

    def test_a_link_too_slow_for_any_layer_changes_nothing(self):
        """
        Feature: choose_recompute with a host link.
        Description: PP 1, a link too slow for any layer's copy.
        Expectation: The choice made without a link.
        """
        evaluator = self._evaluator("pp1", 0.2)
        self.assertEqual(choose_recompute(evaluator, Hard.Device_A2, link=_SLOW_LINK),
                         choose_recompute(evaluator, Hard.Device_A2))

    def test_one_mode_and_two_chunks_per_stage_do_not_offload(self):
        """
        Feature: choose_recompute with a host link.
        Description: One mode for every layer, and two chunks per stage.
        Expectation: Each the choice made without a link: a runtime with one
            mode runs no offload, and offload is priced at one chunk per
            stage.
        """
        evaluator = self._evaluator("pp1", 0.2)
        self.assertEqual(choose_recompute(evaluator, Hard.Device_A2, modes=MODES, link=_FAST_LINK),
                         choose_recompute(evaluator, Hard.Device_A2, modes=MODES))
        evaluator = self._evaluator("vpp", 0.2)
        self.assertEqual(choose_recompute(evaluator, Hard.Device_A2, link=_FAST_LINK),
                         choose_recompute(evaluator, Hard.Device_A2))

    def test_a_mode_per_layer_offloads_its_first_layers_running_off(self):
        """
        Feature: choose_recompute with a host link and a mode per layer.
        Description: PP 1, a device a fifth of the way from all fully
            recomputed to all plain, off or full for each layer, with and
            without a link whose copies take no time.
        Expectation: With the link a run of first layers offloads, each
            running the plain option and naming the mode off; every range
            names a mode, the trainer's plan states each offloaded layer
            off, and the layers save at least as much time as without it.
        """
        evaluator = self._evaluator("pp1", 0.2)
        modes = ("off", "full")
        without = choose_recompute(evaluator, Hard.Device_A2, modes=modes, per_layer=True)
        choice = choose_recompute(evaluator, Hard.Device_A2, modes=modes, per_layer=True, link=_FAST_LINK)
        offloaded = _offloaded_indices(choice)
        self.assertGreater(len(offloaded), 0)
        self.assertEqual(offloaded, list(range(len(offloaded))))
        self.assertTrue(all(item.mode in modes for item in choice.ranges))
        self.assertTrue(all(item.mode == "off" and item.option.recompute == frozenset()
                            for item in choice.ranges if item.option.link_bandwidth))
        plan = _plan_modes(*trainer_plan(choice), sum(item.count for item in choice.ranges))
        self.assertEqual([plan[index] for index in offloaded], ["off"] * len(offloaded))
        self.assertGreaterEqual(sum(choice.stage_savings), sum(without.stage_savings))

    def test_a_mode_per_layer_without_off_does_not_offload(self):
        """
        Feature: choose_recompute with a host link and a mode per layer.
        Description: Full recompute the only mode, chosen per layer, with a
            link whose copies take no time.
        Expectation: The choice made without a link: an offloaded layer
            runs off, which the modes do not offer.
        """
        evaluator = self._evaluator("pp1", 0.2)
        self.assertEqual(choose_recompute(evaluator, Hard.Device_A2, modes=("full",), per_layer=True,
                                          link=_FAST_LINK),
                         choose_recompute(evaluator, Hard.Device_A2, modes=("full",), per_layer=True))

    def test_a_calibrated_link_converts_with_the_compute_ratio(self):
        """
        Feature: HostLink.ms_per_unit and HostLink.of.
        Description: A link priced at its sustained throughput, the same
            link with a COMPUTE ratio stated, and links built from figures.
        Expectation: A copy's seconds convert at the throughput times the
            precision's bytes, or at 1000 over the ratio; a ratio that is
            not positive is refused, as is a device with no link and no
            figures; HostLink.of replaces only the figures stated.
        """
        # pylint: disable=protected-access
        evaluator = self._evaluator("pp1", 0.2)
        link = Hard.HostLink(gib_per_s=8.0, sustained_tflops=100.0)
        seconds = 1.0 / (8.0 * 2 ** 30)
        plain = Candidate._link(link, evaluator)
        self.assertAlmostEqual(plain.per_byte / (seconds * 100e12 * evaluator.ccfg.bytes_p), 1.0, places=12)
        calibrated = Candidate._link(Hard.HostLink.of(link, {"ms_per_unit": 5e-11, "gib_per_s": None}), evaluator)
        self.assertAlmostEqual(calibrated.per_byte / (seconds * 1000.0 / 5e-11), 1.0, places=12)
        self.assertEqual(calibrated.overlap, link.overlap)
        with self.assertRaises(ValueError):
            Hard.HostLink(gib_per_s=8.0, sustained_tflops=100.0, ms_per_unit=0.0)
        with self.assertRaises(ValueError):
            Hard.HostLink.of(None, {"gib_per_s": 8.0, "sustained_tflops": None})
        self.assertEqual(Hard.HostLink.of(None, {"gib_per_s": 8.0, "sustained_tflops": 100.0}), link)
        self.assertEqual(Hard.HostLink.of(link, {"overlap": 1.0}), Hard.HostLink(8.0, 100.0, overlap=1.0))
        # The copies' cost converts with the score's own ratio where one is stated, else as a copy does.
        costly = Hard.HostLink(gib_per_s=8.0, sustained_tflops=100.0, ms_per_unit=2e-11, copy_cost_ms_per_gib=2.8,
                               score_ms_per_unit=5e-11)
        cost_seconds = 2.8e-3 / 2 ** 30
        self.assertAlmostEqual(Candidate._link(costly, evaluator).cost_per_byte / (cost_seconds * 1000.0 / 5e-11), 1.0,
                               places=12)
        unscored = dataclasses.replace(costly, score_ms_per_unit=None)
        self.assertAlmostEqual(Candidate._link(unscored, evaluator).cost_per_byte / (cost_seconds * 1000.0 / 2e-11),
                               1.0, places=12)
        self.assertEqual(Candidate._link(link, evaluator).cost_per_byte, 0.0)
        with self.assertRaises(ValueError):
            Hard.HostLink(gib_per_s=8.0, sustained_tflops=100.0, copy_cost_ms_per_gib=-1.0)

    def test_an_offloading_choice_reads_so(self):
        """
        Feature: describe and to_records.
        Description: A choice that offloads its first layers.
        Expectation: Their line says so and how much of what they keep they
            move, with a link whose copies take no time all of it, and
            their record states offload and the bytes each moves.
        """
        choice = choose_recompute(self._evaluator("pp1", 0.2), Hard.Device_A2, link=_FAST_LINK)
        first = choice.ranges[0]
        self.assertTrue(first.option.link_bandwidth)
        self.assertTrue(describe(choice).splitlines()[0].endswith("no recompute, 100% offloaded to the host"))
        self.assertEqual(to_records(choice)[0]["offload"], True)
        self.assertEqual(to_records(choice)[0]["offloaded_bytes"], first.option.link_bandwidth)
        self.assertTrue(all("offload" not in record for record, item in zip(to_records(choice), choice.ranges)
                            if not item.option.link_bandwidth))


class TestOneMode(unittest.TestCase):
    """One mode for every layer, the way a runtime with one activation checkpoint mode runs them."""

    @classmethod
    def setUpClass(cls) -> None:
        """Seven DeepSeek layers over two stages."""
        cls.folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.path = _small_deepseek(cls.folder.name, 1)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the config."""
        cls.folder.cleanup()

    def _evaluator(self) -> EvaluatorV2:
        """A fresh evaluator of the small DeepSeek."""
        return EvaluatorV2(self.path, framework="mindformers", log_level=0)

    def test_a_roomy_device_runs_every_layer_off(self):
        """
        Feature: choose_recompute modes.
        Description: A 1 TB device, choosing between off and full.
        Expectation: Off, every layer plain, and each stage keeps what the
            memory model says it keeps with every layer plain, to its MB.
        """
        evaluator = _with_capacity(self._evaluator(), 1024 * 1024)
        choice = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"))
        self.assertEqual(choice.mode, "off")
        self.assertTrue(all(_is_plain(item.option) for item in choice.ranges))
        for mine, model in zip(choice.stage_memory, _stage_peaks(evaluator, full_rec=False)):
            self.assertLessEqual(abs(mine - model), 1.0)

    def test_a_device_only_full_recompute_fits_keeps_it(self):
        """
        Feature: choose_recompute modes.
        Description: A device a little larger than the heaviest stage fully
            recomputed, smaller than it plain.
        Expectation: Full, no stage saving any time, at the memory model's
            peak.
        """
        evaluator = self._evaluator()
        full = _stage_peaks(evaluator, full_rec=True)
        self.assertLess(max(full) + 16, max(_stage_peaks(evaluator, full_rec=False)))
        choice = choose_recompute(_with_capacity(evaluator, max(full) + 16), Hard.Device_A2, modes=MODES)
        self.assertEqual(choice.mode, "full")
        self.assertEqual(choice.stage_savings, tuple(0.0 for _ in full))
        self.assertLessEqual(abs(choice.memory - max(full)), 1.0)

    def test_a_choice_per_layer_is_as_fast_as_one_mode(self):
        """
        Feature: choose_recompute modes.
        Description: The heaviest stage, plain, a little too big for the
            device: one mode for every layer, and a choice per layer.
        Expectation: One mode must recompute every layer fully; the choice
            per layer saves more time, still within the device.
        """
        evaluator = self._evaluator()
        capacity = max(_stage_peaks(evaluator, full_rec=False)) - 64
        _with_capacity(evaluator, capacity)
        one = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"))
        each = choose_recompute(evaluator, Hard.Device_A2)
        self.assertEqual(one.mode, "full")
        self.assertIsNone(each.mode)
        self.assertGreater(sum(each.stage_savings), sum(one.stage_savings))
        self.assertLessEqual(each.memory, capacity)

    def test_a_mode_per_layer_fits_and_runs_as_the_trainer_states_it(self):
        """
        Feature: choose_recompute modes per_layer, and trainer_plan.
        Description: The heaviest stage, plain, a little too big for the
            device: off or full for every layer, then for each layer, then
            each layer's own option of its kind's front.
        Expectation: One mode must recompute every layer fully. A mode per
            layer saves time within the device, each range naming off or
            full and running its option, and each stage keeps what the
            config priced whole with those ranges keeps, to its MB; each
            layer's own option saves at least as much. The trainer's plan
            gives every layer its mode, the MTP layer last.
        """
        evaluator = self._evaluator()
        capacity = max(_stage_peaks(evaluator, full_rec=False)) - 64
        _with_capacity(evaluator, capacity)
        one = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"))
        each = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"), per_layer=True)
        own = choose_recompute(evaluator, Hard.Device_A2)
        self.assertEqual(one.mode, "full")
        self.assertIsNone(each.mode)
        for item in each.ranges:
            self.assertIn(item.mode, ("off", "full"))
            self.assertEqual(item.option.recompute, mode_recompute(item.mode, {}))
        self.assertEqual({item.mode for item in each.ranges}, {"off", "full"})
        self.assertGreater(sum(each.stage_savings), sum(one.stage_savings))
        self.assertGreaterEqual(sum(own.stage_savings) * (1 + 1e-12), sum(each.stage_savings))
        self.assertLessEqual(each.memory, capacity)
        for mine, model in zip(each.stage_memory, _stage_peaks_of(evaluator, each)):
            self.assertLessEqual(abs(mine - model), 1.0)
        modes = [item.mode for item in each.ranges for _ in range(item.count)]
        self.assertEqual(len(modes), 8)
        self.assertEqual(_plan_modes(*trainer_plan(each), len(modes)), modes)

    def test_a_mode_per_layer_on_a_roomy_device_is_one_mode(self):
        """
        Feature: choose_recompute modes per_layer.
        Description: A 1 TB device, off or full for each layer.
        Expectation: Every layer off, as one mode for every layer gives,
            which the trainer states as its mode alone.
        """
        evaluator = _with_capacity(self._evaluator(), 1024 * 1024)
        each = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"), per_layer=True)
        one = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"))
        self.assertEqual({item.mode for item in each.ranges}, {"off"})
        self.assertEqual(each.stage_memory, one.stage_memory)
        self.assertEqual(trainer_plan(each), ("off", {}))

    def test_the_modes_recompute_what_their_names_say(self):
        """
        Feature: mode_recompute.
        Description: Each mode, with the switches a config sets.
        Expectation: Off recomputes nothing, full everything, selective the
            switches set to 0; another name is refused.
        """
        configured = {"attBMM": 1, "normOp": 0, "ffAct": 0}
        self.assertEqual(mode_recompute("off", configured), frozenset())
        self.assertIsNone(mode_recompute("full", configured))
        self.assertEqual(mode_recompute("selective", configured), frozenset({"normOp", "ffAct"}))
        with self.assertRaises(ValueError):
            mode_recompute("sometimes", configured)

    def test_a_mode_reads_as_the_ranges_a_config_states(self):
        """
        Feature: mode_ranges.
        Description: Each mode, with the switches a config sets.
        Expectation: Off states no range, full one full range over every
            layer, and selective one selective range stating each op's
            state, the ops the switches set to 0 recomputed.
        """
        configured = {"attBMM": 1, "normOp": 0, "ffAct": 0}
        self.assertEqual(mode_ranges("off", configured), ())
        self.assertEqual(mode_ranges("full", configured), (RecomputeRange(option="full"),))
        (selective,) = mode_ranges("selective", configured)
        self.assertEqual(selective.option, "selective")
        self.assertEqual(selective.switches(), {op: int(op not in ("normOp", "ffAct")) for op in RECOMPUTE_OPS})

    def test_a_model_priced_whole_runs_the_runtimes_selective_mode_only_where_a_census_prices_it(self):
        """
        Feature: whole_modes.
        Description: A tower and a language model, each of two kinds, and
            a census that prices both kinds of both, one kind of the tower,
            or none; with the trainer's own selective switches, and with the
            configs' own.
        Expectation: The trainer's selective mode is offered only where the
            census prices every kind of both; a selective mode of the
            configs' own switches always.
        """
        kinds = [SimpleNamespace(name="dense"), SimpleNamespace(name="moe")]
        record = KindActivations(1.0, 0.0, 1.0, 0.0, 4096)

        def configs(tower_kinds: Sequence[str]) -> List[SimpleNamespace]:
            """A tower whose census prices *tower_kinds*, and a language model whose census prices both kinds."""
            tower = SimpleNamespace(kinds=kinds, census={name: record for name in tower_kinds})
            text = SimpleNamespace(kinds=kinds, census={"dense": record, "moe": record})
            return [tower, text]

        modes = ("off", "selective", "full")
        policy = dict(HYPER_SELECTIVE_REC_OP)
        with patch.object(Candidate, "layer_kinds", side_effect=lambda ccfg: ccfg.kinds):
            self.assertEqual(whole_modes(configs(("dense", "moe")), modes, policy), modes)
            self.assertEqual(whole_modes(configs(("dense",)), modes, policy), ("off", "full"))
            self.assertEqual(whole_modes(configs(()), modes, policy), ("off", "full"))
            self.assertEqual(whole_modes(configs(()), modes, None), modes)


class TestRecomputeAnalysis(unittest.TestCase):
    """The plans one more fully recomputed layer reaches, at one fixed strategy."""

    @classmethod
    def setUpClass(cls) -> None:
        """Seven DeepSeek layers over two stages."""
        cls.folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.path = _small_deepseek(cls.folder.name, 1)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the config."""
        cls.folder.cleanup()

    def _tight(self) -> Tuple[EvaluatorV2, float]:
        """An evaluator whose plain stages a little overfill the device, and that capacity in MB."""
        evaluator = EvaluatorV2(self.path, framework="mindformers", log_level=0)
        capacity = max(_stage_peaks(evaluator, full_rec=False)) - 64
        _with_capacity(evaluator, capacity)
        return evaluator, capacity

    def _analysis(self, evaluator: EvaluatorV2) -> Candidate.RecomputeAnalysis:
        """An analysis of *evaluator*'s strategy, choosing between off and full."""
        return Candidate.RecomputeAnalysis(evaluator, Hard.Device_A2, ("off", "full"), dimensions=_DEGREES)

    def test_every_alternative_recomputes_one_more_layer_and_fits(self):
        """
        Feature: RecomputeAnalysis.alternatives.
        Description: A plan per layer on a device the plain stages overfill.
        Expectation: Each alternative keeps the plan's kinds and degrees,
            runs exactly one more layer full, fits the device and carries no
            score, which only a caller's estimate gives it.
        """
        evaluator, capacity = self._tight()
        plan = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"), per_layer=True)
        fulls = sum(item.count for item in plan.ranges if item.mode == "full")
        results = self._analysis(evaluator).alternatives(plan)
        self.assertTrue(results)
        for result in results:
            self.assertIsNone(result.score)
            self.assertIsNone(result.mode)
            self.assertEqual(result.dimensions, _DEGREES)
            self.assertEqual(sum(item.count for item in result.ranges if item.mode == "full"), fulls + 1)
            self.assertEqual(sum(item.count for item in result.ranges), sum(item.count for item in plan.ranges))
            self.assertLessEqual(result.memory, capacity)
            self.assertLessEqual(result.memory, plan.memory)

    def test_equivalent_layers_give_one_alternative_each(self):
        """
        Feature: RecomputeAnalysis.alternatives.
        Description: The same plan, whose stages hold several layers of one kind.
        Expectation: Swapping which of a group's layers runs full gives one
            alternative, so there are fewer of them than there are layers
            still to turn up, and no two state the same plan.
        """
        evaluator, _ = self._tight()
        plan = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"), per_layer=True)
        off_layers = sum(item.count for item in plan.ranges if item.mode != "full")
        results = self._analysis(evaluator).alternatives(plan)
        plans = [tuple(sorted((item.first, item.count, item.mode) for item in result.ranges))
                 for result in results]
        self.assertEqual(len(plans), len(set(plans)))
        self.assertLess(len(results), off_layers)

    def test_a_plan_of_one_mode_for_every_layer_has_no_alternative(self):
        """
        Feature: RecomputeAnalysis.alternatives.
        Description: A choice of one mode for the whole model.
        Expectation: No alternative: the analysis extends a plan per layer.
        """
        evaluator, _ = self._tight()
        whole = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"))
        self.assertIsNotNone(whole.mode)
        self.assertEqual(self._analysis(evaluator).alternatives(whole), [])

    def test_a_plan_of_other_degrees_or_other_layers_is_refused(self):
        """
        Feature: RecomputeAnalysis.alternatives.
        Description: A plan priced at other degrees, one leaving a body layer
            out, and one naming a layer twice.
        Expectation: Each is refused rather than silently scored against the
            wrong strategy or the wrong model.
        """
        evaluator, _ = self._tight()
        analysis = self._analysis(evaluator)
        plan = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"), per_layer=True)
        with self.assertRaises(ValueError):
            analysis.alternatives(dataclasses.replace(plan, dimensions=(("DP", 8),)))
        with self.assertRaises(ValueError):
            analysis.alternatives(dataclasses.replace(plan, ranges=plan.ranges[:-1]))
        with self.assertRaises(ValueError):
            analysis.alternatives(dataclasses.replace(plan, ranges=plan.ranges + (plan.ranges[0],)))

    def test_the_analysis_refuses_modes_it_cannot_price(self):
        """
        Feature: RecomputeAnalysis.
        Description: No mode at all, a mode that is not one of MODES, and
            modes without full.
        Expectation: The first two are refused; without full there is
            nothing to turn a layer up to, so the analysis offers nothing.
        """
        evaluator, _ = self._tight()
        with self.assertRaises(ValueError):
            Candidate.RecomputeAnalysis(evaluator, Hard.Device_A2, ())
        with self.assertRaises(ValueError):
            Candidate.RecomputeAnalysis(evaluator, Hard.Device_A2, ("off", "nonexistent"))
        plan = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"), per_layer=True)
        off_only = Candidate.RecomputeAnalysis(evaluator, Hard.Device_A2, ("off",), dimensions=_DEGREES)
        self.assertEqual(off_only.alternatives(plan), [])


class TestNoChoice(unittest.TestCase):
    """What the choice does not cover."""

    def test_a_multimodal_model_has_no_choice(self):
        """
        Feature: choose_recompute.
        Description: A multimodal model.
        Expectation: No choice.
        """
        evaluator = SimpleNamespace(ccfg=SimpleNamespace(multimodal=True, pp_sched="1f1b"))
        self.assertIsNone(choose_recompute(evaluator, Hard.Device_A2))

    def test_another_schedule_has_no_choice(self):
        """
        Feature: choose_recompute.
        Description: A V-shaped schedule, whose end of warm-up the budget
            does not follow.
        Expectation: No choice.
        """
        evaluator = SimpleNamespace(ccfg=SimpleNamespace(multimodal=False, pp_sched="zero_bubble_v"))
        self.assertIsNone(choose_recompute(evaluator, Hard.Device_A2))

    def test_a_stage_that_does_not_fit_fully_recomputed_has_no_choice(self):
        """
        Feature: choose_recompute.
        Description: A device smaller than DeepSeek-V3's heaviest stage
            fully recomputed.
        Expectation: No choice.
        """
        evaluator = EvaluatorV2(DEEPSEEK_YAML, framework="mindformers", log_level=0)
        heaviest = max(insight["Static"] + insight["Dynamic"] for insight in evaluator.estimate_peak_insight())
        self.assertIsNone(choose_recompute(_with_capacity(evaluator, heaviest - 1024), Hard.Device_A2))


class TestDescribe(unittest.TestCase):
    """How a choice reads."""

    def test_each_range_reads_as_one_line(self):
        """
        Feature: describe.
        Description: Three ranges of a model without kinds: plain, some ops
            recomputed, and fully recomputed.
        Expectation: One line per range, its ops in the switches' order.
        """
        choice = RecomputeChoice(
            ranges=(LayerRange(0, 2, None, _option(frozenset())),
                    LayerRange(2, 1, None, _option(frozenset({"ffAct", "attBMM"}))),
                    LayerRange(3, 4, None, _option(None))),
            stage_memory=(1.0, 2.0),
            stage_savings=(0.0, 0.0),
        )
        self.assertEqual(describe(choice).splitlines(),
                         ["layers 0-1: no recompute", "layer 2: recompute attBMM+ffAct", "layers 3-6: full recompute"])
        self.assertEqual(option_label(_option(None)), "full recompute")
        self.assertEqual(choice.memory, 2.0)
        self.assertEqual(to_records(choice), [
            {"first": 0, "count": 2, "kind": None, "recompute": "none"},
            {"first": 2, "count": 1, "kind": None, "recompute": ["attBMM", "ffAct"]},
            {"first": 3, "count": 4, "kind": None, "recompute": "full"},
        ])

    def test_one_mode_reads_first(self):
        """
        Feature: describe.
        Description: A choice of one mode for every layer.
        Expectation: Its mode on the first line.
        """
        choice = RecomputeChoice(ranges=(LayerRange(0, 4, None, _option(frozenset())),), stage_memory=(1.0,),
                                 stage_savings=(0.0,), mode="off")
        self.assertEqual(describe(choice).splitlines(), ["every layer: off", "layers 0-3: no recompute"])

    def test_a_mode_per_layer_reads_as_its_modes(self):
        """
        Feature: describe and to_records.
        Description: A choice of a mode per layer over two kinds.
        Expectation: Each range reads as its mode, and its record states it
            beside the switches it recomputes.
        """
        choice = _per_layer((0, 3, "linear_attention", "full"), (3, 1, "full_attention", "selective"))
        self.assertEqual(describe(choice).splitlines(),
                         ["layers 0-2 (linear_attention): full", "layer 3 (full_attention): selective"])
        records = to_records(choice)
        self.assertEqual(records[0], {"first": 0, "count": 3, "kind": "linear_attention", "recompute": "full",
                                      "mode": "full"})
        self.assertEqual(records[1]["mode"], "selective")
        self.assertEqual(records[1]["recompute"], [name for name in SWITCHES if name in _POLICY])


class TestTrainerPlan(unittest.TestCase):
    """A choice of a mode per layer, as HyperParallel's trainer states it."""

    def test_the_mode_is_one_that_recomputes_so_a_forced_mode_cannot_invert_the_plan(self):
        """
        Feature: trainer_plan.
        Description: Plans in which a minority of the layers recompute, one
            of them fully and one selectively; then the same plan stated the
            other way round.
        Expectation: The mode is the one that recomputes, and the entries
            name the layers that do not. The trainer takes an entry equal to
            the mode, so a runner that states the recompute mode itself, as a
            sweep forcing its own full default does, still runs the layers
            the plan states off. Reported the other way round, as off with
            the recomputed layers as entries, the same run would recompute
            every layer and invert the search's choice without a word, so
            the plan is never written that way.
        """
        few_full = _per_layer((0, 1, None, "full"), (1, 7, None, "off"))
        mode, layers = trainer_plan(few_full)
        self.assertEqual((mode, layers), ("full", {"1-7": "off"}))
        self.assertEqual(_plan_modes(mode, layers, 8), ["full"] + ["off"] * 7)
        self.assertEqual(_plan_modes("full", layers, 8), ["full"] + ["off"] * 7)
        inverted = {"0": "full"}
        self.assertEqual(_plan_modes("off", inverted, 8), ["full"] + ["off"] * 7)
        self.assertEqual(_plan_modes("full", inverted, 8), ["full"] * 8)
        self.assertEqual(trainer_plan(_per_layer((0, 1, None, "selective"), (1, 7, None, "off"))),
                         ("selective", {"1-7": "off"}))

    def test_the_layers_that_do_not_run_the_mode_by_range_whatever_their_kinds(self):
        """
        Feature: trainer_plan.
        Description: Eight layers, full attention at 3 and 7: the first three
            fully recomputed and the last five off, ranges split by kind.
        Expectation: Full, the only mode that recomputes though fewer layers
            run it, and one entry for layers 3 to 7.
        """
        choice = _per_layer((0, 3, "linear_attention", "full"), (3, 1, "full_attention", "off"),
                            (4, 3, "linear_attention", "off"), (7, 1, "full_attention", "off"))
        self.assertEqual(trainer_plan(choice), ("full", {"3-7": "off"}))
        self.assertEqual(_plan_modes(*trainer_plan(choice), 8), ["full"] * 3 + ["off"] * 5)

    def test_a_tie_between_the_modes_that_recompute_goes_to_full(self):
        """
        Feature: trainer_plan.
        Description: Two layers selective, two fully recomputed, one off.
        Expectation: Full, and each other layer by its own index.
        """
        choice = _per_layer((0, 1, None, "selective"), (1, 2, None, "full"), (3, 1, None, "off"),
                            (4, 1, None, "selective"))
        self.assertEqual(trainer_plan(choice), ("full", {"0": "selective", "3": "off", "4": "selective"}))

    def test_one_mode_states_no_layers(self):
        """
        Feature: trainer_plan.
        Description: Every layer off per layer, every layer selective per
            layer, and one mode chosen for every layer.
        Expectation: The mode alone each time.
        """
        self.assertEqual(trainer_plan(_per_layer((0, 2, "a", "off"), (2, 2, "b", "off"))), ("off", {}))
        self.assertEqual(trainer_plan(_per_layer((0, 4, None, "selective"))), ("selective", {}))
        one = RecomputeChoice(ranges=(LayerRange(0, 4, None, _option(None)),), stage_memory=(1.0,),
                              stage_savings=(0.0,), mode="full")
        self.assertEqual(trainer_plan(one), ("full", {}))

    def test_options_no_mode_stands_for_are_refused(self):
        """
        Feature: trainer_plan.
        Description: A choice of each layer's own option, one of which
            recomputes the activations alone.
        Expectation: Refused: the trainer runs modes, not switch sets.
        """
        with self.assertRaises(ValueError):
            trainer_plan(_per_layer((0, 2, None, "full"), (2, 2, None, None)))


@dataclasses.dataclass
class _CheckpointWithLayers:
    """A trainer's activation_checkpoint section that takes a plan per layer."""

    mode: str = "off"
    layers: Optional[Dict[str, str]] = None


@dataclasses.dataclass
class _CheckpointWithoutLayers:
    """A trainer's activation_checkpoint section that takes one mode for every layer."""

    mode: str = "off"
    swap_inputs: bool = False


class TestTrainerRunsLayerPlans(unittest.TestCase):
    """Whether the installed trainer takes a plan per layer, read off its configuration."""

    def test_the_field_the_trainers_configuration_declares(self):
        """
        Feature: trainer_runs_layer_plans.
        Description: The installed trainer's ActivationCheckpointConfig,
            standing as a dataclass that declares layers, then as one that
            does not; then a tree with no such configuration.
        Expectation: True only where the dataclass declares layers.
        """
        module = "hyper_parallel.trainer.config.parallelism"
        with patch(f"{module}.ActivationCheckpointConfig", _CheckpointWithLayers):
            self.assertTrue(Candidate.trainer_runs_layer_plans())
        with patch(f"{module}.ActivationCheckpointConfig", _CheckpointWithoutLayers):
            self.assertFalse(Candidate.trainer_runs_layer_plans())
        with patch.object(Candidate, "_TRAINER_CHECKPOINT_CONFIG", ("hyper_parallel.no_such_trainer", "Config")):
            self.assertFalse(Candidate.trainer_runs_layer_plans())


class TestAutoRecomputeSearch(unittest.TestCase):
    """The search scores each candidate with its layers' options."""

    @staticmethod
    def _scored(auto: bool):
        """The search over TP and PP of the dense model on eight devices, and its runner."""
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
        runner = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"),
                                 dimensions=[Dim.TP, Dim.PP], auto_recompute=auto).instance
        results, _ = runner.device_loops(({}, 0), None)
        space = [(config, peak) for config, peak in results.items() if runner.mem_eval.mem_fit(peak)]
        scored, _ = runner.order_search_space(space, None, None)
        return scored, runner

    def test_every_candidate_is_as_fast_or_faster_and_still_fits(self):
        """
        Feature: ParallelizeLayer auto_recompute.
        Description: Order the same space fully recomputed and with auto
            recompute.
        Expectation: Every candidate gets its layers' options, none scores
            worse or stops fitting the device, and some score better.
        """
        set_verbose_level(1)
        full, _ = self._scored(False)
        auto, runner = self._scored(True)
        self.assertEqual(len(auto), len(full))
        own = {str(config): score for config, _, score, _ in full}
        for config, memory, score, _ in auto:
            self.assertIn(config, runner.recompute_choices)
            self.assertLessEqual(score, own[str(config)])
            self.assertTrue(runner.mem_eval.mem_fit(memory))
        self.assertLess(auto[0][2], full[0][2])

    def test_the_potential_sequence_recomputes_one_more_layer_a_step(self):
        """
        Feature: ParallelizeLayer.recompute_potential_sequence.
        Description: The best candidate of a search choosing off or full for
            each layer, then the same search choosing one mode for the whole
            model.
        Expectation: The sequence starts at the candidate's own plan and
            fully recomputes one more layer at each step, keeping its
            degrees and fitting the device, with the cost model's estimate
            and its parts on every step. A search that chose no plan per
            layer refuses to extend one.
        """
        set_verbose_level(1)
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
        runner = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"),
                                 dimensions=[Dim.TP, Dim.PP], auto_recompute=True,
                                 recompute_modes=("off", "full"), recompute_mode_per_layer=True).instance
        results, _ = runner.device_loops(({}, 0), None)
        space = [(config, peak) for config, peak in results.items() if runner.mem_eval.mem_fit(peak)]
        scored, _ = runner.order_search_space(space, None, None)
        config, _, score, _ = scored[0]
        sequence = runner.recompute_potential_sequence(config, base_score=score)
        self.assertGreater(len(sequence), 1)
        fulls = [sum(item.count for item in choice.ranges if item.mode == "full") for choice in sequence]
        self.assertEqual(fulls, list(range(fulls[0], fulls[0] + len(sequence))))
        for choice in sequence:
            self.assertEqual(choice.dimensions, sequence[0].dimensions)
            self.assertEqual(len(choice.score_parts), len(Debug.PerfParts) - 2)
            self.assertIsNotNone(choice.score)
            self.assertTrue(runner.mem_eval.mem_fit(choice.memory))
        self.assertEqual(runner.recompute_potentials[config], sequence)
        self.assertGreaterEqual(len(runner.recompute_results[config]), len(sequence))
        whole = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"),
                                dimensions=[Dim.TP], auto_recompute=True).instance
        with self.assertRaises(ValueError):
            whole.recompute_potential_sequence(config)

    def test_offload_needs_a_choice_per_layer_and_takes_the_device_link(self):
        """
        Feature: ParallelizeLayer auto_offload.
        Description: Ask for offload without auto recompute, then with it,
            then with a link of one's own.
        Expectation: Refused without it; with it, the device's link, or the
            one given.
        """
        with self.assertRaises(ValueError):
            Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                            auto_offload=True)
        runner = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                                 auto_recompute=True, auto_offload=True).instance
        self.assertEqual(runner.offload_link, Hard.Device_A2.host_link)
        runner = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                                 auto_recompute=True, auto_offload=True, host_link=_FAST_LINK).instance
        self.assertEqual(runner.offload_link, _FAST_LINK)

    def test_the_recompute_cannot_also_come_from_the_config(self):
        """
        Feature: ParallelizeLayer auto_recompute.
        Description: Ask for auto recompute and for the config's recompute,
            then for a mode no runtime has.
        Expectation: Both refused.
        """
        with self.assertRaises(ValueError):
            Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                            auto_recompute=True, mppb=True)
        with self.assertRaises(ValueError):
            Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                            auto_recompute=True, recompute_modes=("off", "sometimes"))

    def test_a_mode_per_layer_needs_the_modes_auto_recompute_chooses_among(self):
        """
        Feature: ParallelizeLayer recompute_mode_per_layer.
        Description: Ask for a mode per layer without the modes, then
            without auto recompute.
        Expectation: Both refused.
        """
        with self.assertRaises(ValueError):
            Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                            auto_recompute=True, recompute_mode_per_layer=True)
        with self.assertRaises(ValueError):
            Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                            recompute_modes=("off", "full"), recompute_mode_per_layer=True)

    def test_a_mode_per_layer_scores_each_candidate_no_worse_than_one_mode(self):
        """
        Feature: ParallelizeLayer recompute_mode_per_layer.
        Description: Order the space on a 2.5 GB device, choosing off or
            full for every layer, then for each layer. The candidates use
            the whole machine, the search holding DP at the config's own
, and the model, stating no dtype, is held in bfloat16
: at 2.5 GB one of them fits only fully recomputed.
        Expectation: The same candidates; every one gets a mode per layer,
            each range naming its mode, fits and scores no worse than with
            one mode for every layer, and one that one mode must recompute
            fully scores better. The log gives the best's plan as the
            trainer runs it: its mode and layers where the installed trainer
            takes a plan per layer, and else the mode for every layer, the
            layers as advice.
        """
        set_verbose_level(1)
        scored = {}
        for per_layer in (False, True):
            for dim in Dim.ALL_DIMS:
                dim.reset_bound()
            runner = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"),
                                     dimensions=[Dim.TP, Dim.PP], max_mem=Memory.from_string("2.5GB"),
                                     auto_recompute=True, recompute_modes=("off", "full"),
                                     recompute_mode_per_layer=per_layer).instance
            results, _ = runner.device_loops(({}, 0), None)
            space = [(config, peak) for config, peak in results.items() if runner.mem_eval.mem_fit(peak)]
            ordered, _ = runner.order_search_space(space, None, None)
            scored[per_layer] = {str(config): (score, runner.recompute_choices[config]) for config, _, score, _
                                 in ordered}
            for config, memory, _, _ in ordered:
                self.assertTrue(runner.mem_eval.mem_fit(memory))
        self.assertEqual(set(scored[True]), set(scored[False]))
        better = 0
        for config, (score, choice) in scored[True].items():
            one_score, one = scored[False][config]
            self.assertIsNone(choice.mode)
            self.assertTrue(all(item.mode in ("off", "full") for item in choice.ranges))
            self.assertLessEqual(score, one_score * (1 + 1e-12))
            better += one.mode == "full" and score < one_score
        self.assertGreater(better, 0)
        # The log reads the first configuration as the best: put first one
        # whose plan runs some layers another mode.
        mixed = next(entry for entry in ordered if trainer_plan(runner.recompute_choices[entry[0]])[1])
        mode, layers = trainer_plan(runner.recompute_choices[mixed[0]])
        mixed_first = [mixed, *(entry for entry in ordered if entry is not mixed)]
        lines = {}
        for runs_plans in (True, False):
            with patch.object(Par.logger, "output") as output, \
                    patch.object(Par, "trainer_runs_layer_plans", return_value=runs_plans):
                runner._log_recompute(mixed_first)  # pylint: disable=protected-access
            lines[runs_plans] = [call.args[0] % call.args[1:] for call in output.call_args_list]
        self.assertTrue(lines[True][0].startswith("Recompute was chosen per layer among off, full for "))
        self.assertEqual(lines[True][-1], f"As the trainer runs it: activation_checkpoint mode {mode}, layers {layers}")
        self.assertEqual(lines[False][:-1], lines[True][:-1])
        self.assertTrue(lines[False][-1].startswith(f"As the trainer runs it: activation_checkpoint mode {mode} "
                                                    "for every layer. The installed trainer's"))
        self.assertTrue(lines[False][-1].endswith(f"advice: layers {layers}"))

    def test_one_mode_per_candidate_and_the_per_layer_choice_of_one(self):
        """
        Feature: ParallelizeLayer recompute_modes and recompute_per_layer.
        Description: Order the space choosing off or full for every layer,
            then ask the best configuration for a choice per layer.
        Expectation: Every candidate gets one of the two modes; the choice
            per layer scores no worse than the best candidate's mode.
        """
        set_verbose_level(1)
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
        runner = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"),
                                 dimensions=[Dim.TP, Dim.PP], auto_recompute=True,
                                 recompute_modes=("off", "full")).instance
        results, _ = runner.device_loops(({}, 0), None)
        space = [(config, peak) for config, peak in results.items() if runner.mem_eval.mem_fit(peak)]
        scored, _ = runner.order_search_space(space, None, None)
        self.assertEqual({choice.mode for choice in runner.recompute_choices.values()} - {"off", "full"}, set())
        self.assertEqual(len(runner.recompute_choices), len(scored))
        per_layer, score = runner.recompute_per_layer(scored[0][0])
        self.assertIsNone(per_layer.mode)
        self.assertLessEqual(score, scored[0][2] * (1 + 1e-12))

    def test_a_multimodal_search_prices_the_whole_model(self):
        """
        Feature: ParallelizeMultiModal auto_recompute.
        Description: A vision-language model, whose candidates are priced on
            the tower and the text model together (IR F2), searched with and
            without auto_recompute.
        Expectation: No candidate gets a choice per layer: the options would
            be built for the text model's layers, and their budgets would
            leave the tower out. Each gets one mode for every layer instead,
            the fastest that fits priced whole, and scores no worse than
            fully recomputed.
        """
        set_verbose_level(1)
        scored, choices = {}, {}
        for auto in (True, False):
            for dim in Dim.ALL_DIMS:
                dim.reset_bound()
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "vl.yaml")
                with open(path, "w", encoding="utf-8") as handle:
                    yaml.safe_dump(_VL_TRAINING, handle)
                with patch.object(_hf_model_spec, "_get_hf_config", return_value=_VL), \
                        patch.dict(os.environ, {"MPLCONFIGDIR": folder}):
                    runner = Par.Parallelize("hyper_v2", path, Hard.Machine(8, "A2"), global_batch_size=16,
                                             dimensions=[Dim.DP], auto_recompute=auto).instance
                    self.assertIsInstance(runner, Par.ParallelizeMultiModal)
                    results, _ = runner.device_loops(({}, 0), None)
                    space = [(config, peak) for config, peak in results.items() if runner.mem_eval.mem_fit(peak)]
                    ordered, _ = runner.order_search_space(space, None, None)
                    scored[auto] = {str(config): score for config, _, score, _ in ordered}
                    choices[auto] = dict(runner.recompute_choices)
                    self.assertEqual(runner.recompute_per_layer(ordered[0][0]), (None, None))
        self.assertEqual(choices[False], {})
        self.assertEqual(len(choices[True]), len(scored[True]))
        for choice in choices[True].values():
            self.assertIn(choice.mode, MODES)
            self.assertEqual(choice.ranges, ())
        self.assertEqual(set(scored[True]), set(scored[False]))
        for config, score in scored[True].items():
            self.assertLessEqual(score, scored[False][config])

    @staticmethod
    def _vl_search(folder: str, **extra: Any):
        """The vision-language model's search over DP on eight devices, and its runner."""
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
        path = os.path.join(folder, "vl.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(_VL_TRAINING, handle)
        runner = Par.Parallelize("hyper_v2", path, Hard.Machine(8, "A2"), global_batch_size=16,
                                 dimensions=[Dim.DP], **extra).instance
        results, _ = runner.device_loops(({}, 0), None)
        space = [(config, peak) for config, peak in results.items() if runner.mem_eval.mem_fit(peak)]
        ordered, _ = runner.order_search_space(space, None, None)
        return ordered, runner

    def test_a_multimodal_search_runs_the_trainers_mode_priced_whole(self):
        """
        Feature: ParallelizeMultiModal recompute_modes.
        Description: The vision-language model searched without auto
            recompute, with the trainer's full mode alone, and with off or
            full; each candidate's mode then stated on the tower and the text
            model and the whole model priced.
        Expectation: Every candidate gets a mode, and keeps the memory and
            the score the whole model has under it: full recomputes every
            layer as the config states, so the full mode scores each
            candidate as the search without auto recompute does, and off,
            where it fits, is faster.
        """
        set_verbose_level(1)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(_hf_model_spec, "_get_hf_config", return_value=_VL), \
                    patch.dict(os.environ, {"MPLCONFIGDIR": folder}):
                own, _ = self._vl_search(folder)
                full, _ = self._vl_search(folder, auto_recompute=True, recompute_modes=("full",))
                either, runner = self._vl_search(folder, auto_recompute=True, recompute_modes=("off", "full"))
                self.assertIsInstance(runner, Par.ParallelizeMultiModal)
                whole = runner.priced()
                for config, memory, score, _ in either:
                    choice = runner.recompute_choices[config]
                    runner.config.set_parallel_config(config)
                    priced = copy.deepcopy(runner.priced())
                    for name in priced.mm_order:
                        apply_exec(priced.mm_ccfgs[name], ExecSpec(recompute=mode_ranges(choice.mode, {})))
                    runner.mem_eval.set_config(priced)
                    peaks = [insight["Static"] + insight["Dynamic"]
                             for insight in runner.mem_eval.estimate_peak_insight()]
                    runner.mem_eval.set_config(whole)
                    self.assertEqual(memory, int(round(max(peaks))))
                    self.assertEqual(score, Par.estimate_performance(priced, device_type=runner.machine.device,
                                                                     memory=memory))
        self.assertEqual([(str(c), m, s) for c, m, s, _ in full], [(str(c), m, s) for c, m, s, _ in own])
        self.assertTrue(either)
        self.assertLess(either[0][2], own[0][2])


if __name__ == "__main__":
    unittest.main()
