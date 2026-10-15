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
"""Tests for applying an execution spec to a cost-model config.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_apply_exec.py -v
"""
import copy
import dataclasses
import os
import unittest
from typing import Any, Dict

import hyper_parallel
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel._exec_spec import ExecSpec, RecomputeRange
from hyper_parallel.auto_parallel.sapp_nd.nd.common.apply_exec import apply_exec, apply_layer_strategy, exec_of
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import check_and_apply_custom_hook
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_order import get_model_order
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import estimate_comm
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import estimate_comp
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import (
    STRATEGY_GUARDED,
    CostModelConfig,
    arm_strategy_guard,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
# ND's own yamls, where the package that runs keeps them, which need not be
# beside these tests: a gate runs a copy of them against the installed package.
_YAMLS = os.path.join(os.path.dirname(os.path.abspath(hyper_parallel.__file__)),
                      "auto_parallel", "sapp_nd", "nd", "yamls")
_SOURCES = (
    (os.path.join(_HERE, "deepseek.yaml"), None),
    (os.path.join(_YAMLS, "hyper_deepseek_v3.yaml"), "hyper_v2"),
    (os.path.join(_YAMLS, "hyper_qwen3_72b.yaml"), "hyper_v2"),
)


def _state(ccfg: Any) -> Dict[str, Any]:
    """The config's fields, declared or set, with the ones held in objects read as values."""
    state = {}
    names = set(vars(ccfg)) | {spec_field.name for spec_field in dataclasses.fields(type(ccfg))}
    for key in sorted(names):
        value = getattr(ccfg, key)
        if isinstance(value, Config):
            value = vars(value)
        elif isinstance(value, Memory):
            value = (value.size, value.unit)
        state[key] = value
    return state


class TestApplyExec(unittest.TestCase):
    """apply_exec writes what an ExecSpec states and derives the rest, the one way a degree reaches an armed config."""

    def test_a_spec_derives_every_field_and_a_strategy_derives_them_again(self):
        """
        Feature: exec_of, apply_exec and set_strategy.
        Description: Apply to a parsed MindFormers config and to two Hyper
            configs the ExecSpec read back from each; then state only TP 2
            on the DeepSeek MindFormers config, which runs sequence
            parallelism and slices its recompute input at TP 4; then set TP
            4 again as a search's strategy.
        Expectation: Its own spec changes no field but the recompute, which
            comes back stated as ranges, and no layer's recompute; TP 2
            changes TP and the fields derived from it, not DP; the strategy
            derives them again on that config.
        """
        for path, framework in _SOURCES:
            ccfg = CostModelConfig(path, framework=framework)
            before, layers = _state(ccfg), ccfg.generate_partitions_vpp()
            apply_exec(ccfg, exec_of(ccfg))
            after = _state(ccfg)
            changed = sorted(key for key, value in before.items() if value != after[key])
            self.assertEqual(changed, ["recompute_ranges"], f"{os.path.basename(path)}: fields changed {changed}")
            self.assertEqual(ccfg.generate_partitions_vpp(), layers, path)
        ccfg = CostModelConfig(os.path.join(_HERE, "deepseek.yaml"))
        dp = ccfg.d
        apply_exec(ccfg, ExecSpec(tp=2))
        got = (ccfg.t, ccfg.sp, ccfg.shard_recompute_input, ccfg.d)
        self.assertEqual(got, (2, 2, 2, dp), f"t, sp, shard_recompute_input, d={got}")
        ccfg.set_strategy(mp=4)
        got = (ccfg.t, ccfg.sp, ccfg.shard_recompute_input)
        self.assertEqual(got, (4, 4, 4), f"t, sp, shard_recompute_input={got}")

    def test_an_armed_config_takes_a_degree_only_from_the_sanctioned_writers(self):
        """
        Feature: the strategy guard.
        Description: Arm the DeepSeek MindFormers config as a search arms
            it, and write each guarded field directly; then change it
            through set_strategy, apply_exec and a layer kind's degrees, and
            write a field that is not a degree, such as the recompute
            switches a pricer sets.
        Expectation: AttributeError for each direct write, the field keeping
            its value; every other write lands.
        """
        ccfg = CostModelConfig(os.path.join(_HERE, "deepseek.yaml"))
        arm_strategy_guard(ccfg)
        for name in sorted(STRATEGY_GUARDED):
            before = getattr(ccfg, name)
            with self.assertRaises(AttributeError, msg=f"{name} was written"):
                setattr(ccfg, name, 3)
            self.assertEqual(getattr(ccfg, name), before, f"{name} changed")
        ccfg.set_strategy(mp=2)
        apply_exec(ccfg, ExecSpec(dp=8))
        apply_layer_strategy(ccfg, {"ep": 1})
        ccfg.rec_op = Config({"gather": 0})
        got = (ccfg.t, ccfg.d, ccfg.ep, ccfg.rec_op.gather)
        self.assertEqual(got, (2, 8, 1, 0), f"t, d, ep, rec_op.gather={got}")


def _unit(pp: int = 2, **accelerator: Any) -> CostModelConfig:
    """A small dense Hyper model of 8 layers at *pp* stages, fully recomputed; *accelerator* adds to its section."""
    return CostModelConfig({
        "model": {"name": "unit", "config_overrides": {
            "hidden_size": 1024, "num_hidden_layers": 8, "num_attention_heads": 8,
            "num_key_value_heads": 8, "intermediate_size": 2816, "vocab_size": 32000,
            "max_position_embeddings": 2048,
        }},
        "training": {"global_batch_size": 8, "micro_batch_size": 1},
        "accelerator": {"tp_size": 2, "pp_size": pp, **accelerator},
        "fsdp_config": {"dp_shard_size": 2},
        "activation_checkpoint": {"mode": "full"},
        "dataset": {"data_transform": {"max_seq_len": 2048}},
        "context": {"max_device_memory": "64GB", "device_num": 4 * pp},
    }, framework="hyper_v2")


def _layers(ccfg: Any) -> list:
    """Each layer's recompute type, in model order."""
    stages = ccfg.generate_partitions_vpp()
    return [stages[stage][chunk][lay].name[:3] for stage, chunk, lay in get_model_order(ccfg, stages)]


def _recomputed(switches: Any) -> Any:
    """The ops *switches* recompute, sorted; None for a layer with no switches of its own."""
    if switches is None:
        return None
    values = switches if isinstance(switches, dict) else vars(switches)
    return sorted(op for op, keep in values.items() if not keep)


def _priced(ranges: tuple, **accelerator: Any) -> tuple:
    """Each stage's peak memory, and its compute and communication with recompute, with *ranges* stated."""
    ccfg = _unit(**accelerator)
    apply_exec(ccfg, ExecSpec(recompute=ranges))
    memory = [insight["Static"] + insight["Dynamic"]
              for insight in EvaluatorV2(None, ccfg=ccfg).estimate_peak_insight()]
    hooked = copy.deepcopy(ccfg)
    check_and_apply_custom_hook(hooked)
    stages = hooked.generate_partitions_vpp()
    compute = estimate_comp(copy.deepcopy(hooked), CustomConfig(), stages, with_recomp=True)
    comm = estimate_comm(copy.deepcopy(hooked), CustomConfig(), stages, Hard.Device_A2, with_recomp=True)
    return memory, compute, comm


class TestRecomputeRanges(unittest.TestCase):
    """A config's recompute as ranges over its layers, S2's shape, and its per-layer channel."""

    def test_each_layer_takes_its_ranges_option_and_switches(self):
        """
        Feature: apply_exec and derive, the recompute ranges (S2).
        Description: Three layers selective with the feed-forward activation
            recomputed, one fully recomputed, one no range covers, and the
            last three selective with the norms recomputed; then a per-stage
            form; then a range past the model's last layer.
        Expectation: Each layer in model order takes its range's option and
            switches, none where it is not selective, and rec_op the first
            range's; the per-stage form replaces the ranges; the range past
            the last layer is refused.
        """
        ccfg = _unit()
        apply_exec(ccfg, ExecSpec(recompute=(
            RecomputeRange(first=0, count=3, option="selective", ops={"ffAct": "recompute"}),
            RecomputeRange(first=3, count=1, option="full"),
            RecomputeRange(first=5, option="selective", ops={"normOp": "recompute"}),
        )))
        self.assertEqual(_layers(ccfg), ["SEL"] * 3 + ["FUL", "NOT"] + ["SEL"] * 3)
        self.assertEqual([_recomputed(switches) for switches in ccfg.layer_switches],
                         [["ffAct"]] * 3 + [None, None] + [["normOp"]] * 3)
        self.assertEqual(_recomputed(ccfg.rec_op), ["ffAct"])
        apply_exec(ccfg, ExecSpec(full_recompute=True))
        self.assertEqual((ccfg.recompute_ranges, _layers(ccfg)), (None, ["FUL"] * 8))
        with self.assertRaises(ValueError):
            apply_exec(_unit(), ExecSpec(recompute=(RecomputeRange(first=6, count=4, option="full"),)))

    def test_each_stage_prices_as_the_config_of_its_setting(self):
        """
        Feature: the per-layer channel (S2).
        Description: One config whose layers on the first stage are
            selective with the feed-forward activation and the gathers
            recomputed, and on the second with the norms recomputed, a
            range a layer; and each setting over every layer, in a config
            of its own. Under 1F1B, interleaved 1F1B and a V schedule,
            whose second chunk runs back up the stages.
        Expectation: Each stage keeps the memory, and runs the compute and
            the communication with recompute, of the config whose setting
            it runs, all three of which differ from the other setting's.
        """
        settings = ({"ffAct": "recompute", "gather": "recompute"}, {"normOp": "recompute"})
        for accelerator in ({}, {"pp_interleave_num": 2},
                            {"pp_interleave_num": 2, "pipeline_scheduler": "zero_bubble_v"}):
            with self.subTest(**accelerator):
                probe = _unit(**accelerator)
                order = get_model_order(probe, probe.generate_partitions_vpp())
                mixed = _priced(tuple(RecomputeRange(first=index, count=1, option="selective", ops=settings[stage])
                                      for index, (stage, _, _) in enumerate(order)), **accelerator)
                alone = [_priced((RecomputeRange(option="selective", ops=ops),), **accelerator) for ops in settings]
                for stage in range(2):
                    own = [part[stage] for part in alone[stage]]
                    self.assertEqual([part[stage] for part in mixed], own, stage)
                    for part, other in enumerate(alone[1 - stage]):
                        self.assertNotEqual(other[stage], own[part], (stage, part))


if __name__ == "__main__":
    unittest.main()
