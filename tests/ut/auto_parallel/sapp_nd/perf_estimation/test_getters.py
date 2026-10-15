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
"""Tests for what the performance path prices a layer with: its kind, its FSDP traffic and its recompute.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/perf_estimation/test_getters.py -v
"""
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import patch

import yaml

from hyper_parallel.auto_parallel._model_spec import KindActivations
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalAttn
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import check_and_apply_custom_hook
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import HYPER_SELECTIVE_REC_OP
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import estimate_comm
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import (
    get_layer_configs_by_position,
    get_recomp_factor,
    get_table_quantity,
    selective_shares,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig

LAY = LayerType.NOT_REC_LAYER
FULL, LINEAR = "full_attention", "linear_attention"


def _hybrid_config(folder: str, layer_types: List[str], pp: int = 1, vp: int = 1,
                   sched: str = "1f1b") -> CostModelConfig:
    """A Qwen3.5-style stack of *layer_types*, parsed by the hyper_v2 front end."""
    hf_config = SimpleNamespace(
        model_type="qwen3_5_moe", hidden_size=1024, num_hidden_layers=len(layer_types),
        num_attention_heads=8, num_key_value_heads=2, head_dim=128, vocab_size=32000,
        max_position_embeddings=8192, num_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=256, shared_expert_intermediate_size=256,
        attn_output_gate=True, layer_types=list(layer_types),
        linear_num_key_heads=8, linear_key_head_dim=64, linear_num_value_heads=16,
        linear_value_head_dim=64, linear_conv_kernel_dim=4,
    )
    train = {
        "model": {"pretrained_model_name_or_path": "local/unit", "torch_dtype": "bfloat16"},
        "training": {"global_batch_size": 8, "micro_batch_size": 1},
        "accelerator": {"tp_size": 1, "pp_size": pp, "pp_interleave_num": vp,
                        "pipeline_scheduler": sched},
        "fsdp_config": {"dp_shard_size": 2},
        "dataset": {"data_transform": {"max_seq_len": 2048}},
        "context": {"max_device_memory": "64GB", "device_num": 2 * pp},
    }
    path = os.path.join(folder, f"hybrid_{len(os.listdir(folder))}.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(train, handle)
    with patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config", return_value=hf_config):
        ccfg = CostModelConfig(path, framework="hyper_v2")
    check_and_apply_custom_hook(ccfg)
    return ccfg


class TestPerformanceAgreesWithMemory(unittest.TestCase):
    """Both estimators give every layer the same kind."""

    def setUp(self) -> None:
        """A scratch folder for the stacks' config files."""
        folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def _stack(self, sched: str) -> CostModelConfig:
        """Eight layers on two stages of two chunks, alternating kinds by pairs."""
        return _hybrid_config(self.folder, [FULL, FULL, LINEAR, LINEAR] * 2, pp=2, vp=2, sched=sched)

    def test_v_schedule(self):
        """
        Feature: one layer order for every estimator.
        Description: A V schedule, second chunk back up the stages.
        Expectation: The performance path gives each position the kind the
            memory backbone gives it.
        """
        ccfg = self._stack("zero_bubble_v")
        stages = ccfg.generate_partitions_vpp()
        perf = {pos: cfg.n_softmax for pos, cfg in get_layer_configs_by_position(ccfg, stages).items()}

        memory = {}

        def spy(cfg: CostModelConfig, ctx: Any) -> float:
            """The real score formula, recording the counts at each layer's first visit."""
            stage_id, chunk_id, lay_id = ctx.current_stage_id, ctx.current_chunk_id, ctx.current_lay_id
            # The backward-overhead pass revisits positions, output included.
            if isinstance(lay_id, int) and stages[stage_id][chunk_id][lay_id] == LAY:
                memory.setdefault((stage_id, chunk_id, lay_id), cfg.n_softmax)
            return EvalAttn.attn_score_activations(cfg, ctx)

        evaluator = EvaluatorV2(None, ccfg=ccfg)
        evaluator.set_attn_eval_fun(score=spy)
        evaluator.estimate_peak(stages=stages)
        self.assertEqual(memory, perf)
        self.assertEqual(sorted(perf.values()), [0, 0, 0, 0, 1, 1, 1, 1])


def _factor(rec_op: Any, op_name: str) -> int:
    """The recompute factor of *op_name* in a selective layer with these switches."""
    return get_recomp_factor(SimpleNamespace(rec_op=rec_op), LayerType.SEL_REC_LAYER, op_name)


class TestFsdpTraffic(unittest.TestCase):
    """The bytes FSDP's collectives move, as the time model's DP term."""

    def test_a_ring_moves_n_minus_one_over_n(self):
        """
        Feature: EvalLayerComm.fsdp_traffic.
        Description: A part of 100 parameters over 1 rank, sharded over 4 and
            held by 16, bf16, gathered twice a micro-batch, its copies
            all-reducing every other; and one no rank shards, held by 4.
        Expectation: Each collective over n shards moves (n - 1) / n of the
            part: two gathers and a reduce-scatter of 3 / 4, and an
            all-reduce of the shard over its 4 copies, 2 * 3 / 4, halved;
            the unsharded part only all-reduces.
        """
        ccfg = SimpleNamespace(bytes_p=2, bytes_grad=2)
        self.assertAlmostEqual(EvalLayerComm.fsdp_traffic(ccfg, ((100, 1, 4, 16),), 2, 0.5),
                               0.75 * 100 * (2 * 2 + 2) + 2 * 0.75 * 25 * 2 * 0.5)
        self.assertAlmostEqual(EvalLayerComm.fsdp_traffic(ccfg, ((100, 1, 1, 4),), 2, 1),
                               2 * 0.75 * 100 * 2)

    def test_the_all_reduce_is_its_own_share_of_the_traffic(self):
        """
        Feature: EvalLayerComm.fsdp_reduce_traffic.
        Description: The two parts of the test above.
        Expectation: The sharded part's all-reduce is its copies' 2 * 3 / 4 of
            a quarter, halved, the rest its gathers and reduce-scatter; the
            unsharded part's traffic is all all-reduce.
        """
        ccfg = SimpleNamespace(bytes_p=2, bytes_grad=2)
        self.assertAlmostEqual(EvalLayerComm.fsdp_reduce_traffic(ccfg, ((100, 1, 4, 16),), 0.5),
                               2 * 0.75 * 25 * 2 * 0.5)
        self.assertAlmostEqual(EvalLayerComm.fsdp_reduce_traffic(ccfg, ((100, 1, 1, 4),), 1),
                               EvalLayerComm.fsdp_traffic(ccfg, ((100, 1, 1, 4),), 2, 1))

    def test_the_walk_counts_each_micro_batch_s_collectives(self):
        """
        Feature: the DP term of the communication walk under FSDP.
        Description: A HyperParallel stack at DP shard 2, four micro-batches,
            with its FSDP resharding and keeping the parameters gathered.
        Expectation: Resharded, each layer is gathered twice a micro-batch
            and the root once; kept, each once a step.  No copy, so no
            all-reduce, but the group shards: once a step.
        """
        self.assertEqual(self._rounds(True), [(1, 1, 0.25), (3, 2, 0.25)])
        self.assertEqual(self._rounds(False), [(1, 0.25, 0.25), (3, 0.25, 0.25)])

    @staticmethod
    def _rounds(reshards: bool) -> list:
        """The parts, gathers and all-reduces the walk passes FSDP's traffic, with *reshards*."""
        with tempfile.TemporaryDirectory() as folder:
            ccfg = _hybrid_config(folder, [FULL, LINEAR])
        ccfg.reshards = reshards
        calls = []
        real = EvalLayerComm.fsdp_traffic

        def spy(cfg: CostModelConfig, parts: tuple, gathers: float, reduces: float) -> float:
            """The real traffic, recording its rounds."""
            calls.append((len(parts), gathers, reduces))
            return real(cfg, parts, gathers, reduces)

        with patch.object(EvalLayerComm, "fsdp_traffic", side_effect=spy):
            estimate_comm(ccfg, CustomConfig(), ccfg.generate_partitions_vpp(), Hard.Device_A2)
        return sorted(set(calls))


class TestSelectiveRecompute(unittest.TestCase):
    """A switch at 0 recomputes its op, as the memory model reads it."""

    def test_time_and_memory_agree_on_every_switch(self):
        """In a selective layer an op is kept in memory or recomputed in time, never both or neither."""
        for switch in (0, 1):
            kept = EvalUtils.rec_coeff(True, switch)
            recomputed = _factor(Config({"softmax": switch}), "softmax")
            self.assertEqual(kept + recomputed, 1, f"switch={switch}: kept={kept}, recomputed={recomputed}")

    def test_a_census_prices_the_matmuls_hyperparallels_policy_runs_again(self):
        """
        A selective layer of HyperParallel's policy, whose kind's census
        states the shares of its attention's and feed-forward's matmul FLOPs
        the policy runs again, prices that share of each again, beside the
        ops its switches recompute; a full layer, and a selective one of
        other switches, have no shares.
        """
        census = KindActivations(1.0, 1.0, 1.0, 1.0, 4096, selective_attention_mm=0.25, selective_ffn_mm=0.5)
        lccfg = SimpleNamespace(n_softmax=1, n_attMM=1, n_ffMM=1, kind_activations=census,
                                rec_op=Config(dict(HYPER_SELECTIVE_REC_OP)))
        table = {"n_softmax": 10.0, "n_attMM": 1000.0, "n_ffMM": 100.0}
        shares = selective_shares(lccfg, LayerType.SEL_REC_LAYER)
        self.assertEqual(shares, {"attMM": 0.25, "ffMM": 0.5})
        once = get_table_quantity(lccfg, table, LayerType.SEL_REC_LAYER, False, shares=shares)
        again = get_table_quantity(lccfg, table, LayerType.SEL_REC_LAYER, True, shares=shares)
        self.assertEqual(again - once, 10.0 + 250.0 + 50.0)
        self.assertEqual(selective_shares(lccfg, LayerType.FULL_REC_LAYER), {})
        lccfg.rec_op = Config(dict(HYPER_SELECTIVE_REC_OP, ffAct=1))
        self.assertEqual(selective_shares(lccfg, LayerType.SEL_REC_LAYER), {})


if __name__ == "__main__":
    unittest.main()
