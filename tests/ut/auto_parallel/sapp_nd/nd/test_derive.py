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
"""Tests for the derived fields of a cost-model config.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_derive.py -v
"""
import unittest
from types import SimpleNamespace
from typing import Any

from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import (
    HYPER_SELECTIVE_REC_OP,
    derive,
    derive_embedding_sharding,
    derive_flash_attention_factor,
    derive_recompute_switches,
    optimizer_ranks,
    runs_hyper_selective,
)


def _config(**fields: Any) -> SimpleNamespace:
    """A dense config holding the facts derive reads, *fields* overriding them."""
    facts = {
        "d": 4, "t": 2, "p": 1, "cp": 1, "ep": 1, "etp": 0, "n_exp": 1, "hff_exp": 64,
        "has_op": True, "has_grad_shard": False, "os_max_shard": 8,
        "vocab_emb_dp": False, "emb_dp_sharded": True, "recompute_slice_activation": False,
        "sel_rec": False, "sel_comm_rec": False, "sel_rec_rule": "hyperparallel",
        "has_fa": True, "sequence_parallel": False, "sp": 1, "s": 4096, "a": 32,
    }
    facts.update(fields)
    return SimpleNamespace(**facts)


class TestDerive(unittest.TestCase):
    """derive computes each field from the facts a parser states."""

    def test_embedding_sharding(self):
        """
        Feature: derive_embedding_sharding.
        Description: The embedding facts the four parsers state, at d=4 and t=2.
        Expectation: MindFormers and Hyper split the table over d, and over t
            too unless the vocabulary embedding is data parallel without
            pipelining; MindSpeed splits it over t * d and TOML over t alone.
        """
        cases = [
            ({"vocab_emb_dp": True, "p": 1}, 4),
            ({"vocab_emb_dp": True, "p": 2}, 8),
            ({"vocab_emb_dp": False}, 8),
            ({"vocab_emb_dp": False, "emb_dp_sharded": False}, 2),
        ]
        for fields, expected in cases:
            ccfg = _config(**fields)
            derive_embedding_sharding(ccfg)
            self.assertEqual(ccfg.shard_embed, expected, f"{fields}: shard_embed={ccfg.shard_embed}")

    def test_fsdp_shards_the_embedding_over_its_group(self):
        """
        Feature: derive_embedding_sharding, under FSDP.
        Description: A table over d=8 and t=2 whose FSDP group is 2 ranks,
            as under HSDP, with and without FSDP.
        Expectation: FSDP shards the table over its group and TP, and
            gathers it over the group; otherwise it is split over d.
        """
        got = []
        for fsdp in (True, False):
            ccfg = _config(d=8, os_max_shard=2, grad_shard_as_params=fsdp)
            derive_embedding_sharding(ccfg)
            got.append((ccfg.shard_embed, ccfg.gather_embed))
        self.assertEqual(got, [(4, 2), (16, 1)])

    def test_mindformers_recompute_switches(self):
        """
        Feature: derive_recompute_switches.
        Description: MindFormers' selective recompute under flash attention and
            sequence parallelism, then its communication recompute alone.
        Expectation: The head cast, norm and activation are recomputed and the
            attention kernels kept; the all-gather is recomputed only by
            select_comm_recompute; an MLA layer's up-projections, which
            MindFormers' rule leaves out, are kept.
        """
        ccfg = _config(sel_rec_rule="mindformers", sel_rec=True, sp=2)
        derive_recompute_switches(ccfg)
        expected = {"attBMM": 1, "headCast": 0, "dropout": 1, "softmax": 1, "normOp": 0, "gather": 1, "ffAct": 0,
                    "attUp": 1}
        self.assertEqual(vars(ccfg.rec_op), expected, f"rec_op={vars(ccfg.rec_op)}")

        ccfg = _config(sel_rec_rule="mindformers", sel_comm_rec=[1, 0], sp=2)
        derive_recompute_switches(ccfg)
        expected = {**dict.fromkeys(HYPER_SELECTIVE_REC_OP, 1), "gather": 0}
        self.assertEqual(vars(ccfg.rec_op), expected, f"rec_op={vars(ccfg.rec_op)}")

    def test_a_policy_that_recomputes_the_up_projections_is_not_hyperparallels(self):
        """
        Feature: runs_hyper_selective.
        Description: HyperParallel's switches as the policy states them, as
            a setting stated before attUp existed states them, and with
            attUp at 0.
        Expectation: The first two are its policy, a switch left out keeping
            its op; the third is not, so a census does not price it as the
            policy.
        """
        seven = {name: state for name, state in HYPER_SELECTIVE_REC_OP.items() if name != "attUp"}
        self.assertTrue(runs_hyper_selective(None, dict(HYPER_SELECTIVE_REC_OP)))
        self.assertTrue(runs_hyper_selective(None, seven))
        self.assertFalse(runs_hyper_selective(None, dict(seven, attUp=0)))

    def test_unknown_recompute_rule_is_refused(self):
        """
        Feature: derive_recompute_switches.
        Description: A config naming no known selective recompute rule.
        Expectation: ValueError rather than a guess.
        """
        with self.assertRaises(ValueError):
            derive_recompute_switches(_config(sel_rec_rule="megatron"))

    def test_flash_attention_factor(self):
        """
        Feature: derive_flash_attention_factor.
        Description: With and without flash attention, and a model with no heads.
        Expectation: s / a under flash attention, s otherwise.
        """
        for fields, expected in (({}, 128.0), ({"has_fa": False}, 4096), ({"a": 0}, 4096)):
            ccfg = _config(**fields)
            derive_flash_attention_factor(ccfg)
            self.assertEqual(ccfg.s_fa, expected, f"{fields}: s_fa={ccfg.s_fa}")

    def test_derive_sets_every_derived_field(self):
        """
        Feature: derive.
        Description: One call on a dense config that slices the recompute
            input and recomputes selectively.
        Expectation: The expert degrees, sharding factors, communication
            flags, recompute switches, flash attention factor and byte widths
            are all set, the last the default family's without pipelining.
        """
        ccfg = _config(recompute_slice_activation=True, sel_rec=True)
        derive(ccfg)
        got = (ccfg.t_exp, ccfg.d_exp, ccfg.comm_d_non_exp, ccfg.s_fa)
        self.assertEqual(got, (2, 4, 2, 128.0), f"t_exp, d_exp, comm_d_non_exp, s_fa={got}")
        got = (ccfg.shard_embed, ccfg.shard_recompute_input, ccfg.shard_output_activ)
        self.assertEqual(got, (8, 2, 1), f"shard_embed, shard_recompute_input, shard_output_activ={got}")
        self.assertEqual(vars(ccfg.rec_op), HYPER_SELECTIVE_REC_OP, f"rec_op={vars(ccfg.rec_op)}")
        got = (ccfg.bytes_grad, ccfg.bytes_os, ccfg.bytes_norm, ccfg.bytes_dropout)
        self.assertEqual(got, (0, 4, 4, 0), f"byte widths={got}")


class TestGradientSharding(unittest.TestCase):
    """Gradients are sharded by one of three rules."""

    def test_gradient_sharding(self):
        """
        Feature: derive_optimizer_sharding.
        Description: At d=4, t=2 and an optimizer shard of 2 data-parallel
            ranks: gradients sharded as the parameters are, as FSDP holds
            them; over the whole optimizer shard; and neither.
        Expectation: The non-expert, expert and partial expert gradient
            factors: the parameters' (4, 8, 1), the optimizer's (8, 8, 1),
            and TP alone.
        """
        cases = [
            ({"grad_shard_as_params": True}, (4, 8, 1)),
            ({"has_grad_shard": True}, (8, 8, 1)),
            ({}, (2, 2, 2)),
        ]
        for facts, want in cases:
            ccfg = _config(os_max_shard=2, **facts)
            derive(ccfg)
            got = (ccfg.shard_grad_non_exp, ccfg.shard_grad_exp, ccfg.shard_grad_exp_partial)
            self.assertEqual(got, want, f"{facts}: shard_grad_non_exp, shard_grad_exp, shard_grad_exp_partial={got}")


class TestOptimizerSharding(unittest.TestCase):
    """Optimizer sharding splits a parameter over data-parallel ranks, on top of TP."""

    def test_parameters_are_sharded_over_tp_and_the_optimizer_ranks(self):
        """
        Feature: derive_optimizer_sharding.
        Description: TP 4 and DP 8, with optimizer sharding over 2, 8, 3 and
            16 data-parallel ranks, and without optimizer sharding.
        Expectation: A parameter is sharded over TP times the ranks, never
            fewer than TP; a count that does not divide DP shards over all of
            it; without optimizer sharding, over TP alone.
        """
        cases = [(2, True, 8), (8, True, 32), (3, True, 32), (16, True, 32), (2, False, 4)]
        for ranks, has_op, want in cases:
            ccfg = _config(d=8, t=4, os_max_shard=ranks, has_op=has_op)
            derive(ccfg)
            self.assertEqual(ccfg.shard_p_os_non_exp_partial, want, f"{ranks} ranks, has_op={has_op}")

    def test_a_shard_spanning_dp_and_cp_is_the_runs(self):
        """
        Feature: optimizer_ranks, HyperParallel's shard over DP times CP.
        Description: DP 4 and CP 2 with a stated shard of 8 ranks, for a run
            whose shard spans CP's ranks and for one of the legacy schemas.
        Expectation: 8, the whole domain HyperParallel shards over, where a
            count not dividing DP was read as DP's 4; the legacy rule still
            reads 4.
        """
        got = [optimizer_ranks(_config(d=4, cp=2, os_max_shard=8, shard_spans_cp=spans))
               for spans in (True, False)]
        self.assertEqual(got, [8, 4])


class TestDeriveFamily(unittest.TestCase):
    """What the run does not state, the model's family says."""

    def test_byte_widths(self):
        """
        Feature: derive_byte_widths.
        Description: The default family at PP 2 and PP 1, llama2 and
            pangualpha, then runs that state their own gradients.
        Expectation: Gradients take 4 bytes under PP and none without, but
            llama2's 2, which it accumulates at any PP; pangualpha keeps a
            one-byte dropout mask; what a run states wins over its family.
        """
        cases = [
            ({"p": 2}, (4, 4, 4, 0)),
            ({"p": 1}, (0, 4, 4, 0)),
            ({"arch": "llama2", "p": 1}, (2, 4, 4, 0)),
            ({"arch": "pangualpha", "p": 2}, (4, 4, 4, 1)),
            ({"p": 1, "grad_bytes": 2, "grad_accumulation": True}, (2, 4, 4, 0)),
            ({"arch": "llama2", "p": 1, "grad_accumulation": False}, (0, 4, 4, 0)),
        ]
        for facts, want in cases:
            ccfg = _config(**facts)
            derive(ccfg)
            got = (ccfg.bytes_grad, ccfg.bytes_os, ccfg.bytes_norm, ccfg.bytes_dropout)
            self.assertEqual(got, want, f"{facts}: bytes_grad, bytes_os, bytes_norm, bytes_dropout={got}")

    def test_optimizer_bytes(self):
        """
        Feature: derive_byte_widths, the optimizer's bytes per parameter.
        Description: The default fp32 AdamW; bf16 states; Muon's one bf16
            momentum; fp32 states with an fp32 copy of the parameters.
        Expectation: A layer's parameter keeps its optimizer's states and
            any copy; the embedding and output tables AdamW's two states.
        """
        cases = [
            ({}, (8, 8)),
            ({"optimizer_state_bytes": 2}, (4, 4)),
            ({"optimizer_state_bytes": 2, "optimizer_states": 1}, (2, 4)),
            ({"optimizer_state_bytes": 4, "main_param_bytes": 4}, (12, 12)),
        ]
        for facts, want in cases:
            ccfg = _config(p=2, **facts)
            derive(ccfg)
            self.assertEqual((ccfg.bytes_optim, ccfg.bytes_optim_table), want, f"{facts}")

    def test_resharding(self):
        """
        Feature: derive_resharding.
        Description: A run that states nothing, and runs that state their
            FSDP frees a layer's gathered parameters, or keeps them.
        Expectation: The family keeps them, as MindSpore's optimizer
            parallelism does; what a run states wins.
        """
        got = []
        for facts in ({}, {"reshard_params": True}, {"reshard_params": False}):
            ccfg = _config(**facts)
            derive(ccfg)
            got.append(ccfg.reshards)
        self.assertEqual(got, [False, True, False])

    def test_routed_expert_shard(self):
        """
        Feature: routed_expert_shard.
        Description: A MoE model at DP 8 and optimizer shard 2, without expert
            parallelism and at EP 4, whose run states no expert shard, 1, 2
            or 4.
        Expectation: Stated by no one, the experts shard over the whole
            expert data-parallel group; stated, over as many of its ranks at
            EP 4 (2 of them), and with the other parameters over the
            optimizer's 2 ranks without EP.
        """
        got = {}
        for ep in (1, 4):
            for shard in (None, 1, 2, 4):
                ccfg = _config(d=8, t=1, ep=ep, n_exp=16, os_max_shard=2, expert_shard=shard)
                derive(ccfg)
                got[ep, shard] = ccfg.shard_p_os_exp
        self.assertEqual(got, {
            (1, None): 8, (1, 1): 2, (1, 2): 2, (1, 4): 2,
            (4, None): 2, (4, 1): 1, (4, 2): 2, (4, 4): 2,
        })

    def test_accumulates_grads(self):
        """
        Feature: derive_byte_widths, gradient accumulation without a pipeline.
        Description: A run that states nothing, and one that states it holds
            its gradients without pipeline parallelism.
        Expectation: The family accumulates only in a pipeline; what a run
            states wins, and at PP 1 its gradients then take memory.
        """
        got = []
        for facts in ({}, {"grad_accumulation": True}):
            ccfg = _config(p=1, **facts)
            derive(ccfg)
            got.append((ccfg.accumulates_grads, ccfg.bytes_grad > 0))
        self.assertEqual(got, [(False, False), (True, True)])

    def test_deferred_grad_accumulation(self):
        """
        Feature: derive_resharding, the gradients FSDP holds back.
        Description: A run that states nothing, and one that states its FSDP
            holds each layer's reduce-scatter output until the backward ends.
        Expectation: The family adds each output as soon as it is reduced;
            what a run states wins.
        """
        got = []
        for facts in ({}, {"deferred_grad_accumulation": True}):
            ccfg = _config(**facts)
            derive(ccfg)
            got.append(ccfg.defers_grads)
        self.assertEqual(got, [False, True])

    def test_param_casts(self):
        """
        Feature: derive_param_casts.
        Description: Runs whose optimizer shards and does not, stating
            nothing, and runs that state whether their layers keep a cast of
            their weights, as HyperParallel's states none.
        Expectation: Stated by no one, a layer keeps the casts exactly where
            the optimizer does not shard; what a run states wins.
        """
        got = []
        for facts in ({"has_op": True}, {"has_op": False}, {"has_op": False, "param_casts": False},
                      {"has_op": True, "param_casts": True}):
            ccfg = _config(**facts)
            derive(ccfg)
            got.append(ccfg.keeps_param_casts)
        self.assertEqual(got, [False, True, False, True])

    def test_overlapped_grad_reduce(self):
        """
        Feature: derive_resharding, the gradients FSDP holds whole.
        Description: A run that states nothing, and one that states its FSDP
            reduces a layer's gradients while the next layer's backward runs.
        Expectation: The family holds no whole gradient; what a run states wins.
        """
        got = []
        for facts in ({}, {"overlapped_grad_reduce": True}):
            ccfg = _config(**facts)
            derive(ccfg)
            got.append(ccfg.overlaps_grad_reduce)
        self.assertEqual(got, [False, True])

    def test_activation_sharding(self):
        """
        Feature: derive_activation_sharding.
        Description: At t=2: the default family, with a sliced recompute
            input; Qwen, whose family shards activations, and a run that
            says otherwise; a vision tower, with Qwen's language model and
            with none; Qwen with its loss on logits gathered whole, and on
            logits sharded over the vocabulary.
        Expectation: shard_recompute_input and shard_output_activ: the
            output layer's activations split only where the loss runs on
            sharded logits, as every family's does unless its run says not.
        """
        cases = [
            ({}, (1, 1)),
            ({"recompute_slice_activation": True}, (2, 1)),
            ({"arch": "qwen"}, (2, 2)),
            ({"arch": "qwen", "shard_activations": False}, (1, 1)),
            ({"shard_activations": True}, (2, 2)),
            ({"arch": "vision", "inherited_arch": "qwen"}, (2, 2)),
            ({"arch": "vision"}, (1, 1)),
            ({"arch": "qwen", "loss_parallel": False}, (2, 1)),
            ({"arch": "qwen", "loss_parallel": True}, (2, 2)),
        ]
        for facts, want in cases:
            ccfg = _config(**facts)
            derive(ccfg)
            got = (ccfg.shard_recompute_input, ccfg.shard_output_activ)
            self.assertEqual(got, want, f"{facts}: shard_recompute_input, shard_output_activ={got}")

    def test_an_mla_family_prices_its_value_heads(self):
        """
        Feature: derive_head_dim.
        Description: DeepSeek heads 56 wide by h / a, with a declared
            value-head width and without; a Qwen model declaring one.
        Expectation: The declared width, else the family's 128; Qwen keeps
            its own.
        """
        cases = [
            ({"arch": "deepseek"}, 128),
            ({"arch": "deepseek", "v_head_dim": 96}, 96),
            ({"arch": "qwen", "v_head_dim": 96}, 56),
        ]
        for facts, want in cases:
            ccfg = _config(dh=56, **facts)
            derive(ccfg)
            self.assertEqual(ccfg.dh, want, f"{facts}: dh={ccfg.dh}")


if __name__ == "__main__":
    unittest.main()
