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
    derive_recompute_switches,
    optimizer_ranks,
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


def _derived(**fields: Any) -> SimpleNamespace:
    """:func:`_config` with *fields*, its derived fields set."""
    ccfg = _config(**fields)
    derive(ccfg)
    return ccfg


class TestDerive(unittest.TestCase):
    """derive computes each field from the facts a parser states."""

    def test_derive_sets_every_derived_field(self):
        """
        Feature: derive.
        Description: One call on a dense config that slices the recompute
            input and recomputes selectively.
        Expectation: The expert degrees, sharding factors, communication
            flags, recompute switches, flash attention factor and byte widths
            are all set, the last the default family's without pipelining.
        """
        ccfg = _derived(recompute_slice_activation=True, sel_rec=True)
        got = (ccfg.t_exp, ccfg.d_exp, ccfg.comm_d_non_exp, ccfg.s_fa)
        self.assertEqual(got, (2, 4, 2, 128.0), f"t_exp, d_exp, comm_d_non_exp, s_fa={got}")
        got = (ccfg.shard_embed, ccfg.shard_recompute_input, ccfg.shard_output_activ)
        self.assertEqual(got, (8, 2, 1), f"shard_embed, shard_recompute_input, shard_output_activ={got}")
        self.assertEqual(vars(ccfg.rec_op), HYPER_SELECTIVE_REC_OP, f"rec_op={vars(ccfg.rec_op)}")
        got = (ccfg.bytes_grad, ccfg.bytes_os, ccfg.bytes_norm, ccfg.bytes_dropout, ccfg.layer_fields)
        self.assertEqual(got, (0, 4, 4, 0, None), f"byte widths and layer_fields={got}")

    def test_mindformers_recompute_switches(self):
        """
        Feature: derive_recompute_switches.
        Description: MindFormers' selective recompute under flash attention
            and sequence parallelism.
        Expectation: The head cast, norm and activation are recomputed and
            the attention kernels kept, as are the all-gather, which only
            select_comm_recompute recomputes, and an MLA layer's
            up-projections, which MindFormers' rule leaves out.
        """
        ccfg = _config(sel_rec_rule="mindformers", sel_rec=True, sp=2)
        derive_recompute_switches(ccfg)
        expected = {"attBMM": 1, "headCast": 0, "dropout": 1, "softmax": 1, "normOp": 0, "gather": 1, "ffAct": 0,
                    "attUp": 1}
        self.assertEqual(vars(ccfg.rec_op), expected, f"rec_op={vars(ccfg.rec_op)}")

    def test_sharding(self):
        """
        Feature: derive's sharding of the embedding, the gradients and the
            optimizer states, the routed experts' included.
        Description: At d=4 and t=2: a vocabulary embedding data parallel
            without pipelining, TOML's table, and at d=8 a table FSDP shards
            over its group of 2 ranks, as under HSDP; gradients sharded as
            the parameters, as FSDP holds them, over an optimizer shard of
            2 ranks, and neither. At d=8 and t=4, an optimizer shard of 3
            ranks. At DP 4 and CP 2, a shard of 8 ranks that spans CP's
            ranks, and one of the legacy schemas (C1). A MoE model at d=8,
            t=1 and an optimizer shard of 2, stating no expert shard without
            EP and at EP 4, and stating 4 without EP.
        Expectation: The table over d, over t alone, and over the FSDP
            group and t, gathered over the group; the non-expert, expert and
            partial expert gradient factors the parameters' (4, 8, 1), the
            optimizer's (8, 8, 1), and TP's alone; a count that does not
            divide DP shards over all of it, on top of TP; the shard
            spanning CP over its 8 ranks, the legacy one over DP's 4; the
            experts over their whole expert data-parallel group, else over
            the optimizer's 2 ranks.
        """
        tables = [_derived(**facts) for facts in (
            {"vocab_emb_dp": True}, {"emb_dp_sharded": False},
            {"d": 8, "os_max_shard": 2, "grad_shard_as_params": True})]
        self.assertEqual([(ccfg.shard_embed, ccfg.gather_embed) for ccfg in tables], [(4, 1), (2, 1), (4, 2)])
        grads = [_derived(os_max_shard=2, **facts)
                 for facts in ({"grad_shard_as_params": True}, {"has_grad_shard": True}, {})]
        self.assertEqual([(ccfg.shard_grad_non_exp, ccfg.shard_grad_exp, ccfg.shard_grad_exp_partial)
                          for ccfg in grads], [(4, 8, 1), (8, 8, 1), (2, 2, 2)])
        self.assertEqual(_derived(d=8, t=4, os_max_shard=3).shard_p_os_non_exp_partial, 32)
        self.assertEqual([optimizer_ranks(_config(d=4, cp=2, os_max_shard=8, shard_spans_cp=spans))
                          for spans in (True, False)], [8, 4])
        experts = [_derived(d=8, t=1, ep=ep, n_exp=16, os_max_shard=2, expert_shard=shard).shard_p_os_exp
                   for ep, shard in ((1, None), (4, None), (1, 4))]
        self.assertEqual(experts, [8, 2, 2])


class TestDeriveFamily(unittest.TestCase):
    """What the run does not state, the model's family says."""

    def test_what_the_run_states_wins_over_its_family(self):
        """
        Feature: derive_family and derive_param_casts.
        Description: At t=2: llama2, whose family accumulates its 2-byte
            gradients at any PP, in a run that does not; at PP 2, Muon's
            one bf16 momentum, and fp32 states with an fp32 copy of the
            parameters; a run that states nothing of FSDP's resharding, and
            one that reshards; runs whose optimizer does not shard, stating
            nothing of the weight casts and stating none; Qwen, whose family
            shards activations, in a run that states nothing, one that does
            not shard them, and one whose loss runs on logits gathered
            whole, and a vision tower under Qwen's language model; DeepSeek
            heads 56 wide by h / a, declaring no value-head width and 96,
            and a Qwen model declaring 96.
        Expectation: No gradient bytes; a layer's parameter keeps its
            optimizer's states and any copy, the embedding and output tables
            AdamW's two; the family keeps the gathered parameters, as
            MindSpore's optimizer parallelism does; casts where the
            optimizer does not shard, unless the run says not; Qwen's
            sharded recompute input and output layer, then neither, then
            the output layer's activations whole, and the tower Qwen's; the
            family's 128, then the declared 96, and Qwen keeps its own.
        """
        self.assertEqual(_derived(arch="llama2", p=1, grad_accumulation=False).bytes_grad, 0)
        optim = [_derived(p=2, optimizer_state_bytes=2, optimizer_states=1),
                 _derived(p=2, optimizer_state_bytes=4, main_param_bytes=4)]
        self.assertEqual([(ccfg.bytes_optim, ccfg.bytes_optim_table) for ccfg in optim], [(2, 4), (12, 12)])
        self.assertEqual([_derived(**facts).reshards for facts in ({}, {"reshard_params": True})], [False, True])
        casts = [_derived(has_op=False, **facts).keeps_param_casts for facts in ({}, {"param_casts": False})]
        self.assertEqual(casts, [True, False])
        sharding = [_derived(**facts) for facts in (
            {"arch": "qwen"}, {"arch": "qwen", "shard_activations": False}, {"arch": "qwen", "loss_parallel": False},
            {"arch": "vision", "inherited_arch": "qwen"})]
        self.assertEqual([(ccfg.shard_recompute_input, ccfg.shard_output_activ) for ccfg in sharding],
                         [(2, 2), (1, 1), (2, 1), (2, 2)])
        heads = [_derived(dh=56, **facts).dh for facts in (
            {"arch": "deepseek"}, {"arch": "deepseek", "v_head_dim": 96}, {"arch": "qwen", "v_head_dim": 96})]
        self.assertEqual(heads, [128, 96, 56])


if __name__ == "__main__":
    unittest.main()
