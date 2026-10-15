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
"""Tests for the attention, feed-forward and norm formulas of the memory model.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/memory_estimation/test_layer_block.py -v
"""
import unittest
from types import SimpleNamespace

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import (
    EvalAttn,
    EvalFFn,
    EvalNorm,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType


def _score(softmax: int, layer: LayerType) -> float:
    """Score-tensor activations of one layer with the given softmax switch."""
    ccfg = SimpleNamespace(
        s_fa=4, b=1, a=2, s=4, n_softmax=1, t=1, cp=1,
        bytes_softmax=4, bytes_dropout=1, bytes_compute=2,
        rec_op=Config({"softmax": softmax, "dropout": 1, "headCast": 1}),
    )
    return EvalAttn.attn_score_activations(ccfg, SimpleNamespace(current_node=layer, micro_factor=1))


class TestAttentionScore(unittest.TestCase):
    """The softmax switch acts only in a selective layer, like every other switch."""

    def test_a_plain_layer_keeps_its_softmax(self):
        """A layer that recomputes nothing stores the softmax whatever the switch says."""
        on, off = _score(1, LayerType.NOT_REC_LAYER), _score(0, LayerType.NOT_REC_LAYER)
        self.assertEqual(off, on, f"switch at 0 gives {off}, at 1 gives {on}")


def _mla() -> SimpleNamespace:
    """DeepSeek-V3's attention: width 7168, 128 heads, latents of 1536 and 512, heads 128 and 64 wide."""
    return SimpleNamespace(h=7168, a=128, n_kv=128, dh=128, dhr=64, dc_q=1536, dc_kv=512, n_attMM=4,
                           qk_nope_head_dim=None)


class TestMlaParameters(unittest.TestCase):
    """What an MLA layer's attention holds."""

    def test_the_shared_latent_is_counted_once(self):
        """
        Feature: EvalAttn.num_params_mla.
        Description: DeepSeek-V3's attention.
        Expectation: The query latent's two projections and norm, the one
            down-projection the keys and values share with the rotary key,
            their latent's norm, their up-projection and the output
            projection: 187107328, the count of the model Transformers
            builds.
        """
        self.assertEqual(EvalAttn.num_params_mla(_mla(), None), 187107328)


def _vectors(**stated) -> SimpleNamespace:
    """A layer of width 64, 4 query and 2 key heads 16 wide, gated experts 32 wide, a vocabulary of 100."""
    return SimpleNamespace(**{"h": 64, "a": 4, "n_kv": 2, "dh": 16, "dc_kv": 0, "n_attMM": 4, "attn_output_gate": False,
                              "attn_extra_p": 0, "n_ffMM": 3, "hff": 128, "hff_exp": 32, "etp": 1, "n_exp": 4,
                              "n_shared_exp": 1, "n_normOp": 2, "n_qknorm": 0, "v": 100, "qkv_bias": None,
                              "o_bias": None, "mlp_bias": None, "norm_bias": None, "layer_norms": None,
                              "shared_expert_gate": None, **stated})


class TestVectors(unittest.TestCase):
    """The biases and norm weights a layer holds, as its model states them."""

    def test_stated_each_vector_is_the_models(self):
        """
        Feature: the parameter formulas of a model that states its biases.
        Description: The layer biasing its query, key and value projections
            and its feed-forward, as Qwen2 and a biased Llama do.
        Expectation: Each bias as wide as its projection's output, none
            where stated none.
        """
        biased = _vectors(qkv_bias=True, o_bias=False, mlp_bias=True)
        self.assertEqual(EvalAttn.num_params_attn(biased, None), 64 * 64 * 2 + 64 * 32 * 2 + 64 + 2 * 32)
        self.assertEqual(EvalFFn.num_params_routed_expert(biased, None), 4 * (3 * 32 * 64 + 2 * 32 + 64))


def _experts(hff: int) -> SimpleNamespace:
    """A MoE layer of width 64 over 16 tokens: 4 experts 32 wide, 2 chosen, one shared, the dense width *hff*."""
    return SimpleNamespace(s=16, b=1, h=64, hff=hff, hff_exp=32, n_ffMM=3, n_ffParamCast=0, bytes_compute=2, t=1,
                           cp=1, n_exp=4, n_chosen_exp=2, n_shared_exp=1, gmm=True, cap_fact=1, mlp_bias=None,
                           shared_expert_gate=None, rec_op=Config({"ffAct": 1}))


class TestExpertWidth(unittest.TestCase):
    """A MoE layer's experts are as wide as the model states them, whatever its dense layers' width."""

    def test_every_expert_is_hff_exp_wide(self):
        """
        Feature: EvalFFn.routed_exp_activations.
        Description: The MoE layer of a model whose dense layers are 128
            wide, four times its experts.
        Expectation: Each of the 32 token-expert pairs keeps its input, two
            projections into 32 and the activation's output, in bf16.
        """
        ccfg = _experts(128)
        ctx = SimpleNamespace(current_node=LayerType.NOT_REC_LAYER, micro_factor=1, dropless_tok_factor=1)
        self.assertEqual(EvalFFn.routed_exp_activations(ccfg, ctx), 16 * 2 * 2 * (64 + 3 * 32))


def _norms(n_qk_norm: int, norm_switch: int = 1) -> SimpleNamespace:
    """A layer with two norms, 4 query and 2 key heads 32 wide, and *n_qk_norm* QK-norms."""
    return SimpleNamespace(
        h=64, a=4, n_kv=2, dh=32, n_normOp=2, n_qknorm=n_qk_norm, s=8, b=1, bytes_norm=4, t=1, cp=1,
        rec_op=Config({"normOp": norm_switch}),
    )


class TestQkNorm(unittest.TestCase):
    """A QK-norm is priced as a norm over every head's queries and keys."""

    def test_a_qk_norm_holds_its_weights_and_its_inputs(self):
        """
        Feature: the memory of a QK-norm.
        Description: The same layer with a QK-norm and without.
        Expectation: The QK-norm adds a query and a key weight of one head's
            width, 2 x 32 parameters, and keeps every head's queries and
            keys, 6 heads of 32 per token, at the norms' width in bytes.
        """
        ctx = SimpleNamespace(current_node=LayerType.NOT_REC_LAYER, micro_factor=1)
        params = [EvalNorm.num_params_norm(_norms(n), ctx) for n in (1, 0)]
        activations = [EvalNorm.norm_activations(_norms(n), ctx) for n in (1, 0)]
        self.assertEqual((params[0] - params[1], activations[0] - activations[1]), (2 * 32, 8 * 1 * 4 * 6 * 32))

    def test_a_selective_layer_recomputes_it_with_the_norms(self):
        """
        Feature: the norm switch.
        Description: A selective layer whose normOp switch is 0.
        Expectation: It keeps neither its norms' inputs nor its QK-norm's.
        """
        ctx = SimpleNamespace(current_node=LayerType.SEL_REC_LAYER, micro_factor=1)
        self.assertEqual(EvalNorm.norm_activations(_norms(1, norm_switch=0), ctx), 0)


if __name__ == "__main__":
    unittest.main()
