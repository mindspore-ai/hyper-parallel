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

"""Sinkhorn composition, identity routing and model-owned head views."""

import copy
import unittest

import torch
from torch import nn

from hyper_parallel.components.optim import Sinkhorn as SinkhornBuilder
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.optimizer import get_hyper_optimizer
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.sinkhorn import Sinkhorn
from hyper_parallel.models.deepseek_v41.adapter.optim.headwise_muon import build_reshape_fn
from hyper_parallel.trainer.config.target import Target


def _model():
    model = nn.Module()
    model.embed = nn.Embedding(7, 4)
    model.lm_head = nn.Linear(4, 7, bias=False)
    model.lm_head.weight = model.embed.weight
    model.self_attn = nn.Module()
    model.self_attn.q_b_proj = nn.Linear(4, 6, bias=False)
    model.norm = nn.LayerNorm(4)
    model.scale = nn.Parameter(torch.ones(()))
    return model


class TestSinkhornBuilder(unittest.TestCase):
    """Verify routing without altering the original optimizer runtimes."""

    def test_partition_tied_weights_overrides_and_decay(self):
        """
        Feature: Sinkhorn composition and model-owned head-wise Muon.
        Description: Tied embeddings update once, group LR overrides persist, and norms decay.
        Expectation: Routing, numerical results and state round trips satisfy the asserted invariants.
        """
        model = _model()
        config = {"lr": 0.01, "param_groups": [{"param_patterns": [r"lm_head\.weight$"], "lr": 0.05}]}
        original = copy.deepcopy(config)
        optimizer = SinkhornBuilder(config, {}, {"weight_decay": 0.1}, model).get_optimizer()
        leaves = optimizer.optimizers_dict
        self.assertEqual(config, original)
        self.assertIsInstance(leaves["sinkhorn"], Sinkhorn)
        self.assertIsInstance(leaves["muon"], Muon)
        self.assertEqual(leaves["sinkhorn"].param_groups[0]["lr"], 0.05)
        self.assertEqual(len(leaves["sinkhorn"].param_groups[0]["params"]), 1)
        ids = [id(param) for group in optimizer.param_groups for param in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(param) for param in model.parameters()})
        decay = {id(param): group["weight_decay"] for group in leaves["adamw"].param_groups
                 for param in group["params"]}
        self.assertEqual(decay[id(model.norm.weight)], 0.1)
        self.assertEqual(decay[id(model.norm.bias)], 0.0)
        self.assertEqual(decay[id(model.scale)], 0.0)

    def test_conflicting_aliases_and_overrides_rejected(self):
        """
        Feature: Sinkhorn composition and model-owned head-wise Muon.
        Description: All tied aliases participate in conflict checking.
        Expectation: Routing, numerical results and state round trips satisfy the asserted invariants.
        """
        model = _model()
        with self.assertRaisesRegex(ValueError, "Overlapping optimizer"):
            SinkhornBuilder({"param_patterns": [r"embed\.weight$"]}, {},
                            {"param_patterns": [r"lm_head\.weight$"]}, model)
        with self.assertRaisesRegex(ValueError, "Overlapping Sinkhorn"):
            SinkhornBuilder({"param_groups": [{"param_patterns": ["embed"]},
                                               {"param_patterns": ["lm_head"]}]}, {}, {}, model)

    def test_excluded_and_frozen_parameters(self):
        """
        Feature: Sinkhorn composition and model-owned head-wise Muon.
        Description: Host-owned and frozen parameters never enter the dense optimizer.
        Expectation: Routing, numerical results and state round trips satisfy the asserted invariants.
        """
        model = _model()
        model.scale.requires_grad_(False)
        optimizer = SinkhornBuilder({}, {}, {}, model, excluded_param_ids=[id(model.embed.weight)]).get_optimizer()
        ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
        self.assertNotIn(id(model.embed.weight), ids)
        self.assertNotIn(id(model.scale), ids)
        self.assertNotIn("sinkhorn", optimizer.optimizers_dict)

    def test_all_factory_combinations_checkpoint(self):
        """
        Feature: Sinkhorn composition and model-owned head-wise Muon.
        Description: Every nonempty subset has a collision-free state dict and resumes updates.
        Expectation: Routing, numerical results and state round trips satisfy the asserted invariants.
        """
        for mask in range(1, 8):
            with self.subTest(mask=mask):
                model = _model()
                optimizer = get_hyper_optimizer(
                    model, [model.self_attn.q_b_proj.weight] if mask & 1 else [],
                    [model.norm.weight] if mask & 2 else [],
                    sinkhorn_params=[model.embed.weight] if mask & 4 else [],
                )
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        param.grad = torch.randn_like(param)
                optimizer.step()
                checkpoint = copy.deepcopy(optimizer.state_dict())
                before = [param.detach().clone() for param in model.parameters()]
                optimizer.step()
                expected = [param.detach().clone() for param in model.parameters()]
                with torch.no_grad():
                    for param, value in zip(model.parameters(), before):
                        param.copy_(value)
                optimizer.load_state_dict(checkpoint)
                optimizer.step()
                for param, value in zip(model.parameters(), expected):
                    torch.testing.assert_close(param, value)
                self.assertEqual(optimizer.flatten, mask.bit_count() > 1)

    def test_fp32_main_params_without_sinkhorn_reset(self):
        """
        Feature: Sinkhorn composition and model-owned head-wise Muon.
        Description: The unchanged wrapper replaces Sinkhorn params and preserves resume state.
        Expectation: Routing, numerical results and state round trips satisfy the asserted invariants.
        """
        model = nn.Linear(3, 4, bias=False).bfloat16()
        chain = get_hyper_optimizer(model, [], [], sinkhorn_params=[model.weight])
        leaf = chain.optimizer
        self.assertFalse(hasattr(leaf, "reset_optimizer_parameters"))
        self.assertFalse(hasattr(leaf, "is_muon"))
        wrapped = Float16OptimizerWithFloat16Params(chain, model)
        main = leaf.param_groups[0]["params"][0]
        self.assertIsNot(main, model.weight)
        self.assertEqual(main.dtype, torch.float32)
        # Sinkhorn itself works with a parameter that has no model-name metadata.
        del main.model_name
        model.weight.grad = torch.ones_like(model.weight)
        wrapped.step()
        self.assertIn(main, leaf.state)
        checkpoint = copy.deepcopy(wrapped.state_dict())
        wrapped.load_state_dict(checkpoint)
        self.assertEqual(leaf.state[main]["momentum_buffer"].dtype, torch.float32)

    def test_nested_target_constructs_model_callback(self):
        """
        Feature: Sinkhorn composition and model-owned head-wise Muon.
        Description: The existing nested Target protocol injects a callable without framework changes.
        Expectation: Routing, numerical results and state round trips satisfy the asserted invariants.
        """
        callback = Target(build_reshape_fn, target_path=build_reshape_fn.__module__ + ".build_reshape_fn",
                          attention_head_dim=2, indexer_head_dim=1)
        target = Target(SinkhornBuilder, target_path="hyper_parallel.components.optim.Sinkhorn",
                        sinkhorn_config={}, muon_config={"reshape_fn": callback}, adamw_config={})
        optimizer = target.build(model=_model()).get_optimizer()
        self.assertTrue(callable(optimizer.optimizers_dict["muon"].reshape_fn))


class TestHeadwiseMuon(unittest.TestCase):
    """The model callback returns head views accepted by the unchanged Muon."""

    def test_views_and_identity(self):
        """
        Feature: Sinkhorn composition and model-owned head-wise Muon.
        Description: Backbone, DSpark and indexer head views preserve storage and unrelated weights.
        Expectation: Routing, numerical results and state round trips satisfy the asserted invariants.
        """
        reshape = build_reshape_fn(2, 1)
        matrix = torch.randn(6, 4)
        for name, shape in (("model.layers.0.self_attn.q_b_proj.weight", (3, 2, 4)),
                            ("model.layers.0.self_attn.kv_proj.weight", (3, 2, 4)),
                            ("dspark.stages.0.self_attn.wq_b.weight", (3, 2, 4)),
                            ("model.layers.0.self_attn.indexer.q_b_proj.weight", (6, 1, 4)),
                            ("model.layers.0.self_attn.indexer.wk.weight", (6, 1, 4))):
            view = reshape(name, matrix)[0]
            self.assertEqual(tuple(view.shape), shape)
            self.assertEqual(view.untyped_storage().data_ptr(), matrix.untyped_storage().data_ptr())
        self.assertIs(reshape("self_attn.q_a_proj.weight", matrix)[0], matrix)
        self.assertIs(reshape("dspark.stages.0.self_attn.wq_a.weight", matrix)[0], matrix)
        key_matrix = torch.randn(2, 4)
        key_view = reshape("dspark.stages.0.self_attn.wkv.weight", key_matrix)[0]
        self.assertEqual(tuple(key_view.shape), (1, 2, 4))
        self.assertEqual(key_view.untyped_storage().data_ptr(), key_matrix.untyped_storage().data_ptr())
        with self.assertRaisesRegex(ValueError, "divisible"):
            reshape("self_attn.q_b_proj.weight", torch.randn(5, 4))
        with self.assertRaises(ValueError):
            build_reshape_fn(0, 1)

    def test_matches_independent_head_optimizers(self):
        """
        Feature: Sinkhorn composition and model-owned head-wise Muon.
        Description: Backbone and DSpark multi-step updates equal independently optimized heads.
        Expectation: Routing, numerical results and state round trips satisfy the asserted invariants.
        """
        for name, rows in (("self_attn.q_b_proj.weight", 6),
                           ("dspark.stages.0.self_attn.wq_b.weight", 6),
                           ("dspark.stages.0.self_attn.wkv.weight", 2)):
            with self.subTest(name=name):
                torch.manual_seed(91)
                param = nn.Parameter(torch.randn(rows, 4))
                param.model_name = name
                heads = [nn.Parameter(value.clone()) for value in param.detach().split(2)]
                optimizer = Muon([param], reshape_fn=build_reshape_fn(2, 1))
                reference = Muon(heads)
                for _ in range(3):
                    param.grad = torch.randn_like(param)
                    for head, gradient in zip(heads, param.grad.split(2)):
                        head.grad = gradient.clone()
                    optimizer.step()
                    reference.step()
                    torch.testing.assert_close(param, torch.cat(heads), rtol=1e-5, atol=1e-6)
