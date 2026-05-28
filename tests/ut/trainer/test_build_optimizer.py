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
"""Unit tests for ``BaseTrainer._build_optimizer``.

Decay-vs-no-decay grouping is silent on the surface — a regression won't
break loss values, it'll just slowly mis-train (LayerNorm parameters with
weight decay collapse to zero over thousands of steps). UT pins the rules
that are otherwise only catchable by long-running convergence experiments:

1. Bias / LayerNorm / RMSNorm names route to the no-decay group; the rest
   keep the configured ``weight_decay``.
2. Case-insensitive matching — ``BIAS``, ``LayerNorm``, ``rmsnorm`` all
   trigger no-decay.
3. Tied parameters are deduplicated by ``id()`` so weight decay never
   double-applies to a shared tensor (e.g. tied input embedding + output
   head).
4. Frozen parameters (``requires_grad=False``) are excluded from both groups.
5. The two ``param_groups`` retain order: decay first, no_decay second.
"""
# pylint: disable=protected-access
import os
import types
import unittest
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import nn

from hyper_parallel.trainer import base as trainer_base
from hyper_parallel.trainer.base import BaseTrainer


def _make_args(weight_decay=0.05, lr=1e-3):
    """Args namespace shaped like the fields ``_build_optimizer`` reads."""
    return types.SimpleNamespace(
        model=types.SimpleNamespace(name="toy"),
        train=types.SimpleNamespace(
            max_steps=1,
            optimizer=types.SimpleNamespace(
                lr=lr,
                weight_decay=weight_decay,
                eps=1e-8,
                betas=(0.9, 0.999),
                foreach=None,
            ),
        ),
    )


def _build_trainer(model, args=None):
    """Build a ``BaseTrainer`` with ``self.model`` already populated."""
    args = args or _make_args()
    with patch.object(trainer_base, "get_spec", return_value=object()):
        trainer = BaseTrainer(args)
    trainer.model = model
    return trainer


class _MixedModel(nn.Module):
    """Module with the four name patterns ``_build_optimizer`` discriminates on."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)  # linear.weight (decay), linear.bias (no_decay)
        self.norm = nn.LayerNorm(4)  # norm.weight / norm.bias both no_decay
        self.embed = nn.Embedding(8, 4)  # embed.weight (decay)


class TestDecayNoDecaySplit(unittest.TestCase):
    """Parameter-name classifier must route bias / norm to the no-decay group."""

    def test_bias_layernorm_route_to_no_decay(self):
        """``linear.bias`` + both LayerNorm params go to the no-decay group."""
        model = _MixedModel()
        trainer = _build_trainer(model)
        trainer._build_optimizer()
        groups = trainer.optimizer.param_groups
        self.assertEqual(len(groups), 2, f"Expected 2 param groups, got {len(groups)}")

        decay_ids = {id(p) for p in groups[0]["params"]}
        no_decay_ids = {id(p) for p in groups[1]["params"]}

        self.assertIn(id(model.linear.weight), decay_ids, (
            "Linear weight must be in decay group"
        ))
        self.assertIn(id(model.embed.weight), decay_ids, (
            "Embedding weight must be in decay group"
        ))
        self.assertIn(id(model.linear.bias), no_decay_ids, (
            "Linear bias must be in no-decay group"
        ))
        self.assertIn(id(model.norm.weight), no_decay_ids, (
            "LayerNorm weight must be in no-decay group (name contains 'norm')"
        ))
        self.assertIn(id(model.norm.bias), no_decay_ids, (
            "LayerNorm bias must be in no-decay group"
        ))

    def test_weight_decay_only_applied_to_decay_group(self):
        """``weight_decay`` from config sits on group 0; group 1 is forced to 0."""
        model = _MixedModel()
        trainer = _build_trainer(model, args=_make_args(weight_decay=0.07))
        trainer._build_optimizer()
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["weight_decay"], 0.07)
        self.assertEqual(trainer.optimizer.param_groups[1]["weight_decay"], 0.0, (
            f"no-decay group must have weight_decay=0, "
            f"got {trainer.optimizer.param_groups[1]['weight_decay']}"
        ))

    def test_case_insensitive_match(self):
        """``BIAS`` / ``LayerNorm`` / ``rmsnorm`` patterns are matched case-insensitively.

        ``base.py:568-570`` lowercases the name before substring matching;
        protect against a regression that drops ``.lower()`` (would silently
        push uppercase LayerNorm into the decay group).
        """

        class _UpperCaseModel(nn.Module):

            def __init__(self):
                super().__init__()
                # Use ``register_parameter`` to keep arbitrary names that survive
                # ``named_parameters`` walking.
                self.register_parameter("BIAS_TOKEN", nn.Parameter(torch.ones(2)))
                self.register_parameter("LAYERNORM_W", nn.Parameter(torch.ones(2)))
                self.register_parameter("MyRMSNorm", nn.Parameter(torch.ones(2)))
                self.register_parameter("real_weight", nn.Parameter(torch.ones(2)))

        model = _UpperCaseModel()
        trainer = _build_trainer(model)
        trainer._build_optimizer()
        no_decay_ids = {id(p) for p in trainer.optimizer.param_groups[1]["params"]}
        for attr in ("BIAS_TOKEN", "LAYERNORM_W", "MyRMSNorm"):
            self.assertIn(id(getattr(model, attr)), no_decay_ids, (
                f"{attr!r} must route to no-decay group (case-insensitive match)"
            ))
        self.assertNotIn(id(model.real_weight), no_decay_ids, (
            "'real_weight' has none of the no-decay tokens — must stay in decay group"
        ))


class TestTiedParameterDedup(unittest.TestCase):
    """Tied parameters (shared ``nn.Parameter`` object) must appear in exactly one group."""

    def test_tied_embedding_output_head_not_double_counted(self):
        """A param object reachable under two names lands in the optimizer once."""

        class _TiedModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 4)
                # Tie output head to embedding weight (HF convention).
                self.lm_head = nn.Linear(4, 8, bias=False)
                self.lm_head.weight = self.embed.weight

        model = _TiedModel()
        trainer = _build_trainer(model)
        trainer._build_optimizer()
        all_ids = []
        for grp in trainer.optimizer.param_groups:
            all_ids.extend(id(p) for p in grp["params"])
        # The shared weight should appear once total, not twice.
        self.assertEqual(
            all_ids.count(id(model.embed.weight)), 1,
            f"Tied weight must be deduped: appeared {all_ids.count(id(model.embed.weight))} "
            f"times across groups",
        )


class TestFrozenParametersExcluded(unittest.TestCase):
    """``requires_grad=False`` parameters never reach the optimizer."""

    def test_frozen_param_skipped(self):
        """A frozen param is excluded from both decay and no-decay groups."""
        model = _MixedModel()
        model.embed.weight.requires_grad_(False)
        trainer = _build_trainer(model)
        trainer._build_optimizer()
        all_ids = set()
        for grp in trainer.optimizer.param_groups:
            all_ids.update(id(p) for p in grp["params"])
        self.assertNotIn(id(model.embed.weight), all_ids, (
            "Frozen embedding must not appear in optimizer param_groups"
        ))
        # Sanity: trainable params still present.
        self.assertIn(id(model.linear.weight), all_ids)


if __name__ == "__main__":
    unittest.main()
