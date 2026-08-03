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
"""Unit tests for the direct ``AdamW`` class target."""

import unittest

import torch
from torch import nn
from torch.optim import Optimizer

from hyper_models.components.optim import AdamW
from hyper_models.trainer.config import Target


OPTIMIZER_TARGET = "hyper_models.components.optim.optimizer.optimizer.AdamW"


class _MixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.embed = nn.Embedding(8, 4)


class TestAdamWTarget(unittest.TestCase):
    """The YAML class target returns one runtime optimizer."""

    def test_returns_single_optimizer_with_decay_groups(self):
        model = _MixedModel()

        target = Target(
            AdamW,
            target_path=OPTIMIZER_TARGET,
            lr=1e-3,
            weight_decay=0.07,
        )
        optimizer = target.build(
            model=model,
            device_mesh=object(),
        )

        self.assertIsInstance(optimizer, Optimizer)
        self.assertNotIsInstance(optimizer, list)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.07)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

        decay_ids = {id(param) for param in optimizer.param_groups[0]["params"]}
        no_decay_ids = {
            id(param) for param in optimizer.param_groups[1]["params"]
        }
        self.assertIn(id(model.linear.weight), decay_ids)
        self.assertIn(id(model.embed.weight), decay_ids)
        self.assertIn(id(model.linear.bias), no_decay_ids)
        self.assertIn(id(model.norm.weight), no_decay_ids)
        self.assertIn(id(model.norm.bias), no_decay_ids)

    def test_frozen_parameters_are_excluded(self):
        model = _MixedModel()
        model.embed.weight.requires_grad_(False)

        optimizer = AdamW(model=model)

        parameter_ids = {
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        }
        self.assertNotIn(id(model.embed.weight), parameter_ids)
        self.assertIn(id(model.linear.weight), parameter_ids)

    def test_tied_parameter_is_deduplicated_across_model_parts(self):
        shared = nn.Parameter(torch.ones(2))

        class _FirstPart(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("shared_weight", shared)
                self.register_parameter("first_bias", nn.Parameter(torch.ones(2)))

        class _SecondPart(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("shared_weight", shared)
                self.register_parameter("second_weight", nn.Parameter(torch.ones(2)))

        class _PartitionedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.parts = nn.ModuleList([_FirstPart(), _SecondPart()])

        optimizer = AdamW(model=_PartitionedModel())
        parameter_ids = [
            id(param)
            for group in optimizer.param_groups
            for param in group["params"]
        ]

        self.assertEqual(parameter_ids.count(id(shared)), 1)

    def test_no_decay_matching_is_case_insensitive(self):
        class _NamedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("LAYERNORM_WEIGHT", nn.Parameter(torch.ones(2)))
                self.register_parameter("regular_weight", nn.Parameter(torch.ones(2)))

        model = _NamedModel()
        optimizer = AdamW(model=model)
        no_decay_ids = {
            id(param) for param in optimizer.param_groups[1]["params"]
        }

        self.assertIn(id(model.LAYERNORM_WEIGHT), no_decay_ids)
        self.assertNotIn(id(model.regular_weight), no_decay_ids)


if __name__ == "__main__":
    unittest.main()
