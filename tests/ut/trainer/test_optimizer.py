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
"""Unit tests for auto-model optimizer parameter grouping."""
# Select the Torch backend before importing HyperParallel checkpoint aliases.
# pylint: disable=wrong-import-position

import os
import unittest

import torch
from torch import nn

from tests.ut.platform.mindspore._ensure_mindspore_platform import (
    restore_torch_platform_for_ut,
)

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
restore_torch_platform_for_ut()

from hyper_parallel.auto_models.components.optim.optimizer.optimizer import (
    AdamW,
    get_parameter_names,
)
from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper


class _Block(nn.Module):
    """Small block containing parameters from both AdamW decay groups."""

    def __init__(self) -> None:
        """Initialize projection and normalization parameters."""
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the projection followed by normalization."""
        return self.norm(self.proj(inputs))


class _Model(nn.Module):
    """Model supporting whole-layer and submodule checkpoint wrapping."""

    def __init__(self) -> None:
        """Initialize one checkpointable block and an output projection."""
        super().__init__()
        self.block = _Block()
        self.output = nn.Linear(4, 4, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the block and output projection."""
        return self.output(self.block(inputs))


def _group_weight_decay_by_parameter(model: nn.Module) -> dict[int, float]:
    """Return the configured weight decay for every trainable parameter."""
    groups, _ = AdamW.get_adamw_param_groups(
        model,
        weight_decay=0.1,
        no_decay_params=("bias", "norm"),
    )
    return {
        id(parameter): group["weight_decay"]
        for group in groups
        for parameter in group["params"]
    }


class TestOptimizerParameterGrouping(unittest.TestCase):
    """Tests for wrapper-stable AdamW parameter classification."""

    def test_get_parameter_names_uses_checkpoint_wrapper_public_names(self):
        """Decay names should follow the wrapper's public parameter names."""
        wrapped_model = checkpoint_wrapper(_Model())

        decay_names = get_parameter_names(wrapped_model, ("bias", "norm"))

        self.assertEqual(decay_names, ["block.proj.weight", "output.weight"])
        self.assertTrue(all("_swap_wrapped_module" not in name for name in decay_names))

    def test_checkpoint_wrapping_preserves_adamw_parameter_groups(self):
        """Whole-layer and submodule wrappers should not change weight decay."""
        for wrap_target in ("whole_layer", "submodule"):
            with self.subTest(wrap_target=wrap_target):
                model = _Model()
                expected_groups = _group_weight_decay_by_parameter(model)
                if wrap_target == "whole_layer":
                    wrapped_model = checkpoint_wrapper(model)
                else:
                    model.block.proj = checkpoint_wrapper(model.block.proj)
                    wrapped_model = model

                actual_groups = _group_weight_decay_by_parameter(wrapped_model)

                self.assertEqual(actual_groups, expected_groups)
                self.assertEqual(
                    actual_groups[id(model.block.proj.weight)],
                    0.1,
                )
                self.assertEqual(
                    actual_groups[id(model.block.norm.weight)],
                    0.0,
                )


if __name__ == "__main__":
    unittest.main()
