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
"""Unit tests for dual-mode FSDP mixed-precision configuration."""
# pylint: disable=wrong-import-position

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch  # pylint: disable=wrong-import-position

from hyper_parallel.models.build_options import (
    FSDP2Config,
    FSDP2MixedPrecisionConfig,
)
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from tests.common.mark_utils import arg_mark


class _AdapterDeclaredFSDPModel(torch.nn.Module):
    """Minimal model that declares two non-decoder FSDP execution units."""

    gradient_checkpointing = False

    def __init__(self) -> None:
        """Create visual modules outside the HF decoder discovery convention."""
        super().__init__()
        self.config = SimpleNamespace(
            model_type="adapter_declared_fsdp_test",
            architectures=[],
        )
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.vision = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 2),
        )
        self.aligner = torch.nn.Linear(2, 2)


class TestFSDP2MixedPrecisionConfig(unittest.TestCase):
    """Validate optimizer-driven fp32 main gradients and core-policy mapping."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_policy_maps_cast_and_main_grad_flags(self):
        """Map YAML fields to the core policy.

        Feature: FSDP mixed-precision policy construction.
        Description: Configure forward casting and enable fp32 main params at runtime.
        Expectation: Both values reach the core policy unchanged.
        """
        config = FSDP2Config(
            mix_precision=FSDP2MixedPrecisionConfig(
                param_dtype="bfloat16",
                reduce_dtype="float32",
                cast_forward_inputs=False,
            )
        )
        manager = FSDP2Manager(config, MagicMock(), fp32_main_params=True)

        policy = manager._build_mixed_precision_policy()  # pylint: disable=protected-access

        self.assertEqual(policy.param_dtype, torch.bfloat16)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertFalse(policy.cast_forward_inputs)
        self.assertTrue(policy.apply_grad_on_fp32_main_grad)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_policy_disables_main_grad_without_optimizer_policy(self):
        """Keep ordinary optimizer gradients on ``param.grad``.

        Feature: FSDP mixed-precision policy construction.
        Description: Build FSDP without the fp32 main-parameter optimizer policy.
        Expectation: The core policy does not write gradients to ``main_grad``.
        """
        manager = FSDP2Manager(FSDP2Config(), MagicMock())

        policy = manager._build_mixed_precision_policy()  # pylint: disable=protected-access

        self.assertFalse(policy.apply_grad_on_fp32_main_grad)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_source_layout_reduction_scales_gradient_domain(self):
        """Configure source-layout gradient scaling.

        Feature: FSDP source-layout gradient semantics.
        Description: Configure a unit whose parameters have source layouts.
        Expectation: The unit uses SUM and compensates its full gradient domain.
        """
        hsdp_module = MagicMock()
        mesh_context = MagicMock(
            dp_size=2,
            cp_size=2,
            tp_size=2,
            loss_parallel=False,
        )
        manager = FSDP2Manager(FSDP2Config(), mesh_context)

        configured = manager._configure_source_layout_gradient_scaling(  # pylint: disable=protected-access
            hsdp_module,
            {object(): MagicMock()},
        )

        self.assertTrue(configured)
        hsdp_module.set_reduce_op_type.assert_called_once_with("sum", recurse=False)
        hsdp_module.set_gradient_scaling_factor.assert_called_once_with(1.0 / 8.0)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_model_declares_non_decoder_fsdp_units(self):
        """Discover adapter-declared units alongside ordinary FSDP units.

        Feature: extensible FSDP child-unit discovery.
        Description: A custom model declares vision modules not represented by
            HF's gradient-checkpointing decoder container.
        Expectation: The FSDP manager preserves their full model-qualified
            module paths as independently unsharded forward units.
        """
        manager = FSDP2Manager(FSDP2Config(), MagicMock())
        adapter_spec = SimpleNamespace(
            fsdp_wrap_modules=lambda model: ("vision.0", "aligner"),
            fsdp_excluded_subtrees=lambda model: ("vision",),
            fsdp_execution_order=lambda model, module_fqns: (
                "vision.0",
                "aligner",
                "layers.0",
            ),
        )
        model = _AdapterDeclaredFSDPModel()
        with patch(
            "hyper_parallel.distributed._builder.fsdp_adapter.get_model_adapter",
            return_value=adapter_spec,
        ):
            wrap_modules = manager._find_wrap_modules(  # pylint: disable=protected-access
                model,
            )
            ordered_wrap_modules = manager._order_wrap_modules(  # pylint: disable=protected-access
                model,
                wrap_modules,
            )

        self.assertEqual(
            [wrap_module.fqn for wrap_module in wrap_modules],
            ["layers.0", "vision.0", "aligner"],
        )
        self.assertEqual(
            [wrap_module.fqn for wrap_module in ordered_wrap_modules],
            ["vision.0", "aligner", "layers.0"],
        )

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_model_accepts_nested_declared_fsdp_units(self):
        """Accept adapter declarations with intentional nested ownership.

        Feature: extensible FSDP child-unit validation.
        Description: A model adapter declares a parent on one FSDP mesh and a
            large descendant that will use another mesh.
        Expectation: Both units are retained so the deepest unit can be
            wrapped first and excluded from its parent's managed parameters.
        """
        model = _AdapterDeclaredFSDPModel()
        adapter_spec = SimpleNamespace(
            fsdp_wrap_modules=lambda model: ("vision", "vision.0"),
        )
        with patch(
            "hyper_parallel.distributed._builder.fsdp_adapter.get_model_adapter",
            return_value=adapter_spec,
        ):
            wrap_modules = FSDP2Manager._find_adapter_declared_wrap_modules(  # pylint: disable=protected-access
                model,
            )
        self.assertEqual(
            [wrap_module.fqn for wrap_module in wrap_modules],
            ["vision", "vision.0"],
        )


if __name__ == "__main__":
    unittest.main()
