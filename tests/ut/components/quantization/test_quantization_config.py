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
"""CPU-only contracts for the low-precision policy catalog."""

import unittest

from torch import nn

from hyper_parallel.components.quantization.config import (
    LowPrecisionConfig,
    LowPrecisionDtypeScheme,
)
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.trainer.config.parallelism import (
    PlanOverride,
    entries_to_module_replacements,
)
from hyper_parallel.trainer.config.target import Target
from tests.common.mark_utils import arg_mark


cpu_test = arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)


class _SourceModule(nn.Module):
    """Importable source type used by the plan-override test."""

    def forward(self, value):
        return value


class TestLowPrecisionDtypeScheme(unittest.TestCase):
    """Validation contracts that must fail before an NPU operator is called."""

    @cpu_test
    def test_supported_native_and_fake_schemes(self):
        """W8A8, native W4A8 and fake W4A8 policies retain typed fields."""
        w8a8 = LowPrecisionDtypeScheme()
        native = LowPrecisionDtypeScheme(
            weight_format="mxfp4", act_format="mxfp8", block_size=128
        )
        fake = LowPrecisionDtypeScheme(
            is_fake_quantize=True,
            weight_format="mxfp4",
            act_format="mxfp8",
            block_size=32,
        )

        self.assertEqual((w8a8.weight_format, w8a8.act_format), ("mxfp8", "mxfp8"))
        self.assertEqual(native.block_size, 128)
        self.assertTrue(fake.is_fake_quantize)

    @cpu_test
    def test_invalid_family_block_and_flag_fail_fast(self):
        """Cross-family formats and malformed MX policies are rejected."""
        with self.assertRaisesRegex(ValueError, "same format family"):
            LowPrecisionDtypeScheme(weight_format="mxfp4", act_format="hif8")
        with self.assertRaisesRegex(ValueError, "block_size must be 32 or 128"):
            LowPrecisionDtypeScheme(
                weight_format="mxfp4", act_format="mxfp8", block_size=64
            )
        with self.assertRaisesRegex(ValueError, "native w8a8.*block_size=32"):
            LowPrecisionDtypeScheme(block_size=128)
        with self.assertRaisesRegex(ValueError, "is_fake_quantize must be a bool"):
            LowPrecisionDtypeScheme(is_fake_quantize=1)


class TestLowPrecisionConfig(unittest.TestCase):
    """Named policy selection and legacy compatibility contracts."""

    @cpu_test
    def test_mapping_catalog_is_typed_resolved_and_serialized(self):
        """YAML-like mappings become typed schemes and round-trip to dictionaries."""
        config = LowPrecisionConfig(
            enabled=True,
            dtype_schemes={
                "native": {
                    "weight_format": "mxfp4",
                    "act_format": "mxfp8",
                    "block_size": 32,
                },
                "fake": {
                    "is_fake_quantize": True,
                    "weight_format": "mxfp4",
                    "act_format": "mxfp8",
                    "block_size": 128,
                },
            },
            default_dtype_scheme="native",
        )

        self.assertIsInstance(config.dtype_schemes["native"], LowPrecisionDtypeScheme)
        self.assertIs(config.resolve_dtype_scheme(), config.dtype_schemes["native"])
        self.assertIs(config.resolve_dtype_scheme("fake"), config.dtype_schemes["fake"])
        self.assertEqual(config.to_dict()["dtype_schemes"]["fake"]["block_size"], 128)

    @cpu_test
    def test_catalog_selection_errors_are_explicit(self):
        """A catalog cannot silently fall back when no valid name was selected."""
        config = LowPrecisionConfig(dtype_schemes={"native": LowPrecisionDtypeScheme()})
        with self.assertRaisesRegex(ValueError, "did not select"):
            config.resolve_dtype_scheme()
        with self.assertRaisesRegex(ValueError, "Unknown low-precision dtype scheme"):
            config.resolve_dtype_scheme("missing")
        with self.assertRaisesRegex(ValueError, "is not present"):
            LowPrecisionConfig(
                dtype_schemes={"native": LowPrecisionDtypeScheme()},
                default_dtype_scheme="missing",
            )

    @cpu_test
    def test_legacy_pairs_resolve_to_typed_policies(self):
        """The three existing format/scaling pairs preserve their behavior."""
        w8a8 = LowPrecisionConfig().resolve_dtype_scheme()
        w4a8 = LowPrecisionConfig(format="mxfp4_e2m1").resolve_dtype_scheme()
        hif8 = LowPrecisionConfig(format="hif8", scaling="current").resolve_dtype_scheme()

        self.assertEqual((w8a8.weight_format, w8a8.act_format), ("mxfp8", "mxfp8"))
        self.assertEqual((w4a8.weight_format, w4a8.act_format), ("mxfp4", "mxfp8"))
        self.assertEqual((hif8.weight_format, hif8.act_format), ("hif8", "hif8"))

    @cpu_test
    def test_plan_override_injects_selected_policy_read_only(self):
        """The plan binds one named scheme without exposing mutable context."""
        received = []

        @module_replacement
        def capture(*, module, module_fqn, context):
            del module_fqn
            received.append(context)
            return module

        target = Target(capture, target_path=f"{__name__}.capture")
        entry = PlanOverride(
            match="expert",
            when="low_precision",
            low_precision_dtype_scheme="fake",
            module_type=f"{__name__}._SourceModule",
            replace_module=target,
        )
        fake = LowPrecisionDtypeScheme(
            is_fake_quantize=True,
            weight_format="mxfp4",
            act_format="mxfp8",
        )
        config = LowPrecisionConfig(
            enabled=True,
            dtype_schemes={"fake": fake},
            default_dtype_scheme="fake",
        )

        rules = entries_to_module_replacements(
            [entry], low_precision_enabled=True, low_precision_config=config
        )
        source = _SourceModule()
        self.assertIs(
            rules[0].factory(module=source, module_fqn="expert", context={"tp": None}),
            source,
        )
        self.assertIs(received[0]["low_precision"], fake)
        with self.assertRaises(TypeError):
            received[0]["new_key"] = "not allowed"

    @cpu_test
    def test_plan_override_policy_requires_low_precision_condition(self):
        """A named policy cannot be attached to an unconditional replacement."""
        @module_replacement
        def identity(*, module, module_fqn, context):
            del module_fqn, context
            return module

        entry = PlanOverride(
            match="expert",
            low_precision_dtype_scheme="native",
            module_type=f"{__name__}._SourceModule",
            replace_module=Target(identity, target_path=f"{__name__}.identity"),
        )
        with self.assertRaisesRegex(ValueError, "is not a low-precision entry"):
            entries_to_module_replacements(
                [entry],
                low_precision_enabled=True,
                low_precision_config=LowPrecisionConfig(
                    dtype_schemes={"native": LowPrecisionDtypeScheme()}
                ),
            )


if __name__ == "__main__":
    unittest.main()
