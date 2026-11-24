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
from typing import Any

from torch import nn

from hyper_parallel.components.quantization.config import (
    LowPrecisionConfig,
    LowPrecisionDtypeScheme,
)
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.trainer.config.parallelism import (
    PlanOverride,
    ReplaceAction,
    entries_to_module_replacements,
)
from hyper_parallel.trainer.config.target import Target


class _SourceModule(nn.Module):
    """Importable source type used by the plan-override test."""

    def forward(self, value: Any) -> Any:
        """Return the input unchanged for replacement matching tests."""
        return value


class TestLowPrecisionDtypeScheme(unittest.TestCase):
    """Validation contracts that must fail before an NPU operator is called."""

    def test_supported_native_and_fake_schemes(self):
        """W8A8, native W4A8 and fake W4A8 policies retain typed fields."""
        w8a8 = LowPrecisionDtypeScheme()
        native = LowPrecisionDtypeScheme(
            weight_format="mxfp4", act_format="mxfp8", block_size=32
        )
        fake = LowPrecisionDtypeScheme(
            is_fake_quantize=True,
            weight_format="mxfp4",
            act_format="mxfp8",
            block_size=32,
        )

        self.assertEqual((w8a8.weight_format, w8a8.act_format), ("mxfp8", "mxfp8"))
        self.assertEqual(w8a8.block_size, 32)
        self.assertEqual(native.block_size, 32)
        self.assertTrue(fake.is_fake_quantize)

        fake_default = LowPrecisionDtypeScheme(
            is_fake_quantize=True,
            weight_format="mxfp4",
            act_format="mxfp8",
        )
        self.assertEqual(fake_default.block_size, 32)

    def test_hifloat_scheme_does_not_apply_mx_block_size_restrictions(self):
        """HiFloat keeps the integer field without MX-only validation."""
        scheme = LowPrecisionDtypeScheme(
            weight_format="hif8",
            act_format="hif8",
            block_size=64,
        )

        self.assertEqual(scheme.block_size, 64)

    def test_invalid_family_block_and_flag_fail_fast(self):
        """Cross-family formats and malformed MX policies are rejected."""
        with self.assertRaisesRegex(ValueError, "same format family"):
            LowPrecisionDtypeScheme(weight_format="mxfp4", act_format="hif8")
        with self.assertRaisesRegex(ValueError, "block_size must be 32"):
            LowPrecisionDtypeScheme(
                weight_format="mxfp4", act_format="mxfp8", block_size=64
            )
        with self.assertRaisesRegex(ValueError, "block_size must be 32"):
            LowPrecisionDtypeScheme(block_size=128)
        with self.assertRaisesRegex(ValueError, "is_fake_quantize must be a bool"):
            LowPrecisionDtypeScheme(is_fake_quantize=1)


class TestLowPrecisionConfig(unittest.TestCase):
    """Named policy selection and legacy compatibility contracts."""

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
                    "block_size": 32,
                },
            },
            default_dtype_scheme="native",
        )

        self.assertIsInstance(config.dtype_schemes["native"], LowPrecisionDtypeScheme)
        self.assertIs(config.resolve_dtype_scheme(), config.dtype_schemes["native"])
        self.assertIs(config.resolve_dtype_scheme("fake"), config.dtype_schemes["fake"])
        self.assertEqual(config.to_dict()["dtype_schemes"]["fake"]["block_size"], 32)

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

    def test_legacy_pairs_resolve_to_typed_policies(self):
        """The three existing format/scaling pairs preserve their behavior."""
        w8a8 = LowPrecisionConfig().resolve_dtype_scheme()
        w4a8 = LowPrecisionConfig(format="mxfp4_e2m1").resolve_dtype_scheme()
        hif8 = LowPrecisionConfig(format="hif8", scaling="current").resolve_dtype_scheme()

        self.assertEqual((w8a8.weight_format, w8a8.act_format), ("mxfp8", "mxfp8"))
        self.assertEqual((w4a8.weight_format, w4a8.act_format), ("mxfp4", "mxfp8"))
        self.assertEqual((hif8.weight_format, hif8.act_format), ("hif8", "hif8"))
        self.assertEqual((w8a8.block_size, w4a8.block_size, hif8.block_size), (32, 32, 32))

    def test_plan_override_injects_selected_policy_read_only(self):
        """The plan binds one named scheme without exposing mutable context."""
        received = []

        @module_replacement
        def capture(
            *, module: nn.Module, module_fqn: str, context: dict[str, Any]
        ) -> nn.Module:
            """Record replacement context and return the original module."""
            del module_fqn
            received.append(context)
            return module

        target = Target(capture, target_path=f"{__name__}.capture")
        entry = PlanOverride(
            match="expert",
            when="low_precision",
            low_precision_dtype_scheme="fake",
            replace=ReplaceAction(
                kind="module", source_type=f"{__name__}._SourceModule", implementation=target,
            ),
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

    def test_low_precision_replacement_is_skipped_when_disabled(self):
        """Absent or disabled low precision leaves the source module unreplaced."""
        @module_replacement
        def identity(
            *, module: nn.Module, module_fqn: str, context: dict[str, Any]
        ) -> nn.Module:
            """Keep the source module unchanged if a replacement is invoked."""
            del module_fqn, context
            return module

        entry = PlanOverride(
            match="expert",
            when="low_precision",
            low_precision_dtype_scheme="native",
            replace=ReplaceAction(
                kind="module", source_type=f"{__name__}._SourceModule",
                implementation=Target(identity, target_path=f"{__name__}.identity"),
            ),
        )
        native = LowPrecisionDtypeScheme()
        configs = (
            None,
            LowPrecisionConfig(
                enabled=False,
                dtype_schemes={"native": native},
                default_dtype_scheme="native",
            ),
        )
        for config in configs:
            with self.subTest(configured=config is not None):
                rules = entries_to_module_replacements(
                    [entry],
                    low_precision_enabled=False,
                    low_precision_config=config,
                )
                self.assertEqual(rules, ())

    def test_plan_override_policy_requires_low_precision_condition(self):
        """A named policy cannot be attached to an unconditional replacement."""
        @module_replacement
        def identity(
            *, module: nn.Module, module_fqn: str, context: dict[str, Any]
        ) -> nn.Module:
            """Return the original module without applying a policy."""
            del module_fqn, context
            return module

        entry = PlanOverride(
            match="expert",
            low_precision_dtype_scheme="native",
            replace=ReplaceAction(
                kind="module", source_type=f"{__name__}._SourceModule",
                implementation=Target(identity, target_path=f"{__name__}.identity"),
            ),
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
