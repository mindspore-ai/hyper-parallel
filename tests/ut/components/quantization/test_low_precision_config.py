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
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.quantization.config import (
    CSA2IndexerQKFormat,
    CSA2MainKVFormat,
    CSA2SWAKVFormat,
    LowPrecisionConfig,
    LowPrecisionDtypeScheme,
)
from hyper_parallel.models.module_build import build_module_context, prepare_model_components
from hyper_parallel.models.replacement import _apply_module_replacement_actions, module_replacement
from hyper_parallel.trainer.config.parallelism import (
    PlanOverride,
    ReplaceAction,
    entries_to_module_replacements,
    normalize_distributed_setup_overrides,
)
from hyper_parallel.trainer.config.resolver import normalize_value
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
        with self.assertRaisesRegex(ValueError, "Unknown W/A low-precision dtype scheme"):
            config.resolve_dtype_scheme("missing")
        with self.assertRaisesRegex(ValueError, "is not a W/A scheme"):
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

    def test_csa2_formats_are_independent_and_round_trip(self):
        """Each direct activation format enables only its named CSA2 path."""
        cases = (
            ("indexer_qk", {"act_format": "mxfp4"}, CSA2IndexerQKFormat),
            ("main_kv", {"act_format": "fp4_e2m1", "scale_format": "e4m3"}, CSA2MainKVFormat),
            ("swa_kv", {"act_format": "fp8_e4m3", "scale_format": "e8m0"}, CSA2SWAKVFormat),
        )
        for name, value, format_type in cases:
            with self.subTest(name=name):
                config = LowPrecisionConfig(enabled=True, dtype_schemes={name: value})
                self.assertIsInstance(getattr(config, name), format_type)
                self.assertEqual(config.active_direct_formats(), frozenset({name}))
                self.assertEqual(config.to_dict()["dtype_schemes"][name]["act_format"], value["act_format"])
                self.assertEqual(
                    LowPrecisionConfig(**config.to_dict()).active_direct_formats(), frozenset({name})
                )

    def test_csa2_format_mapping_passes_typed_yaml_resolution(self):
        """The Trainer parser accepts direct format mappings without scheme names."""
        raw = {
            "enabled": True,
            "dtype_schemes": {
                "indexer_qk": {"act_format": "mxfp4"},
                "main_kv": {"act_format": "fp4_e2m1", "scale_format": "e4m3", "group": 16},
                "swa_kv": {"act_format": "fp8_e4m3", "scale_format": "e8m0", "group": 32},
            },
        }
        resolved = normalize_value(raw, LowPrecisionConfig, path="training.low_precision")
        self.assertIsInstance(resolved, LowPrecisionConfig)
        self.assertEqual(
            resolved.active_direct_formats(), frozenset({"indexer_qk", "main_kv", "swa_kv"})
        )

    def test_csa2_rejects_unimplemented_formats(self):
        """The public config rejects unknown scale names and FP8 families."""
        invalid = (
            ("indexer_qk", {"act_format": "mxfp8"}),
            ("main_kv", {"act_format": "fp4_e2m1", "scale_format": "bf16"}),
            ("swa_kv", {"act_format": "fp8_e5m2"}),
            ("main_kv", {"act_format": "fp4_e2m1", "scale_format": "e8m7", "group": 32}),
            ("swa_kv", {"act_format": "fp8_e4m3", "scale_format": "e8m7", "group": 16}),
        )
        for name, value in invalid:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    LowPrecisionConfig(enabled=True, dtype_schemes={name: value})

    def test_csa2_bf16_scale_uses_e8m7_name(self):
        """Both cache formats accept BF16 group scales under one bit-layout name."""
        config = LowPrecisionConfig(enabled=True, dtype_schemes={
            "main_kv": {"act_format": "fp4_e2m1", "scale_format": "e8m7", "group": 16},
            "swa_kv": {"act_format": "fp8_e4m3", "scale_format": "e8m7", "group": 32},
        })
        self.assertEqual(config.main_kv.scale_format, "e8m7")
        self.assertEqual(config.swa_kv.scale_format, "e8m7")
        self.assertEqual(LowPrecisionConfig(**config.to_dict()), config)

    def test_csa2_malformed_mapping_names_the_config_path(self):
        """Missing or misspelled fields report the CSA2 subconfiguration."""
        with self.assertRaisesRegex(ValueError, r"Invalid LowPrecisionConfig\.dtype_schemes\.main_kv.*scale_format"):
            LowPrecisionConfig(enabled=True, dtype_schemes={"main_kv": {"act_format": "fp4_e2m1"}})
        with self.assertRaisesRegex(ValueError, r"Invalid LowPrecisionConfig\.dtype_schemes\.swa_kv.*group_size"):
            LowPrecisionConfig(enabled=True, dtype_schemes={
                "swa_kv": {"act_format": "fp8_e4m3", "group_size": 32},
            })

    def test_csa2_config_reaches_existing_attention_replacement(self):
        """An ordinary replacement consumes direct formats without a scheme selector."""
        received = []

        @module_replacement
        def capture(*, module: nn.Module, module_fqn: str, context: dict[str, Any]) -> nn.Module:
            """Record the factory context and keep the source module."""
            del module_fqn
            received.append(context)
            return module

        capture.low_precision_formats = frozenset({"indexer_qk", "main_kv", "swa_kv"})
        entry = PlanOverride(
            match="attention",
            replace=ReplaceAction(
                kind="module", source_type=f"{__name__}._SourceModule",
                implementation=Target(capture, target_path=f"{__name__}.capture"),
            ),
        )
        config = LowPrecisionConfig(enabled=True, dtype_schemes={
            "main_kv": {"act_format": "fp4_e2m1", "scale_format": "e4m3"},
        })
        source = _SourceModule()
        rules = entries_to_module_replacements([entry], low_precision_enabled=True, low_precision_config=config)
        context = build_module_context(SimpleNamespace(low_precision_config=config))
        self.assertIs(rules[0].factory(module=source, module_fqn="attention", context=context), source)
        self.assertIs(received[-1]["low_precision"], config)
        self.assertEqual(config.active_direct_formats(), frozenset({"main_kv"}))

        disabled = entries_to_module_replacements([entry], low_precision_enabled=False, low_precision_config=config)
        disabled_context = build_module_context(SimpleNamespace(low_precision_config=LowPrecisionConfig(enabled=False)))
        disabled[0].factory(module=source, module_fqn="attention", context=disabled_context)
        self.assertIsNone(received[-1]["low_precision"])

    def test_csa2_single_card_keeps_model_parallel_axis_guard(self):
        """Direct cache formats run on one card and retain the existing parallel-axis guard."""
        @module_replacement
        def capture(*, module: nn.Module, module_fqn: str, context: dict[str, Any]) -> nn.Module:
            """Keep the attention fixture and inspect the effective policy."""
            del module_fqn
            self.assertIsInstance(context["low_precision"], LowPrecisionConfig)
            return module

        capture.low_precision_formats = frozenset({"main_kv"})
        entry = PlanOverride(
            match="attention",
            replace=ReplaceAction(
                kind="module", source_type=f"{__name__}._SourceModule",
                implementation=Target(capture, target_path=f"{__name__}.capture"),
            ),
        )
        cache_only = LowPrecisionConfig(enabled=True, dtype_schemes={
            "main_kv": {"act_format": "fp4_e2m1", "scale_format": "e4m3"},
        })
        rules = entries_to_module_replacements(
            [entry], low_precision_enabled=True, low_precision_config=cache_only
        )

        def apply(policy: LowPrecisionConfig, axes: dict[str, bool]) -> None:
            model = nn.Module()
            model.attention = _SourceModule()
            _apply_module_replacement_actions(
                model, rules, weights_mapping=[],
                context={"low_precision": policy, **axes},
                capture_checkpoint_metadata=False,
            )

        apply(cache_only, {})
        for axis in ("tp", "cp", "pp"):
            with self.subTest(axis=axis):
                with self.assertRaises(NotImplementedError):
                    apply(cache_only, {axis: True})
        mixed = LowPrecisionConfig(enabled=True, dtype_schemes={
            "main_kv": {"act_format": "fp4_e2m1", "scale_format": "e4m3"},
            "w4a8": LowPrecisionDtypeScheme(weight_format="mxfp4", act_format="mxfp8"),
        })
        with self.assertRaises(NotImplementedError):
            apply(mixed, {"cp": True})

    def test_unused_named_formats_do_not_require_replacement(self):
        """An unused format catalog may contain CSA2 and W/A policies."""
        plan = SimpleNamespace(plan_overrides=[])
        previous = torch.utils.deterministic.fill_uninitialized_memory
        for schemes in (
            {"swa_kv": {"act_format": "fp8_e4m3"}},
            {"swa_kv": {"act_format": "fp8_e4m3"}, "w4a8": LowPrecisionDtypeScheme()},
        ):
            with self.subTest(formats=tuple(schemes)):
                setup = SimpleNamespace(low_precision_config=LowPrecisionConfig(
                    enabled=True, dtype_schemes=schemes,
                ))
                normalize_distributed_setup_overrides(setup, plan)
                self.assertEqual(setup.module_replacement_requests, ())
                self.assertEqual(setup.module_replacements, ())
                prepare_model_components(nn.Module(), build_module_context(setup))
        self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, previous)

    def test_csa2_direct_format_uses_latest_replacement_request_on_one_card(self):
        """The normalized CSA2 request reaches the replacement on the single-card build path."""
        received = []

        @module_replacement
        def capture(*, module: nn.Module, module_fqn: str, context: dict[str, Any]) -> nn.Module:
            """Observe the policy delivered by the compiled request."""
            received.append((module_fqn, context["low_precision"]))
            return module

        entry = PlanOverride(
            match="attention",
            replace=ReplaceAction(
                kind="module", source_type=f"{__name__}._SourceModule",
                implementation=Target(capture, target_path=f"{__name__}.capture"),
            ),
        )
        policy = LowPrecisionConfig(enabled=True, dtype_schemes={
            "main_kv": {"act_format": "fp4_e2m1", "scale_format": "e8m7"},
        })
        mesh = SimpleNamespace(tp_size=1, cp_size=1, ep_size=1, pp_size=1)
        setup = SimpleNamespace(low_precision_config=policy, mesh_context=mesh)
        previous = torch.utils.deterministic.fill_uninitialized_memory
        normalize_distributed_setup_overrides(setup, SimpleNamespace(plan_overrides=[entry]))
        self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, previous)
        self.assertFalse(setup._hp_legacy_low_precision_targets)
        self.assertEqual(len(setup.module_replacement_requests), 1)
        self.assertIs(setup.module_replacement_requests[0].target, capture)

        model = nn.Module()
        model.attention = _SourceModule()
        prepare_model_components(model, build_module_context(setup))
        self.assertEqual(received, [("attention", policy)])

    def test_csa2_plan_does_not_change_fp8_prefill_guard(self):
        """Neither Main scale format changes the process-wide fill policy."""
        @module_replacement
        def capture(*, module: nn.Module, module_fqn: str, context: dict[str, Any]) -> nn.Module:
            """Declare a Main KV consumer without constructing a model."""
            del module_fqn, context
            return module

        capture.low_precision_formats = frozenset({"main_kv"})
        entry = PlanOverride(
            match="attention",
            replace=ReplaceAction(
                kind="module", source_type=f"{__name__}._SourceModule",
                implementation=Target(capture, target_path=f"{__name__}.capture"),
            ),
        )
        previous = torch.utils.deterministic.fill_uninitialized_memory
        for scale in ("e8m7", "e4m3"):
            with self.subTest(scale_format=scale):
                setup = SimpleNamespace(low_precision_config=LowPrecisionConfig(
                    enabled=True,
                    dtype_schemes={"main_kv": {"act_format": "fp4_e2m1", "scale_format": scale}},
                ))
                normalize_distributed_setup_overrides(setup, SimpleNamespace(plan_overrides=[entry]))
                self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, previous)

    def test_csa2_consumer_rejects_weight_activation_scheme_on_same_entry(self):
        """One attention replacement cannot receive two incompatible context types."""
        @module_replacement
        def capture(*, module: nn.Module, module_fqn: str, context: dict[str, Any]) -> nn.Module:
            """Return the module without constructing an attention fixture."""
            del module_fqn, context
            return module

        capture.low_precision_formats = frozenset({"main_kv"})
        entry = PlanOverride(
            match="attention", when="low_precision", low_precision_dtype_scheme="w4a8",
            replace=ReplaceAction(
                kind="module", source_type=f"{__name__}._SourceModule",
                implementation=Target(capture, target_path=f"{__name__}.capture"),
            ),
        )
        config = LowPrecisionConfig(
            enabled=True,
            dtype_schemes={
                "main_kv": {"act_format": "fp4_e2m1", "scale_format": "e4m3"},
                "w4a8": LowPrecisionDtypeScheme(weight_format="mxfp4", act_format="mxfp8"),
            },
        )

        with self.assertRaisesRegex(ValueError, "cannot select a W/A dtype scheme and CSA2 formats"):
            entries_to_module_replacements([entry], low_precision_enabled=True, low_precision_config=config)


if __name__ == "__main__":
    unittest.main()
