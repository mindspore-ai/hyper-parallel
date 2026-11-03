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
"""Unit tests for the replacement activation-checkpoint YAML schema."""

import unittest

from hyper_parallel.trainer.config.parallelism import (
    ActivationCheckpointConfig,
    FullCheckpointConfig,
    FullExcludeRegionConfig,
    SelectiveCheckpointConfig,
)
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, resolve_component


class ActivationCheckpointConfigTest(unittest.TestCase):
    """Validate normalization, mode requirements, and config-level conflicts."""

    def test_resolver_normalizes_ranges_and_swap_options(self) -> None:
        """YAML ranges should become ordered tuples and preserve resource domains."""
        config = resolve_component(
            {
                "mode": "hybrid",
                "full": {
                    "recompute_layer": ["0-1"],
                    "exclude_region": {
                        "save": {"module_a": [0]},
                        "swap": {
                            "group_swap": False,
                            "cpu_pool": 512,
                            "module_b": ["0-1"],
                        },
                    },
                },
                "selective": {
                    "recompute": {"module_a": [2]},
                    "swap": {
                        "group_swap": True,
                        "cpu_pool": 256,
                        "module_b": [3],
                    },
                },
            },
            annotation=ActivationCheckpointConfig,
            path="$.activation_checkpoint",
        )

        self.assertEqual(config.full.recompute_layers, (0, 1))
        self.assertEqual(config.full.save_regions[0].layers, (0,))
        self.assertEqual(config.full.swap_regions[0].layers, (0, 1))
        self.assertFalse(config.full.group_swap)
        self.assertEqual(config.full.cpu_pool_mb, 512)
        self.assertTrue(config.selective.group_swap)
        self.assertEqual(config.selective.cpu_pool_mb, 256)

    def test_off_requires_no_subconfiguration(self) -> None:
        """Off mode should be a valid standalone configuration."""
        config = ActivationCheckpointConfig(mode="off")
        self.assertEqual(config.to_dict(), {"mode": "off"})

    def test_active_modes_require_their_sections(self) -> None:
        """Each active mode should require the corresponding config block."""
        with self.assertRaisesRegex(ValueError, "requires activation_checkpoint.full"):
            ActivationCheckpointConfig(mode="full")
        with self.assertRaisesRegex(ValueError, "requires activation_checkpoint.selective"):
            ActivationCheckpointConfig(mode="selective")
        with self.assertRaisesRegex(ValueError, "requires activation_checkpoint.selective"):
            ActivationCheckpointConfig(
                mode="hybrid",
                full=FullCheckpointConfig(recompute_layer=[0]),
            )

    def test_layer_set_rejects_duplicates_and_invalid_ranges(self) -> None:
        """Duplicate, reversed, negative, boolean, and malformed layers should fail."""
        invalid_values = (
            ([0, "0-1"], "duplicate layer index 0"),
            (["3-1"], "reversed layer range"),
            ([-1], "negative layer index"),
            ([True], "boolean layer values"),
            ([""], "invalid layer range"),
        )
        for value, message in invalid_values:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                FullCheckpointConfig(recompute_layer=value)

    def test_pool_and_group_swap_types_are_strict(self) -> None:
        """Pool capacities and group flags should reject ambiguous YAML values."""
        for value in (0, -1, True, 1.5, "512"):
            with self.subTest(cpu_pool=value), self.assertRaisesRegex(ValueError, "positive integer"):
                SelectiveCheckpointConfig(swap={"cpu_pool": value, "module_a": [0]})
        with self.assertRaisesRegex(ValueError, "group_swap.*bool"):
            SelectiveCheckpointConfig(swap={"group_swap": 1, "module_a": [0]})

    def test_full_and_selective_conflicts_are_rejected(self) -> None:
        """Overlapping, nested, and out-of-full targets should fail early."""
        with self.assertRaisesRegex(ValueError, "outside.*recompute_layer"):
            ActivationCheckpointConfig(
                mode="full",
                full=FullCheckpointConfig(
                    recompute_layer=[0],
                    exclude_region=FullExcludeRegionConfig(save={"module_a": [1]}),
                ),
            )
        with self.assertRaisesRegex(ValueError, "save and.*swap conflict"):
            ActivationCheckpointConfig(
                mode="full",
                full=FullCheckpointConfig(
                    recompute_layer=[0],
                    exclude_region=FullExcludeRegionConfig(
                        save={"module_a": [0]},
                        swap={"module_a": [0]},
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "nested targets"):
            ActivationCheckpointConfig(
                mode="selective",
                selective=SelectiveCheckpointConfig(
                    recompute={"module_a": [0]},
                    swap={"module_a.child": [0]},
                ),
            )

    def test_selective_effectiveness_and_hybrid_disjointness(self) -> None:
        """Active selective targets must exist and hybrid layers must be disjoint."""
        with self.assertRaisesRegex(ValueError, "at least one"):
            ActivationCheckpointConfig(
                mode="selective",
                selective=SelectiveCheckpointConfig(swap={"cpu_pool": 512}),
            )
        with self.assertRaisesRegex(ValueError, r"overlap=\[1\]"):
            ActivationCheckpointConfig(
                mode="hybrid",
                full=FullCheckpointConfig(recompute_layer=["0-1"]),
                selective=SelectiveCheckpointConfig(recompute={"module_a": [1]}),
            )

    def test_legacy_fields_and_missing_mode_are_rejected(self) -> None:
        """The resolver should expose no compatibility layer for the old schema."""
        with self.assertRaisesRegex(ConfigResolutionError, "missing required.*mode"):
            resolve_component(
                {"full": {"recompute_layer": [0]}},
                annotation=ActivationCheckpointConfig,
                path="$.activation_checkpoint",
            )
        for legacy_field, legacy_value in (
            ("selection", {"source": "model_adapter_safe_regions", "layer_count": 1}),
            ("swap_inputs", True),
        ):
            with self.subTest(legacy_field=legacy_field), self.assertRaisesRegex(
                ConfigResolutionError,
                "unknown configuration fields",
            ):
                resolve_component(
                    {"mode": "full", legacy_field: legacy_value},
                    annotation=ActivationCheckpointConfig,
                    path="$.activation_checkpoint",
                )


if __name__ == "__main__":
    unittest.main()
