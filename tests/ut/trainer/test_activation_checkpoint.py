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
"""Unit tests for model-agnostic activation-checkpoint block discovery."""
# pylint: disable=wrong-import-position

import os
import unittest
from unittest.mock import MagicMock, call, patch

from torch import Tensor, nn

from tests.ut.platform.mindspore._ensure_mindspore_platform import (
    restore_torch_platform_for_ut,
)

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
restore_torch_platform_for_ut()

from hyper_parallel.auto_models.components.activation_checkpoint.activation_checkpoint import (
    _apply_activation_checkpointing,
    _find_transformer_block_modules,
    _find_transformer_layer_container_infos,
    _wrap_layer_containers,
    apply_submodule_checkpointing,
)


_ACTIVATION_CHECKPOINT_MODULE = (
    "hyper_parallel.auto_models.components.activation_checkpoint.activation_checkpoint"
)


class _DiscoveryBlock(nn.Module):
    """Minimal block used to test model-agnostic layer discovery."""

    def __init__(self) -> None:
        """Create one minimal transformer block."""
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the fixture's linear layer."""
        return self.linear(inputs)


class _DiscoveryOwner(nn.Module):
    """HF-style owner whose repeated block path has an arbitrary name."""

    gradient_checkpointing = False

    def __init__(self) -> None:
        """Create an owner with a non-contiguous repeated block container."""
        super().__init__()
        self.decoder = nn.ModuleDict({"2": _DiscoveryBlock(), "7": _DiscoveryBlock()})

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply every block in registration order."""
        for block in self.decoder.values():
            inputs = block(inputs)
        return inputs


class _DiscoveryModel(nn.Module):
    """Model with multiple marked towers and no architecture-specific class name."""

    def __init__(self) -> None:
        """Create independent text and image towers."""
        super().__init__()
        self.text_tower = _DiscoveryOwner()
        self.image_tower = _DiscoveryOwner()

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply both towers and combine their outputs."""
        return self.text_tower(inputs) + self.image_tower(inputs)


class _UnmarkedDiscoveryModel(nn.Module):
    """Repeated layers without the HF discovery marker."""

    def __init__(self) -> None:
        """Create repeated blocks without a discovery marker."""
        super().__init__()
        self.layers = nn.ModuleList([_DiscoveryBlock(), _DiscoveryBlock()])

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply every unmarked block in order."""
        for block in self.layers:
            inputs = block(inputs)
        return inputs


class _CheckpointableSubmodules(nn.Module):
    """Minimal transformer block exposing all supported checkpoint targets."""

    def __init__(self) -> None:
        """Create submodules recognized by the checkpointing fallback."""
        super().__init__()
        self.mlp = nn.Linear(2, 2)
        self.self_attn = nn.Linear(2, 2)
        self.input_layernorm = nn.LayerNorm(2)
        self.post_attention_layernorm = nn.LayerNorm(2)


class TestTransformerBlockDiscovery(unittest.TestCase):
    """Tests for model-agnostic transformer block discovery."""

    def test_discovery_uses_marker_and_not_model_paths(self):
        """Marked arbitrary towers should all be discovered."""
        blocks, _ = _find_transformer_block_modules(_DiscoveryModel())

        self.assertEqual(
            [block.fqn for block in blocks],
            [
                "text_tower.decoder.2",
                "text_tower.decoder.7",
                "image_tower.decoder.2",
                "image_tower.decoder.7",
            ],
        )

    def test_container_info_preserves_registered_keys(self):
        """Container metadata should retain non-contiguous ModuleDict keys."""
        containers = _find_transformer_layer_container_infos(_DiscoveryModel())

        self.assertEqual(
            [container.path for container in containers],
            ["text_tower.decoder", "image_tower.decoder"],
        )
        self.assertEqual(
            [[block.child_name for block in container.blocks] for container in containers],
            [["2", "7"], ["2", "7"]],
        )

    def test_unmarked_layers_are_not_selected(self):
        """A conventional layers attribute is insufficient without the marker."""
        blocks, _ = _find_transformer_block_modules(_UnmarkedDiscoveryModel())

        self.assertEqual(blocks, [])

    def test_shared_block_is_selected_once(self):
        """Aliased block objects should not be wrapped more than once."""
        owner = _DiscoveryOwner()
        shared = _DiscoveryBlock()
        owner.decoder["2"] = shared
        owner.decoder["7"] = shared

        blocks, _ = _find_transformer_block_modules(owner)

        self.assertEqual([block.child_name for block in blocks], ["2"])

    def test_wrapping_uses_actual_container_child_names(self):
        """Wrapping should preserve arbitrary ModuleDict keys."""
        model = _DiscoveryModel()
        containers = _find_transformer_layer_container_infos(model)

        wrapped_count = _wrap_layer_containers(
            containers,
            nn.Sequential,
        )

        self.assertEqual(wrapped_count, 4)
        for tower in (model.text_tower, model.image_tower):
            self.assertTrue(all(isinstance(block, nn.Sequential) for block in tower.decoder.values()))

    def test_missing_marker_has_clear_activation_checkpoint_error(self):
        """Activation checkpointing should fail instead of guessing a container."""
        with self.assertRaisesRegex(ValueError, "gradient_checkpointing"):
            _apply_activation_checkpointing(_UnmarkedDiscoveryModel(), "selective")


class TestActivationCheckpointSwapInputs(unittest.TestCase):
    """Tests for activation checkpoint input-swapping configuration."""

    def setUp(self) -> None:
        """Keep swap-input tests independent of optional Transformers imports."""
        hf_checkpointing_patch = patch(
            f"{_ACTIVATION_CHECKPOINT_MODULE}._should_use_hf_native_gradient_checkpointing",
            return_value=False,
        )
        hf_checkpointing_patch.start()
        self.addCleanup(hf_checkpointing_patch.stop)

    @staticmethod
    def _wrapped_blocks(model):
        """Return all checkpoint-wrapped blocks from the discovery fixture."""
        return [
            block
            for tower in (model.text_tower, model.image_tower)
            for block in tower.decoder.values()
        ]

    def test_eager_checkpoint_wrappers_receive_swap_inputs(self):
        """Eager full and selective wrappers should receive the configured value."""
        for mode in ("full", "selective"):
            for swap_inputs in (False, True):
                with self.subTest(mode=mode, swap_inputs=swap_inputs):
                    model = _DiscoveryModel()
                    with (
                        patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_profiler_ops_sac_ignored"),
                        patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_fsdp_ops_sac_ignored"),
                    ):
                        _apply_activation_checkpointing(
                            model,
                            mode,
                            enable_compile=False,
                            swap_inputs=swap_inputs,
                        )

                    for block in self._wrapped_blocks(model):
                        self.assertIs(block.checkpoint_kwargs["swap_inputs"], swap_inputs)

    def test_compile_checkpoint_wrappers_omit_swap_inputs(self):
        """Compile wrappers should omit swap_inputs and report that it is disabled."""
        expected_warning = (
            "activation_checkpoint.swap_inputs is not supported with torch.compile; "
            "input swapping will be disabled."
        )
        for mode in ("full", "selective"):
            with self.subTest(mode=mode):
                model = _DiscoveryModel()
                with (
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_profiler_ops_sac_ignored"),
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_fsdp_ops_sac_ignored"),
                    self.assertLogs(_ACTIVATION_CHECKPOINT_MODULE, level="WARNING") as log_context,
                ):
                    _apply_activation_checkpointing(
                        model,
                        mode,
                        enable_compile=True,
                        swap_inputs=True,
                    )

                self.assertIn(expected_warning, "\n".join(log_context.output))
                for block in self._wrapped_blocks(model):
                    self.assertNotIn("swap_inputs", block.checkpoint_kwargs)

    def test_hf_native_checkpointing_warns_when_swap_inputs_enabled(self):
        """HF-native checkpointing should warn that input swapping is disabled."""
        model = _DiscoveryModel()
        model.gradient_checkpointing_enable = MagicMock()
        expected_warning = (
            "activation_checkpoint.swap_inputs is not supported by Hugging Face native "
            "gradient checkpointing for now; input swapping will be disabled."
        )

        with (
            patch(
                f"{_ACTIVATION_CHECKPOINT_MODULE}._should_use_hf_native_gradient_checkpointing",
                return_value=True,
            ),
            self.assertLogs(_ACTIVATION_CHECKPOINT_MODULE, level="WARNING") as log_context,
        ):
            result = _apply_activation_checkpointing(
                model,
                "full",
                swap_inputs=True,
            )

        self.assertIs(result, model)
        self.assertIn(expected_warning, "\n".join(log_context.output))
        model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

    def test_rejects_non_boolean_swap_inputs(self):
        """The component boundary should reject ambiguous swap_inputs values."""
        with self.assertRaisesRegex(
            ValueError,
            "activation_checkpoint.swap_inputs must be bool",
        ):
            _apply_activation_checkpointing(
                _DiscoveryModel(),
                "full",
                swap_inputs=1,
            )

    def test_submodule_checkpointing_handles_swap_inputs_by_compile_mode(self):
        """Submodule wrappers should pass swap_inputs only during eager execution."""
        for enable_compile in (False, True):
            with self.subTest(enable_compile=enable_compile):
                block = _CheckpointableSubmodules()

                wrapped_count = apply_submodule_checkpointing(
                    [block],
                    has_kv_sharing=False,
                    enable_compile=enable_compile,
                    swap_inputs=True,
                )

                self.assertEqual(wrapped_count, 4)
                for attr_name in (
                    "mlp",
                    "self_attn",
                    "input_layernorm",
                    "post_attention_layernorm",
                ):
                    checkpoint_kwargs = getattr(block, attr_name).checkpoint_kwargs
                    if enable_compile:
                        self.assertNotIn("swap_inputs", checkpoint_kwargs)
                    else:
                        self.assertIs(checkpoint_kwargs["swap_inputs"], True)

    def test_swap_inputs_registers_prefetch_within_each_container(self):
        """Eager input swapping should connect adjacent wrapped transformer blocks."""
        for mode in ("full", "selective"):
            with self.subTest(mode=mode):
                model = _DiscoveryModel()
                swap_manager = MagicMock()
                with (
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.SwapManager", return_value=swap_manager),
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_profiler_ops_sac_ignored"),
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_fsdp_ops_sac_ignored"),
                ):
                    _apply_activation_checkpointing(
                        model,
                        mode,
                        swap_inputs=True,
                    )

                swap_manager.set_forward_prefetch_layer.assert_has_calls(
                    [
                        call(model.text_tower.decoder["2"], model.text_tower.decoder["7"]),
                        call(model.image_tower.decoder["2"], model.image_tower.decoder["7"]),
                    ]
                )
                self.assertEqual(swap_manager.set_forward_prefetch_layer.call_count, 2)

    def test_submodule_checkpointing_registers_matching_prefetch_chains(self):
        """KV-shared fallback should connect matching non-attention wrappers."""
        model = _DiscoveryOwner()
        model.decoder["2"] = _CheckpointableSubmodules()
        model.decoder["7"] = _CheckpointableSubmodules()
        swap_manager = MagicMock()

        with (
            patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.SwapManager", return_value=swap_manager),
            patch(
                f"{_ACTIVATION_CHECKPOINT_MODULE}._detect_kv_sharing_and_maybe_disable_cache",
                return_value=True,
            ),
        ):
            _apply_activation_checkpointing(
                model,
                "full",
                swap_inputs=True,
            )

        first_block = model.decoder["2"]
        second_block = model.decoder["7"]
        expected_calls = [
            call(getattr(first_block, attr_name), getattr(second_block, attr_name))
            for attr_name in (
                "mlp",
                "input_layernorm",
                "post_attention_layernorm",
            )
        ]
        swap_manager.set_forward_prefetch_layer.assert_has_calls(expected_calls)
        self.assertEqual(swap_manager.set_forward_prefetch_layer.call_count, 3)
        self.assertIsInstance(first_block.self_attn, nn.Linear)
        self.assertIsInstance(second_block.self_attn, nn.Linear)

    def test_prefetch_registration_requires_effective_swap_inputs(self):
        """Disabled or compile-only input swapping should not register prefetch hooks."""
        for enable_compile, swap_inputs in ((False, False), (True, True)):
            with self.subTest(enable_compile=enable_compile, swap_inputs=swap_inputs):
                model = _DiscoveryModel()
                swap_manager = MagicMock()
                with (
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.SwapManager", return_value=swap_manager),
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.logger.warning"),
                ):
                    _apply_activation_checkpointing(
                        model,
                        "full",
                        enable_compile=enable_compile,
                        swap_inputs=swap_inputs,
                    )

                swap_manager.set_forward_prefetch_layer.assert_not_called()
