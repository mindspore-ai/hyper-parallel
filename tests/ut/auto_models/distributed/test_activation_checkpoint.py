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
"""Unit tests for the replacement activation-checkpoint runtime."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import Tensor, nn

from hyper_parallel.core.activation_memory import (
    CheckpointExcludeWrapper,
    CheckpointWrapper,
    SwapManager,
    SwapWrapper,
)
from hyper_parallel.core.activation_memory import checkpoint_wrapper as real_checkpoint_wrapper
from hyper_parallel.distributed.activation_checkpoint import (
    _apply_activation_checkpointing,
    _find_transformer_block_modules,
    _find_transformer_layer_container_infos,
)
from hyper_parallel.trainer.config.parallelism import (
    ActivationCheckpointConfig,
    FullCheckpointConfig,
    FullExcludeRegionConfig,
    SelectiveCheckpointConfig,
)

_ACTIVATION_CHECKPOINT_MODULE = "hyper_parallel.distributed.activation_checkpoint"


class _Block(nn.Module):
    """Minimal transformer block with ordered child modules."""

    def __init__(self) -> None:
        """Create independently wrappable submodules."""
        super().__init__()
        self.module_a = nn.Linear(2, 2)
        self.module_b = nn.Linear(2, 2)
        self.value = 1

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply children in registration order."""
        return self.module_b(self.module_a(inputs))


class _Owner(nn.Module):
    """HF-style owner with an arbitrary repeated-block container name."""

    gradient_checkpointing = False

    def __init__(self, keys: tuple[str, ...] = ("2", "7", "9", "11")) -> None:
        """Create non-contiguous registered block keys."""
        super().__init__()
        self.decoder = nn.ModuleDict({key: _Block() for key in keys})

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply every block in registration order."""
        for block in self.decoder.values():
            inputs = block(inputs)
        return inputs


class _StemHeadOwner(nn.Module):
    """HF-style owner mixing decoder layers with ordinary sequential branches."""

    gradient_checkpointing = False

    def __init__(self) -> None:
        """Register a stem before decoder layers and a head after them."""
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 2))


class _Model(nn.Module):
    """Model exposing one marked transformer owner."""

    def __init__(self) -> None:
        """Create the marked owner."""
        super().__init__()
        self.tower = _Owner()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 2)
        self.lm_head = nn.Linear(2, 8, bias=False)
        self.gradient_checkpointing_disable_calls = 0

    def gradient_checkpointing_disable(self) -> None:
        """Disable the fixture's HuggingFace-style native checkpointing."""
        self.gradient_checkpointing_disable_calls += 1
        self.tower.gradient_checkpointing = False

    def forward(self, inputs: Tensor) -> Tensor:
        """Delegate to the marked owner."""
        return self.tower(inputs)


class _UnmarkedModel(nn.Module):
    """Repeated layers without the structural discovery marker."""

    def __init__(self) -> None:
        """Create an unmarked repeated container."""
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.embed_tokens = nn.Embedding(8, 2)
        self.lm_head = nn.Linear(2, 8, bias=False)
        self.gradient_checkpointing_disable_calls = 0

    def gradient_checkpointing_disable(self) -> None:
        """Record an unexpected native-checkpoint disable call."""
        self.gradient_checkpointing_disable_calls += 1

    def forward(self, input_ids: Tensor) -> Tensor:
        """Run model-level modules without a discoverable transformer owner."""
        return self.lm_head(self.embed_tokens(input_ids))


class ActivationCheckpointRuntimeTest(unittest.TestCase):
    """Validate model discovery, wrapper installation, resources, and rollback."""

    def setUp(self) -> None:
        """Reset the process-wide swap manager before each test."""
        SwapManager._instance = None  # pylint: disable=protected-access

    def tearDown(self) -> None:
        """Reset the process-wide swap manager after each test."""
        SwapManager._instance = None  # pylint: disable=protected-access

    def test_discovery_uses_marker_and_registration_order(self) -> None:
        """Arbitrary container names and keys should retain stable numbering."""
        model = _Model()
        blocks, _ = _find_transformer_block_modules(model)
        containers = _find_transformer_layer_container_infos(model)

        self.assertEqual(
            [block.fqn for block in blocks],
            [
                "tower.decoder.2",
                "tower.decoder.7",
                "tower.decoder.9",
                "tower.decoder.11",
            ],
        )
        self.assertEqual([container.path for container in containers], ["tower.decoder"])

    def test_discovery_skips_heterogeneous_sequential_stem_and_head(self) -> None:
        """Ordinary sequential branches should not enter global layer numbering."""
        model = _StemHeadOwner()

        blocks, _ = _find_transformer_block_modules(model)
        containers = _find_transformer_layer_container_infos(model)

        self.assertEqual([block.fqn for block in blocks], ["layers.0", "layers.1"])
        self.assertEqual([container.path for container in containers], ["layers"])

    def test_full_checkpoint_does_not_wrap_sequential_stem_or_head(self) -> None:
        """Layer zero should select the first decoder block without touching branches."""
        model = _StemHeadOwner()
        stem_modules = tuple(model.stem.children())
        head_modules = tuple(model.head.children())
        first, second = model.layers
        config = ActivationCheckpointConfig(mode="full", full=FullCheckpointConfig(recompute_layer=[0]))

        _apply_activation_checkpointing(model, config)

        self.assertIsInstance(model.layers[0], CheckpointWrapper)
        self.assertIs(model.layers[0]._wrapped_module, first)
        self.assertIs(model.layers[1], second)
        self.assertEqual(tuple(model.stem.children()), stem_modules)
        self.assertEqual(tuple(model.head.children()), head_modules)

    def test_discovery_preserves_homogeneous_sequential_blocks(self) -> None:
        """Empty, single-layer, and partially wrapped sequences retain their behavior."""
        for layer_count in (0, 1, 2):
            for wrapped_count in range(layer_count + 1):
                with self.subTest(layer_count=layer_count, wrapped_count=wrapped_count):
                    owner = _Owner()
                    layers = [_Block() for _ in range(layer_count)]
                    for index in range(wrapped_count):
                        layers[index] = real_checkpoint_wrapper(layers[index])
                    owner.decoder = nn.Sequential(*layers)

                    blocks, _ = _find_transformer_block_modules(owner)

                    self.assertEqual([block.fqn for block in blocks], [f"decoder.{i}" for i in range(layer_count)])

    def test_discovery_rejects_heterogeneous_sequential_wrapped_blocks(self) -> None:
        """Checkpoint wrappers must not hide differing underlying child types."""
        for wrapped_count in (0, 1, 2):
            with self.subTest(wrapped_count=wrapped_count):
                owner = _Owner()
                layers = [nn.Linear(2, 2), nn.ReLU()]
                for index in range(wrapped_count):
                    layers[index] = real_checkpoint_wrapper(layers[index])
                owner.decoder = nn.Sequential(*layers)

                blocks, _ = _find_transformer_block_modules(owner)

                self.assertEqual(blocks, [])

    def test_discovery_preserves_heterogeneous_module_list_and_dict(self) -> None:
        """Explicit list and dictionary containers still permit differing block types."""
        for container in (
            nn.ModuleList([_Block(), nn.Sequential(nn.Linear(2, 2), nn.ReLU())]),
            nn.ModuleDict({"2": _Block(), "7": nn.Sequential(nn.Linear(2, 2), nn.ReLU())}),
        ):
            with self.subTest(container_type=type(container).__name__):
                owner = _Owner()
                owner.decoder = container

                blocks, _ = _find_transformer_block_modules(owner)

                self.assertEqual(
                    [block.fqn for block in blocks],
                    [f"decoder.{key}" for key, _ in container.named_children()],
                )

    def test_hybrid_installs_exact_wrappers_and_independent_pools(self) -> None:
        """Hybrid mode should preserve wrapper types, parameters, and one chain order."""
        model = _Model()
        config = ActivationCheckpointConfig(
            mode="hybrid",
            full=FullCheckpointConfig(
                recompute_layer=["0-1"],
                exclude_region=FullExcludeRegionConfig(
                    save={"module_a": [0]},
                    swap={"group_swap": False, "cpu_pool": 2, "module_b": ["0-1"]},
                ),
            ),
            selective=SelectiveCheckpointConfig(
                recompute=[
                    {"module_a": [2]},
                    "model.embed_tokens",
                    "lm_head",
                ],
                swap={"group_swap": True, "cpu_pool": 2, "module_b": [3]},
            ),
        )

        with patch(
            f"{_ACTIVATION_CHECKPOINT_MODULE}.SwapManager.set_forward_prefetch_layer",
            autospec=True,
        ) as set_prefetch:
            result = _apply_activation_checkpointing(model, config)

        self.assertIs(result, model)
        layer0, layer1, layer2, layer3 = model.tower.decoder.values()
        self.assertIsInstance(layer0, CheckpointWrapper)
        self.assertIsInstance(layer1, CheckpointWrapper)
        self.assertIsInstance(layer0._wrapped_module.module_a, CheckpointExcludeWrapper)
        self.assertFalse(layer0._wrapped_module.module_a.swap)
        self.assertIsInstance(layer0._wrapped_module.module_b, CheckpointExcludeWrapper)
        self.assertTrue(layer0._wrapped_module.module_b.swap)
        self.assertIsInstance(layer2.module_a, CheckpointWrapper)
        self.assertIsInstance(layer3.module_b, SwapWrapper)
        self.assertTrue(layer3.module_b.group_swap)
        self.assertIsInstance(model.model.embed_tokens, CheckpointWrapper)
        self.assertIsInstance(model.lm_head, CheckpointWrapper)

        resources = model._hp_activation_checkpoint_resources  # pylint: disable=protected-access
        self.assertEqual(resources.full_swap_cpu_pool.max_host_bytes, 2 * 1024 * 1024)
        self.assertEqual(resources.selective_swap_cpu_pool.max_host_bytes, 2 * 1024 * 1024)
        self.assertIsNot(resources.full_swap_cpu_pool, resources.selective_swap_cpu_pool)
        self.assertIs(layer0._wrapped_module.module_b.cpu_pool, resources.full_swap_cpu_pool)
        self.assertIs(layer3.module_b.cpu_pool, resources.selective_swap_cpu_pool)

        self.assertEqual(set_prefetch.call_count, 2)
        self.assertEqual(
            [item.args[1:] for item in set_prefetch.call_args_list],
            [(layer0, layer1), (layer1, layer3.module_b)],
        )

    def test_null_pool_is_forwarded_as_none(self) -> None:
        """Missing pool settings should not allocate a pinned-memory pool."""
        model = _Model()
        config = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(
                swap={"module_b": ["0-1"]},
            ),
        )

        _apply_activation_checkpointing(model, config)

        first, second, _, _ = model.tower.decoder.values()
        self.assertIsInstance(first.module_b, SwapWrapper)
        self.assertIsNone(first.module_b.cpu_pool)
        self.assertTrue(first.module_b.group_swap)
        resources = model._hp_activation_checkpoint_resources  # pylint: disable=protected-access
        self.assertIsNone(resources.selective_swap_cpu_pool)
        self.assertEqual(SwapManager().active_group_count(), 2)

    def test_off_mode_is_a_no_op(self) -> None:
        """Off mode should not discover layers, allocate pools, or install wrappers."""
        model = _UnmarkedModel()

        result = _apply_activation_checkpointing(model, ActivationCheckpointConfig(mode="off"))

        self.assertIs(result, model)
        self.assertEqual(model.gradient_checkpointing_disable_calls, 0)
        self.assertFalse(hasattr(model, "_hp_activation_checkpoint_resources"))
        self.assertEqual(SwapManager().active_group_count(), 0)

    def test_model_targets_do_not_require_transformer_layer_discovery(self) -> None:
        """Model-root targets should work on modules without discoverable transformer layers."""
        model = _UnmarkedModel()
        embed_tokens = model.embed_tokens
        lm_head = model.lm_head
        config = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(
                recompute=["embed_tokens", "lm_head"],
            ),
        )

        _apply_activation_checkpointing(model, config)

        self.assertIsInstance(model.embed_tokens, CheckpointWrapper)
        self.assertIs(model.embed_tokens._wrapped_module, embed_tokens)
        self.assertIsInstance(model.lm_head, CheckpointWrapper)
        self.assertIs(model.lm_head._wrapped_module, lm_head)
        self.assertEqual(model.gradient_checkpointing_disable_calls, 1)

    def test_model_target_recompute_preserves_embedding_and_head_gradients(self) -> None:
        """Integer embedding inputs and a model-level head should retain baseline gradients."""
        model = _UnmarkedModel()
        baseline = copy.deepcopy(model)
        input_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        config = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(
                recompute=["embed_tokens", "lm_head"],
            ),
        )

        baseline_output = baseline(input_ids)
        baseline_output.sum().backward()
        _apply_activation_checkpointing(model, config)
        output = model(input_ids)
        output.sum().backward()

        self.assertTrue(torch.equal(output, baseline_output))
        self.assertTrue(
            torch.equal(model.embed_tokens.weight.grad, baseline.embed_tokens.weight.grad)
        )
        self.assertTrue(torch.equal(model.lm_head.weight.grad, baseline.lm_head.weight.grad))

    def test_model_target_recompute_preserves_tied_weights_and_state_dict_names(self) -> None:
        """Wrapping embedding and head modules should preserve their shared parameter contract."""
        model = _UnmarkedModel()
        model.lm_head.weight = model.embed_tokens.weight
        tied_weight = model.embed_tokens.weight
        state_dict_keys = tuple(model.state_dict())
        config = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(
                recompute=["embed_tokens", "lm_head"],
            ),
        )

        _apply_activation_checkpointing(model, config)

        self.assertIs(model.embed_tokens.weight, tied_weight)
        self.assertIs(model.lm_head.weight, tied_weight)
        self.assertEqual(tuple(model.state_dict()), state_dict_keys)

    def test_model_targets_reject_layer_subtrees_and_module_aliases(self) -> None:
        """Model paths should not bypass layer scope or select one module through aliases."""
        for module_path in ("tower.decoder.2.module_a", "tower.decoder", "tower"):
            with self.subTest(module_path=module_path):
                model = _Model()
                config = ActivationCheckpointConfig(
                    mode="selective",
                    selective=SelectiveCheckpointConfig(recompute=[module_path]),
                )
                with self.assertRaisesRegex(ValueError, "must be outside transformer layers"):
                    _apply_activation_checkpointing(model, config)
                self.assertEqual(model.gradient_checkpointing_disable_calls, 0)

        model = _Model()
        model.alias_head = model.lm_head
        config = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(recompute=["lm_head", "alias_head"]),
        )
        with self.assertRaisesRegex(ValueError, "aliases"):
            _apply_activation_checkpointing(model, config)
        self.assertIs(model.lm_head, model.alias_head)
        self.assertEqual(model.gradient_checkpointing_disable_calls, 0)

    def test_invalid_model_path_fails_before_any_mutation(self) -> None:
        """A missing model-root path should fail before native checkpointing is disabled."""
        model = _Model()
        embed_tokens = model.model.embed_tokens
        lm_head = model.lm_head
        config = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(recompute=["model.missing"]),
        )

        with self.assertRaisesRegex(ValueError, "model.missing.*does not exist"):
            _apply_activation_checkpointing(model, config)

        self.assertIs(model.model.embed_tokens, embed_tokens)
        self.assertIs(model.lm_head, lm_head)
        self.assertEqual(model.gradient_checkpointing_disable_calls, 0)

    def test_model_target_apply_failure_rolls_back_wrappers_and_marks(self) -> None:
        """A model-level wrapper failure should restore prior model targets."""
        model = _Model()
        embed_tokens = model.model.embed_tokens
        lm_head = model.lm_head
        config = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(
                recompute=["model.embed_tokens", "lm_head"],
            ),
        )
        wrapper_calls = 0

        def fail_second_wrapper(module: nn.Module) -> nn.Module:
            """Wrap the first model target and fail on the second target."""
            nonlocal wrapper_calls
            wrapper_calls += 1
            if wrapper_calls == 2:
                raise RuntimeError("injected model wrapper failure")
            return real_checkpoint_wrapper(module)

        with patch(
            f"{_ACTIVATION_CHECKPOINT_MODULE}.checkpoint_wrapper",
            side_effect=fail_second_wrapper,
        ), self.assertRaisesRegex(RuntimeError, "injected model wrapper failure"):
            _apply_activation_checkpointing(model, config)

        self.assertIs(model.model.embed_tokens, embed_tokens)
        self.assertIs(model.lm_head, lm_head)
        self.assertTrue(all(not hasattr(module, "_is_wrapped") for module in model.modules()))

    def test_active_mode_disables_native_checkpointing_before_wrapping(self) -> None:
        """HyperParallel wrappers should not nest under HuggingFace checkpointing."""
        model = _Model()
        model.tower.gradient_checkpointing = True
        config = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(recompute={"module_a": [0]}),
        )

        def wrap_after_native_disable(module: nn.Module) -> nn.Module:
            """Verify native checkpointing is disabled before wrapper construction."""
            self.assertFalse(model.tower.gradient_checkpointing)
            return real_checkpoint_wrapper(module)

        with patch(
            f"{_ACTIVATION_CHECKPOINT_MODULE}.checkpoint_wrapper",
            side_effect=wrap_after_native_disable,
        ):
            _apply_activation_checkpointing(model, config)

        self.assertEqual(model.gradient_checkpointing_disable_calls, 1)
        self.assertIsInstance(model.tower.decoder["2"].module_a, CheckpointWrapper)

    def test_invalid_module_path_fails_before_any_mutation(self) -> None:
        """Resolve failures should leave every selected layer untouched."""
        model = _Model()
        originals = tuple(model.tower.decoder.values())
        config = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(
                recompute={"missing": [0], "module_a": [1]},
            ),
        )

        with self.assertRaisesRegex(ValueError, "layer 0.*missing.*does not exist"):
            _apply_activation_checkpointing(model, config)

        self.assertEqual(tuple(model.tower.decoder.values()), originals)
        self.assertEqual(model.gradient_checkpointing_disable_calls, 0)
        self.assertTrue(all(not hasattr(module, "_is_wrapped") for module in model.modules()))

    def test_apply_failure_rolls_back_modules_and_overlap_marks(self) -> None:
        """A wrapper-construction failure should restore the original model tree."""
        model = _Model()
        originals = tuple(model.tower.decoder.values())
        config = ActivationCheckpointConfig(
            mode="full",
            full=FullCheckpointConfig(recompute_layer=["0-1"]),
        )
        wrapper_calls = 0

        def fail_second_wrapper(module: nn.Module) -> nn.Module:
            """Wrap the first layer and inject a failure for the second."""
            nonlocal wrapper_calls
            wrapper_calls += 1
            if wrapper_calls == 2:
                raise RuntimeError("injected wrapper failure")
            return real_checkpoint_wrapper(module)

        with patch(
            f"{_ACTIVATION_CHECKPOINT_MODULE}.checkpoint_wrapper",
            side_effect=fail_second_wrapper,
        ), self.assertRaisesRegex(RuntimeError, "injected wrapper failure"):
            _apply_activation_checkpointing(model, config)

        self.assertEqual(tuple(model.tower.decoder.values()), originals)
        self.assertTrue(all(not hasattr(module, "_is_wrapped") for module in model.modules()))
        self.assertEqual(SwapManager().active_group_count(), 0)

    def test_prefetch_failure_rolls_back_hooks_groups_and_wrappers(self) -> None:
        """A partial prefetch registration should be fully torn down."""
        model = _Model()
        originals = tuple(model.tower.decoder.values())
        config = ActivationCheckpointConfig(
            mode="selective",
            selective=SelectiveCheckpointConfig(
                swap={"module_b": ["0-2"]},
            ),
        )
        real_set_prefetch = SwapManager.set_forward_prefetch_layer
        prefetch_calls = 0

        def fail_second_prefetch(
            manager: SwapManager,
            first_layer: nn.Module,
            second_layer: nn.Module,
        ) -> None:
            """Register the pair, then fail after the second registration."""
            nonlocal prefetch_calls
            prefetch_calls += 1
            real_set_prefetch(manager, first_layer, second_layer)
            if prefetch_calls == 2:
                raise RuntimeError("injected prefetch failure")

        with patch.object(
            SwapManager,
            "set_forward_prefetch_layer",
            autospec=True,
            side_effect=fail_second_prefetch,
        ), self.assertRaisesRegex(RuntimeError, "injected prefetch failure"):
            _apply_activation_checkpointing(model, config)

        self.assertEqual(tuple(model.tower.decoder.values()), originals)
        self.assertTrue(all(not hasattr(module, "_is_wrapped") for module in model.modules()))
        self.assertEqual(SwapManager().active_group_count(), 0)

    def test_compile_and_layer_range_errors_are_explicit(self) -> None:
        """Compile conflicts and out-of-range layers should fail before wrapping."""
        model = _Model()
        config = ActivationCheckpointConfig(
            mode="full",
            full=FullCheckpointConfig(recompute_layer=[4]),
        )
        with self.assertRaisesRegex(ValueError, "layer 4.*range 0-3"):
            _apply_activation_checkpointing(model, config)

        compile_config = ActivationCheckpointConfig(
            mode="full",
            full=FullCheckpointConfig(recompute_layer=[0]),
        )
        with self.assertRaisesRegex(ValueError, "not supported with torch.compile"):
            _apply_activation_checkpointing(model, compile_config, enable_compile=True)


if __name__ == "__main__":
    unittest.main()
