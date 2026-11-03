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

import unittest
from unittest.mock import patch

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


class _Model(nn.Module):
    """Model exposing one marked transformer owner."""

    def __init__(self) -> None:
        """Create the marked owner."""
        super().__init__()
        self.tower = _Owner()

    def forward(self, inputs: Tensor) -> Tensor:
        """Delegate to the marked owner."""
        return self.tower(inputs)


class _UnmarkedModel(nn.Module):
    """Repeated layers without the structural discovery marker."""

    def __init__(self) -> None:
        """Create an unmarked repeated container."""
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])


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
                recompute={"module_a": [2]},
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
        self.assertFalse(first.module_b.group_swap)
        resources = model._hp_activation_checkpoint_resources  # pylint: disable=protected-access
        self.assertIsNone(resources.selective_swap_cpu_pool)
        self.assertEqual(SwapManager().active_group_count(), 2)

    def test_off_mode_is_a_no_op(self) -> None:
        """Off mode should not discover layers, allocate pools, or install wrappers."""
        model = _UnmarkedModel()

        result = _apply_activation_checkpointing(model, ActivationCheckpointConfig(mode="off"))

        self.assertIs(result, model)
        self.assertFalse(hasattr(model, "_hp_activation_checkpoint_resources"))
        self.assertEqual(SwapManager().active_group_count(), 0)

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
