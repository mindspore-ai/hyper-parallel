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
"""CPU regression tests for graph parameter/gradient ownership and layouts."""

from types import SimpleNamespace
import unittest
from typing import Callable, Optional
from unittest.mock import PropertyMock, patch

import torch
from torch import nn

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from tests.common.mark_utils import arg_mark


class _LocalMomentum(torch.optim.SGD):
    """Consume local storage while retaining DTensor state, without collectives."""

    is_muon = True

    def reset_optimizer_parameters(self) -> None:
        """The unit test does not build distributed owner/communication plans."""

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> None:
        """Apply one local SGD update with momentum."""
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param.grad)
                momentum = to_local_if_dtensor(state["momentum_buffer"])
                momentum.mul_(0.5).add_(to_local_if_dtensor(param.grad))
                to_local_if_dtensor(param).add_(momentum, alpha=-group["lr"])


def _layout(shape, placements=(Shard(0), Replicate())):
    """Describe a logical tensor on the test data- and tensor-parallel mesh."""
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp_shard", "tp"),
                            rank_list=(0, 1, 2, 3), init_backend=False)
    layout = Layout.from_device_mesh(mesh)
    layout.set_placements(placements)
    layout.placement_to_tensor_map(len(shape))
    layout.set_tensor_meta(shape, tuple(torch.empty(shape).stride()), torch.float32)
    return layout


def _optimizer(dtype=torch.float32, fp32_main_params=False):
    """Build a graph optimizer with a local momentum implementation."""
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones((2, 3), dtype=dtype))
    model.weight.model_name = "weight"
    leaf = _LocalMomentum(model.parameters(), lr=0.1)
    chain = ChainedOptimizer(model, {"muon": leaf})
    chain.defer_parameter_binding()
    optimizer = Float16OptimizerWithFloat16Params(chain, model) if fp32_main_params else chain
    return model, leaf, optimizer


class TestGraphOptimizer(unittest.TestCase):
    """Binding must preserve matrix semantics without cloning FP32 weights."""

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_shared_storage_and_two_step_gradient_lifecycle(self):
        """Feature: FP32 graph views.

        Description: Accumulate two gradients on the same parameter view.
        Expectation: Weights and momentum match the reference and gradients clear.
        """
        model, leaf, optimizer = _optimizer()
        layouts = {model.weight: (_layout((4, 3)), (2, 3))}
        optimizer.bind_parameter_layouts(layouts)
        view = optimizer.optimizer_param_by_model_param[model.weight]
        self.assertIsInstance(view, DTensor)
        self.assertEqual(tuple(view.shape), (4, 3))
        self.assertEqual(view.to_local().data_ptr(), model.weight.data_ptr())
        with torch.no_grad():
            expected = torch.clone(model.weight)
        momentum = torch.zeros_like(expected)
        for value in (1.0, 3.0):
            model.weight.grad = torch.full_like(model.weight, value)
            momentum.mul_(0.5).add_(value)
            expected.add_(momentum, alpha=-0.1)
            optimizer.bind_parameter_layouts(layouts)
            optimizer.step()
            optimizer.zero_grad()
            torch.testing.assert_close(model.weight, expected)
            self.assertIsNone(model.weight.grad)
            self.assertIsNone(view.grad)
            self.assertIs(optimizer.optimizer_param_by_model_param[model.weight], view)
        state = leaf.state[view]["momentum_buffer"]
        self.assertIsInstance(state, DTensor)
        self.assertEqual(tuple(state.shape), (4, 3))
        torch.testing.assert_close(state.to_local(), momentum)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_zero_grad_preserves_existing_buffers_when_requested(self):
        """Feature: Graph optimizer gradient reset.

        Description: Zero an existing model gradient and main gradient after an update.
        Expectation: False preserves and zeroes the buffers, and True releases them.
        """
        for fp32 in (False, True):
            with self.subTest(fp32_main_params=fp32):
                dtype = torch.bfloat16 if fp32 else torch.float32
                model, _, optimizer = _optimizer(dtype, fp32_main_params=fp32)
                optimizer.bind_parameter_layouts({model.weight: (_layout((4, 3)), (2, 3))})
                view = optimizer.optimizer_param_by_model_param[model.weight]
                grad = torch.ones_like(model.weight)
                main_grad = torch.ones_like(model.weight, dtype=torch.float32)
                model.weight.grad = grad
                model.weight.main_grad = main_grad
                model.weight.grad_added_to_main_grad = True
                optimizer.step()

                optimizer.zero_grad(set_to_none=False)
                self.assertIs(model.weight.grad, grad)
                self.assertIs(model.weight.main_grad, main_grad)
                torch.testing.assert_close(grad, torch.zeros_like(grad))
                torch.testing.assert_close(main_grad, torch.zeros_like(main_grad))
                torch.testing.assert_close(view.grad.to_local(), torch.zeros_like(main_grad))
                self.assertFalse(model.weight.grad_added_to_main_grad)

                optimizer.zero_grad(set_to_none=True)
                self.assertIsNone(model.weight.grad)
                self.assertIsNone(model.weight.main_grad)
                self.assertIsNone(view.grad)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_strided_shard_retains_logical_shape(self):
        """Feature: Strided sharding.

        Description: Bind two axes that shard the row dimension.
        Expectation: Optimizer parameter and momentum preserve the placements.
        """
        model, leaf, optimizer = _optimizer()
        layout = _layout((8, 3), (StridedShard(0, 2), Shard(0)))
        optimizer.bind_parameter_layouts({model.weight: (layout, (2, 3))})
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        view = optimizer.optimizer_param_by_model_param[model.weight]
        self.assertEqual(tuple(view.shape), (8, 3))
        self.assertEqual(tuple(view.placements), tuple(layout.placements))
        self.assertEqual(tuple(leaf.state[view]["momentum_buffer"].shape), (8, 3))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_fp32_main_is_created_from_final_local_shard(self):
        """Feature: FP32 main weights.

        Description: Bind a BF16 shard after compilation.
        Expectation: Only the final local shard is copied into main storage.
        """
        model, _, optimizer = _optimizer(torch.bfloat16, fp32_main_params=True)
        layouts = {model.weight: (_layout((4, 3)), (2, 3))}
        self.assertFalse(optimizer.model_param_by_optimizer_param)
        optimizer.bind_parameter_layouts(layouts)
        main = optimizer.optimizer_param_by_model_param[model.weight]
        self.assertEqual(main.dtype, torch.float32)
        self.assertEqual(tuple(main.to_local().shape), (2, 3))
        self.assertNotEqual(main.to_local().data_ptr(), model.weight.data_ptr())
        self.assertIs(optimizer.model_param_by_optimizer_param[main], model.weight)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        torch.testing.assert_close(main.to_local(), torch.full((2, 3), 0.9))
        torch.testing.assert_close(model.weight, main.to_local().to(torch.bfloat16))
        original = model.weight
        with optimizer.optimizer._optimizer_parameter_view():
            self.assertIs(model.weight, main)
        self.assertIs(model.weight, original)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_padding_is_excluded_and_empty_shards_are_retained(self):
        """Feature: Padded graph weights.

        Description: Run updates with one or zero valid rows.
        Expectation: Padding stays zero and optimizer state uses valid rows.
        """
        for valid_rows in (0, 1):
            with self.subTest(valid_rows=valid_rows):
                model, leaf, optimizer = _optimizer()
                optimizer.bind_parameter_layouts({model.weight: (_layout((1, 3)), (valid_rows, 3))})
                model.weight.grad = torch.ones_like(model.weight)
                optimizer.step()
                view = optimizer.optimizer_param_by_model_param[model.weight]
                self.assertEqual(tuple(view.shape), (1, 3))
                self.assertEqual(view.to_local().shape[0], valid_rows)
                self.assertEqual(leaf.state[view]["momentum_buffer"].to_local().shape[0], valid_rows)
                torch.testing.assert_close(model.weight[valid_rows:], torch.zeros((2 - valid_rows, 3)))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_changed_storage_or_layout_is_rejected(self):
        """Feature: Binding invalidation.

        Description: Change layout identity and then replace physical storage.
        Expectation: Each invalidation is detected before an update.
        """
        model, _, optimizer = _optimizer()
        layouts = {model.weight: (_layout((4, 3)), (2, 3))}
        optimizer.bind_parameter_layouts(layouts)
        with self.assertRaisesRegex(RuntimeError, "layout changed"):
            optimizer.bind_parameter_layouts(dict(layouts))
        with torch.no_grad():
            model.weight.data = torch.clone(model.weight)
        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaisesRegex(RuntimeError, "storage changed"):
            optimizer.step()

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_binding_requires_an_uninitialized_optimizer(self):
        """Feature: First binding.

        Description: Populate leaf state before registering layouts.
        Expectation: Binding rejects initialized leaf state.
        """
        model, leaf, optimizer = _optimizer()
        leaf.state[model.weight]["momentum_buffer"] = torch.ones_like(model.weight)
        with self.assertRaisesRegex(RuntimeError, "before initializing optimizer state"):
            optimizer.bind_parameter_layouts({model.weight: (_layout((4, 3)), (2, 3))})

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_incomplete_layout_can_be_retried_without_replacing_groups(self):
        """Feature: Retryable layout binding.

        Description: Omit the second layout, then retry with both.
        Expectation: The failed bind preserves original parameters and scheduler groups.
        """
        model, leaf, optimizer = _optimizer()
        model.second = nn.Parameter(torch.zeros_like(model.weight))
        model.second.model_name = "second"
        group = leaf.param_groups[0]
        group["params"].append(model.second)
        layout = (_layout((4, 3)), (2, 3))
        with self.assertRaisesRegex(ValueError, "Missing graph parameter layout for second"):
            optimizer.bind_parameter_layouts({model.weight: layout})
        self.assertIs(group["params"][0], model.weight)
        optimizer.bind_parameter_layouts({model.weight: layout, model.second: layout})
        self.assertIs(optimizer.param_groups[0], group)
        self.assertIs(group["params"][1], optimizer.optimizer_param_by_model_param[model.second])

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_binding_rejects_unregistered_and_duplicate_parameters(self):
        """Feature: Graph parameter ownership.

        Description: Add an external or repeated parameter to another leaf.
        Expectation: Bind rejects it before changing either parameter group.
        """
        for duplicate, message in ((False, "not registered"), (True, "Duplicate.*weight")):
            with self.subTest(duplicate=duplicate):
                model, leaf, optimizer = _optimizer()
                extra = model.weight if duplicate else nn.Parameter(torch.ones(2, 3))
                fallback = torch.optim.AdamW([extra], lr=0.1, foreach=False)
                optimizer.chained_optimizers.append(fallback)
                with self.assertRaisesRegex(ValueError, message):
                    optimizer.bind_parameter_layouts({model.weight: (_layout((4, 3)), (2, 3))})
                self.assertIs(leaf.param_groups[0]["params"][0], model.weight)
                self.assertIs(fallback.param_groups[0]["params"][0], extra)
                self.assertFalse(optimizer.parameter_views.bound)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_binding_reset_failure_restores_parameter_groups_and_leaf_cache(self):
        """Feature: Graph binding rollback.

        Description: Fail after a leaf has written a parameter-derived cache.
        Expectation: Groups and caches roll back without triggering a second reset.
        """
        model, leaf, optimizer = _optimizer()
        layout = {model.weight: (_layout((4, 3)), (2, 3))}
        original_cache = (model.weight,)
        leaf.param_cache = original_cache
        group = leaf.param_groups[0]

        def fail_reset():
            """Publish partial metadata before the injected reset failure."""
            leaf.param_cache = tuple(group["params"])
            leaf.state[group["params"][0]]["partial_state"] = True
            raise RuntimeError("injected reset failure")

        with patch.object(leaf, "reset_optimizer_parameters", side_effect=fail_reset) as reset:
            with self.assertRaisesRegex(RuntimeError, "injected reset failure"):
                optimizer.bind_parameter_layouts(layout)
            reset.assert_called_once_with()
        self.assertIs(group["params"][0], model.weight)
        self.assertIs(leaf.param_cache, original_cache)
        self.assertFalse(leaf.state)
        self.assertFalse(optimizer.optimizer_param_by_model_param)
        self.assertFalse(optimizer.parameter_views.bound)
        optimizer.bind_parameter_layouts(layout)
        self.assertTrue(optimizer.parameter_views.bound)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_fp32_main_weights_bind_after_weights_only_reload(self):
        """Feature: Weights-only graph restore.

        Description: Reload model weights before lazily binding optimizer layouts.
        Expectation: Binding initializes FP32 main weights from restored values.
        """
        model, _, optimizer = _optimizer(torch.bfloat16, fp32_main_params=True)
        with torch.no_grad():
            model.weight.fill_(3)
        self.assertIsInstance(optimizer, Float16OptimizerWithFloat16Params)
        optimizer.reload_model_params()  # pylint: disable=no-member
        optimizer.bind_parameter_layouts({model.weight: (_layout((4, 3)), (2, 3))})
        main = optimizer.optimizer_param_by_model_param[model.weight]
        torch.testing.assert_close(main.to_local(), torch.full((2, 3), 3.0))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_unsharded_graph_keeps_original_parameters(self):
        """Feature: Local graph execution.

        Description: Bind a graph without distributed layouts.
        Expectation: The optimizer updates the original parameter.
        """
        model, _, optimizer = _optimizer()
        optimizer.bind_parameter_layouts(None)
        self.assertIs(optimizer.optimizer_param_by_model_param[model.weight], model.weight)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        torch.testing.assert_close(model.weight, torch.full((2, 3), 0.9))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_fp32_adamw_identity_binding_consumes_main_grad(self):
        """Feature: FP32 graph gradients.
        Description: AdamW reuses an FP32 live parameter and reads its main_grad.
        Expectation: AdamW updates the reused parameter.
        """
        model = nn.Linear(2, 1, bias=False)
        nn.init.ones_(model.weight)
        leaf = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0)
        chain = ChainedOptimizer(model, {"adamw": leaf})
        chain.defer_parameter_binding()
        optimizer = Float16OptimizerWithFloat16Params(chain, model)
        optimizer.bind_parameter_layouts(None)
        self.assertIs(optimizer.optimizer_param_by_model_param[model.weight], model.weight)
        model.weight.main_grad = torch.ones_like(model.weight)
        optimizer.step()
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 0.9))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_adam_fallback_keeps_padded_local_shape_with_fp32_main(self):
        """Feature: AdamW fallback.

        Description: Bind a mixed chain over BF16 parameters.
        Expectation: AdamW uses local main weights and Muon keeps its matrix layout.
        """
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones((2, 3), dtype=torch.bfloat16))
        model.scale = nn.Parameter(torch.ones((2,), dtype=torch.bfloat16))
        for name, param in model.named_parameters():
            param.model_name = name
        muon = _LocalMomentum([model.weight], lr=0.1)
        # Keep the CPU test independent of NPU capability discovery.
        adam = torch.optim.AdamW([model.scale], lr=0.1, weight_decay=0.1, foreach=False)
        chain = ChainedOptimizer(model, {"muon": muon, "adamw": adam}, flatten=True)
        chain.defer_parameter_binding()
        optimizer = Float16OptimizerWithFloat16Params(chain, model)
        optimizer.bind_parameter_layouts({
            model.weight: (_layout((4, 3)), (2, 3)),
            model.scale: (_layout((1,)), (1,)),
        })
        main = optimizer.optimizer_param_by_model_param[model.scale]
        self.assertNotIsInstance(main, DTensor)
        self.assertEqual(tuple(main.shape), (2,))
        self.assertEqual(main.dtype, torch.float32)
        model.weight.grad = torch.ones_like(model.weight)
        model.scale.grad = torch.tensor([1, 0], dtype=torch.bfloat16)
        optimizer.step()
        optimizer.zero_grad()
        torch.testing.assert_close(model.scale, torch.tensor([0.89, 0], dtype=torch.bfloat16))
        torch.testing.assert_close(main.detach(), torch.tensor([0.89, 0]))
        self.assertIsNone(model.scale.grad)
        self.assertIsNone(main.grad)

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_checkpoint_round_trip_preserves_fqn_and_main_weights(self):
        """Feature: Optimizer checkpoint.

        Description: Load moments and FP32 main weights into another bound optimizer.
        Expectation: State and model parameter associations survive the round trip.
        """
        model, leaf, optimizer = _optimizer(torch.bfloat16, fp32_main_params=True)
        optimizer.bind_parameter_layouts({model.weight: (_layout((4, 3)), (2, 3))})
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad()
        state = optimizer.state_dict()
        saved_main = optimizer.optimizer_param_by_model_param[model.weight].to_local().clone()
        saved_momentum = to_local_if_dtensor(next(iter(leaf.state.values()))["momentum_buffer"]).clone()
        model2, leaf2, restored = _optimizer(torch.bfloat16, fp32_main_params=True)
        restored.bind_parameter_layouts({model2.weight: (_layout((4, 3)), (2, 3))})
        restored.load_state_dict(state)
        view = restored.optimizer_param_by_model_param[model2.weight]
        torch.testing.assert_close(view.to_local(), saved_main)
        torch.testing.assert_close(to_local_if_dtensor(leaf2.state[view]["momentum_buffer"]), saved_momentum)
        self.assertIs(restored.model_param_by_optimizer_param[view], model2.weight)


class TestOptimizerParameterLayouts(unittest.TestCase):
    """Matrix layouts must describe unpadded logical shapes and ceil chunks."""

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_compiler_preserves_storage_layout_and_removes_logical_padding(self):
        """Feature: Compiler layout metadata.

        Description: Query logical layouts for a physically padded parameter.
        Expectation: Logical rows exclude padding and storage metadata is preserved.
        """
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones((2, 3)))
        storage_layout = _layout((4, 3))
        compiler = GraphCompiler.__new__(GraphCompiler)
        compiler.model = model
        compiler._parameter_shapes = {"weight": (3, 3)}
        compiler._parameter_meshes = {"weight": storage_layout.mesh["fsdp_shard"]}
        compiler._joint_graph = SimpleNamespace(graph_module=SimpleNamespace(fsdp_sharded_param_names={"weight"}))
        compiler._optimizer_layout_source = None
        with patch.object(GraphCompiler, "parameter_layouts", new_callable=PropertyMock) as layouts:
            layouts.return_value = {model.weight: storage_layout}
            with patch.object(compiler, "source_layout_for_parameter", return_value=None):
                result = compiler.optimizer_parameter_layouts
                self.assertIs(compiler.optimizer_parameter_layouts, result)
        layout, local_shape = result[model.weight]
        self.assertEqual(layout.tensor_shape, (3, 3))
        self.assertEqual(local_shape, (2, 3))
        self.assertEqual(storage_layout.tensor_shape, (4, 3))

    @arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_invalid_shard_metadata_has_actionable_errors(self):
        """Feature: Validation of optimizer layout metadata.

        Description: Query a physically sharded parameter with incomplete metadata.
        Expectation: The error identifies a missing shape, mesh, or axis.
        """
        model = nn.Linear(3, 2, bias=False)
        compiler = GraphCompiler(model, lambda *_: None)
        compiler._joint_graph = SimpleNamespace(graph_module=SimpleNamespace(fsdp_sharded_param_names={"weight"}))
        storage_layout = _layout((4, 3))
        with patch.object(GraphCompiler, "parameter_layouts", new_callable=PropertyMock) as layouts:
            layouts.return_value = {model.weight: storage_layout}
            with self.assertRaisesRegex(ValueError, "Missing pre-shard shape.*weight"):
                _ = compiler.optimizer_parameter_layouts
            compiler._parameter_shapes = {"weight": (3, 3)}
            with self.assertRaisesRegex(ValueError, "weight.*requires an initialized FSDP mesh"):
                _ = compiler.optimizer_parameter_layouts
            compiler._parameter_meshes["weight"] = SimpleNamespace(mesh_dim_names=None)
            with self.assertRaisesRegex(ValueError, "requires a shard mesh axis"):
                _ = compiler.optimizer_parameter_layouts
            compiler._parameter_meshes["weight"] = SimpleNamespace(
                mesh_dim_names=("fsdp_shard",), get_coordinate=lambda: None,
            )
            with self.assertRaisesRegex(ValueError, "outside the FSDP mesh.*weight"):
                _ = compiler.optimizer_parameter_layouts


if __name__ == "__main__":
    unittest.main()
