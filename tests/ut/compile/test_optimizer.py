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
from typing import Any
import unittest
from unittest.mock import PropertyMock, patch

import torch
from torch import nn

from tests.common.mark_utils import arg_mark

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.optimizer import GraphOptimizer
from hyper_parallel.components.optim.mixed_precision_optimizer import (
    FP32_MAIN_PARAM_STATE_KEY,
    MIXED_PRECISION_OPTIMIZER_STATE_KEY,
    Float16OptimizerWithFloat16Params,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


class _LocalMomentum(torch.optim.SGD):
    """Consume local storage while retaining DTensor state, without collectives."""

    is_muon = True

    def reset_optimizer_parameters(self) -> None:
        """The unit test does not build distributed owner/communication plans."""

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        """Apply a local momentum update while retaining DTensor state."""
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
    return model, leaf, GraphOptimizer(chain, model, fp32_main_params=fp32_main_params)


class TestGraphOptimizer(unittest.TestCase):
    """Binding must preserve matrix semantics without cloning FP32 weights."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_shared_storage_and_two_step_gradient_lifecycle(self):
        """
        description: Verify repeated updates reuse the FP32 storage alias and clear both gradients.
        expectation: Weights and momentum follow two updates without replacing aliased storage.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model, leaf, optimizer = _optimizer()
        layouts = {model.weight: (_layout((4, 3)), (2, 3))}
        optimizer.bind(layouts)
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
            optimizer.bind(layouts)
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_strided_shard_retains_logical_shape(self):
        """
        description: Check that a strided shard preserves global metadata in both weights and momentum.
        expectation: Weight and momentum retain the strided global layout.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model, leaf, optimizer = _optimizer()
        layout = _layout((8, 3), (StridedShard(0, 2), Shard(0)))
        optimizer.bind({model.weight: (layout, (2, 3))})
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        view = optimizer.optimizer_param_by_model_param[model.weight]
        self.assertEqual(tuple(view.shape), (8, 3))
        self.assertEqual(tuple(view.placements), tuple(layout.placements))
        self.assertEqual(tuple(leaf.state[view]["momentum_buffer"].shape), (8, 3))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_fp32_main_is_created_from_final_local_shard(self):
        """
        description: Check FP32 main weights use the final local shard and write back to the model dtype.
        expectation: Only the valid shard owns FP32 main storage and the model receives its rounded update.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model, _, optimizer = _optimizer(torch.bfloat16, fp32_main_params=True)
        layouts = {model.weight: (_layout((4, 3)), (2, 3))}
        self.assertFalse(optimizer.model_param_by_optimizer_param)
        optimizer.bind(layouts)
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_padding_is_excluded_and_empty_shards_are_retained(self):
        """
        description: Check padding stays out of optimizer state even when a rank owns no valid rows.
        expectation: State excludes padding and unused model rows remain zero.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        for valid_rows in (0, 1):
            with self.subTest(valid_rows=valid_rows):
                model, leaf, optimizer = _optimizer()
                optimizer.bind({model.weight: (_layout((1, 3)), (valid_rows, 3))})
                model.weight.grad = torch.ones_like(model.weight)
                optimizer.step()
                view = optimizer.optimizer_param_by_model_param[model.weight]
                self.assertEqual(tuple(view.shape), (1, 3))
                self.assertEqual(view.to_local().shape[0], valid_rows)
                self.assertEqual(leaf.state[view]["momentum_buffer"].to_local().shape[0], valid_rows)
                torch.testing.assert_close(model.weight[valid_rows:], torch.zeros((2 - valid_rows, 3)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_changed_storage_or_layout_is_rejected(self):
        """
        description: Reject layout changes and replacement of aliased storage after binding.
        expectation: Rebinding or replacing physical storage raises before an update.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model, _, optimizer = _optimizer()
        layouts = {model.weight: (_layout((4, 3)), (2, 3))}
        optimizer.bind(layouts)
        with self.assertRaisesRegex(RuntimeError, "layout changed"):
            optimizer.bind(dict(layouts))
        with torch.no_grad():
            model.weight.data = torch.clone(model.weight)
        model.weight.grad = torch.ones_like(model.weight)
        with self.assertRaisesRegex(RuntimeError, "storage changed"):
            optimizer.step()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_binding_requires_an_uninitialized_optimizer(self):
        """
        description: Reject binding after the leaf optimizer has initialized state.
        expectation: Binding an optimizer with existing state raises.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model, leaf, optimizer = _optimizer()
        leaf.state[model.weight]["momentum_buffer"] = torch.ones_like(model.weight)
        with self.assertRaisesRegex(RuntimeError, "before initializing optimizer state"):
            optimizer.bind({model.weight: (_layout((4, 3)), (2, 3))})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_unsharded_graph_keeps_original_parameters(self):
        """
        description: Update the original parameter directly when there is no distributed layout.
        expectation: The original parameter receives the expected local update.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model, _, optimizer = _optimizer()
        optimizer.bind(None)
        self.assertIs(optimizer.optimizer_param_by_model_param[model.weight], model.weight)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        torch.testing.assert_close(model.weight, torch.full((2, 3), 0.9))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_adam_fallback_keeps_padded_local_shape_with_fp32_main(self):
        """
        description: Check the AdamW fallback uses plain FP32 main weights with the physical shard shape.
        expectation: AdamW owns plain FP32 storage and writes the expected BF16 model value.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones((2, 3), dtype=torch.bfloat16))
        model.scale = nn.Parameter(torch.ones((2,), dtype=torch.bfloat16))
        for name, param in model.named_parameters():
            param.model_name = name
        muon = _LocalMomentum([model.weight], lr=0.1)
        # Keep the CPU test independent of NPU capability discovery.
        adam = torch.optim.AdamW([model.scale], lr=0.1, weight_decay=0.0, foreach=False)
        chain = ChainedOptimizer(model, {"muon": muon, "adamw": adam}, flatten=True)
        optimizer = GraphOptimizer(chain, model, fp32_main_params=True)
        optimizer.bind({model.weight: (_layout((4, 3)), (2, 3))})
        main = optimizer.optimizer_param_by_model_param[model.scale]
        self.assertNotIsInstance(main, DTensor)
        self.assertEqual(tuple(main.shape), (2,))
        self.assertEqual(main.dtype, torch.float32)
        model.weight.grad = torch.ones_like(model.weight)
        model.scale.grad = torch.ones_like(model.scale)
        optimizer.step()
        optimizer.zero_grad()
        torch.testing.assert_close(model.scale, torch.full((2,), 0.9, dtype=torch.bfloat16))
        self.assertIsNone(model.scale.grad)
        self.assertIsNone(main.grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_checkpoint_round_trip_preserves_fqn_and_main_weights(self):
        """
        description: Restore the model-name mapping, FP32 main weights, and momentum into a fresh optimizer.
        expectation: Fresh bindings restore full-precision weights and momentum by name.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model, leaf, optimizer = _optimizer(torch.bfloat16, fp32_main_params=True)
        optimizer.bind({model.weight: (_layout((4, 3)), (2, 3))})
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad()
        state = optimizer.state_dict()
        saved_main = optimizer.optimizer_param_by_model_param[model.weight].to_local().clone()
        saved_momentum = to_local_if_dtensor(next(iter(leaf.state.values()))["momentum_buffer"]).clone()
        model2, leaf2, restored = _optimizer(torch.bfloat16, fp32_main_params=True)
        restored.bind({model2.weight: (_layout((4, 3)), (2, 3))})
        restored.load_state_dict(state)
        view = restored.optimizer_param_by_model_param[model2.weight]
        torch.testing.assert_close(view.to_local(), saved_main)
        torch.testing.assert_close(to_local_if_dtensor(leaf2.state[view]["momentum_buffer"]), saved_momentum)
        self.assertIs(restored.model_param_by_optimizer_param[view], model2.weight)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_eager_and_graph_updates_match_across_precisions(self):
        """
        description: Compare repeated updates and momentum with eager DTensor ownership.
        expectation: All six dtype/main-weight combinations produce identical weights and momentum.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for fp32_main in (False, True):
                with self.subTest(dtype=dtype, fp32_main=fp32_main):
                    graph_model, graph_leaf, graph = _optimizer(dtype, fp32_main)
                    layout = _layout((4, 3))
                    layout.set_tensor_meta((4, 3), (3, 1), dtype)
                    graph.bind({graph_model.weight: (layout, (2, 3))})
                    eager_model = nn.Module()
                    eager_model.weight = nn.Parameter(DTensor.from_local_with_layout(
                        torch.ones((2, 3), dtype=dtype), layout, shape=(4, 3)))
                    eager_model.weight.model_name = "weight"
                    eager_leaf = _LocalMomentum(eager_model.parameters(), lr=0.1)
                    eager = ChainedOptimizer(eager_model, {"muon": eager_leaf})
                    if fp32_main:
                        eager = Float16OptimizerWithFloat16Params(eager, eager_model)
                    graph_param = graph.optimizer_param_by_model_param[graph_model.weight]
                    eager_param = eager_leaf.param_groups[0]["params"][0]
                    for value in (0.125, -0.375, 0.25):
                        graph_model.weight.grad = torch.full_like(graph_model.weight, value)
                        eager_model.weight.grad = DTensor.from_local_with_layout(
                            torch.full((2, 3), value, dtype=dtype), layout, shape=(4, 3))
                        graph.step()
                        eager.step()
                        torch.testing.assert_close(graph_model.weight, eager_model.weight.to_local())
                        torch.testing.assert_close(to_local_if_dtensor(graph_param),
                                                   to_local_if_dtensor(eager_param))
                        torch.testing.assert_close(
                            to_local_if_dtensor(graph_leaf.state[graph_param]["momentum_buffer"]),
                            to_local_if_dtensor(eager_leaf.state[eager_param]["momentum_buffer"]))
                        graph.zero_grad()
                        eager.zero_grad()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_fp32_identity_fallback_consumes_main_grad(self):
        """
        description: Update an unchanged FP32 AdamW parameter supplied only through main_grad.
        expectation: The unchanged FP32 parameter updates from main_grad alone.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones((2, 3)))
        model.scale = nn.Parameter(torch.ones(2))
        for name, param in model.named_parameters():
            param.model_name = name
        leaf = _LocalMomentum([model.weight], lr=0.1)
        adam = torch.optim.AdamW([model.scale], lr=0.1, weight_decay=0.0, foreach=False)
        graph = GraphOptimizer(ChainedOptimizer(model, {"muon": leaf, "adamw": adam}), model)
        graph.bind({model.weight: (_layout((4, 3)), (2, 3))})
        model.scale.main_grad = torch.ones_like(model.scale)
        self.assertIsNone(model.scale.grad)
        graph.step()
        torch.testing.assert_close(model.scale, torch.full((2,), 0.9))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_checkpoint_validation_precedes_inner_restore(self):
        """
        description: Reject missing main weights without partially restoring leaf state.
        expectation: Invalid main-weight mappings raise before leaf state restoration.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model, _, optimizer = _optimizer(torch.bfloat16, fp32_main_params=True)
        optimizer.bind({model.weight: (_layout((4, 3)), (2, 3))})
        state = optimizer.state_dict()
        invalid_states = (
            (None, "must be a mapping"),
            ({}, "missing fp32_from_fp16_params"),
            ({FP32_MAIN_PARAM_STATE_KEY: {}}, "missing fp32 main parameters: weight"),
        )
        for mixed_state, message in invalid_states:
            with self.subTest(mixed_state=mixed_state):
                invalid = dict(state, **{MIXED_PRECISION_OPTIMIZER_STATE_KEY: mixed_state})
                with patch.object(optimizer.optimizer, "load_state_dict") as restore:
                    with self.assertRaisesRegex(RuntimeError, message):
                        optimizer.load_state_dict(invalid)
                    restore.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_legacy_checkpoint_reloads_main_weights_and_warns(self):
        """
        description: Retain eager's explicit precision warning when main weights are absent.
        expectation: Model values initialize main weights and a precision-loss warning is emitted.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model, _, optimizer = _optimizer(torch.bfloat16, fp32_main_params=True)
        optimizer.bind({model.weight: (_layout((4, 3)), (2, 3))})
        state = optimizer.state_dict()
        state.pop(MIXED_PRECISION_OPTIMIZER_STATE_KEY)
        with torch.no_grad():
            model.weight.fill_(0.5)
        with self.assertLogs("hyper_parallel.components.optim.mixed_precision_optimizer", level="WARNING") as logs:
            optimizer.load_state_dict(state)
        self.assertIn("cannot be recovered", logs.output[0])
        main = optimizer.optimizer_param_by_model_param[model.weight]
        torch.testing.assert_close(main.to_local(), torch.full((2, 3), 0.5))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_incompatible_dtensor_gradient_is_rejected_without_consuming_it(self):
        """
        description: Keep existing DTensor metadata validation instead of relabeling a bad gradient.
        expectation: Invalid logical shape raises and leaves the producer gradient intact.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        model, _, optimizer = _optimizer()
        optimizer.bind({model.weight: (_layout((4, 3)), (2, 3))})
        gradient = DTensor.from_local_with_layout(torch.ones(2, 3), _layout((8, 3)), shape=(8, 3))
        model.weight.main_grad = gradient
        with self.assertRaisesRegex(ValueError, "global shape must match"):
            optimizer.prepare_grads()
        self.assertIs(model.weight.main_grad, gradient)
        self.assertIsNone(optimizer.optimizer_param_by_model_param[model.weight].grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_zero_grad_preserves_inner_policy_and_resets_model_flags(self):
        """
        description: Clear producer state while preserving the requested inner gradient policy.
        expectation: Model state clears and inner gradients follow set_to_none.
        feature: Shared graph/eager optimizer parameter ownership.
        """
        for set_to_none in (False, True):
            with self.subTest(set_to_none=set_to_none):
                model, _, optimizer = _optimizer()
                optimizer.bind({model.weight: (_layout((4, 3)), (2, 3))})
                model.weight.main_grad = torch.ones_like(model.weight)
                model.weight.grad_added_to_main_grad = True
                optimizer.prepare_grads()
                param = optimizer.optimizer_param_by_model_param[model.weight]
                optimizer.zero_grad(set_to_none=set_to_none)
                self.assertIsNone(model.weight.grad)
                self.assertIsNone(model.weight.main_grad)
                self.assertFalse(model.weight.grad_added_to_main_grad)
                if set_to_none:
                    self.assertIsNone(param.grad)
                else:
                    torch.testing.assert_close(param.grad.to_local(), torch.zeros((2, 3)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_eager_checkpoint_uses_module_fqn_with_an_optimizer_alias(self):
        """
        description: Preserve the eager checkpoint key when optimizer metadata uses another name.
        expectation: Main weights use the module FQN and parameter-view mappings remain live.
        feature: Eager checkpoint compatibility after sharing runtime flows.
        """
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones((2, 3), dtype=torch.bfloat16))
        leaf = _LocalMomentum(model.parameters(), lr=0.1)
        chain = ChainedOptimizer(model, {"muon": leaf})
        model.weight.model_name = "optimizer_alias"
        eager = Float16OptimizerWithFloat16Params(chain, model)
        state = eager.state_dict()[MIXED_PRECISION_OPTIMIZER_STATE_KEY][FP32_MAIN_PARAM_STATE_KEY]
        self.assertEqual(set(state), {"weight"})
        main = eager.optimizer_param_by_model_param[model.weight]
        with chain._optimizer_parameter_view():
            self.assertIs(model.weight, main)


class TestOptimizerParameterLayouts(unittest.TestCase):
    """Matrix layouts must describe unpadded logical shapes and ceil chunks."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_compiler_preserves_storage_layout_and_removes_logical_padding(self):
        """
        description: Describe the logical optimizer shape while preserving the padded storage layout.
        expectation: Cached optimizer layout excludes padding without mutating storage metadata.
        feature: Shared graph/eager optimizer parameter ownership.
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


if __name__ == "__main__":
    unittest.main()
