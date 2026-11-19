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
from unittest.mock import PropertyMock, patch

import torch
from torch import nn

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.optimizer import GraphOptimizer
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


class _LocalMomentum(torch.optim.SGD):
    """Consume local storage while retaining DTensor state, without collectives."""

    is_muon = True

    def reset_optimizer_parameters(self):
        """The unit test does not build distributed owner/communication plans."""

    @torch.no_grad()
    def step(self, closure=None):
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
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp_shard", "tp"),
                            rank_list=(0, 1, 2, 3), init_backend=False)
    layout = Layout.from_device_mesh(mesh)
    layout.set_placements(placements)
    layout.placement_to_tensor_map(len(shape))
    layout.set_tensor_meta(shape, tuple(torch.empty(shape).stride()), torch.float32)
    return layout


def _optimizer(dtype=torch.float32, fp32_main_params=False):
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones((2, 3), dtype=dtype))
    model.weight.model_name = "weight"
    leaf = _LocalMomentum(model.parameters(), lr=0.1)
    chain = ChainedOptimizer(model, {"muon": leaf})
    return model, leaf, GraphOptimizer(chain, model, fp32_main_params=fp32_main_params)


class TestGraphOptimizer(unittest.TestCase):
    """Binding must preserve matrix semantics without cloning FP32 weights."""

    def test_shared_storage_and_two_step_gradient_lifecycle(self):
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

    def test_strided_shard_retains_logical_shape(self):
        model, leaf, optimizer = _optimizer()
        layout = _layout((8, 3), (StridedShard(0, 2), Shard(0)))
        optimizer.bind({model.weight: (layout, (2, 3))})
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        view = optimizer.optimizer_param_by_model_param[model.weight]
        self.assertEqual(tuple(view.shape), (8, 3))
        self.assertEqual(tuple(view.placements), tuple(layout.placements))
        self.assertEqual(tuple(leaf.state[view]["momentum_buffer"].shape), (8, 3))

    def test_fp32_main_is_created_from_final_local_shard(self):
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

    def test_padding_is_excluded_and_empty_shards_are_retained(self):
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

    def test_changed_storage_or_layout_is_rejected(self):
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

    def test_binding_requires_an_uninitialized_optimizer(self):
        model, leaf, optimizer = _optimizer()
        leaf.state[model.weight]["momentum_buffer"] = torch.ones_like(model.weight)
        with self.assertRaisesRegex(RuntimeError, "before initializing optimizer state"):
            optimizer.bind({model.weight: (_layout((4, 3)), (2, 3))})

    def test_unsharded_graph_keeps_original_parameters(self):
        model, _, optimizer = _optimizer()
        optimizer.bind(None)
        self.assertIs(optimizer.optimizer_param_by_model_param[model.weight], model.weight)
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        torch.testing.assert_close(model.weight, torch.full((2, 3), 0.9))

    def test_adam_fallback_keeps_padded_local_shape_with_fp32_main(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones((2, 3), dtype=torch.bfloat16))
        model.scale = nn.Parameter(torch.ones((2,), dtype=torch.bfloat16))
        for name, param in model.named_parameters():
            param.model_name = name
        muon = _LocalMomentum([model.weight], lr=0.1)
        adam = torch.optim.AdamW([model.scale], lr=0.1, weight_decay=0.0)
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

    def test_checkpoint_round_trip_preserves_fqn_and_main_weights(self):
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


class TestOptimizerParameterLayouts(unittest.TestCase):
    """Matrix layouts must describe unpadded logical shapes and ceil chunks."""

    def test_compiler_preserves_storage_layout_and_removes_logical_padding(self):
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
