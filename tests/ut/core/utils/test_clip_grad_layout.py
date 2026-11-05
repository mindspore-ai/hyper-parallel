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
"""Public clipping parity for local parameters carrying explicit final Layouts."""

from contextlib import contextmanager
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.utils.clip_grad import _clip_grads_with_norm_, _get_param_mesh_info, clip_grad_norm_


def _model(gradient=True):
    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(1, 2))
    if gradient:
        model.weight.grad = torch.tensor([[3., 4.]])
    return model


def _mesh(shape, names):
    return init_device_mesh("cpu", shape, mesh_dim_names=names,
                            rank_list=tuple(range(math.prod(shape))), init_backend=False)


@contextmanager
def _norm_collectives():
    groups = {"axis0": [0, 2], "axis1": [0, 1]}
    with patch.object(DeviceMesh, "get_group", autospec=True,
                      side_effect=lambda mesh, dim: f"axis{dim}"), \
            patch("hyper_parallel.core.utils.clip_grad.dist.get_process_group_ranks",
                  side_effect=lambda group: groups[group]), \
            patch("hyper_parallel.core.utils.clip_grad.dist.all_reduce",
                  side_effect=lambda value, **kwargs: value.mul_(2)) as reduce:
        yield reduce


class TestExplicitGradientLayouts(unittest.TestCase):
    """Clip the actual gradient objects with the same mesh-driven norm math."""

    def test_single_process_without_mesh(self):
        """Preserve the ordinary local clipping entry point."""
        model = _model()
        self.assertEqual(float(clip_grad_norm_(model, 2.)), 5.)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[3., 4.]]) * (2. / (5. + 1e-6)))

    def test_default_entry_preserves_duplicate_parameter_behavior(self):
        """Keep repeated inputs in the legacy path, including repeated scaling."""
        for kwargs in ({}, {"parameter_layouts": None}):
            with self.subTest(kwargs=kwargs):
                model = _model()
                norm = clip_grad_norm_([model.weight, model.weight], 1., foreach=False, **kwargs)
                expected_norm = torch.tensor(50.).sqrt()
                coefficient = 1. / (expected_norm + 1e-6)
                expected = (torch.tensor([[3., 4.]]) * coefficient) * coefficient
                torch.testing.assert_close(norm, expected_norm, rtol=0, atol=0)
                torch.testing.assert_close(model.weight.grad, expected, rtol=0, atol=0)

    def test_default_entry_preserves_frozen_gradients_and_foreach_selection(self):
        """Keep the existing frozen-gradient handling and caller's foreach choice."""
        for foreach in (None, False, True):
            with self.subTest(foreach=foreach):
                model = _model()
                model.frozen = nn.Parameter(torch.zeros(1), requires_grad=False)
                model.frozen.grad = torch.tensor([12.])
                with patch("hyper_parallel.core.utils.clip_grad._clip_grads_with_norm_",
                           wraps=_clip_grads_with_norm_) as clip:
                    norm = clip_grad_norm_(model, float("inf"), foreach=foreach)
                self.assertEqual(float(norm), 5.)
                self.assertTrue(torch.equal(model.frozen.grad, torch.tensor([12.])))
                self.assertIs(clip.call_args.args[3], foreach)
                torch.testing.assert_close(model.frozen.grad, torch.tensor([12.]), rtol=0, atol=0)

    def test_default_dtensor_lookup_uses_gradient_spec_before_parameter(self):
        """Read native placements/device_mesh without requiring explicit Layout metadata."""
        gradient = SimpleNamespace(placements=(Shard(0),), device_mesh=object())
        parameter = SimpleNamespace(main_grad=gradient, grad=None,
                                    placements=(Replicate(),), device_mesh=object())
        with patch("hyper_parallel.core.utils.clip_grad._is_dtensor",
                   side_effect=lambda value: value is gradient or value is parameter):
            mesh, shards, partials = _get_param_mesh_info(parameter)
            self.assertIs(mesh, gradient.device_mesh)
            self.assertEqual((shards, partials), ((0,), ()))
            parameter.main_grad = torch.ones(1)
            mesh, shards, partials = _get_param_mesh_info(parameter)
            self.assertIs(mesh, parameter.device_mesh)
            self.assertEqual((shards, partials), ((), ()))

    def test_local_clipping_preserves_parameter_and_gradient_objects(self):
        """Scale the optimizer's actual gradient without replacing its objects."""
        model = _model()
        parameter, gradient = model.weight, model.weight.grad
        mesh = _mesh((1,), ("fsdp_shard",))
        layout = GraphCompiler._build_parameter_layout(
            parameter, mesh, None, sharded=False, unsharded_shape=parameter.shape)
        norm = clip_grad_norm_(model, 2., parameter_layouts={parameter: layout})
        self.assertEqual(float(norm), 5.)
        self.assertIs(model.weight, parameter)
        self.assertIs(model.weight.grad, gradient)
        torch.testing.assert_close(gradient, torch.tensor([[3., 4.]]) * (2. / (5. + 1e-6)))
        torch.testing.assert_close(model.weight, torch.zeros(1, 2), rtol=0, atol=0)

    def test_tp_and_fsdp_shards_contribute_once(self):
        """Reduce the global norm over both source and FSDP shard axes."""
        mesh = _mesh((2, 2), ("fsdp_shard", "tp"))
        model = _model()
        layout = GraphCompiler._build_parameter_layout(
            model.weight, mesh["fsdp_shard"], ((Shard(0),), mesh["tp"]), sharded=True, unsharded_shape=(2, 2))
        self.assertIsInstance(layout.placements[0], StridedShard)
        self.assertEqual(layout.placements[0].split_factor, 2)
        self.assertEqual(layout.tensor_shape, (4, 2))
        with _norm_collectives() as reduce:
            norm = clip_grad_norm_(model, 5., parameter_layouts={model.weight: layout})
        self.assertEqual(float(norm), 10.)
        self.assertEqual([call.kwargs["group"] for call in reduce.call_args_list], ["axis0", "axis1"])
        torch.testing.assert_close(model.weight.grad, torch.tensor([[3., 4.]]) * (5. / (10. + 1e-6)))

    def test_replicated_scalar_and_shared_parameters_contribute_once(self):
        """Count each parameter identity once and avoid reducing replicated scalars."""
        mesh = _mesh((2, 2), ("fsdp_shard", "tp"))
        model = _model()
        model.scalar = nn.Parameter(torch.tensor(0.))
        model.scalar.grad = torch.tensor(2.)
        layouts = {param: GraphCompiler._build_parameter_layout(param, mesh["fsdp_shard"], ((Replicate(),), mesh["tp"]),
                   sharded=name == "weight", unsharded_shape=(2, 2) if name == "weight" else ())
                   for name, param in model.named_parameters()}
        with _norm_collectives() as reduce:
            norm = clip_grad_norm_([model.weight, model.scalar, model.weight], float("inf"),
                                   parameter_layouts=layouts)
        self.assertAlmostEqual(float(norm), math.sqrt(2 * 25 + 4), places=5)
        self.assertEqual([call.kwargs["group"] for call in reduce.call_args_list], ["axis0"])

    def test_expert_layout_retains_ep_when_edp_is_one(self):
        """Include source EP shards even when EDP requires no communication."""
        for edp in (1, 2):
            with self.subTest(edp=edp):
                mesh = _mesh((edp, 2), ("edp_shard", "ep"))
                model = _model()
                layout = GraphCompiler._build_parameter_layout(
                    model.weight, mesh["edp_shard"], ((Shard(0),), mesh["ep"]),
                    sharded=edp > 1, unsharded_shape=(edp, 2))
                self.assertTrue(layout.placements[1].is_shard(0))
                with _norm_collectives() as reduce:
                    norm = clip_grad_norm_(model, float("inf"), parameter_layouts={model.weight: layout})
                self.assertAlmostEqual(float(norm), 5 * math.sqrt(2 * edp), places=5)
                self.assertEqual(reduce.call_count, 1 if edp == 1 else 2)

    def test_missing_gradient_keeps_collective_order(self):
        """Enter the same norm collective when a rank has no local gradient."""
        mesh = _mesh((2,), ("fsdp_shard",))
        model = _model(gradient=False)
        layout = GraphCompiler._build_parameter_layout(
            model.weight, mesh, None, sharded=True, unsharded_shape=(2, 2))
        with _norm_collectives() as reduce:
            reduce.side_effect = lambda value, **kwargs: value.fill_(25.)
            with self.assertWarnsRegex(UserWarning, "no gradients"):
                norm = clip_grad_norm_(model, 1., parameter_layouts={model.weight: layout})
        self.assertEqual(float(norm), 5.)
        self.assertEqual(reduce.call_count, 1)
        self.assertIsNone(model.weight.grad)

    def test_fp32_main_gradient_frozen_parameter_and_layout_coverage(self):
        """Prefer main gradients, ignore frozen parameters and require full layouts."""
        mesh = _mesh((1,), ("fsdp_shard",))
        model = _model(gradient=False).to(torch.bfloat16)
        model.weight.main_grad = torch.tensor([[3., 4.]])
        model.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
        model.frozen.grad = torch.tensor([5., 12.])
        with self.assertRaisesRegex(ValueError, "cover every local parameter"):
            clip_grad_norm_(model, 1., parameter_layouts={})
        layout = GraphCompiler._build_parameter_layout(
            model.weight, mesh, None, sharded=False, unsharded_shape=(1, 2))
        self.assertEqual(float(clip_grad_norm_(model, 1., parameter_layouts={model.weight: layout})), 5.)
        torch.testing.assert_close(model.weight.main_grad, torch.tensor([[3., 4.]]) / (5. + 1e-6))
        self.assertIsNone(model.weight.grad)
        torch.testing.assert_close(model.frozen, torch.ones(2), rtol=0, atol=0)
        torch.testing.assert_close(model.frozen.grad, torch.tensor([5., 12.]), rtol=0, atol=0)

    def test_explicit_layout_matches_dtensor_path_and_rejects_conflicts(self):
        """Match real DTensor clipping and reject conflicting explicit metadata."""
        mesh = _mesh((2,), ("fsdp_shard",))
        model = _model()
        layout = GraphCompiler._build_parameter_layout(
            model.weight, mesh, None, sharded=True, unsharded_shape=(2, 2))
        local = model.weight.detach().requires_grad_(True)
        local.grad = model.weight.grad.clone()
        dtensor = DTensor.from_local_with_layout(local, layout)
        with _norm_collectives():
            eager_norm = clip_grad_norm_([dtensor], .1, foreach=False)
            graph_norm = clip_grad_norm_(model, .1, parameter_layouts={model.weight: layout})
        torch.testing.assert_close(graph_norm, eager_norm, rtol=0, atol=0)
        torch.testing.assert_close(model.weight.grad, local.grad, rtol=0, atol=0)
        conflict = GraphCompiler._build_parameter_layout(
            model.weight, mesh, None, sharded=False, unsharded_shape=(2, 2))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            clip_grad_norm_([dtensor], .1, parameter_layouts={dtensor: conflict})

    def test_padding_describes_actual_physical_storage(self):
        """Keep graph storage padding in the final layout shape."""
        mesh = _mesh((2,), ("fsdp_shard",))
        parameter = nn.Parameter(torch.zeros(2, 3))
        layout = GraphCompiler._build_parameter_layout(
            parameter, mesh, None, sharded=True, unsharded_shape=(3, 3))
        self.assertEqual(layout.tensor_shape, (4, 3))
        self.assertTrue(layout.placements[0].is_shard(0))
