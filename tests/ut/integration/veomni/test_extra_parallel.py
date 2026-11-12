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
"""CPU checks of VeOmni ExtraParallel boundary translation."""

import math
import gc
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import weakref

import torch

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.integration.veomni import extra_parallel
from hyper_parallel.integration.veomni.extra_parallel import (
    _clipping_layout,
    bind_gradient_divisor,
    parameters_for_clipping,
)


class _ClippingSpec(SimpleNamespace):
    """Reject expensive submesh lookup when complete metadata is available."""

    @property
    def para_mesh(self):
        """Catch accidental use of the native dynamic submesh property."""
        raise AssertionError("Clipping must use the existing full mesh")


class TestExtraParallel(unittest.TestCase):
    """Keep divisor scaling equivalent and ordinary parameters unchanged."""

    def setUp(self):
        """Keep metadata caches isolated across test cases."""
        _clipping_layout.cache_clear()

    def tearDown(self):
        """Release mesh metadata created by this test."""
        _clipping_layout.cache_clear()

    @staticmethod
    def _expert_model(dtype=torch.float32):
        """Create real CPU tensor wrappers without initializing process groups."""
        shard_mesh = DeviceMesh("cpu", [0, 2], mesh_dim_names=("ep_fsdp",), _init_backend=False)
        full_mesh = DeviceMesh("cpu", [[0, 1], [2, 3]],
                               mesh_dim_names=("ep_fsdp", "ep"), _init_backend=False)
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(DTensor.from_local(
            torch.ones(2, 2, 3, dtype=dtype), shard_mesh, [Shard(1)], shape=(2, 4, 3)))
        module.weight.grad = DTensor.from_local(
            torch.ones(2, 2, 3, dtype=dtype), shard_mesh, [Shard(1)], shape=(2, 4, 3))
        setattr(module, "_fqn2spec_info", {"weight": SimpleNamespace(
            persistent_fsdp_shard_dim=None, placement=Shard(0),
            para_mesh=SimpleNamespace(size=lambda: 2), para_fsdp_mesh=full_mesh,
            fsdp_shard_dim=1)})
        return module

    def test_expert_views_reuse_only_layout_metadata(self):
        """Repeated calls reuse layouts while exposing fresh tensor wrappers."""
        module = self._expert_model()
        with patch.object(extra_parallel, "to_hyper_mesh", wraps=extra_parallel.to_hyper_mesh) as conversion:
            first = parameters_for_clipping(module)[0]
            second = parameters_for_clipping(module)[0]
        conversion.assert_called_once()
        self.assertIs(first.layout, second.layout)
        self.assertIsNot(first, second)
        self.assertIsNot(first.grad, second.grad)
        self.assertEqual(tuple(first.shape), (4, 4, 3))
        self.assertEqual(first.placements, (Shard(1), Shard(0)))
        self.assertEqual(first.to_local().data_ptr(), module.weight.to_local().data_ptr())
        self.assertEqual(second.grad.to_local().data_ptr(), module.weight.grad.to_local().data_ptr())

    def test_expert_views_refresh_and_clear_gradients(self):
        """Replacing or clearing a gradient never reuses an earlier gradient."""
        module = self._expert_model()
        first = parameters_for_clipping(module)[0]
        replacement = torch.full_like(module.weight.grad.to_local(), 3)
        module.weight.grad = DTensor.from_local(replacement, module.weight.device_mesh,
                                                module.weight.placements, shape=module.weight.shape)
        second = parameters_for_clipping(module)[0]
        self.assertEqual(second.grad.to_local().data_ptr(), replacement.data_ptr())
        self.assertNotEqual(first.grad.to_local().data_ptr(), replacement.data_ptr())
        module.weight.grad = None
        self.assertIsNone(parameters_for_clipping(module)[0].grad)

    def test_expert_clipping_traverses_model_once(self):
        """Planned EP parameters need only one complete named-parameter walk."""
        module = self._expert_model()
        with patch.object(module, "named_parameters", wraps=module.named_parameters) as named:
            views = parameters_for_clipping(module)
        named.assert_called_once_with()
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].grad.to_local().data_ptr(), module.weight.grad.to_local().data_ptr())

    def test_expert_views_do_not_construct_submeshes(self):
        """Read the EP axis directly for unequal FSDP/EP and HSDP meshes."""
        for shape, names, placements in (
            ((2, 4), ("ep_fsdp", "ep"), (Shard(1), Shard(0))),
            ((2, 2, 4), ("replicate", "ep_fsdp", "ep"), (Replicate(), Shard(1), Shard(0))),
        ):
            with self.subTest(mesh_shape=shape):
                module = self._expert_model()
                specification = vars(module)["_fqn2spec_info"]["weight"]
                values = {key: value for key, value in vars(specification).items() if key != "para_mesh"}
                values["para_fsdp_mesh"] = DeviceMesh(
                    "cpu", torch.arange(math.prod(shape)).reshape(shape),
                    mesh_dim_names=names, _init_backend=False)
                setattr(module, "_fqn2spec_info", {"weight": _ClippingSpec(**values)})
                view = parameters_for_clipping(module)[0]
                self.assertEqual(tuple(view.shape), (8, 4, 3))
                self.assertEqual(view.placements, placements)
                self.assertEqual(view.grad.to_local().data_ptr(), module.weight.grad.to_local().data_ptr())

    def test_clip_view_sets_autograd_on_local_storage(self):
        """Preparing metadata does not dispatch a distributed requires-grad op."""
        module = self._expert_model()
        with patch.object(DTensor, "requires_grad_", side_effect=AssertionError("Unexpected distributed dispatch")):
            view = parameters_for_clipping(module)[0]
        self.assertTrue(view.requires_grad)
        self.assertTrue(module.weight.requires_grad)

    def test_changed_shape_does_not_reuse_old_layout(self):
        """A new logical shape selects new metadata rather than a stale view."""
        module = self._expert_model()
        first = parameters_for_clipping(module)[0]
        original = module.weight
        module.weight = torch.nn.Parameter(DTensor.from_local(
            torch.ones(2, 2, 5), original.device_mesh, original.placements, shape=(2, 4, 5)))
        second = parameters_for_clipping(module)[0]
        self.assertEqual(tuple(second.shape), (4, 4, 5))
        self.assertIsNot(first.layout, second.layout)

    def test_expert_main_gradient_preserves_its_dtype(self):
        """FP32 main gradients keep distinct metadata from BF16 parameters."""
        module = self._expert_model(torch.bfloat16)
        gradient = torch.full((2, 2, 3), 5.0)
        module.weight.main_grad = gradient
        view = parameters_for_clipping(module)[0]
        self.assertEqual(view.dtype, torch.bfloat16)
        self.assertEqual(view.main_grad.dtype, torch.float32)
        self.assertEqual(view.main_grad.to_local().data_ptr(), gradient.data_ptr())
        self.assertIsNone(view.grad)
        self.assertIsNot(view.layout, view.main_grad.layout)

    def test_clipping_cache_does_not_retain_gradients(self):
        """Only layouts survive after temporary clipping views are released."""
        module = self._expert_model()
        local = module.weight.grad.to_local()
        reference = weakref.ref(local)
        views = parameters_for_clipping(module)
        module.weight.grad = None
        del local, views
        gc.collect()
        self.assertIsNone(reference())

    def test_replaced_parameter_uses_current_storage(self):
        """Parameter replacement does not require tensor-cache invalidation."""
        module = self._expert_model()
        original = module.weight
        first = parameters_for_clipping(module)[0]
        module.weight = torch.nn.Parameter(DTensor.from_local(
            torch.full_like(original.to_local(), 7), original.device_mesh,
            original.placements, shape=original.shape))
        second = parameters_for_clipping(module)[0]
        self.assertIs(first.layout, second.layout)
        self.assertNotEqual(first.to_local().data_ptr(), second.to_local().data_ptr())
        self.assertEqual(second.to_local().data_ptr(), module.weight.to_local().data_ptr())

    def test_gradient_divisor_uses_native_scaling(self):
        """Convert sum/divisor without changing the reduction implementation."""
        module = torch.nn.Module()
        module.set_gradient_scaling_factor = MagicMock()
        bind_gradient_divisor(module, 2)
        module.set_reduce_scatter_divide_factor(4)
        module.set_gradient_scaling_factor.assert_called_once_with(0.5)
        module.set_gradient_divide_factor(8)
        module.set_gradient_scaling_factor.assert_called_with(0.25)
        self.assertFalse(hasattr(torch.nn.Module(), "set_gradient_divide_factor"))

    def test_invalid_divisor_does_not_change_scaling(self):
        """Reject invalid divisors before touching the Hyper module state."""
        module = torch.nn.Module()
        module.set_gradient_scaling_factor = MagicMock()
        bind_gradient_divisor(module, 2)
        for factor in (0, -1, True, math.inf, math.nan, "4"):
            with (
                self.subTest(factor=factor),
                self.assertRaisesRegex(ValueError, "finite positive"),
            ):
                module.set_gradient_divide_factor(factor)
        module.set_gradient_scaling_factor.assert_not_called()

    def test_plain_parameters_retain_identity(self):
        """Ordinary models need neither layout wrappers nor gradient copies."""
        module = torch.nn.Linear(2, 2)
        original = list(module.parameters())
        actual = parameters_for_clipping(module)
        self.assertEqual(len(original), len(actual))
        for expected, parameter in zip(original, actual):
            self.assertIs(expected, parameter)
