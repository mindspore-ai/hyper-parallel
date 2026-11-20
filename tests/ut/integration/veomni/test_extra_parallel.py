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
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch.distributed.tensor import DTensor as TorchDTensor

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.integration.veomni.extra_parallel import (
    bind_gradient_divisor,
    native_clipping_view,
)


class TestExtraParallel(unittest.TestCase):
    """Check native clipping views and gradient-divisor translation."""

    def test_plain_fsdp_clip_uses_torch_gradient_views_without_ep_groups(self):
        """EP=1 must reach the same native norm branch as ordinary Torch FSDP."""
        model = self._expert_model()
        delattr(model, "_fqn2spec_info")
        gradient = model.weight.grad
        converted = MagicMock(spec=TorchDTensor)
        with patch("hyper_parallel.integration.veomni.extra_parallel.to_torch_view",
                   return_value=converted) as convert:
            proxy = native_clipping_view(model)
        convert.assert_called_once_with(gradient, mesh_cache={}, detached_spec_cache={})
        self.assertFalse(hasattr(proxy, "_extra_parallel_param_groups"))
        self.assertIs(next(proxy.parameters()).grad, converted)

    def test_plain_fsdp_metadata_caches_do_not_survive_the_clip_call(self):
        """Scope metadata reuse to one invocation instead of retaining old gradients."""
        model = self._expert_model()
        delattr(model, "_fqn2spec_info")
        with patch("hyper_parallel.integration.veomni.extra_parallel.to_torch_view",
                   return_value=MagicMock(spec=TorchDTensor)) as convert:
            native_clipping_view(model)
            native_clipping_view(model)
        for name in ("mesh_cache", "detached_spec_cache"):
            first, second = [call.kwargs[name] for call in convert.call_args_list]
            self.assertIsNot(first, second)

    def test_plain_fsdp_clip_preserves_missing_gradients(self):
        """Unused parameters must not enter native Torch's norm list as extra zeros."""
        model = torch.nn.Linear(2, 2)
        proxy = native_clipping_view(model)
        self.assertTrue(all(view.grad is None for view in proxy.parameters()))
        self.assertFalse(hasattr(proxy, "_extra_parallel_param_groups"))

    def test_cpu_offload_clip_uses_native_offload_branch(self):
        """CPU gradients keep VeOmni's explicit shard reduction and host scaling."""
        model = torch.nn.Linear(2, 2)
        setattr(model, "_fsdp_cpu_offload_enabled", True)
        proxy = native_clipping_view(model)
        self.assertTrue(getattr(proxy, "_fsdp_cpu_offload_enabled", False))
        self.assertFalse(hasattr(proxy, "_extra_parallel_param_groups"))

    def test_native_clip_view_shares_live_gradient_storage(self):
        """VeOmni scaling mutates the original Hyper gradient, not a copy."""
        model = self._expert_model()
        proxy = native_clipping_view(model)
        view = next(proxy.parameters())
        gradient = model.weight.grad.to_local()
        self.assertEqual(view.grad.data_ptr(), gradient.data_ptr())
        view.grad.mul_(0.5)
        self.assertTrue(torch.equal(gradient, torch.full_like(gradient, 0.5)))
        self.assertEqual(getattr(proxy, "_extra_parallel_param_groups")["ep"], [view])

    def test_native_clip_view_prefers_fp32_main_gradient(self):
        """A main gradient is clipped without changing the parameter or .grad."""
        model = self._expert_model()
        original = model.weight.grad
        model.weight.main_grad = model.weight.grad.detach().clone()
        proxy = native_clipping_view(model)
        view = next(proxy.parameters())
        self.assertEqual(view.grad.data_ptr(), model.weight.main_grad.to_local().data_ptr())
        self.assertIs(model.weight.grad, original)

    def test_native_clip_missing_gradient_keeps_collective_membership(self):
        """An unused trainable shard contributes zero without allocating a full gradient."""
        model = self._expert_model()
        model.weight.grad = None
        proxy = native_clipping_view(model)
        view = next(proxy.parameters())
        self.assertEqual(view.grad.numel(), 1)
        self.assertEqual(float(view.grad), 0.0)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(getattr(proxy, "_extra_parallel_param_groups")["ep"], [view])

    def test_native_clip_preserves_replicated_and_persistent_ownership(self):
        """Shared LoRA skips EP summation while persistent experts use the flat group."""
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                model = self._expert_model()
                specification = getattr(model, "_fqn2spec_info")["weight"]
                specification.placement = Replicate()
                specification.para_name = "ep"
                specification.persistent_fsdp_shard_dim = 0 if persistent else None
                proxy = native_clipping_view(model)
                view = next(proxy.parameters())
                self.assertEqual(getattr(proxy, "_extra_parallel_param_groups")["ep"], [view])
                self.assertEqual(getattr(proxy, "_persistent_extra_parallel_param_ids"),
                                 {id(view)} if persistent else set())
                self.assertEqual(getattr(proxy, "_ep_replicated_lora_param_ids"),
                                 set() if persistent else {id(view)})

    def test_native_clip_empty_gradient_contributes_zero(self):
        """An empty shard contributes zero to infinity norm without leaving its group."""
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.empty(0, 3))
        model.weight.grad = torch.empty_like(model.weight)
        setattr(model, "_fsdp_cpu_offload_enabled", True)

        proxy = native_clipping_view(model)
        view = next(proxy.parameters())

        self.assertEqual(float(torch.max(torch.abs(view.grad))), 0.0)
        self.assertEqual(model.weight.grad.numel(), 0)
        self.assertTrue(getattr(proxy, "_fsdp_cpu_offload_enabled", False))

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
