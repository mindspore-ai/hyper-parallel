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
"""Unit tests for checkpoint-exclusion boundary options.

Covers the two switches an early (layer-wise) auxiliary backward needs:

* ``preserve_inputs`` keeps real forward inputs as the excluded region's saved
  tensors so its backward may run before the surrounding checkpoint replays.
* ``exclude_single_op`` moves the exclusion boundary down to the first
  custom-shard local kernel, so replay re-runs the DTensor expansion helpers
  while the kernel output is reused from cache.
"""
# Tests intentionally inspect private helper state to cover the boundary contract.
# pylint: disable=protected-access
import copy
import importlib
import os
import unittest
from collections.abc import Callable

import torch
from torch import nn
import torch.distributed as dist

from hyper_parallel.core.activation_memory.api import checkpoint_wrapper
from hyper_parallel.core.activation_memory.wrapper import (
    CheckpointExcludeWrapper,
    checkpoint_exclude_wrapper,
)
from hyper_parallel.distributed.activation_checkpoint import _wrap_excluded_targets
from tests.common.mark_utils import arg_mark


_checkpoint_exclude = importlib.import_module("hyper_parallel.core.activation_memory.checkpoint_exclude")


class _EarlyBackwardRegion(nn.Module):
    """Excluded region whose backward must run before the checkpoint replays."""

    def __init__(self) -> None:
        """Initialize the trainable gain and the execution counter."""
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Save an input view for backward and count real executions."""
        self.calls += 1
        # The view keeps an input alias alive for backward while the parameter
        # gives the early backward an observable gradient target.
        return (tensor.view(-1) * self.gain).square().sum()


class _EarlyBackwardBlock(nn.Module):
    """Checkpoint body with a main path and a detachable auxiliary branch."""

    def __init__(self, excluded: nn.Module) -> None:
        """Initialize the excluded region and the main-path scale."""
        super().__init__()
        self.excluded = excluded
        self.scale = nn.Parameter(torch.tensor(3.0))
        self.aux_value = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the auxiliary branch detached and the main path on top."""
        t = x * self.scale
        # Detached like the DSA indexer loss: the auxiliary branch must never
        # backpropagate into the main stem through an early backward.
        self.aux_value = self.excluded(t.detach())
        return (x * t).sum()


class TestPreserveInputsEarlyBackward(unittest.TestCase):
    """Feature: preserve_inputs enables backward before checkpoint replay."""

    def setUp(self) -> None:
        """Start each test with empty checkpoint-boundary caches."""
        _checkpoint_exclude._get_recompute_trigger.cache_clear()
        _checkpoint_exclude._get_recompute_boundary.cache_clear()
        torch.manual_seed(2026)

    def tearDown(self) -> None:
        """Release helper cache references after each test."""
        _checkpoint_exclude._get_recompute_trigger.cache_clear()
        _checkpoint_exclude._get_recompute_boundary.cache_clear()

    @staticmethod
    def _build(preserve_inputs: bool):
        """Return the checkpointed model, its excluded region, and a reference copy."""
        excluded = _EarlyBackwardRegion()
        block = _EarlyBackwardBlock(checkpoint_exclude_wrapper(excluded, preserve_inputs=preserve_inputs))
        model = checkpoint_wrapper(block)
        reference = copy.deepcopy(_EarlyBackwardBlock(_EarlyBackwardRegion()))
        reference.excluded.load_state_dict(excluded.state_dict())
        reference.scale.data.copy_(block.scale.data)
        return model, block, excluded, reference

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_early_backward_runs_before_replay(self):
        """
        Feature: preserve_inputs boundary for checkpoint exclusion.
        Description: Run the excluded region auxiliary backward before the surrounding checkpoint replays.
        Expectation: The early backward uses real saved inputs, matches the plain-graph gradients, and never forces a replay.
        """
        model, block, excluded, reference = self._build(preserve_inputs=True)
        x = torch.randn(6, requires_grad=True)

        output = model(x)
        aux = block.aux_value
        # Early backward: no outer replay has happened at this point.
        aux.backward()

        self.assertIsNotNone(excluded.gain.grad)
        # Real saved inputs mean the early backward never forced a replay.
        self.assertEqual(excluded.calls, 1)
        output.backward()

        ref_x = x.detach().clone().requires_grad_(True)
        ref_out = reference(ref_x)
        (ref_out + reference.aux_value).backward()

        self.assertTrue(torch.allclose(excluded.gain.grad, reference.excluded.gain.grad))
        self.assertTrue(torch.allclose(block.scale.grad, reference.scale.grad))
        self.assertTrue(torch.allclose(x.grad, ref_x.grad))
        # Region exclusion also keeps the region body out of the main replay.
        self.assertEqual(excluded.calls, 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_default_deferred_inputs_break_after_early_backward(self):
        """
        Feature: preserve_inputs boundary for checkpoint exclusion.
        Description: Trigger an early backward under the default deferred-alias boundary.
        Expectation: The hidden replay consumes the exclusion cache and the main backward fails with "No cached forward output".
        """
        model, block, excluded, _ = self._build(preserve_inputs=False)
        x = torch.randn(6, requires_grad=True)

        output = model(x)
        # The early backward forces one hidden replay of the outer region.
        block.aux_value.backward()
        self.assertEqual(excluded.calls, 1)
        with self.assertRaisesRegex(RuntimeError, "No cached forward output"):
            output.backward()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_preserve_inputs_full_backward_matches_reference(self):
        """
        Feature: preserve_inputs boundary for checkpoint exclusion.
        Description: Complete both the early auxiliary backward and the main backward.
        Expectation: All gradients match the plain non-excluded reference graph.
        """
        model, block, excluded, reference = self._build(preserve_inputs=True)
        x = torch.randn(6, requires_grad=True)

        output = model(x)
        block.aux_value.backward()
        output.backward()

        ref_x = x.detach().clone().requires_grad_(True)
        ref_out = reference(ref_x)
        (ref_out + reference.aux_value).backward()

        self.assertTrue(torch.allclose(excluded.gain.grad, reference.excluded.gain.grad))
        self.assertTrue(torch.allclose(block.scale.grad, reference.scale.grad))
        self.assertTrue(torch.allclose(x.grad, ref_x.grad))


class _DispatchingBlock(nn.Module):
    """Checkpoint body whose excluded callable dispatches a custom-shard matmul."""

    def __init__(self, excluded: nn.Module) -> None:
        """Initialize the excluded callable holder."""
        super().__init__()
        self.excluded = excluded
        self.calls = 0

    def forward(self, dx: torch.Tensor, dw: torch.Tensor) -> torch.Tensor:
        """Run the excluded dispatched region and count region executions."""
        self.calls += 1
        return self.excluded(dx, dw)


class TestExcludeSingleOpDispatch(unittest.TestCase):
    """Feature: exclude_single_op caches only the dispatched local kernel."""

    _PORT = 29861

    def setUp(self) -> None:
        """Start one gloo rank and clear the mesh caches."""
        _checkpoint_exclude._get_recompute_trigger.cache_clear()
        _checkpoint_exclude._get_recompute_boundary.cache_clear()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self._PORT)
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
        from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS
        from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP

        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        torch.manual_seed(2026)

    def tearDown(self) -> None:
        """Release meshes and the process group between tests."""
        from hyper_parallel.core.utils.communication import EXISTING_COMM_GROUPS
        from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP

        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()
        _checkpoint_exclude._get_recompute_trigger.cache_clear()
        _checkpoint_exclude._get_recompute_boundary.cache_clear()
        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def _make_distributed_inputs(mesh):
        """Create replicate DTensor operands for the dispatched matmul."""
        from hyper_parallel.core.dtensor.dtensor import DTensor
        from hyper_parallel.core.dtensor.placement_types import Replicate

        x = torch.randn(4, 8, requires_grad=True)
        w = torch.randn(8, 6, requires_grad=True)
        dx = DTensor.from_local(x, mesh, (Replicate(),))
        dw = DTensor.from_local(w, mesh, (Replicate(),))
        return x, w, dx, dw

    def _make_mesh(self):
        """Create a single-rank CPU mesh."""
        from hyper_parallel.core.dtensor.device_mesh import init_device_mesh

        return init_device_mesh(device_type="cpu", mesh_shape=(1,), mesh_dim_names=("dp",))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_local_kernel_runs_once_and_region_replays(self):
        """
        Feature: exclude_single_op caching only the dispatched local kernel.
        Description: Backpropagate through a checkpoint whose excluded callable dispatches a custom-shard matmul.
        Expectation: Replay re-runs the region expansion once while the dispatched kernel executes once, with reference gradients.
        """
        import hyper_parallel.core.shard._op_dispatch as op_dispatch

        mesh = self._make_mesh()
        x, w, dx, dw = self._make_distributed_inputs(mesh)
        region_calls = []
        kernel_executions = []
        real_runner = op_dispatch.checkpoint_exclude.run_excludable_local_kernel

        def counting_runner(
            op_impl: Callable, args: tuple, kwargs: dict
        ) -> torch.Tensor:
            """Count kernel executions and delegate to the real runner."""
            scope = _checkpoint_exclude._ACTIVE_LOCAL_EXCLUSION.get()
            executes = (
                scope is None
                or scope.consumed
                or not scope.recompute_state.is_recomputing
            )
            if executes:
                kernel_executions.append(1)
            return real_runner(op_impl, args, kwargs)

        op_dispatch.checkpoint_exclude.run_excludable_local_kernel = counting_runner
        try:

            def dispatched_region(
                dx_: torch.Tensor, dw_: torch.Tensor
            ) -> torch.Tensor:
                """Record the region call and dispatch the matmul."""
                region_calls.append(1)
                return torch.matmul(dx_, dw_)

            excluded = checkpoint_exclude_wrapper(dispatched_region, exclude_single_op=True)
            block = _DispatchingBlock(excluded)
            model = checkpoint_wrapper(block)

            y = model(dx, dw)
            y.to_local().sum().backward()
        finally:
            op_dispatch.checkpoint_exclude.run_excludable_local_kernel = real_runner

        # The region callable re-ran on replay while the kernel executed once.
        self.assertEqual(len(region_calls), 2)
        self.assertEqual(len(kernel_executions), 1)
        ref_x = x.detach().clone().requires_grad_(True)
        ref_w = w.detach().clone().requires_grad_(True)
        torch.matmul(ref_x, ref_w).sum().backward()
        self.assertTrue(torch.allclose(x.grad, ref_x.grad))
        self.assertTrue(torch.allclose(w.grad, ref_w.grad))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_exclude_single_op_requires_dispatched_kernel(self):
        """
        Feature: exclude_single_op caching only the dispatched local kernel.
        Description: Wrap a plain callable without custom-shard dispatch under exclude_single_op.
        Expectation: The forward fails fast reporting that it dispatched no custom-shard operator.
        """
        excluded = checkpoint_exclude_wrapper(lambda a, b: a * b + 1.0, exclude_single_op=True)
        block = _DispatchingBlock(excluded)
        model = checkpoint_wrapper(block)
        x = torch.randn(4, requires_grad=True)

        with self.assertRaisesRegex(RuntimeError, "dispatched no custom-shard operator"):
            model(x, x)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_local_op_backward_without_checkpoint_runs_plain(self):
        """
        Feature: exclude_single_op caching only the dispatched local kernel.
        Description: Call the exclusion wrapper outside any checkpoint.
        Expectation: The wrapper behaves as a transparent call and returns the matmul result.
        """
        mesh = self._make_mesh()
        _, _, dx, dw = self._make_distributed_inputs(mesh)
        excluded = checkpoint_exclude_wrapper(torch.matmul, exclude_single_op=True)

        y = excluded(dx, dw)
        self.assertTrue(torch.allclose(y.to_local(), torch.matmul(dx.to_local(), dw.to_local())))


class TestExcludeWrapperOptionWiring(unittest.TestCase):
    """Feature: declared exclusions can request preserve_inputs centrally."""

    @staticmethod
    def _build_model():
        """Return a two-level model and its FQN map with an excludable region."""
        outer = nn.Module()
        outer.mid = nn.Module()
        outer.mid.inner = _EarlyBackwardRegion()
        module_by_fqn = {
            "": outer,
            "mid": outer.mid,
            "mid.inner": outer.mid.inner,
        }
        return outer, module_by_fqn

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_wrap_excluded_targets_passes_wrapper_options(self):
        """
        Feature: declared exclusions can request preserve_inputs centrally.
        Description: Install exclusions through _wrap_excluded_targets with wrapper options.
        Expectation: The installed wrapper receives preserve_inputs set to True.
        """
        outer, module_by_fqn = self._build_model()

        count = _wrap_excluded_targets(module_by_fqn, [("mid.inner", outer.mid.inner)], {"preserve_inputs": True})

        self.assertEqual(count, 1)
        self.assertIsInstance(outer.mid.inner, CheckpointExcludeWrapper)
        self.assertTrue(outer.mid.inner.preserve_inputs)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_wrap_excluded_targets_defaults_to_deferred_inputs(self):
        """
        Feature: declared exclusions can request preserve_inputs centrally.
        Description: Install exclusions through _wrap_excluded_targets without options.
        Expectation: The installed wrapper keeps the default deferred-input boundary.
        """
        outer, module_by_fqn = self._build_model()

        count = _wrap_excluded_targets(module_by_fqn, [("mid.inner", outer.mid.inner)])

        self.assertEqual(count, 1)
        self.assertFalse(outer.mid.inner.preserve_inputs)


if __name__ == "__main__":
    unittest.main()
