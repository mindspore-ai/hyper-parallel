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
"""Unit tests for distributed clip_grad_norm_ norm aggregation.

These tests mock ``torch.distributed`` so they run on CPU without any
multi-card / HCCL setup, focusing on how the finite p-norm path buckets
grads by the *process group* they reduce over:

* grads on the **same** process group (even across different ``DeviceMesh``
  objects) must be pooled into ONE reduce -- bit-exact with FSDP2;
* grads on **different** process groups (TP+FSDP / expert parallel) must
  each reduce over their own group;
* replicate grads (empty shard dims) must not be reduced at all;
* a ``Partial`` grad's norm uses its group-reduced (already-global) value,
  not the raw local shard, with no further reduce in the ``()`` bucket;
* the bucketed norm reproduces upstream ``torch.nn.utils.get_total_norm``'s
  global value (a real-FSDP2-semantics oracle, minus the collective).
"""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import numpy as np  # pylint: disable=C0413
import torch  # pylint: disable=C0413
import torch.distributed as dist  # pylint: disable=C0413

from hyper_parallel.core.dtensor.placement_types import Partial  # pylint: disable=C0413
from hyper_parallel.platform.torch import clip_grad as clip_grad_mod  # pylint: disable=C0413
from tests.common.mark_utils import arg_mark  # pylint: disable=C0413


def _make_mesh(dim_to_group):
    """Build a mock DeviceMesh whose ``get_group(dim)`` returns a sentinel."""
    mesh = MagicMock()
    mesh.get_group.side_effect = lambda dim: dim_to_group[dim]
    return mesh


def _run_total_norm(grad_groups, all_grads, key_per_grad, mesh_cache,
                    group_world_size, group_ranks):
    """Call ``_get_total_norm`` with mocked collectives; return (norm, groups).

    ``group_world_size`` maps each sentinel group to the SUM factor applied
    by the mocked all-reduce (simulating SUM over identical shards).
    ``group_ranks`` maps each sentinel group to its global ranks tuple.
    """
    recorded_groups = []

    def fake_all_reduce(tensor: torch.Tensor, op: object = None,  # pylint: disable=W0613
                        group: object = None) -> None:
        """Record the group and simulate a SUM over identical shards."""
        recorded_groups.append(group)
        tensor.mul_(group_world_size[group])

    with patch.object(clip_grad_mod.dist, "all_reduce", side_effect=fake_all_reduce), \
         patch.object(clip_grad_mod.dist, "get_process_group_ranks",
                      side_effect=lambda group: list(group_ranks[group])):
        total_norm = clip_grad_mod._get_total_norm(  # pylint: disable=W0212
            grad_groups, 2.0, mesh_cache, torch.device("cpu"),
            all_grads, key_per_grad,
        )
    return total_norm, recorded_groups


class _FakeParam:
    """Minimal stand-in for a Parameter whose ``grad`` is a (mock) DTensor.

    Assigning a mock to ``torch.nn.Parameter.grad`` trips the C++ grad
    validation and segfaults, so the Partial-path test drives
    ``clip_grad_norm_`` with this plain object instead. Only the attributes
    the implementation reads (``grad`` / ``requires_grad`` / ``device``) are
    provided; the absence of ``main_grad`` makes ``_get_grad_obj`` fall back
    to ``grad``.
    """

    def __init__(self, grad: object, device: torch.device) -> None:
        """Store the (mock DTensor) grad object and the local device."""
        self.grad = grad
        self.requires_grad = True
        self.device = device


class TestClipGradPerGroupReduction(unittest.TestCase):
    """Finite-p norm buckets grads by the process group they reduce over."""

    def test_different_groups_reduce_separately(self):
        """Grads on different process groups each reduce over their own group.

        A regression that pools everything and reduces over only the first
        group would record one group and use the wrong world size.
        """
        torch.manual_seed(0)
        grad_a = torch.randn(8, 16)
        grad_b = torch.randn(8, 16)

        mesh = _make_mesh({0: "GROUP_DIM0", 1: "GROUP_DIM1"})
        mesh_cache = {1: mesh}
        grad_groups = {
            (1, (0,)): [grad_a],
            (1, (1,)): [grad_b],
        }
        all_grads = [grad_a, grad_b]
        key_per_grad = [(1, (0,)), (1, (1,))]

        total_norm, recorded = _run_total_norm(
            grad_groups, all_grads, key_per_grad, mesh_cache,
            group_world_size={"GROUP_DIM0": 4, "GROUP_DIM1": 2},
            group_ranks={"GROUP_DIM0": (0, 1, 2, 3), "GROUP_DIM1": (0, 1)},
        )

        self.assertEqual(
            sorted(recorded), ["GROUP_DIM0", "GROUP_DIM1"],
            f"expected reduction over both groups, got {recorded}",
        )
        norm_a_sq = grad_a.double().pow(2).sum().item()
        norm_b_sq = grad_b.double().pow(2).sum().item()
        expected = (4.0 * norm_a_sq + 2.0 * norm_b_sq) ** 0.5
        self.assertTrue(
            np.allclose(total_norm.item(), expected, rtol=1e-4, atol=1e-4),
            f"per-group norm mismatch: got={total_norm.item()}, expected={expected}",
        )

    def test_same_group_different_mesh_is_pooled(self):
        """Grads on different meshes sharing one DP group pool into one reduce.

        Regression guard: splitting them into per-group reduces changes the
        float accumulation order (1-ULP drift that compounds over steps and
        breaks FSDP2 bit-exactness). Same process group -> a single reduce.
        """
        torch.manual_seed(2)
        grad_a = torch.randn(8, 16)
        grad_b = torch.randn(8, 16)

        mesh_a = _make_mesh({0: "GA0"})
        mesh_b = _make_mesh({0: "GB0"})
        mesh_cache = {1: mesh_a, 2: mesh_b}
        grad_groups = {
            (1, (0,)): [grad_a],
            (2, (0,)): [grad_b],
        }
        all_grads = [grad_a, grad_b]
        key_per_grad = [(1, (0,)), (2, (0,))]

        total_norm, recorded = _run_total_norm(
            grad_groups, all_grads, key_per_grad, mesh_cache,
            # Both meshes' dim-0 group covers the SAME ranks -> one bucket.
            group_world_size={"GA0": 4, "GB0": 4},
            group_ranks={"GA0": (0, 1, 2, 3), "GB0": (0, 1, 2, 3)},
        )

        self.assertEqual(
            len(recorded), 1,
            f"same-group meshes must pool into ONE reduce, got {recorded}",
        )
        norm_a_sq = grad_a.double().pow(2).sum().item()
        norm_b_sq = grad_b.double().pow(2).sum().item()
        expected = (4.0 * (norm_a_sq + norm_b_sq)) ** 0.5
        self.assertTrue(
            np.allclose(total_norm.item(), expected, rtol=1e-4, atol=1e-4),
            f"pooled norm mismatch: got={total_norm.item()}, expected={expected}",
        )

    def test_replicate_group_is_not_reduced(self):
        """A replicate group (empty shard_dims) contributes locally, no reduce."""
        torch.manual_seed(1)
        grad_sharded = torch.randn(8, 16)
        grad_replicate = torch.randn(8, 16)

        mesh = _make_mesh({0: "GROUP_DIM0"})
        mesh_cache = {1: mesh}
        grad_groups = {
            (1, (0,)): [grad_sharded],
            (1, ()): [grad_replicate],
        }
        all_grads = [grad_sharded, grad_replicate]
        key_per_grad = [(1, (0,)), (1, ())]

        total_norm, recorded = _run_total_norm(
            grad_groups, all_grads, key_per_grad, mesh_cache,
            group_world_size={"GROUP_DIM0": 4},
            group_ranks={"GROUP_DIM0": (0, 1, 2, 3)},
        )

        self.assertEqual(
            recorded, ["GROUP_DIM0"],
            f"replicate group must not be all-reduced, got {recorded}",
        )
        norm_sharded_sq = grad_sharded.double().pow(2).sum().item()
        norm_replicate_sq = grad_replicate.double().pow(2).sum().item()
        expected = (4.0 * norm_sharded_sq + norm_replicate_sq) ** 0.5
        self.assertTrue(
            np.allclose(total_norm.item(), expected, rtol=1e-4, atol=1e-4),
            f"replicate norm mismatch: got={total_norm.item()}, expected={expected}",
        )

    def test_matches_upstream_get_total_norm(self):
        """hp's bucketed reduction reproduces the global p-norm oracle.

        The oracle is upstream ``torch.nn.utils.get_total_norm`` when the
        running torch exposes it (added in 2.6), else the mathematically
        identical norm of the concatenated full-gradient vector. A sharded
        param's global grad is the concatenation of ``ws`` identical shards;
        a replicate param is already global. Asserts hp equals the global
        norm, that the replicate group is never reduced, and that the result
        is NOT the naive over-count (which would all-reduce the replicate
        term too).
        """
        torch.manual_seed(3)
        ws = 4
        shard = torch.randn(2, 16)                  # this rank's FSDP shard
        full_sharded = torch.cat([shard] * ws, 0)   # ws shards -> global grad
        rep = torch.randn(8, 16)                     # replicate grad (global)

        oracle_grads = [full_sharded, rep]
        upstream_get_total_norm = getattr(torch.nn.utils, "get_total_norm", None)
        if upstream_get_total_norm is not None:
            oracle = upstream_get_total_norm(oracle_grads, 2.0)
        else:
            # Older torch lacks get_total_norm; the global p-norm equals the
            # norm of the concatenated full-gradient vector (identical value).
            oracle = torch.linalg.vector_norm(
                torch.cat([g.reshape(-1) for g in oracle_grads]), 2.0,
            )

        mesh = _make_mesh({0: "DP"})
        mesh_cache = {1: mesh}
        grad_groups = {
            (1, (0,)): [shard],
            (1, ()): [rep],
        }
        all_grads = [shard, rep]
        key_per_grad = [(1, (0,)), (1, ())]

        total_norm, recorded = _run_total_norm(
            grad_groups, all_grads, key_per_grad, mesh_cache,
            group_world_size={"DP": ws},
            group_ranks={"DP": (0, 1, 2, 3)},
        )

        self.assertEqual(
            recorded, ["DP"],
            f"replicate must not be reduced, got {recorded}",
        )
        self.assertTrue(
            np.allclose(total_norm.item(), oracle.item(), rtol=1e-4, atol=1e-4),
            f"hp norm must match upstream get_total_norm: "
            f"got={total_norm.item()}, oracle={oracle.item()}",
        )
        overcount = (
            ws * shard.double().pow(2).sum() + ws * rep.double().pow(2).sum()
        ).pow(0.5).item()
        self.assertFalse(
            np.allclose(total_norm.item(), overcount, rtol=1e-4, atol=1e-4),
            f"hp must avoid replicate over-count: "
            f"got={total_norm.item()}, overcount={overcount}",
        )


class TestClipGradPartialPath(unittest.TestCase):
    """Finite-p norm of a ``Partial`` grad must use the group-reduced value."""

    def test_partial_grad_uses_reduced_value_in_finite_p_norm(self):
        """A ``Partial('sum')`` grad's norm is ``||sum_g g_local||``.

        Regression: the finite-p path bucketed the RAW local grads and
        ignored the ``_coalesce_partial_reduce`` result, so a Partial grad's
        norm was the un-reduced local value ``||g_local||``. It must instead
        take the norm of the group-summed (already-global) gradient, issuing
        exactly ONE collective (the Partial value-reduce) with no further
        reduce in the empty ``()`` signature bucket.
        """
        torch.manual_seed(5)
        ws = 4
        g_local = torch.randn(8, 16)
        g_global = g_local * ws   # SUM over the Partial group of identical locals

        mesh = _make_mesh({0: "PG"})
        grad = MagicMock(spec=clip_grad_mod.DTensor)
        grad._local_tensor = g_local
        grad.placements = (Partial("sum"),)
        grad.device_mesh = mesh
        param = _FakeParam(grad, torch.device("cpu"))

        recorded = []

        def fake_all_reduce(tensor: torch.Tensor, op: object = None,  # pylint: disable=W0613
                            group: object = None) -> None:
            """Record the group and simulate a SUM over identical locals."""
            recorded.append(group)
            tensor.mul_(ws)

        with patch.object(clip_grad_mod.dist, "all_reduce", side_effect=fake_all_reduce), \
             patch.object(clip_grad_mod.dist, "get_process_group_ranks",
                          side_effect=lambda group: [0, 1, 2, 3]), \
             patch.object(clip_grad_mod.dist, "get_world_size",
                          side_effect=lambda group=None: ws):
            total_norm = clip_grad_mod.clip_grad_norm_(
                [param], max_norm=1e9, norm_type=2.0,
            )

        self.assertEqual(
            recorded, ["PG"],
            f"expected one Partial value-reduce only, got {recorded}",
        )
        expected = g_global.double().pow(2).sum().pow(0.5).item()
        raw_local = g_local.double().pow(2).sum().pow(0.5).item()
        self.assertTrue(
            np.allclose(total_norm.item(), expected, rtol=1e-4, atol=1e-4),
            f"Partial norm must use the reduced value: got={total_norm.item()}, "
            f"expected||sum gi||={expected}, raw_local={raw_local}",
        )
        self.assertFalse(
            np.allclose(total_norm.item(), raw_local, rtol=1e-4, atol=1e-4),
            f"Partial norm must NOT equal the un-reduced local norm: "
            f"got={total_norm.item()}, raw_local={raw_local}",
        )


class TestClipGradScalarReductionDevice(unittest.TestCase):
    """Norm-scalar reductions pick their device from the group's backend.

    A CPU scalar (FSDP CPU offload) may only be moved to an accelerator when
    the reduction group itself cannot reduce CPU tensors: moving it for a
    CPU-capable backend such as Gloo makes the collective fail with
    ``No backend type associated with device type <accelerator>``.
    """

    def _reduce_scalar(self, backend):
        """Reduce a scalar over a mocked group reporting *backend*.

        Returns the tensor handed to ``all_reduce`` and the accelerator mock.
        """
        scalar = torch.tensor(3.0)
        group = MagicMock(name="group")
        accelerator = MagicMock(return_value=torch.device("cpu"))
        with patch.object(clip_grad_mod.dist, "get_backend", return_value=backend), \
             patch.object(clip_grad_mod, "_accelerator_device", accelerator), \
             patch.object(clip_grad_mod.dist, "all_reduce") as all_reduce:
            clip_grad_mod._reduce_scalar_norm(  # pylint: disable=W0212
                scalar, clip_grad_mod.dist.ReduceOp.SUM, group,
            )
        return all_reduce.call_args[0][0], accelerator

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_cpu_capable_backend_keeps_scalar_on_cpu(self):
        """Reduce a CPU scalar over a Gloo group.

        Feature: backend-aware norm-scalar reduction for FSDP CPU offload.

        Description: Gloo accepts CPU tensors, so the scalar of a CPU-resident
        norm is reduced where it is.

        Expectation: ``all_reduce`` receives the original CPU tensor and no
        accelerator device is resolved.
        """
        reduced, accelerator = self._reduce_scalar("gloo")
        self.assertTrue(reduced.is_cpu, "Gloo must reduce the CPU scalar on the CPU")
        accelerator.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_composite_backend_with_cpu_sub_backend_keeps_scalar_on_cpu(self):
        """Reduce a CPU scalar over a group exposing both Gloo and NCCL.

        Feature: backend-aware norm-scalar reduction for FSDP CPU offload.

        Description: A multi-backend group routes a CPU tensor through its CPU
        sub-backend, so the accelerator sub-backend must not take over.

        Expectation: the scalar stays on the CPU and no accelerator device is
        resolved.
        """
        reduced, accelerator = self._reduce_scalar("cpu:gloo,cuda:nccl")
        self.assertTrue(reduced.is_cpu, "a CPU-capable sub-backend wins over the accelerator one")
        accelerator.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_nccl_backend_moves_scalar_to_cuda(self):
        """Reduce a CPU scalar over an NCCL group.

        Feature: backend-aware norm-scalar reduction for FSDP CPU offload.

        Description: NCCL cannot reduce CPU tensors, so the scalar has to be
        moved to the CUDA device of the local rank.

        Expectation: the accelerator device is resolved for ``cuda``.
        """
        self._reduce_scalar("nccl")[1].assert_called_once_with("cuda")

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_hccl_backend_moves_scalar_to_npu(self):
        """Reduce a CPU scalar over an HCCL group.

        Feature: backend-aware norm-scalar reduction for FSDP CPU offload.

        Description: HCCL cannot reduce CPU tensors, so the scalar has to be
        moved to the NPU of the local rank.

        Expectation: the accelerator device is resolved for ``npu``.
        """
        self._reduce_scalar("hccl")[1].assert_called_once_with("npu")

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_unknown_backend_keeps_scalar_on_cpu(self):
        """Reduce a CPU scalar over a group whose backend is outside the map.

        Feature: backend-aware norm-scalar reduction for FSDP CPU offload.

        Description: An unmapped backend must not be assumed to be an
        accelerator one.

        Expectation: the scalar stays on the CPU and no accelerator device is
        resolved.
        """
        reduced, accelerator = self._reduce_scalar("undefined")
        self.assertTrue(reduced.is_cpu, "an unmapped backend must not move the scalar")
        accelerator.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_unregistered_group_keeps_scalar_on_cpu(self):
        """Reduce a CPU scalar over a group outside the distributed world.

        Feature: backend-aware norm-scalar reduction for FSDP CPU offload.

        Description: ``dist.get_backend`` rejects a group that was never
        registered, and that rejection must not turn into a device move.

        Expectation: the scalar stays on the CPU and no accelerator device is
        resolved.
        """
        with patch.object(clip_grad_mod, "_accelerator_device") as accelerator, \
             patch.object(clip_grad_mod.dist, "all_reduce") as all_reduce:
            clip_grad_mod._reduce_scalar_norm(  # pylint: disable=W0212
                torch.tensor(3.0), clip_grad_mod.dist.ReduceOp.SUM,
                MagicMock(name="unregistered"),
            )
        self.assertTrue(all_reduce.call_args[0][0].is_cpu)
        accelerator.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_real_gloo_group_reduces_cpu_scalar(self):
        """Reduce a CPU scalar over a real single-rank Gloo group.

        Feature: backend-aware norm-scalar reduction for FSDP CPU offload.

        Description: Exercise the Gloo branch against a real process group
        instead of a mocked backend string.

        Expectation: the scalar is reduced in place and the result is the
        single-rank value.
        """
        if dist.is_initialized():
            self.skipTest("a process group is already initialized by another test")
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore(),
        )
        try:
            group = dist.new_group(backend="gloo")
            reduced = clip_grad_mod._reduce_scalar_norm(  # pylint: disable=W0212
                torch.tensor(3.0), dist.ReduceOp.SUM, group,
            )
        finally:
            dist.destroy_process_group()
        self.assertEqual(reduced.item(), 3.0)


if __name__ == "__main__":
    unittest.main()
