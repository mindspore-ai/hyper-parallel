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
"""Unit tests for exact dense-shard norm accounting in clip_grad."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.core.utils.clip_grad import (
    NormShardPlan,
    _check_dense_grad_presence,
    _effective_dense_grads,
    _norm_coverage,
    _norm_representative,
    build_norm_shard_plan,
    dense_grad_norm_sq_and_refs,
    scale_grad_refs_,
)


def _fake_gather(mapping: dict) -> Any:
    """Return an all_gather_object stub filling one entry per rank."""
    def gather(out: list, local: Any) -> None:
        """Fill every output slot from the rank-indexed mapping."""
        for index, slot in enumerate(out):
            out[index] = mapping.get(index, local)
    return gather


class TestNormCoverage(unittest.TestCase):
    """Global rectangle description for plain and sharded parameters."""

    def test_plain_parameter_covers_its_full_shape(self):
        """A non-DTensor parameter represents its entire global shape."""
        param = nn.Parameter(torch.zeros(2, 3))
        self.assertEqual(_norm_coverage(param), ((2, 3), ((0, 2), (0, 3))))

    def test_dtensor_layout_slice_must_match_local_shape(self):
        """The layout slice must describe the exact local tensor shape."""
        param = SimpleNamespace(shape=(2, 2), layout=SimpleNamespace(rank_list=[0, 1]),
                                _local_tensor=torch.zeros(2, 1))
        with patch("hyper_parallel.core.utils.clip_grad._is_dtensor", return_value=True), \
                patch("hyper_parallel.core.dtensor.layout.infer_slice_area_by_layout",
                      return_value=((0, 2), (0, 1))) as infer:
            self.assertEqual(_norm_coverage(param), ((2, 2), ((0, 2), (0, 1))))
            infer.assert_called_once()

    def test_dtensor_layout_without_current_rank_is_rejected(self):
        """A layout that lacks the current rank has no local coverage."""
        param = SimpleNamespace(shape=(2, 2), layout=SimpleNamespace(rank_list=[1]),
                                _local_tensor=torch.zeros(2, 2))
        with patch("hyper_parallel.core.utils.clip_grad._is_dtensor", return_value=True):
            with self.assertRaisesRegex(ValueError, "does not contain the current global rank"):
                _norm_coverage(param)

    def test_dtensor_layout_mismatched_local_shape_is_rejected(self):
        """A layout slice inconsistent with the local shape is rejected."""
        param = SimpleNamespace(shape=(2, 2), layout=SimpleNamespace(rank_list=[0]),
                                _local_tensor=torch.zeros(2, 2))
        with patch("hyper_parallel.core.utils.clip_grad._is_dtensor", return_value=True), \
                patch("hyper_parallel.core.dtensor.layout.infer_slice_area_by_layout",
                      return_value=((0, 2), (0, 1))):
            with self.assertRaisesRegex(ValueError, "cannot describe the exact local"):
                _norm_coverage(param)


class TestNormRepresentative(unittest.TestCase):
    """Shard coverage validation and representative election."""

    def test_disjoint_shards_pick_one_representative_per_copy(self):
        """Disjoint areas with full volume elect the lowest copy rank."""
        stage_data = {
            0: {"w": ((2, 2), ((0, 1), (0, 2)))},
            1: {"w": ((2, 2), ((1, 2), (0, 2)))},
            2: {"w": ((2, 2), ((1, 1), (0, 2)))},
            3: {"w": ((2, 2), ((1, 2), (0, 2)))},
        }
        is_rep, copies = _norm_representative(
            "w", (2, 2), stage_data, 1, ((1, 2), (0, 2)))
        self.assertTrue(is_rep)
        self.assertEqual(copies, (1, 3))
        is_rep, copies = _norm_representative(
            "w", (2, 2), stage_data, 3, ((1, 2), (0, 2)))
        self.assertFalse(is_rep)

    def test_global_shape_mismatch_is_rejected(self):
        """All members must agree on the parameter global shape."""
        stage_data = {
            0: {"w": ((2, 2), ((0, 2), (0, 2)))},
            1: {"w": ((4, 2), ((0, 4), (0, 2)))},
        }
        with self.assertRaisesRegex(ValueError, "global shape mismatch"):
            _norm_representative("w", (2, 2), stage_data, 0, ((0, 2), (0, 2)))

    def test_incomplete_coverage_is_rejected(self):
        """Shard volumes must add up to the global shape volume."""
        stage_data = {
            0: {"w": ((2, 2), ((0, 1), (0, 2)))},
            1: {"w": ((2, 2), ((0, 1), (0, 2)))},
        }
        with self.assertRaisesRegex(ValueError, "coverage is incomplete or duplicated"):
            _norm_representative("w", (2, 2), stage_data, 0, ((0, 1), (0, 2)))

    def test_overlapping_rectangles_are_rejected(self):
        """Nonempty shard rectangles must be pairwise disjoint."""
        stage_data = {
            0: {"w": ((2, 2), ((0, 1), (0, 2)))},
            1: {"w": ((2, 2), ((0, 2), (0, 1)))},
        }
        with self.assertRaisesRegex(ValueError, "rectangles overlap"):
            _norm_representative("w", (2, 2), stage_data, 0, ((0, 1), (0, 2)))


class TestBuildNormShardPlan(unittest.TestCase):
    """Plan construction with and without torch.distributed."""

    def test_single_process_plan_excludes_external_params(self):
        """Excluded parameters stay out of the plan on a single rank."""
        model = nn.Module()
        model.dense = nn.Parameter(torch.zeros(2, 2))
        model.external = nn.Parameter(torch.zeros(4, 4))
        plan = build_norm_shard_plan(model, [model.external], SimpleNamespace(device_mesh=None))
        self.assertIsInstance(plan, NormShardPlan)
        self.assertEqual(set(plan.representatives), {"dense"})
        self.assertEqual(plan.representatives, {"dense": True})
        self.assertEqual(plan.replica_ranks, {"dense": (0,)})

    def test_distributed_plan_gathers_stage_members(self):
        """All-gathered coverage elects one representative across ranks."""
        model = nn.Module()
        model.dense = nn.Parameter(torch.zeros(2, 2))
        local = {"dense": ((2, 2), ((0, 2), (0, 2)))}
        patches = (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "get_rank", return_value=0),
            patch.object(dist, "all_gather_object", side_effect=_fake_gather({1: local})),
        )
        mesh = SimpleNamespace(rank_list=[0, 1])
        with patches[0], patches[1], patches[2], patches[3]:
            plan = build_norm_shard_plan(model, [], SimpleNamespace(device_mesh=mesh))
        self.assertEqual(plan.replica_ranks, {"dense": (0, 1)})
        self.assertEqual(plan.representatives, {"dense": True})

    def test_different_stage_fqns_are_rejected(self):
        """Members of one PP stage must share identical dense FQNs."""
        model = nn.Module()
        model.dense = nn.Parameter(torch.zeros(2, 2))
        patches = (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "get_rank", return_value=0),
            patch.object(dist, "all_gather_object",
                         side_effect=_fake_gather({1: {"other": ((2, 2), ((0, 2), (0, 2)))}})),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaisesRegex(ValueError, "different dense FQNs"):
                build_norm_shard_plan(model, [], SimpleNamespace(device_mesh=None))


class TestEffectiveDenseGrads(unittest.TestCase):
    """Local gradient collection for plain parameters."""

    def test_plain_parameters_collect_local_grads(self):
        """Plain parameters contribute their grads with no partial reduce."""
        param = nn.Parameter(torch.zeros(2))
        param.grad = torch.ones(2)
        infos, reduced = _effective_dense_grads([("w", param)])
        self.assertEqual(len(infos), 1)
        self.assertIs(infos[0][1], param.grad)
        self.assertEqual(reduced, {})

    def test_meshed_parameters_cache_the_mesh_once(self):
        """Parameters on a device mesh register the mesh in the cache."""
        param = nn.Parameter(torch.zeros(2))
        param.grad = torch.ones(2)
        mesh = SimpleNamespace()
        with patch("hyper_parallel.core.utils.clip_grad._get_param_mesh_info",
                   return_value=(mesh, (0,), ())):
            infos, reduced = _effective_dense_grads([("w", param)])
        self.assertEqual(len(infos), 1)
        self.assertEqual(reduced, {})


class TestCheckDenseGradPresence(unittest.TestCase):
    """Replica gradient presence consistency checks."""

    def test_single_process_is_a_noop(self):
        """No consistency check runs without torch.distributed."""
        _check_dense_grad_presence({"w": True}, {"w": (0,)})

    def test_consistent_replicas_pass(self):
        """Replicas agreeing on gradient presence are accepted."""
        patches = (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "all_gather_object",
                         side_effect=_fake_gather({1: {"w": True}})),
        )
        with patches[0], patches[1], patches[2]:
            _check_dense_grad_presence({"w": True}, {"w": (0, 1)})

    def test_mismatched_replicas_are_rejected(self):
        """Replicas disagreeing on gradient presence raise an error."""
        patches = (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2),
            patch.object(dist, "all_gather_object",
                         side_effect=_fake_gather({1: {"w": False}})),
        )
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(ValueError, "presence differs across replicas"):
                _check_dense_grad_presence({"w": True}, {"w": (0, 1)})


class TestDenseGradNormSqAndRefs(unittest.TestCase):
    """Exact dense norm summation and gradient reference collection."""

    def test_only_representatives_contribute_to_the_norm(self):
        """Non-representative copies and excluded params are skipped."""
        model = nn.Module()
        model.kept = nn.Parameter(torch.zeros(2))
        model.copy = nn.Parameter(torch.zeros(2))
        model.nograd = nn.Parameter(torch.zeros(2))
        model.external = nn.Parameter(torch.zeros(2))
        model.kept.grad = torch.full((2,), 2.0)
        model.copy.grad = torch.ones(2)
        model.external.grad = torch.full((2,), 100.0)
        plan = NormShardPlan(
            representatives={"kept": True, "copy": False, "nograd": True},
            replica_ranks={"kept": (0,), "copy": (0,), "nograd": (0,)},
        )
        local_sq, refs = dense_grad_norm_sq_and_refs(
            model, [model.external], plan, SimpleNamespace(device_mesh=None))
        torch.testing.assert_close(local_sq, torch.tensor(8.0))
        self.assertEqual(len(refs), 2)

    def test_empty_stage_uses_the_mesh_device(self):
        """A stage without dense parameters returns a zero norm on the mesh device."""
        model = nn.Module()
        model.external = nn.Parameter(torch.zeros(2))
        plan = NormShardPlan(representatives={}, replica_ranks={})
        local_sq, refs = dense_grad_norm_sq_and_refs(
            model, [model.external], plan,
            SimpleNamespace(device_mesh=SimpleNamespace(device_type="cpu")))
        torch.testing.assert_close(local_sq, torch.tensor(0.0))
        self.assertEqual(refs, [])

    def test_mesh_axes_all_reduce_the_local_sum(self):
        """Every multi-rank mesh axis reduces the local norm square."""
        model = nn.Module()
        model.kept = nn.Parameter(torch.zeros(2))
        model.kept.grad = torch.ones(2)
        plan = NormShardPlan(representatives={"kept": True}, replica_ranks={"kept": (0,)})
        axis = SimpleNamespace(size=lambda: 2, get_group=lambda: "group")

        class _Mesh:
            """Minimal device mesh exposing one two-rank axis."""

            mesh_dim_names = ("dp",)
            device_type = "cpu"

            def __getitem__(self, name: str) -> Any:
                """Return the fake axis for its dimension name."""
                return axis

        def all_reduce(tensor: torch.Tensor, op: Any = None, group: Any = None) -> None:
            """Emulate a two-rank SUM all-reduce by doubling in place."""
            tensor.mul_(2)

        patches = (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=1),
            patch.object(dist, "all_gather_object",
                         side_effect=_fake_gather({})),
            patch.object(dist, "all_reduce", side_effect=all_reduce),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            local_sq, _ = dense_grad_norm_sq_and_refs(model, [], plan, SimpleNamespace(device_mesh=_Mesh()))
        torch.testing.assert_close(local_sq, torch.tensor(4.0))


class TestScaleGradRefs(unittest.TestCase):
    """In-place clipping of collected gradient references."""

    def test_refs_are_scaled_in_place(self):
        """Every referenced gradient is multiplied by the coefficient."""
        grads = [torch.full((2,), 2.0), torch.full((3,), 4.0)]
        scale_grad_refs_(grads, torch.tensor(0.5))
        torch.testing.assert_close(grads[0], torch.ones(2))
        torch.testing.assert_close(grads[1], torch.full((3,), 2.0))


if __name__ == "__main__":
    unittest.main()
