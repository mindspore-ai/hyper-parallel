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
"""Unit tests for graph-mode optimizer-state resharding helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.compile.checkpoint_metadata import (
    shard_chunk_metadata,
    trim_padded_dim0,
)
from hyper_parallel.compile.compiler import GraphCompiler
from hyper_parallel.compile.optimizer_state import (
    iter_optimizer_leaves,
    reshard_state_tensor,
)
from hyper_parallel.core.distributed_checkpoint.metadata import CHUNK_INFO
from hyper_parallel.core.dtensor.placement_types import Shard, StridedShard


def _layout(mesh_size: int = 4, ndim: int = 2) -> SimpleNamespace:
    """Build the layout surface ``shard_chunk_metadata`` consumes."""
    return SimpleNamespace(
        mesh_shape=(mesh_size,),
        tensor_map=(0, *(tuple(-1 for _ in range(ndim - 1)))),
        rank_list=tuple(range(mesh_size)),
        uneven_shard_mesh_dims=(),
        placements=(Shard(0),),
    )


def _single_rank_layout(tensor_shape) -> SimpleNamespace:
    """Build a one-rank layout whose tensor shape carries the padded extent."""
    return SimpleNamespace(
        mesh_shape=(1,),
        tensor_map=(0, -1),
        tensor_shape=tensor_shape,
        rank_list=(0,),
        uneven_shard_mesh_dims=(),
        placements=(Shard(0),),
    )


class _TiedModel(nn.Module):
    """Two parameter FQNs sharing one tensor, as tied embeddings do."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 2, bias=False)
        self.second.weight = self.first.weight


class _StubCompiler(GraphCompiler):
    """GraphCompiler with a pre-built layout map and no live graph passes."""

    def __init__(self, model, layouts, parameter_shapes) -> None:
        super().__init__(model, lambda **_kwargs: None)
        self._stub_layouts = layouts
        self._parameter_shapes = parameter_shapes
        self._joint_graph = SimpleNamespace(
            graph_module=SimpleNamespace(fsdp_padded_params={}),
        )

    @property
    def parameter_layouts(self) -> dict:
        """Return the injected parameter layouts."""
        return self._stub_layouts

    def attach_shard_metadata(self, *, strict_strided: bool = True) -> None:
        """No-op: the injected layouts already describe live storage."""
        return None


class _AttachCompiler(GraphCompiler):
    """GraphCompiler whose real ``attach_shard_metadata`` guard runs."""

    def __init__(self, model, layouts) -> None:
        super().__init__(model, lambda **_kwargs: None)
        self._stub_layouts = layouts
        self._joint_graph = SimpleNamespace(
            graph_module=SimpleNamespace(fsdp_padded_params={}),
        )

    @property
    def parameter_layouts(self) -> dict:
        """Return the injected parameter layouts."""
        return self._stub_layouts


class TestOptimizerStateReshard(unittest.TestCase):
    """Restored moments follow the live FSDP padded chunk geometry."""

    def test_padded_shard_matches_live_parameter(self) -> None:
        """A non-divisible parameter is padded before the equal-size chunk."""
        tensor = torch.arange(20, dtype=torch.float32).reshape(10, 2)

        shard = reshard_state_tensor(
            tensor, (10, 2), shard_index=2, degree=4, n=10, n_pad=12
        )
        padded_shard = reshard_state_tensor(
            tensor, (10, 2), shard_index=3, degree=4, n=10, n_pad=12
        )

        self.assertEqual(tuple(shard.shape), (3, 2))
        self.assertTrue(torch.equal(shard, tensor[6:9]))
        self.assertTrue(torch.equal(padded_shard[0], tensor[9]))
        self.assertTrue(torch.equal(padded_shard[1:], torch.zeros(2, 2)))

    def test_divisible_shard_is_a_plain_chunk(self) -> None:
        """A divisible parameter needs no padding."""
        tensor = torch.arange(16, dtype=torch.float32).reshape(8, 2)

        shard = reshard_state_tensor(
            tensor, (8, 2), shard_index=1, degree=4, n=8, n_pad=8
        )

        self.assertTrue(torch.equal(shard, tensor[2:4]))

    def test_fully_padded_shard_is_zero_filled(self) -> None:
        """A rank covering only padding rows yields an all-zero local shard."""
        tensor = torch.arange(2, dtype=torch.float32).reshape(1, 2)

        shard = reshard_state_tensor(
            tensor, (1, 2), shard_index=3, degree=4, n=1, n_pad=4
        )

        self.assertEqual(tuple(shard.shape), (1, 2))
        self.assertTrue(torch.equal(shard, torch.zeros(1, 2)))

    def test_reshard_before_compile_is_rejected(self) -> None:
        """Resharding without a completed compile must fail loudly."""
        model = nn.Linear(2, 2, bias=False)
        parameter = model.weight
        compiler = _StubCompiler(model, {parameter: SimpleNamespace(placements=())}, {})
        compiler._joint_graph = None
        optimizer = SimpleNamespace(state={parameter: {"exp_avg": torch.zeros(2, 2)}})

        with self.assertRaisesRegex(RuntimeError, "requires a completed compile"):
            compiler.reshard_optimizer_state(optimizer)

    def test_other_shapes_are_left_alone(self) -> None:
        """Scalars and already-sharded states are not touched."""
        self.assertIsNone(
            reshard_state_tensor(
                torch.zeros(()), (), shard_index=0, degree=2, n=0, n_pad=0
            )
        )
        self.assertIsNone(
            reshard_state_tensor(
                torch.zeros(3, 2), (10, 2), shard_index=0, degree=4, n=10, n_pad=12
            )
        )

    def test_padded_global_shape_is_trimmed(self) -> None:
        """Checkpoint metadata stores the true dim-0 size, not the padded size."""
        self.assertEqual(
            trim_padded_dim0((12, 4), n=10, n_pad=12),
            (10, 4),
        )
        self.assertEqual(
            trim_padded_dim0((24, 4), n=10, n_pad=12),
            (20, 4),
        )
        self.assertIsNone(trim_padded_dim0((12, 4), n=12, n_pad=12))

    def test_chained_optimizer_leaves_are_visited(self) -> None:
        """Chained optimizers expose their leaves to the reshard walk."""
        first = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))])
        second = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))])
        chained = type("Chained", (), {"chained_optimizers": [first, second]})()

        self.assertEqual(list(iter_optimizer_leaves(chained)), [first, second])


class TestPaddedChunkMetadata(unittest.TestCase):
    """Padded chunks follow local storage but never describe rows past the tensor."""

    def test_non_padded_chunk_is_unchanged(self) -> None:
        """A divisible parameter keeps the plain layout slice."""
        global_shape, chunk = shard_chunk_metadata(_layout(), 1, (8, 2))

        self.assertEqual(global_shape, (8, 2))
        self.assertEqual(chunk.offsets, (2, 0))
        self.assertEqual(chunk.sizes, (2, 2))

    def test_padded_middle_rank_stays_aligned(self) -> None:
        """A middle rank covers its full equal-size padded shard."""
        global_shape, chunk = shard_chunk_metadata(_layout(), 2, (12, 2), (10, 12))

        self.assertEqual(global_shape, (10, 2))
        self.assertEqual(chunk.offsets, (6, 0))
        self.assertEqual(chunk.sizes, (3, 2))

    def test_padded_last_rank_is_clipped_to_true_size(self) -> None:
        """The last rank's chunk is clipped at the true global extent."""
        global_shape, chunk = shard_chunk_metadata(_layout(), 3, (12, 2), (10, 12))

        self.assertEqual(global_shape, (10, 2))
        self.assertEqual(chunk.offsets, (9, 0))
        self.assertEqual(chunk.sizes, (1, 2))

    def test_extra_global_factor_is_trimmed(self) -> None:
        """A source shard multiplying dim 0 is trimmed without dropping padding."""
        global_shape, chunk = shard_chunk_metadata(_layout(), 3, (24, 2), (10, 12))

        self.assertEqual(global_shape, (20, 2))
        self.assertEqual(chunk.offsets, (18, 0))
        self.assertEqual(chunk.sizes, (2, 2))

    def test_fully_padded_rank_has_zero_size_chunk(self) -> None:
        """A rank whose rows are all padding is clamped to a zero-size chunk."""
        global_shape, chunk = shard_chunk_metadata(_layout(), 3, (4, 2), (1, 4))

        self.assertEqual(global_shape, (1, 2))
        self.assertEqual(chunk.offsets, (1, 0))
        self.assertEqual(chunk.sizes, (0, 2))


class TestPaddedRankResume(unittest.TestCase):
    """Fully-padded ranks map dense state through the physical FSDP layout."""

    def test_fully_padded_rank_is_resharded_to_zeros(self) -> None:
        """A rank owning only padding rows must not be read as unmappable.

        Feature: Graph-mode optimizer-state resharding.
        Description: With n=5 and degree=4 the padded extent is 8, so rank 3
            stores only padding rows (physical ``[6, 8)``). Its DCP chunk is
            clipped to the true ``[5, 3]`` extent, reporting a zero-size chunk
            at ``offset=5``, which is not a multiple of the 2-row shard.
        Expectation: The shard index comes from the unpadded FSDP layout
            (offset 6 -> shard index 3), so rank 3 yields an all-zero local
            shard instead of raising.
        """
        model = nn.Linear(3, 2, bias=False)
        parameter = model.weight
        layout = SimpleNamespace(
            mesh_shape=(4,),
            tensor_map=(0, -1),
            tensor_shape=(8, 3),
            rank_list=(0, 1, 2, 3),
            uneven_shard_mesh_dims=(),
            placements=(Shard(0),),
        )
        compiler = _AttachCompiler(model, {parameter: layout})
        compiler._parameter_shapes = {"weight": (5, 3)}
        compiler._joint_graph.graph_module.fsdp_padded_params = {"weight": (5, 8)}
        dense = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        optimizer = SimpleNamespace(state={parameter: {"exp_avg": dense}})

        mock_dist = MagicMock()
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 3
        with (
            patch("hyper_parallel.compile.checkpoint_metadata.dist", mock_dist),
            patch("hyper_parallel.compile.optimizer_state.dist", mock_dist),
        ):
            compiler.reshard_optimizer_state(optimizer)

        sharded = optimizer.state[parameter]["exp_avg"]
        self.assertEqual(tuple(sharded.shape), (2, 3))
        self.assertTrue(torch.equal(sharded, torch.zeros(2, 3)))


class TestOffAxisResume(unittest.TestCase):
    """A non-dim-0 FSDP shard cannot map dense state and reports it accurately."""

    def test_off_axis_resume_is_rejected(self) -> None:
        """A dim-1 shard rejects dense restored state without calling it strided."""
        model = nn.Linear(2, 2, bias=False)
        parameter = model.weight
        layout = SimpleNamespace(placements=(Shard(1),))
        compiler = _StubCompiler(model, {parameter: layout}, {"weight": (10, 2)})
        optimizer = SimpleNamespace(state={parameter: {"exp_avg": torch.zeros(10, 2)}})

        with self.assertRaisesRegex(RuntimeError, "not a single contiguous dim-0"):
            compiler.reshard_optimizer_state(optimizer)


class TestStridedResume(unittest.TestCase):
    """Strided dim-0 layouts cannot map dense state and must fail loudly."""

    def test_strided_dim0_resume_is_rejected(self) -> None:
        """A TP-and-FSDP strided dim-0 shard rejects dense restored state."""
        model = nn.Linear(2, 2, bias=False)
        parameter = model.weight
        layout = SimpleNamespace(placements=(StridedShard(0, 2),))
        compiler = _StubCompiler(model, {parameter: layout}, {"weight": (10, 2)})
        optimizer = SimpleNamespace(state={parameter: {"exp_avg": torch.zeros(10, 2)}})

        with self.assertRaisesRegex(RuntimeError, "strided"):
            compiler.reshard_optimizer_state(optimizer)

    def test_strided_shard_metadata_is_rejected(self) -> None:
        """A strided local shard cannot be described as one rectangular chunk."""
        model = nn.Linear(2, 2, bias=False)
        parameter = model.weight
        layout = SimpleNamespace(
            tensor_shape=(10, 2),
            placements=(StridedShard(0, 2),),
        )
        compiler = _AttachCompiler(model, {parameter: layout})

        with self.assertRaisesRegex(RuntimeError, "strided"):
            compiler.attach_shard_metadata()

    def test_strided_metadata_can_skip_strict_guard(self) -> None:
        """Optimizer resharding can annotate non-strided params independently."""
        model = nn.Linear(2, 2, bias=False)
        parameter = model.weight
        layout = SimpleNamespace(
            tensor_shape=(10, 2),
            placements=(StridedShard(0, 2),),
        )
        compiler = _AttachCompiler(model, {parameter: layout})

        compiler.attach_shard_metadata(strict_strided=False)

        self.assertFalse(hasattr(parameter, CHUNK_INFO))


class TestUnmappableResume(unittest.TestCase):
    """Sharded params without usable geometry reject dense restored state."""

    def test_missing_chunk_geometry_rejects_dense_state(self) -> None:
        """Dense moments cannot silently survive a sharded parameter."""
        model = nn.Linear(2, 2, bias=False)
        parameter = model.weight
        layout = SimpleNamespace(placements=(Shard(0),))
        compiler = _StubCompiler(model, {parameter: layout}, {"weight": (10, 2)})
        optimizer = SimpleNamespace(state={parameter: {"exp_avg": torch.zeros(10, 2)}})

        with self.assertRaisesRegex(RuntimeError, "shard geometry"):
            compiler.reshard_optimizer_state(optimizer)

    def test_missing_chunk_geometry_leaves_local_state_alone(self) -> None:
        """Already-local moments need no geometry and are left untouched."""
        model = nn.Linear(2, 2, bias=False)
        parameter = model.weight
        layout = SimpleNamespace(placements=(Shard(0),))
        compiler = _StubCompiler(model, {parameter: layout}, {"weight": (10, 2)})
        local = torch.zeros(2, 2)
        optimizer = SimpleNamespace(state={parameter: {"exp_avg": local}})

        compiler.reshard_optimizer_state(optimizer)

        self.assertIs(optimizer.state[parameter]["exp_avg"], local)


class TestTiedParameterResume(unittest.TestCase):
    """Tied parameters resolve to the canonical FQN that carries shard metadata."""

    def test_tied_parameter_uses_canonical_padding_entry(self) -> None:
        """Attach trims a pad-sharded tied param via its canonical name."""
        model = _TiedModel()
        parameter = model.first.weight
        compiler = _AttachCompiler(model, {parameter: _single_rank_layout((12, 2))})
        compiler._joint_graph.graph_module.fsdp_padded_params = {
            "first.weight": (10, 12)
        }

        compiler.attach_shard_metadata()

        chunk_info = getattr(parameter, CHUNK_INFO)
        self.assertEqual(chunk_info.global_shape, (10, 2))
        self.assertEqual(chunk_info.chunk.offsets, (0, 0))
        self.assertEqual(chunk_info.chunk.sizes, (10, 2))

    def test_tied_parameter_state_is_resharded(self) -> None:
        """Dense moments of a tied param follow the canonical padded geometry."""
        model = _TiedModel()
        parameter = model.first.weight
        compiler = _AttachCompiler(model, {parameter: _single_rank_layout((12, 2))})
        compiler._parameter_shapes = {"first.weight": (10, 2)}
        compiler._joint_graph.graph_module.fsdp_padded_params = {
            "first.weight": (10, 12)
        }
        dense = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        optimizer = SimpleNamespace(state={parameter: {"exp_avg": dense}})

        compiler.reshard_optimizer_state(optimizer)

        sharded = optimizer.state[parameter]["exp_avg"]
        self.assertEqual(tuple(sharded.shape), (2, 2))
        self.assertTrue(torch.equal(sharded, dense[:2]))


class TestExoticStateResume(unittest.TestCase):
    """Non-parameter-shaped state is reported instead of silently kept dense."""

    def test_exotic_state_is_reported_and_left_alone(self) -> None:
        """Factored moments cannot map onto one shard and trigger a warning."""
        model = nn.Linear(2, 2, bias=False)
        parameter = model.weight
        compiler = _AttachCompiler(model, {parameter: _single_rank_layout((12, 2))})
        compiler._parameter_shapes = {"weight": (10, 2)}
        compiler._joint_graph.graph_module.fsdp_padded_params = {"weight": (10, 12)}
        dense = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        factored = torch.zeros(10, 1)
        optimizer = SimpleNamespace(
            state={parameter: {"exp_avg": dense, "row": factored}}
        )

        with self.assertLogs(
            "hyper_parallel.compile.optimizer_state", level="WARNING"
        ) as logs:
            compiler.reshard_optimizer_state(optimizer)

        self.assertTrue(any("un-resharded" in line for line in logs.output))
        self.assertEqual(tuple(optimizer.state[parameter]["exp_avg"].shape), (2, 2))
        self.assertIs(optimizer.state[parameter]["row"], factored)


if __name__ == "__main__":
    unittest.main()
