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
"""Unit tests for MindSpore fully_shard parameter lifecycle and communication."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ms = pytest.importorskip("mindspore")
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_fully_shard,
)

ensure_mindspore_platform_for_fully_shard()

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor as CoreDTensor
from hyper_parallel.core.dtensor.placement_types import Placement, Replicate, Shard, StridedShard
from hyper_parallel.core.fully_shard.hsdp_utils import ParamModuleInfo, ShardedState
from hyper_parallel.core.fully_shard.utils import (
    DDPMeshInfo,
    FSDPMeshInfo,
    MixedPrecisionPolicy,
    SourceShardMetaInfo,
)
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat
from hyper_parallel.platform.mindspore.dtensor import DTensorBase as MindSporeDTensorBase
from hyper_parallel.platform.mindspore.fully_shard.param import (
    AllGatherCommCtx,
    AllReduceCommCtx,
    MindSporeHSDPParamV2,
    ParameterHookMigrator,
    ReduceScatterCommCtx,
    make_contiguous_strides_for,
    set_requires_grad_if_needed,
)
from tests.ut.platform.mindspore.fully_shard.conftest import MindSporeFullyShardUnitTest

enable_mindspore_backward_compat()


def _bare_param():
    """Build a parameter wrapper with constructor-owned contexts initialized."""
    hsdp_param = object.__new__(MindSporeHSDPParamV2)
    hsdp_param.unsharded_param_buffers = []
    hsdp_param.unsharded_accumulated_grad = None
    hsdp_param.allgather_comm_ctx = AllGatherCommCtx()
    hsdp_param.reduce_scatter_comm_ctx = ReduceScatterCommCtx()
    hsdp_param.all_reduce_comm_ctx = AllReduceCommCtx()
    hsdp_param._grad = None
    hsdp_param._reduce_partial_output = None
    hsdp_param.gradient_scaling_factor = None
    hsdp_param.mp_policy = MixedPrecisionPolicy()
    hsdp_param.orig_dtype = ms.float32
    hsdp_param.param_dtype = None
    hsdp_param.reduce_dtype = None
    return hsdp_param


def _clone_with_mint_cat(tensor):
    """Materialize a clone in CPU UTs where MindSpore has no Clone kernel."""
    return ms.mint.cat((tensor,), dim=0)


class _CpuMindSporeDTensor(MindSporeDTensorBase):
    """Run production DTensor logic with a MindSpore base under the Torch-pinned UT bootstrap."""

    def __new__(
        cls,
        local_tensor,
        device_mesh=None,
        placements=None,
        layout=None,
        shape=None,
    ):
        if isinstance(local_tensor, _CpuMindSporeDTensor) and layout is None:
            return super().__new__(
                cls,
                local_tensor._local_tensor,
                local_tensor.device_mesh,
                local_tensor.placements,
                local_tensor.layout,
                local_tensor.shape,
            )
        return super().__new__(
            cls,
            local_tensor,
            device_mesh,
            placements,
            layout,
            shape,
        )

    __init_data__ = CoreDTensor.__init_data__
    device_mesh = CoreDTensor.device_mesh
    placements = CoreDTensor.placements
    layout = CoreDTensor.layout
    shape = CoreDTensor.shape
    ndim = CoreDTensor.ndim
    to_local = CoreDTensor.to_local

    @staticmethod
    def from_local(
        local_tensor: ms.Tensor,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        *,
        run_check: bool = False,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> "_CpuMindSporeDTensor":
        """Call production ``from_local`` while replacing only its backend class.

        Args:
            local_tensor: Local MindSpore shard.
            device_mesh: Mesh used to build the real layout.
            placements: Placements applied to the real layout.
            run_check: Whether to run distributed metadata checks.
            shape: Explicit logical global shape.
            stride: Explicit logical global stride.

        Returns:
            A CPU-backed MindSpore DTensor test instance.
        """
        with patch(
            "hyper_parallel.core.dtensor.dtensor.DTensor",
            _CpuMindSporeDTensor,
        ):
            return CoreDTensor.from_local(
                local_tensor,
                device_mesh,
                placements,
                run_check=run_check,
                shape=shape,
                stride=stride,
            )


def _prepare_sharded_param_init(hsdp_param, param, shard_rank, shard_world_size):
    """Configure a parameter wrapper with real mesh, layout, and module ownership."""
    module = ms.nn.Cell()
    module.weight = param
    hsdp_param.device = "cpu"
    hsdp_param._orig_param_is_dtensor = False
    hsdp_param.source_shard_info = None
    hsdp_param._orig_dtensor_mesh = None
    hsdp_param._orig_dtensor_placements = None
    hsdp_param._module_info = ParamModuleInfo(module, "weight")
    hsdp_param._parameter_hook_migrator = ParameterHookMigrator()
    hsdp_param._spmd_shard_mesh_dim = 0
    hsdp_param._spmd_replicate_mesh_dim = None
    hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)
    hsdp_param.mesh_info.mesh = DeviceMesh(
        "cpu",
        list(range(shard_world_size)),
        _init_backend=False,
    )
    hsdp_param.mesh_info.shard_mesh_dim = 0
    hsdp_param.mesh_info.replicate_mesh_dim = None
    hsdp_param.mesh_info.shard_mesh_rank = shard_rank
    hsdp_param.mesh_info.shard_mesh_size = shard_world_size
    hsdp_param.offload_to_cpu = False
    hsdp_param.pin_memory = False
    return module


class TestPlacementConstruction(MindSporeFullyShardUnitTest):
    """Test source-layout preservation and explicit DP placement application."""

    def test_base_placements_prefix_dp_axes_for_source_layout(self):
        """Native DTensor source placements should follow the explicit DP prefix."""
        hsdp_param = _bare_param()
        dp_mesh = MagicMock(ndim=1)
        source_mesh = MagicMock()
        hsdp_param.mesh_info = SimpleNamespace(mesh=dp_mesh)
        hsdp_param.source_shard_info = SimpleNamespace(
            mesh=source_mesh,
            placements=(Shard(1), Replicate()),
        )
        with patch(
            "hyper_parallel.platform.mindspore.fully_shard.param.DeviceMesh.concatenate",
            return_value=MagicMock(ndim=3),
        ) as concatenate:
            placements = hsdp_param._get_base_spmd_placements()

        concatenate.assert_called_once_with([dp_mesh, source_mesh])
        self.assertTrue(placements[0].is_replicate())
        self.assertEqual(placements[1:], (Shard(1), Replicate()))

    def test_apply_data_parallel_placement_builds_strided_shard(self):
        """Same-dimension TP and FSDP sharding should use a StridedShard."""
        hsdp_param = _bare_param()
        hsdp_param._orig_param_is_dtensor = True
        hsdp_param._spmd_shard_mesh_dim = 0
        hsdp_param._spmd_replicate_mesh_dim = None
        hsdp_param._spmd_mesh = SimpleNamespace(ndim=2, mesh_shape=(2, 4))
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)

        placements = hsdp_param._apply_data_parallel_placements(
            [Replicate(), Shard(1)],
            Shard(1),
        )

        self.assertEqual(placements[0], StridedShard(1, split_factor=4))

    def test_replicate_param_keeps_replicate_placement(self):
        """A DDP-managed plain parameter should not gain an FSDP shard placement."""
        hsdp_param = _bare_param()
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param._spmd_shard_mesh_dim = None
        hsdp_param._spmd_replicate_mesh_dim = 0
        hsdp_param._spmd_mesh = SimpleNamespace(ndim=1)
        hsdp_param.mesh_info = object.__new__(DDPMeshInfo)

        placements = hsdp_param._apply_data_parallel_placements([Shard(0)], Shard(0))

        self.assertTrue(placements[0].is_replicate())

    def test_uneven_hsdp_retains_replicate_and_marks_shard(self):
        """
        Feature: Uneven HSDP placement construction.
        Description: Apply dim-0 FSDP sharding after a replicate mesh dimension.
        Expectation: Replication is preserved and the FSDP placement is marked uneven.
        """
        hsdp_param = _bare_param()
        hsdp_param.hsdp_placement = Shard(0)
        hsdp_param._spmd_shard_mesh_dim = 1
        hsdp_param._spmd_replicate_mesh_dim = 0
        hsdp_param._spmd_mesh = SimpleNamespace(ndim=2, mesh_shape=(2, 2))
        hsdp_param.shard_world_size = 2
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)

        hsdp_param._init_shard_placements(
            ms.Tensor(np.ones((5, 3), dtype=np.float32)),
            0,
            [Replicate(), Replicate()],
        )

        self.assertEqual(
            hsdp_param._spmd_placements,
            (Replicate(), Shard(0, uneven_shard=True)),
        )

    def test_uneven_same_dim_source_marks_strided_shard(self):
        """
        Feature: Uneven same-dimension TP and FSDP placement construction.
        Description: Apply uneven FSDP after an existing shard on the same tensor dimension.
        Expectation: The FSDP axis becomes an uneven StridedShard with the source order retained.
        """
        hsdp_param = _bare_param()
        hsdp_param.hsdp_placement = Shard(0)
        hsdp_param._spmd_shard_mesh_dim = 0
        hsdp_param._spmd_replicate_mesh_dim = None
        hsdp_param._spmd_mesh = SimpleNamespace(ndim=2, mesh_shape=(2, 2))
        hsdp_param.shard_world_size = 2
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)

        hsdp_param._init_shard_placements(
            ms.Tensor(np.ones((5, 3), dtype=np.float32)),
            0,
            [Replicate(), Shard(0)],
        )

        self.assertEqual(
            hsdp_param._spmd_placements,
            (StridedShard(0, split_factor=2, uneven_shard=True), Shard(0)),
        )

    def test_plain_source_layout_survives_uneven_fsdp_initialization(self):
        """
        Feature: AutoModel/Trainer source-layout preservation during parameter initialization.
        Description: Apply uneven FSDP to a plain parameter carrying a TP source layout.
        Expectation: The final registered parameter retains the combined logical layout and local chunk.
        """
        full_param = ms.Parameter(
            ms.Tensor(np.arange(15, dtype=np.float32).reshape(5, 3)),
            name="weight",
        )
        hsdp_param = _bare_param()
        module = _prepare_sharded_param_init(
            hsdp_param,
            full_param,
            shard_rank=0,
            shard_world_size=2,
        )
        with patch(
            "hyper_parallel.core.dtensor.device_mesh.platform.get_rank",
            return_value=0,
        ):
            root_mesh = DeviceMesh(
                "cpu",
                [[0, 1], [2, 3]],
                mesh_dim_names=("fsdp", "tp"),
                _init_backend=False,
            )
            fsdp_mesh = root_mesh["fsdp"]
            source_mesh = root_mesh["tp"]
        source_shard_info = SourceShardMetaInfo(
            source_mesh,
            (Shard(0),),
            origin_is_dtensor=False,
        )
        hsdp_param.mesh_info.mesh = fsdp_mesh
        hsdp_param.source_shard_info = source_shard_info

        with (
            patch.object(ms.Tensor, "clone", _clone_with_mint_cat),
            patch(
                "hyper_parallel.platform.mindspore.fully_shard.param.DTensor",
                _CpuMindSporeDTensor,
            ),
        ):
            hsdp_param._init_sharded_param(full_param, shard_placement_fn=None)

        expected_placements = (
            StridedShard(0, split_factor=2, uneven_shard=True),
            Shard(0),
        )
        sharded_param = hsdp_param.sharded_param
        self.assertIs(module.weight, sharded_param)
        self.assertIs(hsdp_param.source_shard_info, source_shard_info)
        self.assertIs(source_shard_info.mesh, source_mesh)
        self.assertEqual(source_shard_info.placements, (Shard(0),))
        self.assertEqual(sharded_param.placements, expected_placements)
        self.assertEqual(sharded_param.shape, (10, 3))
        self.assertEqual(sharded_param._local_tensor.shape, (3, 3))
        self.assertEqual(sharded_param.layout.tensor_shape, (10, 3))
        self.assertEqual(sharded_param.layout.tensor_stride, (3, 1))
        self.assertEqual(sharded_param.layout.tensor_dtype, ms.float32)
        np.testing.assert_allclose(
            sharded_param._local_tensor.asnumpy(),
            np.arange(9, dtype=np.float32).reshape(3, 3),
        )


class TestParameterHelpers(MindSporeFullyShardUnitTest):
    """Test local parameter buffers, shapes, and lifecycle state helpers."""

    def test_build_detaches_initial_communication_storage_from_logical_shard(self):
        """Initial communication storage should be detached without copying the logical shard."""
        hsdp_param = _bare_param()
        full_param = ms.Parameter(
            ms.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3)),
            requires_grad=True,
            name="weight",
        )
        module = _prepare_sharded_param_init(
            hsdp_param,
            full_param,
            shard_rank=0,
            shard_world_size=1,
        )

        with (
            patch.object(ms.Tensor, "clone", _clone_with_mint_cat),
            patch(
                "hyper_parallel.platform.mindspore.fully_shard.param.DTensor",
                _CpuMindSporeDTensor,
            ),
        ):
            hsdp_param._init_sharded_param(full_param, shard_placement_fn=None)

        sharded_param = hsdp_param.sharded_param
        local_param = sharded_param._local_tensor
        self.assertIsInstance(sharded_param, _CpuMindSporeDTensor)
        self.assertIs(module.weight, sharded_param)
        self.assertEqual(sharded_param.name, "weight")
        self.assertTrue(sharded_param.requires_grad)
        self.assertIsNone(sharded_param.grad)
        self.assertEqual(sharded_param.shape, (2, 3))
        self.assertEqual(sharded_param.layout.tensor_shape, (2, 3))
        self.assertEqual(sharded_param.layout.tensor_stride, (3, 1))
        self.assertEqual(sharded_param.placements, (Shard(0),))
        self.assertFalse(hsdp_param._sharded_param_data.requires_grad)
        self.assertTrue(hsdp_param._sharded_param_data.is_leaf)
        self.assertEqual(
            hsdp_param._sharded_param_data.untyped_storage().data_ptr(),
            local_param.untyped_storage().data_ptr(),
        )
        self.assertEqual(full_param.untyped_storage().size(), 0)
        self.assertTrue(sharded_param._hsdp_param_initialized)
        self.assertEqual(hsdp_param.sharded_state, ShardedState.SHARDED)

    def test_refresh_detaches_communication_storage_without_replacing_local_tensor(self):
        """Communication storage should be a detached leaf while the optimizer shard stays unchanged."""
        hsdp_param = _bare_param()
        hsdp_param.sharded_size = (2, 3)
        hsdp_param.padded_sharded_param_size = (2, 3)
        hsdp_param.pin_memory = False
        local_tensor = ms.Parameter(
            ms.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3)),
            requires_grad=True,
        )
        hsdp_param.sharded_param = SimpleNamespace(
            _local_tensor=local_tensor,
            requires_grad=True,
        )

        hsdp_param._refresh_sharded_local_tensor(local_tensor)

        self.assertIs(hsdp_param.sharded_param._local_tensor, local_tensor)
        self.assertTrue(local_tensor.requires_grad)
        self.assertTrue(local_tensor.is_leaf)
        self.assertFalse(hsdp_param._sharded_param_data.requires_grad)
        self.assertTrue(hsdp_param._sharded_param_data.is_leaf)
        self.assertEqual(
            hsdp_param._sharded_param_data.untyped_storage().data_ptr(),
            local_tensor.untyped_storage().data_ptr(),
        )

    def test_dim0_uneven_init_and_reset_keep_logical_shard_separate_from_padding(self):
        """
        Feature: Uneven parameter storage lifecycle.
        Description: Initialize and refresh the short rank of a five-row parameter.
        Expectation: The optimizer shard stays independent from the zero-padded communication buffer.
        """
        hsdp_param = _bare_param()
        full_param = ms.Parameter(
            ms.Tensor(np.arange(15, dtype=np.float32).reshape(5, 3)),
            name="weight",
        )
        module = _prepare_sharded_param_init(
            hsdp_param,
            full_param,
            shard_rank=1,
            shard_world_size=2,
        )

        with (
            patch.object(ms.Tensor, "clone", _clone_with_mint_cat),
            patch(
                "hyper_parallel.platform.mindspore.fully_shard.param.DTensor",
                _CpuMindSporeDTensor,
            ),
        ):
            hsdp_param._init_sharded_param(full_param, shard_placement_fn=None)

        sharded_param = hsdp_param.sharded_param
        local_param = sharded_param._local_tensor
        self.assertIsInstance(sharded_param, _CpuMindSporeDTensor)
        self.assertIs(module.weight, sharded_param)
        self.assertEqual(hsdp_param.sharded_size, (2, 3))
        self.assertEqual(hsdp_param.padded_sharded_param_size, (3, 3))
        self.assertEqual(sharded_param.shape, (5, 3))
        self.assertEqual(local_param.shape, (2, 3))
        np.testing.assert_allclose(
            local_param.asnumpy(),
            np.array([[9, 10, 11], [12, 13, 14]], dtype=np.float32),
        )
        self.assertEqual(sharded_param.layout.tensor_shape, (5, 3))
        self.assertEqual(sharded_param.layout.tensor_stride, (3, 1))
        self.assertEqual(sharded_param.placements, (Shard(0, uneven_shard=True),))
        np.testing.assert_allclose(
            hsdp_param._sharded_param_data.asnumpy(),
            np.array([9, 10, 11, 12, 13, 14, 0, 0, 0], dtype=np.float32),
        )
        self.assertNotEqual(
            local_param.untyped_storage().data_ptr(),
            hsdp_param._sharded_param_data.untyped_storage().data_ptr(),
        )

        loaded_local_tensor = ms.Tensor(np.full((2, 3), 9.0, dtype=np.float32))
        sharded_param._local_tensor = loaded_local_tensor

        with patch(
            "hyper_parallel.platform.mindspore.fully_shard.param.DTensor",
            _CpuMindSporeDTensor,
        ):
            hsdp_param.reset_sharded_param()

        self.assertIs(hsdp_param.sharded_param, sharded_param)
        self.assertIs(module.weight, sharded_param)
        self.assertIs(sharded_param._local_tensor, loaded_local_tensor)
        np.testing.assert_allclose(
            sharded_param._local_tensor.asnumpy(),
            loaded_local_tensor.asnumpy(),
        )
        np.testing.assert_allclose(
            hsdp_param._sharded_param_data.asnumpy(),
            np.array([9, 9, 9, 9, 9, 9, 0, 0, 0], dtype=np.float32),
        )
        self.assertEqual(sharded_param.layout.tensor_shape, (5, 3))
        self.assertEqual(sharded_param.layout.tensor_stride, (3, 1))
        self.assertEqual(sharded_param.placements, (Shard(0, uneven_shard=True),))

    def test_dim0_balanced_chunk_initializes_every_rank(self):
        """
        Feature: MindSpore balanced dim-0 parameter sharding.
        Description: Shard six rows over four ranks through the complete initialization path.
        Expectation: Logical row counts are 2, 2, 1, 1 and short ranks use padded communication storage.
        """
        full_data = np.arange(24, dtype=np.float32).reshape(6, 4)
        expected_row_ranges = ((0, 2), (2, 4), (4, 5), (5, 6))
        for shard_rank, (start, stop) in enumerate(expected_row_ranges):
            with self.subTest(shard_rank=shard_rank):
                hsdp_param = _bare_param()
                full_param = ms.Parameter(ms.Tensor(full_data), name="weight")
                module = _prepare_sharded_param_init(
                    hsdp_param,
                    full_param,
                    shard_rank=shard_rank,
                    shard_world_size=4,
                )

                with (
                    patch.object(ms.Tensor, "clone", _clone_with_mint_cat),
                    patch(
                        "hyper_parallel.platform.mindspore.fully_shard.param.DTensor",
                        _CpuMindSporeDTensor,
                    ),
                ):
                    hsdp_param._init_sharded_param(full_param, shard_placement_fn=None)

                sharded_param = hsdp_param.sharded_param
                local_tensor = sharded_param._local_tensor
                expected_local = full_data[start:stop]
                expected_communication = np.zeros((2, 4), dtype=np.float32)
                expected_communication[: stop - start] = expected_local
                self.assertIs(module.weight, sharded_param)
                self.assertEqual(hsdp_param.sharded_size, expected_local.shape)
                self.assertEqual(hsdp_param.padded_sharded_param_size, (2, 4))
                self.assertEqual(sharded_param.shape, (6, 4))
                self.assertEqual(sharded_param.placements, (Shard(0, uneven_shard=True),))
                np.testing.assert_allclose(local_tensor.asnumpy(), expected_local)
                np.testing.assert_allclose(
                    hsdp_param._sharded_param_data.asnumpy().reshape(2, 4),
                    expected_communication,
                )
                if stop - start == 2:
                    self.assertEqual(
                        local_tensor.untyped_storage().data_ptr(),
                        hsdp_param._sharded_param_data.untyped_storage().data_ptr(),
                    )
                else:
                    self.assertNotEqual(
                        local_tensor.untyped_storage().data_ptr(),
                        hsdp_param._sharded_param_data.untyped_storage().data_ptr(),
                    )

    def test_dim0_smaller_than_world_size_is_rejected(self):
        """
        Feature: MindSpore zero-shape parameter validation.
        Description: Attempt balanced chunking when there are fewer rows than shard ranks.
        Expectation: Initialization fails before an optimizer receives a zero-sized local shard.
        """
        for rows, shard_world_size in ((7, 8), (0, 4)):
            with self.subTest(rows=rows, shard_world_size=shard_world_size):
                hsdp_param = _bare_param()
                param = ms.Parameter(
                    ms.Tensor(np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)),
                    name="weight",
                )
                module = _prepare_sharded_param_init(
                    hsdp_param,
                    param,
                    shard_rank=shard_world_size - 1,
                    shard_world_size=shard_world_size,
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "balanced chunking requires sharded dimension size",
                ):
                    hsdp_param._init_sharded_param(param, shard_placement_fn=None)
                self.assertIs(module.weight, param)

    def test_uneven_non_dim0_sharding_is_rejected(self):
        """
        Feature: Uneven parameter shard validation.
        Description: Request an uneven FSDP parameter split on dimension one.
        Expectation: MindSpore rejects the same unsupported boundary as Torch.
        """
        hsdp_param = _bare_param()
        hsdp_param.hsdp_placement = Shard(1)
        hsdp_param.shard_world_size = 2
        hsdp_param._module_info = SimpleNamespace(param_name="weight")
        hsdp_param._spmd_shard_mesh_dim = 0
        hsdp_param._spmd_replicate_mesh_dim = None
        hsdp_param._spmd_mesh = SimpleNamespace(ndim=1, mesh_shape=(2,))
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)

        with self.assertRaisesRegex(NotImplementedError, "only supports uneven sharding on dim=0"):
            hsdp_param._init_shard_placements(
                ms.Tensor(np.arange(15, dtype=np.float32).reshape(3, 5)),
                1,
                [Replicate()],
            )

    def test_make_contiguous_strides_for(self):
        """Row-major and column-major stride helpers should match tensor shapes."""
        self.assertEqual(make_contiguous_strides_for((2, 3, 4)), (12, 4, 1))
        self.assertEqual(make_contiguous_strides_for((2, 3, 4), row_major=False), (12, 1, 3))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            make_contiguous_strides_for((2, -1))

    def test_init_unsharded_buffers_reuses_and_force_recreates(self):
        """Stable buffers should be reused unless explicit recreation is requested."""
        hsdp_param = _bare_param()
        hsdp_param.init_unsharded_param_buffers([2], [ms.float32], 2, "cpu")
        original = hsdp_param.unsharded_param_buffers[0]
        hsdp_param.init_unsharded_param_buffers([3], [ms.float16], 2, "cpu")
        self.assertIs(hsdp_param.unsharded_param_buffers[0], original)
        hsdp_param.init_unsharded_param_buffers(
            [3], [ms.float16], 2, "cpu", force_recreate=True
        )
        self.assertIsNot(hsdp_param.unsharded_param_buffers[0], original)
        self.assertEqual(hsdp_param.unsharded_param_buffers[0].dtype, ms.float16)

    def test_init_unsharded_buffers_rejects_recreation_after_parameter_binding(self):
        """
        Feature: Stable unsharded parameter buffer ownership.
        Description: Request buffer recreation after binding the stable Parameter.
        Expectation: Recreation is rejected before the backing buffer can be replaced.
        """
        hsdp_param = _bare_param()
        hsdp_param.unsharded_param_buffers = [ms.mint.empty((4,), dtype=ms.float32)]
        hsdp_param._unsharded_param = object()

        with self.assertRaisesRegex(RuntimeError, "stable unsharded parameter"):
            hsdp_param.init_unsharded_param_buffers(
                [3],
                [ms.float16],
                2,
                "cpu",
                force_recreate=True,
            )

    def test_init_unsharded_param_returns_before_rebinding_stable_parameter(self):
        """
        Feature: Stable unsharded Parameter reuse.
        Description: Initialize an unshard cycle after the stable Parameter already exists.
        Expectation: Initialization returns without creating a new buffer view or rebinding Parameter data.
        """
        hsdp_param = _bare_param()
        stable_param = object()
        stable_buffer = MagicMock()
        hsdp_param._unsharded_param = stable_param
        hsdp_param.unsharded_param_buffers = [stable_buffer]

        hsdp_param.init_unsharded_param()

        self.assertIs(hsdp_param._unsharded_param, stable_param)
        stable_buffer.narrow.assert_not_called()

    def test_init_unsharded_param_restores_non_dim_zero_layout(self):
        """Per-parameter all-gather should inline chunk-cat reconstruction for dimension one."""
        hsdp_param = _bare_param()
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param._orig_size = (2, 4)
        hsdp_param.sharded_size = (2, 2)
        hsdp_param.shard_world_size = 2
        hsdp_param.hsdp_placement = Shard(1)
        hsdp_param.sharded_param = SimpleNamespace(name="weight", requires_grad=False)
        hsdp_param.unsharded_param_buffers = [ms.mint.empty((8,), dtype=ms.float32)]
        hsdp_param.allgather_comm_ctx.allgather_output = ms.Tensor(
            [0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0],
            ms.float32,
        )

        hsdp_param.init_unsharded_param()

        np.testing.assert_allclose(
            hsdp_param.unsharded_param.asnumpy(),
            np.arange(8, dtype=np.float32).reshape(2, 4),
        )
        self.assertIsNone(hsdp_param.allgather_comm_ctx.allgather_output)

    def test_init_unsharded_param_hides_dim_zero_padding(self):
        """An uneven dim-0 parameter should expose only logical all-gather elements."""
        hsdp_param = _bare_param()
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param._orig_size = (5,)
        hsdp_param.sharded_param = SimpleNamespace(name="weight", requires_grad=False)
        hsdp_param.unsharded_param_buffers = [
            ms.Tensor([0.0, 1.0, 2.0, 3.0, 4.0, 0.0], ms.float32)
        ]

        hsdp_param.init_unsharded_param()

        self.assertEqual(hsdp_param.unsharded_param.shape, (5,))
        self.assertEqual(
            hsdp_param.unsharded_param.untyped_storage().data_ptr(),
            hsdp_param.unsharded_param_buffers[0].untyped_storage().data_ptr(),
        )
        np.testing.assert_allclose(
            hsdp_param.unsharded_param.asnumpy(),
            np.arange(5, dtype=np.float32),
        )

    def test_init_unsharded_param_removes_each_balanced_chunk_padding(self):
        """
        Feature: Balanced dim-0 all-gather reconstruction.
        Description: Gather four communication slots for a six-row parameter split as 2, 2, 1, 1.
        Expectation: The final parameter contains all six logical rows and no inter-rank padding.
        """
        full_data = np.arange(24, dtype=np.float32).reshape(6, 4)
        packed_data = np.zeros((4, 2, 4), dtype=np.float32)
        packed_data[0] = full_data[0:2]
        packed_data[1] = full_data[2:4]
        packed_data[2, 0] = full_data[4]
        packed_data[3, 0] = full_data[5]
        hsdp_param = _bare_param()
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param._orig_size = (6, 4)
        hsdp_param.sharded_size = (1, 4)
        hsdp_param.padded_sharded_param_size = (2, 4)
        hsdp_param.shard_world_size = 4
        hsdp_param.hsdp_placement = Shard(0, uneven_shard=True)
        hsdp_param.sharded_param = SimpleNamespace(name="weight", requires_grad=False)
        hsdp_param.unsharded_param_buffers = [ms.mint.empty((32,), dtype=ms.float32)]
        hsdp_param.allgather_comm_ctx.allgather_output = ms.Tensor(packed_data)

        hsdp_param.init_unsharded_param()

        self.assertEqual(hsdp_param.unsharded_param.shape, (6, 4))
        np.testing.assert_allclose(
            hsdp_param.unsharded_param.asnumpy(),
            full_data,
        )
        self.assertIsNone(hsdp_param.allgather_comm_ctx.allgather_output)

    def test_init_unsharded_param_rejects_temporary_output_for_even_dim0(self):
        """
        Feature: Even dim-0 all-gather fast-path invariant.
        Description: Pass a temporary all-gather output to the even dim-0 consumer path.
        Expectation: Initialization reports the internal producer/consumer state mismatch.
        """
        hsdp_param = _bare_param()
        hsdp_param._module_info = SimpleNamespace(param_name="weight")
        hsdp_param._orig_size = (4, 2)
        hsdp_param.shard_world_size = 2
        hsdp_param.hsdp_placement = Shard(0)
        hsdp_param.unsharded_param_buffers = [ms.mint.empty((8,), dtype=ms.float32)]
        hsdp_param.allgather_comm_ctx.allgather_output = ms.mint.empty(
            (8,),
            dtype=ms.float32,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "parameter 'weight'.*allgather_comm_ctx.allgather_output.*separate temporary buffer",
        ):
            hsdp_param.init_unsharded_param()

    def test_to_sharded_only_frees_distinct_unsharded_storage(self):
        """Replicate parameters must retain storage shared by sharded and unsharded views."""
        hsdp_param = _bare_param()
        hsdp_param.sharded_param = object()
        hsdp_param._setattr_on_modules = MagicMock()
        hsdp_param.free_unsharded_param = MagicMock()
        shared_storage = object()
        hsdp_param._sharded_param_data = shared_storage
        hsdp_param.unsharded_param_buffers = [shared_storage]

        hsdp_param.to_sharded()

        hsdp_param.free_unsharded_param.assert_not_called()
        self.assertEqual(hsdp_param.sharded_state, ShardedState.SHARDED)

        hsdp_param.unsharded_param_buffers = [object()]
        hsdp_param.to_sharded()

        hsdp_param.free_unsharded_param.assert_called_once_with()

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.set_requires_grad_if_needed")
    def test_wait_for_unshard_preserves_parameter_identity(self, mock_requires_grad):
        """Waiting a prefetched all-gather should install one stable full parameter."""
        hsdp_param = _bare_param()
        hsdp_param.sharded_state = ShardedState.SHARDED
        hsdp_param.sharded_param = MagicMock()
        hsdp_param.init_unsharded_param = MagicMock()
        hsdp_param.to_unsharded = MagicMock()
        handle = MagicMock()
        hsdp_param.allgather_comm_ctx.allgather_handle = handle

        hsdp_param.wait_for_unshard()

        handle.wait.assert_called_once_with()
        hsdp_param.init_unsharded_param.assert_called_once_with()
        hsdp_param.to_unsharded.assert_called_once_with()
        self.assertIsNone(hsdp_param.allgather_comm_ctx.allgather_handle)
        mock_requires_grad.assert_not_called()

    def test_set_requires_grad_if_needed_only_updates_changes(self):
        """Requires-grad propagation should avoid redundant writes."""
        source = SimpleNamespace(requires_grad=True)
        destination = MagicMock(requires_grad=False)
        set_requires_grad_if_needed(source, destination)
        destination.requires_grad_.assert_called_once_with(True)
        destination.requires_grad = True
        destination.requires_grad_.reset_mock()
        set_requires_grad_if_needed(source, destination)
        destination.requires_grad_.assert_not_called()


class TestCommunicationContexts(MindSporeFullyShardUnitTest):
    """Test async handle waiting and mint collective routing."""

    def test_reduce_scatter_and_all_reduce_outputs_wait_once(self):
        """Each communication output accessor should consume its async handle."""
        hsdp_param = _bare_param()
        rs_handle = MagicMock()
        ar_handle = MagicMock()
        rs_output = ms.Tensor([1.0, 2.0])
        ar_output = object()
        hsdp_param._grad = ms.Tensor([3.0, 4.0])
        hsdp_param.reduce_scatter_comm_ctx = ReduceScatterCommCtx(rs_output, rs_handle)
        hsdp_param.all_reduce_comm_ctx = AllReduceCommCtx(ar_output, ar_handle)

        self.assertIs(hsdp_param.reduce_scatter_output(), rs_output)
        self.assertIs(hsdp_param.all_reduce_output(), ar_output)
        rs_handle.wait.assert_called_once_with()
        ar_handle.wait.assert_called_once_with()
        self.assertIsNone(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle)
        self.assertIsNone(hsdp_param.all_reduce_comm_ctx.all_reduce_handle)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_uses_mint_string_and_context(self, mock_reduce_scatter):
        """Per-parameter RS should use parameter mesh info and cache async work."""
        hsdp_param = _bare_param()
        hsdp_param._unsharded_param = SimpleNamespace(
            grad=ms.Tensor([1.0, 2.0, 3.0, 4.0])
        )
        hsdp_param._to_local_unsharded_grad = MagicMock(side_effect=lambda grad: grad)
        hsdp_param.is_sharded = True
        hsdp_param.shard_world_size = 2
        hsdp_param.hsdp_placement = Shard(0)
        hsdp_param._orig_size = (4,)
        hsdp_param.padded_sharded_param_size = (2,)
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)
        hsdp_param.mesh_info.shard_process_group = "fsdp"
        handle = MagicMock()
        mock_reduce_scatter.return_value = handle

        hsdp_param.reduce_scatter_grad(reduce_op="sum")

        self.assertEqual(mock_reduce_scatter.call_args.kwargs["op"], "sum")
        self.assertEqual(mock_reduce_scatter.call_args.kwargs["group"], "fsdp")
        self.assertIs(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_handle, handle)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.apply_gradient_scaling_factor")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_scales_gradient_view_without_materializing(
        self,
        mock_reduce_scatter,
        mock_apply_scaling,
    ):
        """An even dim-0 gradient view should retain the source storage during inplace scaling."""
        source_grad = ms.Tensor([1.0, 2.0, 3.0, 4.0])
        hsdp_param = _bare_param()
        hsdp_param._unsharded_param = SimpleNamespace(grad=source_grad)
        hsdp_param._to_local_unsharded_grad = MagicMock(side_effect=lambda grad: grad)
        hsdp_param.shard_world_size = 2
        hsdp_param.hsdp_placement = Shard(0)
        hsdp_param.padded_sharded_param_size = (2,)
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)
        hsdp_param.mesh_info.shard_process_group = "fsdp"
        hsdp_param.gradient_scaling_factor = 0.5
        mock_reduce_scatter.return_value = MagicMock()

        hsdp_param.reduce_scatter_grad(reduce_op="sum")

        scaled_grad = mock_apply_scaling.call_args.args[0]
        reduce_scatter_input = mock_reduce_scatter.call_args.args[1]
        self.assertIs(reduce_scatter_input, scaled_grad)
        self.assertEqual(
            scaled_grad.untyped_storage().data_ptr(),
            source_grad.untyped_storage().data_ptr(),
        )

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.apply_gradient_scaling_factor")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_zero_pads_uneven_dim0_gradient(
        self,
        mock_reduce_scatter,
        mock_apply_scaling,
    ):
        """
        Feature: Per-parameter uneven reduce-scatter.
        Description: Prepare a five-element gradient for scaling and reduction over two FSDP ranks.
        Expectation: The inplace helper receives the six-element padded communication base tensor.
        """
        hsdp_param = _bare_param()
        hsdp_param._unsharded_param = SimpleNamespace(
            grad=ms.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        hsdp_param._to_local_unsharded_grad = MagicMock(side_effect=lambda grad: grad)
        hsdp_param.is_sharded = True
        hsdp_param.shard_world_size = 2
        hsdp_param.hsdp_placement = Shard(0, uneven_shard=True)
        hsdp_param._orig_size = (5,)
        hsdp_param.padded_sharded_param_size = (3,)
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)
        hsdp_param.mesh_info.shard_process_group = "fsdp"
        hsdp_param.gradient_scaling_factor = 0.5
        mock_reduce_scatter.return_value = MagicMock()
        mock_apply_scaling.return_value = ms.Tensor([-1.0], ms.float32)

        hsdp_param.reduce_scatter_grad(reduce_op="sum")

        output, packed_grad = mock_reduce_scatter.call_args.args
        self.assertEqual(output.shape, (3,))
        mock_apply_scaling.assert_called_once_with(packed_grad, 0.5)
        self.assertIsNot(packed_grad, mock_apply_scaling.return_value)
        np.testing.assert_allclose(
            packed_grad.asnumpy(),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0], dtype=np.float32),
        )

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.apply_gradient_scaling_factor")
    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_pads_each_balanced_dim0_chunk(
        self,
        mock_reduce_scatter,
        mock_apply_scaling,
    ):
        """
        Feature: Balanced dim-0 reduce-scatter packing.
        Description: Pack six gradient elements for a 2, 2, 1, 1 shard assignment.
        Expectation: Rank two and rank three each receive one value followed by local padding.
        """
        hsdp_param = _bare_param()
        hsdp_param._unsharded_param = SimpleNamespace(
            grad=ms.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        )
        hsdp_param._to_local_unsharded_grad = MagicMock(side_effect=lambda grad: grad)
        hsdp_param.is_sharded = True
        hsdp_param.shard_world_size = 4
        hsdp_param.hsdp_placement = Shard(0, uneven_shard=True)
        hsdp_param._orig_size = (6,)
        hsdp_param.padded_sharded_param_size = (2,)
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)
        hsdp_param.mesh_info.shard_process_group = "fsdp"
        mock_reduce_scatter.return_value = MagicMock()

        hsdp_param.reduce_scatter_grad(reduce_op="sum")

        output, packed_grad = mock_reduce_scatter.call_args.args
        self.assertEqual(output.shape, (2,))
        mock_apply_scaling.assert_called_once_with(packed_grad, None)
        np.testing.assert_allclose(
            packed_grad.asnumpy(),
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 6.0, 0.0], dtype=np.float32),
        )

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_all_reduce_consumes_rs_output_and_uses_replicate_group(self, mock_all_reduce):
        """
        Feature: HSDP all-reduce input layout.
        Description: Pass a contiguous reduce-scatter tensor into the replicate-group collective.
        Expectation: The existing tensor is reduced with the requested group and operation.
        """
        hsdp_param = _bare_param()
        output = ms.Tensor([1.0, 2.0])
        hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output = output
        hsdp_param.replicate_world_size = 2
        hsdp_param.mesh_info = object.__new__(DDPMeshInfo)
        hsdp_param.mesh_info.replicate_process_group = "dp"
        handle = MagicMock()
        mock_all_reduce.return_value = handle

        hsdp_param.all_reduce_grad(reduce_op="avg")

        self.assertIs(hsdp_param.all_reduce_comm_ctx.all_reduce_output, output)
        self.assertIs(hsdp_param.all_reduce_comm_ctx.all_reduce_handle, handle)
        self.assertEqual(mock_all_reduce.call_args.kwargs["group"], "dp")
        self.assertEqual(mock_all_reduce.call_args.kwargs["op"], "avg")

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_tp_replicate_reduce_uses_flattened_source_mesh(self, mock_all_reduce):
        """Final gradients should all-reduce across replicated source-layout axes."""
        hsdp_param = _bare_param()
        replicate_mesh = MagicMock()
        replicate_mesh.size.return_value = 2
        replicate_mesh.get_group.return_value = "tp-replicate"
        source_mesh = MagicMock(mesh_dim_names=("tp", "cp"))
        source_mesh.__getitem__.return_value.flatten.return_value = replicate_mesh
        hsdp_param.source_shard_info = SimpleNamespace(
            mesh=source_mesh,
            placements=(Shard(0), Replicate()),
        )
        grad = ms.Tensor([1.0, 2.0])

        hsdp_param.all_reduce_source_replicate_grad_inplace(grad, "sum")

        source_mesh.__getitem__.assert_called_once_with(("cp",))
        mock_all_reduce.assert_called_once_with(
            grad,
            op="sum",
            group="tp-replicate",
            async_op=False,
        )


class TestGradientApplication(MindSporeFullyShardUnitTest):
    """Test reduced-gradient assignment and source-gradient cleanup."""

    def test_apply_reduced_grad_assigns_and_clears_source(self):
        """A reduced local shard should replace the optimizer grad and release full grad."""
        hsdp_param = _bare_param()
        hsdp_param.sharded_size = (2,)
        hsdp_param.sharded_param = SimpleNamespace(
            grad=None,
            _local_tensor=ms.Tensor([0.0, 0.0]),
        )
        hsdp_param._unsharded_param = SimpleNamespace(grad=ms.Tensor([3.0, 4.0]))
        hsdp_param.offload_to_cpu = False
        hsdp_param.pin_memory = False
        hsdp_param._sharded_param_storage_dtype = MagicMock(return_value=ms.float32)
        hsdp_param.to_sharded_dtensor = MagicMock(side_effect=lambda tensor: tensor)

        need_synchronize = hsdp_param.apply_reduced_grad(ms.Tensor([1.0, 2.0], ms.float16))

        self.assertFalse(need_synchronize)
        np.testing.assert_allclose(
            hsdp_param.sharded_param.grad.asnumpy(),
            np.array([1.0, 2.0], dtype=np.float32),
        )
        self.assertIsNone(hsdp_param._unsharded_param.grad)

    def test_clear_output_helpers_release_context_tensors(self):
        """Explicit clear helpers should drop completed communication outputs."""
        hsdp_param = _bare_param()
        hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output = object()
        hsdp_param.all_reduce_comm_ctx.all_reduce_output = object()
        hsdp_param.clear_reduce_scatter_output()
        hsdp_param.clear_all_reduce_output()
        self.assertIsNone(hsdp_param.reduce_scatter_comm_ctx.reduce_scatter_output)
        self.assertIsNone(hsdp_param.all_reduce_comm_ctx.all_reduce_output)


class TestParameterHookMigrator(MindSporeFullyShardUnitTest):
    """Verify parameter backward hooks retain their existing migration semantics."""

    def test_save_deduplicates_hooks_and_migrate_runs_once(self):
        """Repeated saves should deduplicate hooks and each target should migrate once."""
        hook_a = MagicMock()
        hook_b = MagicMock()
        source_param = MagicMock()
        source_param.hooks.return_value = [hook_a, hook_a, hook_b]
        migrator = ParameterHookMigrator()

        migrator._save_backward_hooks(source_param)
        migrator._save_backward_hooks(source_param)

        self.assertEqual(migrator._orig_param_hooks, [hook_a, hook_b])

        class _TargetParam:
            """Minimal parameter double for hook registration."""

            requires_grad = True

            def __init__(self) -> None:
                """Initialize the mocked hook registration method."""
                self.register_hook = MagicMock()

        target_param = _TargetParam()
        migrator._migrate_backward_hooks(target_param)
        migrator._migrate_backward_hooks(target_param)

        registered_hooks = [hook_call.args[0] for hook_call in target_param.register_hook.call_args_list]
        self.assertEqual(registered_hooks, [hook_a, hook_b])
        self.assertTrue(vars(target_param)["migrate_backward_hooks_run_once"])

    def test_migrate_continues_after_registration_error_and_marks_frozen_target(self):
        """Registration errors should not stop later hooks, and frozen targets should be marked."""
        hook_a = MagicMock()
        hook_b = MagicMock()
        source_param = MagicMock()
        source_param.hooks.return_value = [hook_a, hook_b]
        migrator = ParameterHookMigrator()
        migrator._save_backward_hooks(source_param)

        class _TargetParam:
            """Minimal parameter double for hook migration."""

            def __init__(self, requires_grad: bool) -> None:
                """Initialize the gradient flag and mocked registration method."""
                self.requires_grad = requires_grad
                self.register_hook = MagicMock()

        target_param = _TargetParam(requires_grad=True)
        target_param.register_hook.side_effect = [RuntimeError("cannot register"), None]
        migrator._migrate_backward_hooks(target_param)

        self.assertEqual(target_param.register_hook.call_count, 2)
        self.assertTrue(vars(target_param)["migrate_backward_hooks_run_once"])

        frozen_param = _TargetParam(requires_grad=False)
        migrator._migrate_backward_hooks(frozen_param)

        frozen_param.register_hook.assert_not_called()
        self.assertTrue(vars(frozen_param)["migrate_backward_hooks_run_once"])


if __name__ == "__main__":
    unittest.main()
