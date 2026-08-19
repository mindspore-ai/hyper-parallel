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
"""Unit tests for MindSpore fully_shard DTensor-aware param handling."""
# pylint: disable=protected-access

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_fully_shard,
)

ensure_mindspore_platform_for_fully_shard()

import mindspore as ms

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, StridedShard
from hyper_parallel.core.fully_shard.hsdp_utils import FullyShardParamMode, GroupInfo, ShardedState
from hyper_parallel.core.fully_shard.utils import FSDPMeshInfo, MixedPrecisionPolicy
from hyper_parallel.platform.mindspore.fully_shard._version_utils import copy_without_bumping_version
from hyper_parallel.platform.mindspore.fully_shard.param import (
    MindSporeHSDPParamV2,
    make_contiguous_strides_for,
    set_requires_grad_if_needed,
)


def _new_hsdp_param_v2() -> MindSporeHSDPParamV2:
    """Build a bare :class:`MindSporeHSDPParamV2` with fields ``__init__`` normally sets."""
    obj = object.__new__(MindSporeHSDPParamV2)
    obj.unsharded_param_buffers = []
    obj.gradient_scaling_factor = None
    obj.mp_policy = MixedPrecisionPolicy()
    return obj


class FakeDTensorPayload:
    """Lightweight DTensor-like object accepted by unwrap_dtensor_param()."""

    def __init__(self, mesh="mesh-a", placements=("tp",)):
        self._device_mesh = mesh
        self._placements = placements
        self._local_tensor = object()

    @property
    def device_mesh(self):
        return self._device_mesh

    @property
    def placements(self):
        return self._placements

    def to_local(self):
        return self._local_tensor


class HookSourceParam:
    """Small hook source exposing the MindSpore Tensor.hooks() shape."""

    def __init__(self, hooks):
        self._hooks = hooks

    def hooks(self):
        return list(self._hooks)


class HookableParam:
    """Replacement parameter that records migrated hooks."""

    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad
        self.registered_hooks = []

    def register_hook(self, hook):
        self.registered_hooks.append(hook)
        return SimpleNamespace(remove=lambda: None)


class TestMindSporeParam(unittest.TestCase):
    """Cover DTensor-aware constructor and helper behavior."""

    def _make_module_info(self):
        module = MagicMock()
        module.register_load_state_dict_post_hook.return_value = "hook-handle"
        return SimpleNamespace(
            module=module,
            param_name="weight",
            shared_modules=[],
            shared_param_names=[],
        )

    def test_param_init_requires_resolved_param_mode(self):
        """Construction should fail until state/scheduler resolves the parameter mode."""
        mesh_info = MagicMock(spec=FSDPMeshInfo)

        with self.assertRaisesRegex(AssertionError, "param_mode must be resolved"):
            MindSporeHSDPParamV2(
                param=MagicMock(),
                module_info=self._make_module_info(),
                mesh_info=mesh_info,
                device="npu",
                param_mode=None,
            )

    @patch.object(MindSporeHSDPParamV2, "_init_group_infos")
    @patch.object(MindSporeHSDPParamV2, "_init_sharded_param")
    def test_param_init_tracks_dtensor_metadata_and_group_init(self, mock_init_sharded_param, mock_init_group_infos):
        """DTensor-managed params should record original mesh/layout metadata up front."""
        mesh_info = MagicMock(spec=FSDPMeshInfo)
        mesh_info.shard_mesh_dim = 0
        mesh_info.replicate_mesh_dim = 1
        param = FakeDTensorPayload(mesh="tp-mesh", placements=("tp",))

        hsdp_param = MindSporeHSDPParamV2(
            param=param,
            module_info=self._make_module_info(),
            mesh_info=mesh_info,
            device="npu",
            param_mode=FullyShardParamMode.DTENSOR_UNIFIED,
        )

        mock_init_sharded_param.assert_called_once_with(param, None)
        mock_init_group_infos.assert_called_once()
        self.assertTrue(hsdp_param._orig_param_is_dtensor)
        self.assertEqual(hsdp_param._orig_dtensor_mesh, "tp-mesh")
        self.assertEqual(hsdp_param._orig_dtensor_placements, ("tp",))
        self.assertEqual(hsdp_param._spmd_shard_mesh_dim, 0)
        self.assertEqual(hsdp_param._spmd_replicate_mesh_dim, 1)
        self.assertTrue(hsdp_param.uses_param_shard)

    def test_unsharded_grad_data_normalizes_dtensor_grad(self):
        """Gradient accessor should normalize DTensor grads back to local tensors."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._unsharded_param = SimpleNamespace(grad="dtensor-grad")
        hsdp_param._to_local_unsharded_grad = MagicMock(return_value="local-grad")

        grad = hsdp_param.unsharded_grad_data

        hsdp_param._to_local_unsharded_grad.assert_called_once_with("dtensor-grad")
        self.assertEqual(grad, "local-grad")

    def test_get_data_parallel_shard_placement_writes_strided_shard_for_same_dim_layout(self):
        """Same-dim TP/FSDP layouts should materialize a StridedShard on the explicit fully_shard axis."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._spmd_shard_mesh_dim = 1
        hsdp_param._spmd_mesh = SimpleNamespace(mesh_shape=(2, 4, 2))

        placement = MindSporeHSDPParamV2._get_data_parallel_shard_placement(
            hsdp_param,
            [Replicate(), Replicate(), Shard(0)],
            Shard(0),
        )

        self.assertEqual(placement, StridedShard(0, split_factor=2))

    def test_get_data_parallel_shard_placement_keeps_plain_shard_without_same_dim_tp(self):
        """No same-dim TP shard means the fully_shard placement stays plain."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._spmd_shard_mesh_dim = 1
        hsdp_param._spmd_mesh = SimpleNamespace(mesh_shape=(2, 4))

        placement = MindSporeHSDPParamV2._get_data_parallel_shard_placement(
            hsdp_param,
            [Replicate(), Replicate()],
            Shard(0),
        )

        self.assertEqual(placement, Shard(0))

    def test_make_contiguous_strides_for_row_and_column_major_shapes(self):
        """Stride helper should match row-major and supported column-major layouts."""
        self.assertEqual(make_contiguous_strides_for((2, 3, 4)), (12, 4, 1))
        self.assertEqual(make_contiguous_strides_for((2, 3, 4), row_major=False), (12, 1, 3))
        self.assertEqual(make_contiguous_strides_for((5,), row_major=False), (1,))
        self.assertEqual(make_contiguous_strides_for(()), ())
        self.assertEqual(make_contiguous_strides_for((0, 3)), (3, 1))

    def test_make_contiguous_strides_for_rejects_invalid_shapes(self):
        """Stride helper validates the shape contract at its public boundary."""
        with self.assertRaisesRegex(TypeError, "shape must be a tuple or list"):
            make_contiguous_strides_for("bad-shape")
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            make_contiguous_strides_for((2, -1))

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.Parameter")
    @patch(
        "hyper_parallel.platform.mindspore.fully_shard.param."
        "MindSporeHSDPParamV2._get_unsharded_param_from_all_gather_output"
    )
    def test_init_unsharded_param_preserves_dtensor_wrapper(
        self,
        mock_get_unsharded_param,
        mock_parameter,
    ):
        """DTensor-managed params should create the wrapper once and then refresh it in place."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._orig_param_is_dtensor = True
        hsdp_param._unsharded_param = None
        hsdp_param.sharded_param = SimpleNamespace(name="weight", requires_grad=True)
        mock_get_unsharded_param.return_value = "dtensor-unsharded"
        wrapped_parameter = MagicMock(name="wrapped-parameter")
        mock_parameter.return_value = wrapped_parameter

        MindSporeHSDPParamV2.init_unsharded_param(hsdp_param)

        mock_get_unsharded_param.assert_called_once_with()
        mock_parameter.assert_called_once_with(
            "dtensor-unsharded",
            name="weight",
            requires_grad=True,
        )
        self.assertIs(hsdp_param._unsharded_param, wrapped_parameter)

        mock_parameter.reset_mock()
        mock_get_unsharded_param.reset_mock()
        mock_get_unsharded_param.return_value = "next-dtensor-unsharded"
        wrapped_parameter.grad = "old-grad"

        MindSporeHSDPParamV2.init_unsharded_param(hsdp_param)

        mock_parameter.assert_not_called()
        wrapped_parameter.set_data.assert_called_once_with("next-dtensor-unsharded")
        self.assertIsNone(wrapped_parameter.grad)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.Parameter")
    @patch(
        "hyper_parallel.platform.mindspore.fully_shard.param."
        "MindSporeHSDPParamV2._get_unsharded_param_from_all_gather_output"
    )
    def test_init_unsharded_param_preserves_frozen_requires_grad_on_creation(
        self,
        mock_get_unsharded_param,
        mock_parameter,
    ):
        """Frozen local params should pass requires_grad=False when creating the unsharded Parameter."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param._unsharded_param = None
        hsdp_param.sharded_param = SimpleNamespace(name="frozen_weight", requires_grad=False)
        mock_get_unsharded_param.return_value = "frozen-local"
        created_parameter = MagicMock(name="created-parameter")
        mock_parameter.return_value = created_parameter

        MindSporeHSDPParamV2.init_unsharded_param(hsdp_param)

        mock_get_unsharded_param.assert_called_once_with()
        mock_parameter.assert_called_once_with(
            [],
            name="frozen_weight",
            requires_grad=False,
        )
        self.assertIs(hsdp_param._unsharded_param, created_parameter)
        self.assertEqual(created_parameter.data, "frozen-local")

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.set_requires_grad_if_needed")
    @patch(
        "hyper_parallel.platform.mindspore.fully_shard.param."
        "MindSporeHSDPParamV2._get_unsharded_param_from_all_gather_output"
    )
    def test_init_unsharded_param_refreshes_existing_local_parameter_data(
        self,
        mock_get_unsharded_param,
        mock_set_requires_grad_if_needed,
    ):
        """Existing local Parameters should be refreshed in place with the latest unpacked tensor."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._orig_param_is_dtensor = False
        hsdp_param.sharded_param = SimpleNamespace(name="weight", requires_grad=True)
        existing_param = MagicMock(name="unsharded_param")
        existing_param.grad = "old-grad"
        hsdp_param._unsharded_param = existing_param
        mock_get_unsharded_param.return_value = "new-local"

        MindSporeHSDPParamV2.init_unsharded_param(hsdp_param)

        mock_get_unsharded_param.assert_called_once_with()
        mock_set_requires_grad_if_needed.assert_called_once_with(hsdp_param.sharded_param, existing_param)
        self.assertEqual(existing_param.data, "new-local")
        self.assertIsNone(existing_param.grad)

    def test_get_unsharded_param_data_uses_cast_all_gather_input_on_no_comm_path(self):
        """The no-communication path should preserve dtype casts already applied in all_gather_inputs."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.is_sharded = False
        hsdp_param.sharded_state = MagicMock()
        hsdp_param.unsharded_param_buffers = []
        hsdp_param.reset_sharded_param = MagicMock()
        cast_input = MagicMock()
        cast_input.numel.return_value = 8
        cast_input.dtype = "float16"
        cast_input.device = "npu:0"
        output = MagicMock()

        hsdp_param.init_unsharded_param_buffers = MagicMock(
            side_effect=lambda **kwargs: setattr(
                hsdp_param, "unsharded_param_buffers", [output]
            )
        )
        hsdp_param.alloc_unsharded_param_buffers = MagicMock()

        with patch.object(
            MindSporeHSDPParamV2,
            "all_gather_inputs",
            new_callable=PropertyMock,
            return_value=[cast_input],
        ):
            gathered, handle = MindSporeHSDPParamV2._get_unsharded_param_data(hsdp_param)

        hsdp_param.init_unsharded_param_buffers.assert_called_once_with(
            all_gather_input_numels=[8],
            all_gather_input_dtypes=["float16"],
            world_size=1,
            device="npu",
        )
        hsdp_param.alloc_unsharded_param_buffers.assert_called_once()
        output.copy_.assert_not_called()
        output.data.copy_.assert_called_once_with(cast_input)
        self.assertIs(gathered, output)
        self.assertIsNone(handle)

    def test_copy_without_bumping_version_prefers_data_alias(self):
        """Shared helper should write through ``dst.data``."""
        dst = MagicMock(name="dst")
        src = MagicMock(name="src")

        copy_without_bumping_version(dst, src)

        dst.copy_.assert_not_called()
        dst.data.copy_.assert_called_once_with(src)

    def test_release_full_param_storage_if_safe_shrinks_plain_tensor_storage(self):
        """Plain full params should release their original storage after sharded replacement."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._orig_param_is_dtensor = False
        storage = MagicMock()
        storage.size.return_value = 128
        param_data = MagicMock()
        param_data.is_meta = False
        param_data.untyped_storage.return_value = storage

        MindSporeHSDPParamV2._release_full_param_storage_if_safe(hsdp_param, param_data)

        storage.resize_.assert_called_once_with(0)

    def test_release_full_param_storage_if_safe_shrinks_dtensor_local_storage(self):
        """DTensor local tensors should also release their original storage after sharding."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._orig_param_is_dtensor = True
        storage = MagicMock()
        storage.size.return_value = 128
        param_data = MagicMock()
        param_data.is_meta = False
        param_data.untyped_storage.return_value = storage

        MindSporeHSDPParamV2._release_full_param_storage_if_safe(hsdp_param, param_data)

        storage.resize_.assert_called_once_with(0)

    def test_release_full_param_storage_if_safe_skips_meta_inputs(self):
        """Meta tensors should keep their storage untouched."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._orig_param_is_dtensor = False
        storage = MagicMock()
        storage.size.return_value = 128
        param_data = MagicMock()
        param_data.is_meta = True
        param_data.untyped_storage.return_value = storage

        MindSporeHSDPParamV2._release_full_param_storage_if_safe(hsdp_param, param_data)

        storage.resize_.assert_not_called()

    def test_init_unsharded_param_buffers_reuses_existing_buffers(self):
        """Existing all-gather outputs should be reused unless recreation is forced."""
        hsdp_param = _new_hsdp_param_v2()
        existing_output = MagicMock()
        hsdp_param.unsharded_param_buffers = [existing_output]

        MindSporeHSDPParamV2.init_unsharded_param_buffers(
            hsdp_param,
            [4],
            [ms.float32],
            world_size=2,
            device="Ascend:0",
        )

        self.assertEqual(hsdp_param.unsharded_param_buffers, [existing_output])

    def test_init_unsharded_param_buffers_force_recreates_buffers(self):
        """force_recreate should allocate buffers using the normalized device string."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.unsharded_param_buffers = [MagicMock()]

        MindSporeHSDPParamV2.init_unsharded_param_buffers(
            hsdp_param,
            [4, 2],
            [ms.float32, ms.float16],
            world_size=2,
            device="CPU:0",
            force_recreate=True,
        )

        self.assertEqual(len(hsdp_param.unsharded_param_buffers), 2)
        self.assertEqual(hsdp_param.unsharded_param_buffers[0].numel(), 8)
        self.assertEqual(hsdp_param.unsharded_param_buffers[1].numel(), 4)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.DTensor.from_local")
    def test_get_unsharded_param_from_all_gather_output_restores_dtensor_wrapper(self, mock_from_local):
        """DTensor-origin params should wrap the unpacked local tensor with the original layout."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.unsharded_param_buffers = [ms.Tensor(np.arange(16, dtype=np.float32))]
        hsdp_param.sharded_param = SimpleNamespace(
            _local_tensor=ms.Tensor(np.arange(8, dtype=np.float32).reshape(2, 4))
        )
        hsdp_param.mesh_info = object.__new__(FSDPMeshInfo)
        hsdp_param.mesh_info.shard_mesh_size = 2
        hsdp_param.is_sharded = True
        hsdp_param.hsdp_placement = Shard(0)
        hsdp_param._orig_size = (4, 4)
        hsdp_param._orig_param_is_dtensor = True
        hsdp_param._orig_dtensor_mesh = "orig-mesh"
        hsdp_param._orig_dtensor_placements = (Shard(0),)
        mock_from_local.return_value = "wrapped-dtensor"

        unsharded_param = MindSporeHSDPParamV2._get_unsharded_param_from_all_gather_output(hsdp_param)

        mock_from_local.assert_called_once()
        np.testing.assert_allclose(mock_from_local.call_args.args[0].asnumpy(), np.arange(16, dtype=np.float32).reshape(4, 4))
        self.assertEqual(mock_from_local.call_args.args[1:], ("orig-mesh", (Shard(0),)))
        self.assertEqual(unsharded_param, "wrapped-dtensor")

    def test_get_unsharded_param_from_all_gather_output_requires_single_output(self):
        """Per-param all-gather reconstruction expects one fused output buffer."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.unsharded_param_buffers = []

        with self.assertRaisesRegex(AssertionError, "Expected 1 all_gather_output"):
            MindSporeHSDPParamV2._get_unsharded_param_from_all_gather_output(hsdp_param)

    def test_reduce_scatter_output_waits_for_async_handle(self):
        """Cached reduce-scatter outputs should wait on outstanding async work."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._reduce_scatter_output = "reduced"
        hsdp_param.reduce_scatter_handle = MagicMock()

        reduced = MindSporeHSDPParamV2.reduce_scatter_output(hsdp_param)

        self.assertEqual(reduced, "reduced")
        self.assertIsNone(hsdp_param.reduce_scatter_handle)

    def test_clear_reduce_scatter_output_clears_cached_tensor(self):
        """Clear helper should drop the cached reduce-scatter output."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._reduce_scatter_output = "reduced"

        MindSporeHSDPParamV2.clear_reduce_scatter_output(hsdp_param)

        self.assertIsNone(hsdp_param._reduce_scatter_output)

    def test_all_reduce_output_waits_for_async_handle(self):
        """Cached all-reduce outputs should wait on outstanding async work."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._all_reduce_output = "reduced"
        hsdp_param.all_reduce_handle = MagicMock()

        reduced = MindSporeHSDPParamV2.all_reduce_output(hsdp_param)

        self.assertEqual(reduced, "reduced")
        self.assertIsNone(hsdp_param.all_reduce_handle)

    def test_clear_all_reduce_output_clears_cached_tensor(self):
        """Clear helper should drop the cached all-reduce output."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._all_reduce_output = "reduced"

        MindSporeHSDPParamV2.clear_all_reduce_output(hsdp_param)

        self.assertIsNone(hsdp_param._all_reduce_output)

    def test_save_backward_hooks_reads_mindspore_hooks_and_deduplicates(self):
        """MindSpore Tensor.hooks() should be used as the source for user param hooks."""
        hsdp_param = _new_hsdp_param_v2()

        def hook_a(grad):
            return grad

        def hook_b(grad):
            return grad

        source = HookSourceParam([hook_a, hook_a, hook_b])

        MindSporeHSDPParamV2._save_backward_hooks(hsdp_param, source)
        MindSporeHSDPParamV2._save_backward_hooks(hsdp_param, source)

        self.assertEqual(hsdp_param._orig_param_hooks, [hook_a, hook_b])

    def test_setattr_on_modules_migrates_saved_hooks_once(self):
        """Swapping module params should migrate saved hooks to the active replacement once."""
        hsdp_param = _new_hsdp_param_v2()
        primary = SimpleNamespace()
        shared = SimpleNamespace()
        hsdp_param._module_info = SimpleNamespace(
            module=primary,
            param_name="weight",
            shared_modules=[shared],
            shared_param_names=["tied_weight"],
        )

        def hook(grad):
            return grad

        hsdp_param.sharded_param = HookSourceParam([hook])
        new_param = HookableParam(requires_grad=True)

        MindSporeHSDPParamV2._setattr_on_modules(hsdp_param, new_param)
        MindSporeHSDPParamV2._setattr_on_modules(hsdp_param, new_param)

        self.assertIs(primary.weight, new_param)
        self.assertIs(shared.tied_weight, new_param)
        self.assertEqual(new_param.registered_hooks, [hook])
        self.assertTrue(new_param.migrate_backward_hooks_run_once)

    def test_migrate_backward_hooks_skips_frozen_params(self):
        """Saved hooks should not be registered on parameters that do not require gradients."""
        hsdp_param = _new_hsdp_param_v2()

        def hook(grad):
            return grad

        hsdp_param._orig_param_hooks = [hook]
        frozen_param = HookableParam(requires_grad=False)

        MindSporeHSDPParamV2._migrate_backward_hooks(hsdp_param, frozen_param)

        self.assertEqual(frozen_param.registered_hooks, [])
        self.assertTrue(frozen_param.migrate_backward_hooks_run_once)

    def test_setattr_on_modules_updates_primary_and_shared_owners(self):
        """Parameter swapping should keep primary and shared module owners in sync."""
        hsdp_param = _new_hsdp_param_v2()
        primary = SimpleNamespace()
        shared = SimpleNamespace()
        hsdp_param._module_info = SimpleNamespace(
            module=primary,
            param_name="weight",
            shared_modules=[shared],
            shared_param_names=["tied_weight"],
        )

        MindSporeHSDPParamV2._setattr_on_modules(hsdp_param, "new-param")

        self.assertEqual(primary.weight, "new-param")
        self.assertEqual(shared.tied_weight, "new-param")

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.DTensor.from_local")
    def test_to_sharded_dtensor_uses_current_sharding_spec(self, mock_from_local):
        """Reduced grads should be wrapped with the same mesh and placements as the shard."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param._sharding_spec = SimpleNamespace(mesh="mesh", placements=("fsdp",))
        tensor = ms.Tensor(np.ones((2,), dtype=np.float32))
        mock_from_local.return_value = "dtensor"

        self.assertEqual(MindSporeHSDPParamV2.to_sharded_dtensor(hsdp_param, tensor), "dtensor")
        mock_from_local.assert_called_once_with(tensor, "mesh", ("fsdp",))

    def test_grad_accumulation_helpers_cast_and_merge_unsharded_grads(self):
        """Unsharded grads should be accumulated in reduce dtype and cleared from Parameter.grad."""
        hsdp_param = _new_hsdp_param_v2()
        first_grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        hsdp_param._unsharded_param = SimpleNamespace(grad=first_grad)
        hsdp_param.reduce_dtype = ms.float16
        hsdp_param.unsharded_accumulated_grad = None

        MindSporeHSDPParamV2.to_accumulated_grad_if_needed(hsdp_param)

        self.assertIsNone(hsdp_param._unsharded_param.grad)
        self.assertEqual(hsdp_param.unsharded_accumulated_grad.dtype, ms.float16)

        hsdp_param._unsharded_param.grad = ms.Tensor(np.ones((2,), dtype=np.float16))
        MindSporeHSDPParamV2.to_accumulated_grad_if_needed(hsdp_param)
        np.testing.assert_allclose(hsdp_param.unsharded_accumulated_grad.asnumpy(), np.full((2,), 2.0, dtype=np.float16))

    def test_accumulate_unsharded_grad_if_needed_normalizes_new_grad(self):
        """Pending accumulated grad should absorb the latest local unsharded grad."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.unsharded_accumulated_grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        hsdp_param._unsharded_param = SimpleNamespace(grad="dtensor-grad")
        hsdp_param._to_local_unsharded_grad = MagicMock(return_value=ms.Tensor(np.full((2,), 3.0, dtype=np.float32)))

        MindSporeHSDPParamV2.accumulate_unsharded_grad_if_needed(hsdp_param)

        np.testing.assert_allclose(hsdp_param.unsharded_accumulated_grad.asnumpy(), np.full((2,), 4.0, dtype=np.float32))
        self.assertIsNone(hsdp_param._unsharded_param.grad)

    def test_all_gather_inputs_respects_state_offload_and_param_dtype(self):
        """All-gather inputs should cast only after sharded-state validation and optional offload."""
        hsdp_param = _new_hsdp_param_v2()
        source = MagicMock()
        offloaded = MagicMock()
        casted = MagicMock()
        source.dtype = ms.float32
        source.to.return_value = offloaded
        offloaded.dtype = ms.float32
        offloaded.to.return_value = casted
        hsdp_param._sharded_param_data = source
        hsdp_param.offload_to_cpu = True
        hsdp_param.device = "npu"
        hsdp_param.param_dtype = ms.float16
        hsdp_param._assert_in_states = MagicMock()

        self.assertEqual(MindSporeHSDPParamV2.all_gather_inputs.__get__(hsdp_param), [casted])
        hsdp_param._assert_in_states.assert_called_once_with(ShardedState.SHARDED)
        source.to.assert_called_once_with("npu", non_blocking=True)
        offloaded.to.assert_called_once_with(ms.float16)

    def test_state_transition_helpers_wait_and_delegate_once(self):
        """unshard/wait/shard should update prefetch state through existing helpers."""
        hsdp_param = _new_hsdp_param_v2()
        handle = MagicMock()
        hsdp_param.prefetch_handle = None
        hsdp_param._get_unsharded_param_data = MagicMock(return_value=("output", handle))

        MindSporeHSDPParamV2.unshard(hsdp_param, async_op=True)
        MindSporeHSDPParamV2.unshard(hsdp_param, async_op=True)

        hsdp_param._get_unsharded_param_data.assert_called_once_with(async_op=True)
        self.assertIs(hsdp_param.prefetch_handle, handle)

        hsdp_param._assert_in_states = MagicMock()
        hsdp_param.init_unsharded_param = MagicMock()
        hsdp_param.to_unsharded = MagicMock()
        MindSporeHSDPParamV2.wait_for_unshard(hsdp_param)
        handle.wait.assert_called_once_with()
        self.assertIsNone(hsdp_param.prefetch_handle)
        hsdp_param.init_unsharded_param.assert_called_once_with()
        hsdp_param.to_unsharded.assert_called_once_with()

        hsdp_param.to_sharded = MagicMock()
        MindSporeHSDPParamV2.shard(hsdp_param)
        hsdp_param.to_sharded.assert_called_once_with()

    def test_assert_in_states_and_zero_grad_validate_param_lifecycle(self):
        """State checks should fail loudly and zero_grad should clear sharded grad buffers."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.sharded_state = ShardedState.SHARDED
        hsdp_param.sharded_param = SimpleNamespace(grad="grad", main_grad="main-grad")

        with self.assertRaisesRegex(AssertionError, "Expected sharded_state"):
            MindSporeHSDPParamV2._assert_in_states(hsdp_param, ShardedState.UNSHARDED)
        MindSporeHSDPParamV2.zero_grad(hsdp_param)
        self.assertIsNone(hsdp_param.sharded_param.grad)
        self.assertIsNone(hsdp_param.sharded_param.main_grad)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_all_reduce_grad_uses_accumulated_grad_when_no_explicit_grad(self, mock_all_reduce):
        """All-reduce should prefer accumulated unsharded grad when no explicit grad is passed."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.unsharded_accumulated_grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        hsdp_param.unsharded_group_info = GroupInfo("replica", "replica-group", 2)
        mock_all_reduce.return_value = "handle"

        reduced_grad, handle = MindSporeHSDPParamV2.all_reduce_grad(hsdp_param, dtype=ms.float16, async_op=False)

        self.assertEqual(reduced_grad.dtype, ms.float16)
        self.assertEqual(handle, "handle")
        mock_all_reduce.assert_called_once_with(
            reduced_grad,
            op=mock_all_reduce.call_args.kwargs["op"],
            group="replica-group",
            async_op=False,
        )

    def test_apply_reduced_grad_offloads_new_grad_and_reports_synchronization(self):
        """CPU offload branch should move a newly assigned reduced grad to host memory."""
        hsdp_param = _new_hsdp_param_v2()
        viewed_grad = MagicMock()
        cpu_grad = MagicMock()
        reduced_grad = MagicMock()
        reduced_grad.view.return_value = viewed_grad
        viewed_grad.dtype = ms.float32
        viewed_grad.to.return_value = cpu_grad
        hsdp_param.sharded_size = (2,)
        hsdp_param.sharded_param = SimpleNamespace(grad=None)
        hsdp_param.offload_to_cpu = True
        hsdp_param.pin_memory = True
        hsdp_param.to_sharded_dtensor = MagicMock(return_value="dtensor-grad")
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param._unsharded_param = SimpleNamespace(grad="old-unsharded-grad")

        need_synchronize = MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, ms.float32)

        self.assertTrue(need_synchronize)
        viewed_grad.to.assert_called_once_with("cpu", non_blocking=True)
        hsdp_param.to_sharded_dtensor.assert_called_once_with(cpu_grad)
        self.assertEqual(hsdp_param.sharded_param.grad, "dtensor-grad")
        self.assertIsNone(hsdp_param._unsharded_param.grad)

    def test_set_requires_grad_if_needed_only_updates_when_values_differ(self):
        """requires_grad propagation should avoid unnecessary setter calls."""
        src = SimpleNamespace(requires_grad=True)
        dst = SimpleNamespace(requires_grad=False, requires_grad_=MagicMock())

        set_requires_grad_if_needed(src, dst)
        dst.requires_grad_.assert_called_once_with(True)

        dst.requires_grad = True
        dst.requires_grad_.reset_mock()
        set_requires_grad_if_needed(src, dst)
        dst.requires_grad_.assert_not_called()

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.reduce_scatter_tensor")
    def test_reduce_scatter_grad_supports_same_dim_strided_non_dim0_layout(self, mock_reduce_scatter):
        """same-dim StridedShard(dim!=0) should reuse the non-dim0 chunk-cat packing path."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.sharded_state = ShardedState.UNSHARDED
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param._unsharded_param = SimpleNamespace(
            grad=ms.Tensor(np.arange(32, dtype=np.float32).reshape(4, 8))
        )
        hsdp_param.is_sharded = True
        hsdp_param.mesh_info = MagicMock(spec=FSDPMeshInfo)
        hsdp_param.mesh_info.shard_process_group = "pg"
        hsdp_param.mesh_info.shard_mesh_size = 2
        hsdp_param.enable_fsdp_shard = True
        hsdp_param.hsdp_placement = Shard(1)
        hsdp_param._orig_size = (4, 8)
        hsdp_param._orig_param_is_dtensor = True
        hsdp_param._orig_dtensor_placements = (Shard(1),)
        hsdp_param._spmd_shard_mesh_dim = 0
        hsdp_param._spmd_placements = (StridedShard(1, split_factor=2), Shard(1))

        reduced_grad, _ = MindSporeHSDPParamV2.reduce_scatter_grad(hsdp_param, async_op=False)

        expected_packed = np.concatenate(
            np.array_split(np.arange(32, dtype=np.float32).reshape(4, 8), 2, axis=1),
            axis=0,
        ).reshape(-1)
        self.assertEqual(reduced_grad.numel(), 16)
        np.testing.assert_allclose(mock_reduce_scatter.call_args.args[1].asnumpy(), expected_packed)

    @patch("hyper_parallel.platform.mindspore.fully_shard.param.dist.all_reduce")
    def test_all_reduce_grad_uses_layout_driven_unsharded_group(self, mock_all_reduce):
        """
        Gradient all-reduce should follow layout-driven group bookkeeping.

        This should no longer depend on ``mesh_info.replicate_process_group``.
        """
        raw_grad = MagicMock(name="raw-grad")
        normalized_grad = MagicMock(name="normalized-grad")
        normalized_grad.contiguous.return_value = normalized_grad
        mock_all_reduce.return_value = "reduce-handle"

        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param.unsharded_group_info = GroupInfo("unsharded-group", "layout-group", 4)
        hsdp_param._to_local_unsharded_grad = MagicMock(return_value=normalized_grad)

        reduced_grad, handle = MindSporeHSDPParamV2.all_reduce_grad(
            hsdp_param,
            grad=raw_grad,
            async_op=True,
            reduce_op="sum",
        )

        hsdp_param._to_local_unsharded_grad.assert_called_once_with(raw_grad)
        normalized_grad.contiguous.assert_called_once_with()
        mock_all_reduce.assert_called_once_with(
            normalized_grad,
            op="sum",
            group="layout-group",
            async_op=True,
        )
        self.assertIs(reduced_grad, normalized_grad)
        self.assertEqual(handle, "reduce-handle")

    def test_all_reduce_grad_returns_local_grad_for_single_rank_group(self):
        """rank_size <= 1 should avoid distributed all-reduce."""
        grad = MagicMock(name="grad")
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param.unsharded_group_info = GroupInfo("unsharded-group", None, 1)
        hsdp_param._to_local_unsharded_grad = MagicMock(return_value=grad)

        reduced_grad, handle = MindSporeHSDPParamV2.all_reduce_grad(hsdp_param, grad=grad)

        self.assertIs(reduced_grad, grad)
        self.assertIsNone(handle)
        self.assertIs(hsdp_param._all_reduce_output, grad)

    def test_all_reduce_grad_rejects_missing_group_for_multi_rank(self):
        """Multi-rank all-reduce requires a concrete process group."""
        grad = MagicMock(name="grad")
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param.unsharded_group_info = GroupInfo("unsharded-group", None, 2)
        hsdp_param._to_local_unsharded_grad = MagicMock(return_value=grad)

        with self.assertRaisesRegex(RuntimeError, "valid unsharded all-reduce group"):
            MindSporeHSDPParamV2.all_reduce_grad(hsdp_param, grad=grad)

    def test_apply_reduced_grad_assigns_new_sharded_dtensor_grad(self):
        """Reduced grads should be reshaped, cast, wrapped, and assigned when no grad exists."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.sharded_size = (2, 2)
        hsdp_param.sharded_param = SimpleNamespace(grad=None)
        hsdp_param.offload_to_cpu = False
        hsdp_param.to_sharded_dtensor = MagicMock(return_value="dtensor-grad")
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param._unsharded_param = SimpleNamespace(grad="old-unsharded-grad")
        reduced_grad = ms.Tensor(np.arange(4, dtype=np.float32))

        need_synchronize = MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, ms.float16)

        self.assertFalse(need_synchronize)
        hsdp_param.to_sharded_dtensor.assert_called_once()
        self.assertEqual(hsdp_param.sharded_param.grad, "dtensor-grad")
        self.assertIsNone(hsdp_param._unsharded_param.grad)

    def test_apply_reduced_grad_aligns_to_sharded_storage_dtype(self):
        """Issue #215: align reduced grad dtype with sharded param storage before writeback."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.sharded_size = (4,)
        hsdp_param.sharded_param = SimpleNamespace(dtype=ms.bfloat16, grad=None)
        hsdp_param.offload_to_cpu = False
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param._unsharded_param = SimpleNamespace(grad="old-unsharded-grad")
        captured = {}

        def _capture_dtensor(tensor):
            captured["dtype"] = tensor.dtype
            return "dtensor-grad"

        hsdp_param.to_sharded_dtensor = _capture_dtensor
        reduced_grad = ms.Tensor(np.ones((4,), dtype=np.float32))

        MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, ms.float32)

        self.assertEqual(captured["dtype"], ms.bfloat16)

    def test_apply_reduced_grad_accumulates_existing_local_grad(self):
        """Existing sharded DTensor grads should accumulate in-place on the local tensor."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.sharded_size = (2, 2)
        local_grad = ms.Tensor(np.ones((2, 2), dtype=np.float32))
        hsdp_param.sharded_param = SimpleNamespace(grad=SimpleNamespace(_local_tensor=local_grad))
        hsdp_param.offload_to_cpu = False
        hsdp_param.unsharded_accumulated_grad = ms.Tensor(np.ones((2, 2), dtype=np.float32))
        hsdp_param._unsharded_param = SimpleNamespace(grad=None)
        reduced_grad = ms.Tensor(np.ones(4, dtype=np.float32))

        need_synchronize = MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, None)

        self.assertFalse(need_synchronize)
        np.testing.assert_allclose(
            hsdp_param.sharded_param.grad._local_tensor.asnumpy(),
            np.full((2, 2), 2.0, dtype=np.float32),
        )
        self.assertIsNone(hsdp_param.unsharded_accumulated_grad)

    def test_apply_reduced_grad_assigns_main_grad_without_casting_reduced_dtype(self):
        """Main-grad path should keep the reduced-gradient dtype, matching torch."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.mp_policy = MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True)
        hsdp_param.sharded_size = (2, 2)
        hsdp_param.sharded_param = SimpleNamespace(grad="old-grad")
        hsdp_param.offload_to_cpu = False
        hsdp_param.to_sharded_dtensor = MagicMock(side_effect=lambda tensor: SimpleNamespace(_local_tensor=tensor))
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param._unsharded_param = SimpleNamespace(grad="old-unsharded-grad")
        reduced_grad = ms.Tensor(np.arange(4, dtype=np.float16))

        need_synchronize = MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, ms.float16)

        self.assertFalse(need_synchronize)
        self.assertIsNone(hsdp_param.sharded_param.grad)
        main_grad = hsdp_param.sharded_param.main_grad
        self.assertEqual(main_grad._local_tensor.dtype, ms.float16)
        np.testing.assert_allclose(
            main_grad._local_tensor.asnumpy(),
            np.arange(4, dtype=np.float16).reshape(2, 2),
        )
        self.assertIsNone(hsdp_param._unsharded_param.grad)

    def test_apply_reduced_grad_assigns_main_grad_without_unsharded_param(self):
        """Direct DTENSOR_COMPAT applies reduced grads without an unsharded param."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.mp_policy = MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True)
        hsdp_param.sharded_size = (2,)
        hsdp_param.sharded_param = SimpleNamespace(grad="old-grad")
        hsdp_param.offload_to_cpu = False
        hsdp_param.to_sharded_dtensor = MagicMock(side_effect=lambda tensor: SimpleNamespace(_local_tensor=tensor))
        hsdp_param.unsharded_accumulated_grad = None
        hsdp_param._unsharded_param = None
        reduced_grad = ms.Tensor(np.array([1.0, 2.0], dtype=np.float16))

        need_synchronize = MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, ms.float16)

        self.assertFalse(need_synchronize)
        self.assertIsNone(hsdp_param.sharded_param.grad)
        main_grad = hsdp_param.sharded_param.main_grad
        self.assertEqual(main_grad._local_tensor.dtype, ms.float16)
        np.testing.assert_allclose(
            main_grad._local_tensor.asnumpy(),
            np.array([1.0, 2.0], dtype=np.float16),
        )

    def test_apply_reduced_grad_accumulates_existing_main_grad(self):
        """Existing main_grad should accumulate in place and keep grad cleared."""
        hsdp_param = _new_hsdp_param_v2()
        hsdp_param.mp_policy = MixedPrecisionPolicy(apply_grad_on_fp32_main_grad=True)
        hsdp_param.sharded_size = (2,)
        local_main_grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        hsdp_param.sharded_param = SimpleNamespace(
            grad="old-grad",
            main_grad=SimpleNamespace(_local_tensor=local_main_grad),
        )
        hsdp_param.offload_to_cpu = False
        hsdp_param.unsharded_accumulated_grad = ms.Tensor(np.ones((2,), dtype=np.float32))
        hsdp_param._unsharded_param = SimpleNamespace(grad=None)
        reduced_grad = ms.Tensor(np.array([2.0, 3.0], dtype=np.float16))

        need_synchronize = MindSporeHSDPParamV2.apply_reduced_grad(hsdp_param, reduced_grad, ms.float16)

        self.assertFalse(need_synchronize)
        self.assertIsNone(hsdp_param.sharded_param.grad)
        np.testing.assert_allclose(
            hsdp_param.sharded_param.main_grad._local_tensor.asnumpy(),
            np.array([3.0, 4.0], dtype=np.float32),
        )
        self.assertIsNone(hsdp_param.unsharded_accumulated_grad)


if __name__ == "__main__":
    unittest.main()
