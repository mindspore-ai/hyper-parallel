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
"""Parameter-specific mixed precision and weak ownership on CPU."""

import unittest
import weakref
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, clear_device_mesh_cache
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.common.mark_utils import arg_mark


class TestMixedPrecisionCustomParams(unittest.TestCase):
    """Verify overrides through real parameter replacement, mocking only the mesh backend."""

    def setUp(self) -> None:
        """Isolate metadata-only meshes between tests."""
        clear_device_mesh_cache()
        self.addCleanup(clear_device_mesh_cache)

    def _shard_model(self, model: nn.Module, policy: MixedPrecisionPolicy, comm_fusion: bool = False) -> None:
        """Apply FSDP on CPU without a process group or accelerator stream."""
        group = MagicMock(spec=torch.distributed.ProcessGroup)
        group.rank.return_value = 0
        group.size.return_value = 2
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0), \
                patch.object(DeviceMesh, "get_group", return_value=group), \
                patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu")):
            mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("fsdp",), _init_backend=False)
            fully_shard(model, mesh=mesh, mp_policy=policy, comm_fusion=comm_fusion)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fields_override_policy_and_preserve_storage_dtype(self) -> None:
        """Check field precedence and preserve both FP32 and BF16 parameter storage."""
        cases = (
            (None, torch.float32, torch.bfloat16, torch.float32),
            ({"param_dtype": torch.float32, "reduce_dtype": torch.float32},
             torch.float32, torch.float32, torch.float32),
            ({"param_dtype": torch.float16}, torch.float32, torch.float16, torch.float32),
            ({"reduce_dtype": torch.float16}, torch.float32, torch.bfloat16, torch.float16),
            ({"param_dtype": None}, torch.float32, torch.float32, torch.float32),
            ({"reduce_dtype": None}, torch.float32, torch.bfloat16, torch.bfloat16),
            ({"param_dtype": torch.float32, "reduce_dtype": torch.float32},
             torch.bfloat16, torch.float32, torch.float32),
        )
        for custom_dtypes, storage_dtype, compute_dtype, reduce_dtype in cases:
            with self.subTest(custom_dtypes=custom_dtypes, storage_dtype=storage_dtype):
                model = nn.Linear(4, 4, dtype=storage_dtype)
                policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    custom_params={model.weight: custom_dtypes} if custom_dtypes is not None else None,
                )

                self._shard_model(model, policy)
                state = model.hsdp_scheduler.hsdp_state
                state._init_mp_dtypes()  # pylint: disable=protected-access

                self.assertEqual(state.hsdp_params[0].all_gather_inputs[0].dtype, compute_dtype)
                self.assertEqual(
                    state.hsdp_params[0].reduce_comm_dtype(torch.ones(4, dtype=compute_dtype)), reduce_dtype
                )
                self.assertEqual(model.weight.dtype, storage_dtype)
                self.assertEqual(state.hsdp_params[1].all_gather_inputs[0].dtype, torch.bfloat16)
                self.assertEqual(state.hsdp_params[1].reduce_comm_dtype(), torch.float32)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_original_parameter_and_storage_are_released(self) -> None:
        """Preserve identity-specific overrides without retaining replaced weights or storage."""
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones(8, 4))
        model.register_parameter("tied_weight", model.weight)
        model.other_weight = nn.Parameter(torch.ones(8, 4))
        parameter_refs = [weakref.ref(model.weight), weakref.ref(model.other_weight)]
        storage_refs = [StorageWeakRef(model.weight.untyped_storage()),
                        StorageWeakRef(model.other_weight.untyped_storage())]
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            custom_params={
                model.weight: {"param_dtype": torch.float32, "reduce_dtype": torch.float32},
                model.other_weight: {"param_dtype": torch.float16, "reduce_dtype": torch.float16},
            },
        )

        self._shard_model(model, policy)
        state = model.hsdp_scheduler.hsdp_state
        state._init_mp_dtypes()  # pylint: disable=protected-access
        state._init_mp_dtypes()  # pylint: disable=protected-access

        self.assertTrue(all(reference() is None for reference in parameter_refs))
        self.assertTrue(all(reference.expired() for reference in storage_refs))
        self.assertIs(model.weight, model.tied_weight)
        self.assertEqual(len(state.hsdp_params), 2)
        self.assertEqual([param.all_gather_inputs[0].dtype for param in state.hsdp_params],
                         [torch.float32, torch.float16])
        self.assertEqual(
            [param.reduce_comm_dtype(torch.ones(4, dtype=dtype)) for param, dtype in
             zip(state.hsdp_params, (torch.float32, torch.float16))],
            [torch.float32, torch.float16],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_native_dtensor_parameter_uses_its_override(self) -> None:
        """Feature: Native DTensor precision overrides.
        Description: Apply FSDP to an original DTensor Parameter with an FP32 override.
        Expectation: Source metadata stays valid and dtype overrides survive weak-key expiration.
        """
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0), \
                patch.object(DeviceMesh, "get_group", return_value=None), \
                patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu")):
            mesh = DeviceMesh(
                "cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("fsdp", "tp"), _init_backend=False
            )
            model = nn.Module()
            model.weight = nn.Parameter(DTensor.from_local(torch.ones(8, 4), mesh["tp"], (Replicate(),)))
            policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                custom_params={model.weight: {"param_dtype": torch.float32, "reduce_dtype": torch.float32}},
            )

            fully_shard(model, mesh=mesh["fsdp"], mp_policy=policy)

        state = model.hsdp_scheduler.hsdp_state
        state._init_mp_dtypes()  # pylint: disable=protected-access

        self.assertEqual(len(policy.custom_params), 0)
        self.assertTrue(state.hsdp_params[0].source_shard_info.origin_is_dtensor)
        self.assertEqual(state.hsdp_params[0].all_gather_inputs[0].dtype, torch.float32)
        self.assertEqual(state.hsdp_params[0].reduce_comm_dtype(torch.ones(4)), torch.float32)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fused_communication_uses_overridden_dtypes(self) -> None:
        """Feature: Mixed-dtype communication fusion.
        Description: Build all-gather and reduce-scatter buckets for FP32 and BF16 weights in one FSDP unit.
        Expectation: Buckets use each parameter's effective compute and reduction dtype.
        """
        model = nn.Linear(4, 4)
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            custom_params={model.weight: {"param_dtype": torch.float32, "reduce_dtype": torch.float32}},
        )

        self._shard_model(model, policy, comm_fusion=True)
        state = model.hsdp_scheduler.hsdp_state
        state._init_mp_dtypes()  # pylint: disable=protected-access
        state.param_group._init_all_gather_buckets()  # pylint: disable=protected-access
        for hsdp_param in state.hsdp_params:
            hsdp_param.unsharded_accumulated_grad = torch.ones(
                hsdp_param._orig_size,  # pylint: disable=protected-access
                dtype=hsdp_param.param_dtype or hsdp_param.orig_dtype,
            )
        reduce_buckets = state.param_group._build_reduce_scatter_buckets(  # pylint: disable=protected-access
            torch.distributed.ReduceOp.SUM
        )

        self.assertEqual([bucket.dtype for bucket in state.param_group.all_gather_buckets],
                         [torch.float32, torch.bfloat16])
        self.assertEqual([bucket.dtype for bucket in reduce_buckets], [torch.float32, torch.bfloat16])
        self.assertIs(state.param_group.all_gather_buckets[0].hsdp_params[0], state.hsdp_params[0])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_custom_params_fail_at_policy_construction(self) -> None:
        """Feature: Core precision configuration validation.
        Description: Supply FQN keys, tuple values, unknown fields, and non-Torch dtypes.
        Expectation: Policy construction raises ValueError before any parameter is sharded.
        """
        parameter = nn.Parameter(torch.ones(4))
        for custom_params in (
            [],
            {"weight": {"param_dtype": torch.float32}},
            {torch.ones(4): {"param_dtype": torch.float32}},
            {parameter: (torch.float32, torch.float32)},
            {parameter: {"output_dtype": torch.float32}},
            {parameter: {"param_dtype": "float32"}},
        ):
            with self.subTest(custom_params=custom_params), self.assertRaisesRegex(ValueError, "custom_params"):
                MixedPrecisionPolicy(custom_params=custom_params)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_override_survives_meta_materialization(self) -> None:
        """Feature: Deferred precision initialization.
        Description: Materialize an FP32 meta weight as BF16 after the original weak key has expired.
        Expectation: Lazy dtype refresh restores the FP32 compute override from the parameter's resolved policy.
        """
        model = nn.Module()
        model.weight = nn.Parameter(torch.empty(8, 4, device="meta", dtype=torch.float32))
        parameter_ref = weakref.ref(model.weight)
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            custom_params={model.weight: {"param_dtype": torch.float32, "reduce_dtype": torch.float32}},
        )
        self._shard_model(model, policy)
        state = model.hsdp_scheduler.hsdp_state
        self.assertIsNone(parameter_ref())
        self.assertEqual(len(policy.custom_params), 0)

        model.weight = nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
        state.lazy_init()
        state.lazy_init()

        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(state.hsdp_params[0].mp_policy.param_dtype, torch.float32)
        self.assertEqual(state.hsdp_params[0].all_gather_inputs[0].dtype, torch.float32)
        self.assertEqual(state.hsdp_params[0].reduce_comm_dtype(torch.ones(4)), torch.float32)
