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
"""Parameter-specific mixed precision and source identity on CPU."""

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

    def test_fields_override_policy_and_preserve_storage_dtype(self) -> None:
        """Feature: Per-field precision precedence.
        Description: Shard parameters with full, partial, absent, and explicit None overrides.
        Expectation: Compute and reduction use effective dtypes while unlisted parameters keep policy defaults.
        """
        cases = (
            (None, torch.bfloat16, torch.float32),
            ({}, torch.bfloat16, torch.float32),
            ({"param_dtype": torch.float32, "reduce_dtype": torch.float32}, torch.float32, torch.float32),
            ({"param_dtype": torch.float16}, torch.float16, torch.float32),
            ({"reduce_dtype": torch.float16}, torch.bfloat16, torch.float16),
            ({"param_dtype": None}, torch.float32, torch.float32),
            ({"reduce_dtype": None}, torch.bfloat16, torch.bfloat16),
            ({"param_dtype": None, "reduce_dtype": None}, torch.float32, torch.float32),
        )
        for custom_dtypes, compute_dtype, reduce_dtype in cases:
            with self.subTest(custom_dtypes=custom_dtypes):
                model = nn.Linear(4, 4)
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
                self.assertEqual(model.weight.dtype, torch.float32)
                self.assertEqual(state.hsdp_params[1].all_gather_inputs[0].dtype, torch.bfloat16)
                self.assertEqual(state.hsdp_params[1].reduce_comm_dtype(), torch.float32)

    def test_original_parameter_and_storage_are_released(self) -> None:
        """Feature: Source parameter identity without storage retention.
        Description: Keep a policy and shared-weight FSDP module alive after sharding replaces its weight.
        Expectation: Original weight and storage expire; repeated dtype initialization retains the override.
        """
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones(8, 4))
        model.register_parameter("tied_weight", model.weight)
        parameter_ref = weakref.ref(model.weight)
        parameter_id = id(model.weight)
        storage_ref = StorageWeakRef(model.weight.untyped_storage())
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            custom_params={model.weight: {"param_dtype": torch.float32, "reduce_dtype": torch.float32}},
        )

        self._shard_model(model, policy)
        state = model.hsdp_scheduler.hsdp_state
        state._init_mp_dtypes()  # pylint: disable=protected-access
        state._init_mp_dtypes()  # pylint: disable=protected-access

        self.assertIsNone(parameter_ref())
        self.assertTrue(storage_ref.expired())
        self.assertEqual(policy.custom_params, {
            parameter_id: {"param_dtype": torch.float32, "reduce_dtype": torch.float32},
        })
        self.assertIs(model.weight, model.tied_weight)
        self.assertEqual(len(state.hsdp_params), 1)
        self.assertEqual(state.hsdp_params[0]._src_parameter_id, parameter_id)  # pylint: disable=protected-access
        self.assertFalse(hasattr(state.hsdp_params[0], "mp_policy"))
        self.assertEqual(state.hsdp_params[0].all_gather_inputs[0].dtype, torch.float32)
        self.assertEqual(state.hsdp_params[0].reduce_comm_dtype(torch.ones(4)), torch.float32)

    def test_equal_valued_parameters_have_independent_overrides(self) -> None:
        """Feature: Parameter identity matching.
        Description: Configure different dtypes for two multi-element weights containing identical values.
        Expectation: Lookup preserves independent overrides without Tensor equality or value-based matching.
        """
        first = nn.Parameter(torch.ones(2, 3))
        second = nn.Parameter(torch.ones(2, 3))
        policy = MixedPrecisionPolicy(custom_params={
            first: {"param_dtype": torch.float32},
            second: {"param_dtype": torch.float16},
        })

        self.assertEqual(policy.custom_params[id(first)], {"param_dtype": torch.float32})
        self.assertEqual(policy.custom_params[id(second)], {"param_dtype": torch.float16})
        self.assertNotIn(id(nn.Parameter(torch.ones(2, 3))), policy.custom_params)

    def test_native_dtensor_parameter_uses_its_override(self) -> None:
        """Feature: Native DTensor precision overrides.
        Description: Apply FSDP to an original DTensor Parameter with an FP32 override.
        Expectation: Source metadata stays valid and ID-keyed overrides survive parameter replacement.
        """
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0), \
                patch.object(DeviceMesh, "get_group", return_value=None), \
                patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu")):
            mesh = DeviceMesh(
                "cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("fsdp", "tp"), _init_backend=False
            )
            model = nn.Module()
            model.weight = nn.Parameter(DTensor.from_local(torch.ones(8, 4), mesh["tp"], (Replicate(),)))
            parameter_id = id(model.weight)
            policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                custom_params={model.weight: {"param_dtype": torch.float32, "reduce_dtype": torch.float32}},
            )

            fully_shard(model, mesh=mesh["fsdp"], mp_policy=policy)

        state = model.hsdp_scheduler.hsdp_state
        state._init_mp_dtypes()  # pylint: disable=protected-access

        self.assertIn(parameter_id, policy.custom_params)
        self.assertTrue(state.hsdp_params[0].source_shard_info.origin_is_dtensor)
        self.assertEqual(state.hsdp_params[0].all_gather_inputs[0].dtype, torch.float32)
        self.assertEqual(state.hsdp_params[0].reduce_comm_dtype(torch.ones(4)), torch.float32)

    def test_override_values_are_copied(self) -> None:
        """Feature: Policy configuration ownership.
        Description: Modify the input dtype dictionary after creating a policy.
        Expectation: The normalized policy retains its original dtype values.
        """
        parameter = nn.Parameter(torch.ones(4))
        custom_dtypes = {"param_dtype": torch.float32}
        policy = MixedPrecisionPolicy(custom_params={parameter: custom_dtypes})

        custom_dtypes["param_dtype"] = torch.float16

        self.assertEqual(policy.custom_params[id(parameter)], {"param_dtype": torch.float32})

    def test_fused_communication_uses_overridden_dtypes(self) -> None:
        """Feature: Mixed-dtype communication fusion.
        Description: Unshard FP32 and BF16 parameters and bucket direct and accumulated gradients.
        Expectation: All-gather and reduce-scatter use their independently overridden dtypes.
        """
        model = nn.Linear(4, 4)
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            custom_params={model.weight: {"param_dtype": torch.float32, "reduce_dtype": torch.bfloat16}},
        )

        self._shard_model(model, policy, comm_fusion=True)
        state = model.hsdp_scheduler.hsdp_state
        state._init_mp_dtypes()  # pylint: disable=protected-access

        def _fake_all_gather(output, input_tensor, **kwargs):
            output.copy_(input_tensor.repeat(kwargs["group"].size()))

        def _fake_reduce_scatter(**kwargs):
            output = kwargs["output"]
            output.copy_(kwargs["input"].view(kwargs["group"].size(), -1).sum(dim=0))
            return MagicMock()

        with patch("hyper_parallel.core.fully_shard.param_group.dist.all_gather_into_tensor",
                   side_effect=_fake_all_gather) as mock_all_gather:
            state.param_group.unshard()
            state.param_group.wait_for_unshard()

        self.assertEqual([bucket.dtype for bucket in state.param_group.all_gather_buckets],
                         [torch.float32, torch.bfloat16])
        self.assertEqual([call.args[1].dtype for call in mock_all_gather.call_args_list],
                         [torch.float32, torch.bfloat16])
        self.assertEqual([call.args[0].dtype for call in mock_all_gather.call_args_list],
                         [torch.float32, torch.bfloat16])
        self.assertIs(state.param_group.all_gather_buckets[0].hsdp_params[0], state.hsdp_params[0])
        self.assertIs(state.param_group.all_gather_buckets[1].hsdp_params[0], state.hsdp_params[1])

        for accumulated in (False, True):
            with self.subTest(accumulated=accumulated):
                for hsdp_param in state.hsdp_params:
                    hsdp_param.unsharded_accumulated_grad = None
                    hsdp_param.unsharded_param.grad = torch.ones_like(hsdp_param.unsharded_param)
                    if accumulated:
                        hsdp_param.to_accumulated_grad_if_needed()
                with patch("hyper_parallel.core.fully_shard.param_group.dist.reduce_scatter_tensor",
                           side_effect=_fake_reduce_scatter) as mock_reduce_scatter:
                    state.param_group.foreach_reducescatter(torch.distributed.ReduceOp.SUM)

                reduce_buckets = state.param_group.reduce_scatter_buckets
                self.assertEqual([bucket.dtype for bucket in reduce_buckets], [torch.bfloat16, torch.float32])
                self.assertEqual([call.kwargs["input"].dtype for call in mock_reduce_scatter.call_args_list],
                                 [torch.bfloat16, torch.float32])
                self.assertIs(reduce_buckets[0].hsdp_params[0], state.hsdp_params[0])
                self.assertIs(reduce_buckets[1].hsdp_params[0], state.hsdp_params[1])
                state.param_group.wait_reduce_scatter_and_issue_all_reduce()
                for hsdp_param in state.hsdp_params:
                    reduced_grad = hsdp_param.reduce_scatter_output()
                    torch.testing.assert_close(reduced_grad, torch.full_like(reduced_grad, 2))
                    hsdp_param.clear_reduce_scatter_output()
                state.param_group.reset_iter_state()

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

    def test_fp32_override_keeps_original_low_precision_storage(self) -> None:
        """Feature: Separate compute and storage precision.
        Description: Shard a BF16 weight with an FP32 compute and reduction override.
        Expectation: All-gather inputs and reduction use FP32 while sharded storage remains BF16.
        """
        model = nn.Linear(4, 4, dtype=torch.bfloat16)
        policy = MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            custom_params={model.weight: {"param_dtype": torch.float32, "reduce_dtype": torch.float32}},
        )

        self._shard_model(model, policy)
        state = model.hsdp_scheduler.hsdp_state
        state._init_mp_dtypes()  # pylint: disable=protected-access

        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(state.hsdp_params[0].all_gather_inputs[0].dtype, torch.float32)
        self.assertEqual(state.hsdp_params[0].reduce_comm_dtype(torch.ones(4)), torch.float32)

    def test_override_survives_meta_materialization(self) -> None:
        """Feature: Deferred precision initialization.
        Description: Materialize an FP32 meta weight as BF16 after the original Parameter is released.
        Expectation: Lazy dtype initialization resolves the FP32 override using the source parameter ID.
        """
        model = nn.Module()
        model.weight = nn.Parameter(torch.empty(8, 4, device="meta", dtype=torch.float32))
        parameter_ref = weakref.ref(model.weight)
        parameter_id = id(model.weight)
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            custom_params={model.weight: {"param_dtype": torch.float32, "reduce_dtype": torch.float32}},
        )
        self._shard_model(model, policy)
        state = model.hsdp_scheduler.hsdp_state
        self.assertIsNone(parameter_ref())
        self.assertIn(parameter_id, policy.custom_params)

        model.weight = nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16))
        state.lazy_init()
        state.lazy_init()

        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertFalse(hasattr(state.hsdp_params[0], "mp_policy"))
        self.assertEqual(state.hsdp_params[0].param_dtype, torch.float32)
        self.assertEqual(state.hsdp_params[0].all_gather_inputs[0].dtype, torch.float32)
        self.assertEqual(state.hsdp_params[0].reduce_comm_dtype(torch.ones(4)), torch.float32)
