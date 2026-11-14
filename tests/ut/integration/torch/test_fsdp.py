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
"""Framework-independent checks of native Hyper execution through Torch APIs."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.fsdp import CPUOffloadPolicy as TorchCPUOffloadPolicy
from torch.distributed.fsdp import OffloadPolicy as TorchOffloadPolicy
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.distributed.tensor import Shard as TorchShard

from hyper_parallel.core.dtensor.dtensor import DTensor as HyperDTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.core.fully_shard.utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from hyper_parallel.integration import torch as compat
from hyper_parallel.integration.torch import fsdp


class TestTorchFSDPCompatibility(unittest.TestCase):
    """Do not require VeOmni, devices or process-group initialization."""

    def test_exports_are_hyper_types_not_torch_replacements(self):
        """The opt-in namespace uses Hyper objects; Torch itself is unchanged."""
        self.assertIs(compat.DTensor, HyperDTensor)
        self.assertIs(compat.FSDPModule, HSDPModule)
        self.assertIs(torch.distributed.tensor.DTensor, TorchDTensor)
        self.assertIsNot(compat.DTensor, TorchDTensor)

    def test_default_execution_delegates_without_policy_changes(self):
        """No root lifetime, communication order or scheduling options are injected."""
        model = torch.nn.Linear(2, 2)
        with patch.object(fsdp, "hyper_fully_shard", return_value=model) as native:
            self.assertIs(compat.fully_shard(model), model)
        native.assert_called_once_with(model)

    def test_ignored_torch_parameters_remain_model_owned(self):
        """Ignored persistent tables must not enter the managed-layout guard."""
        model = torch.nn.Linear(2, 2)
        ignored = set(model.parameters())
        with (
            patch.object(fsdp, "TorchDTensor", torch.nn.Parameter),
            patch.object(fsdp, "hyper_fully_shard", return_value=model) as native,
        ):
            self.assertIs(compat.fully_shard(model, ignored_params=ignored), model)
        native.assert_called_once_with(model, ignored_params=ignored)

    def test_managed_torch_parameters_still_require_layout_adapter(self):
        """Excluding persistent tables must not allow arbitrary Torch shards."""
        model = torch.nn.Linear(2, 2)
        with (
            patch.object(fsdp, "TorchDTensor", torch.nn.Parameter),
            patch.object(fsdp, "hyper_fully_shard") as native,
            self.assertRaisesRegex(NotImplementedError, "dedicated layout adapter"),
        ):
            compat.fully_shard(model, ignored_params={model.bias})
        native.assert_not_called()

    def test_invalid_fsdp_options_fail_before_mesh_conversion(self):
        """Unknown arguments and integer reshard modes must not start setup."""
        model = torch.nn.Linear(2, 2)
        for options, error in (
            ({"unexpected_option": True}, TypeError),
            ({"reshard_after_forward": 2}, NotImplementedError),
        ):
            with self.subTest(options=options):
                with (
                    patch.object(fsdp, "hyper_fully_shard", autospec=True) as native,
                    patch.object(fsdp, "to_hyper_mesh") as convert,
                    self.assertRaises(error),
                ):
                    compat.fully_shard(model, mesh=object(), **options)
                native.assert_not_called()
                convert.assert_not_called()

    def test_descriptors_and_placement_callback_translate_before_delegation(self):
        """Translate API descriptions but keep all Hyper execution in its core."""
        model = torch.nn.Linear(2, 2)
        mesh, converted_mesh = object(), object()
        policy = SimpleNamespace(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=True,
        )
        placement = MagicMock(return_value=TorchShard(0))
        with (
            patch.object(fsdp, "to_hyper_mesh", return_value=converted_mesh) as convert,
            patch.object(fsdp, "hyper_fully_shard", return_value=model) as native,
        ):
            compat.fully_shard(
                model,
                mesh=mesh,
                mp_policy=policy,
                offload_policy=TorchCPUOffloadPolicy(pin_memory=False),
                shard_placement_fn=placement,
                reshard_after_forward=True,
            )
        convert.assert_called_once_with(mesh)
        actual = native.call_args.kwargs
        self.assertIs(actual["mesh"], converted_mesh)
        self.assertIsInstance(actual["mp_policy"], MixedPrecisionPolicy)
        self.assertEqual(actual["mp_policy"].reduce_dtype, torch.float32)
        self.assertIsInstance(actual["offload_policy"], CPUOffloadPolicy)
        self.assertFalse(actual["offload_policy"].pin_memory)
        self.assertIsInstance(actual["shard_placement_fn"](model.weight), Shard)
        placement.assert_called_once_with(model.weight)
        self.assertTrue(actual["reshard_after_forward"])

    def test_native_policies_are_preserved(self):
        """Existing Hyper policy objects are not rebuilt or downgraded."""
        model = torch.nn.Linear(2, 2)
        policy = MixedPrecisionPolicy()
        for offload in (
            OffloadPolicy(),
            CPUOffloadPolicy(pin_memory=False),
            CPUOffloadPolicy(),
        ):
            with self.subTest(offload=offload):
                with patch.object(fsdp, "hyper_fully_shard") as native:
                    compat.fully_shard(model, mp_policy=policy, offload_policy=offload)
                self.assertIs(native.call_args.kwargs["mp_policy"], policy)
                self.assertIs(native.call_args.kwargs["offload_policy"], offload)

    def test_torch_no_offload_policy_keeps_offload_disabled(self):
        """An explicit Torch base policy must not require CPU pinning options."""
        model = torch.nn.Linear(2, 2)
        with patch.object(fsdp, "hyper_fully_shard", return_value=model) as native:
            self.assertIs(
                compat.fully_shard(model, offload_policy=TorchOffloadPolicy()), model
            )
        self.assertIs(type(native.call_args.kwargs["offload_policy"]), OffloadPolicy)

    def test_torch_cpu_offload_preserves_pin_memory(self):
        """Translate CPU offload separately and preserve either pinning value."""
        model = torch.nn.Linear(2, 2)
        for pin_memory in (False, True):
            with self.subTest(pin_memory=pin_memory):
                with patch.object(
                    fsdp, "hyper_fully_shard", return_value=model
                ) as native:
                    compat.fully_shard(
                        model,
                        offload_policy=TorchCPUOffloadPolicy(pin_memory=pin_memory),
                    )
                actual = native.call_args.kwargs["offload_policy"]
                self.assertIs(type(actual), CPUOffloadPolicy)
                self.assertIs(actual.pin_memory, pin_memory)

    def test_unsupported_offload_is_rejected_before_native_execution(self):
        """An unknown offload descriptor must not silently request CPU offload."""

        class CustomOffload(TorchOffloadPolicy):
            """Represent a policy with unrecognized offload semantics."""

        class CustomCPUOffload(TorchCPUOffloadPolicy):
            """Represent CPU offload with additional unrecognized semantics."""

        for policy in (
            object(),
            SimpleNamespace(pin_memory=False),
            CustomOffload(),
            CustomCPUOffload(),
        ):
            with self.subTest(policy=policy):
                with (
                    patch.object(fsdp, "hyper_fully_shard") as native,
                    self.assertRaisesRegex(NotImplementedError, "offload"),
                ):
                    compat.fully_shard(torch.nn.Linear(2, 2), offload_policy=policy)
                native.assert_not_called()

    def test_checkpoint_restore_preserves_live_parameter_and_optimizer(self):
        """Checkpoint transport must not replace an existing training parameter."""
        model = torch.nn.Linear(2, 2)
        parameter = model.weight
        optimizer = torch.optim.AdamW(model.parameters())
        payload = {
            name: torch.ones_like(value) for name, value in model.state_dict().items()
        }
        compat.set_model_state_dict(model, payload)
        self.assertIs(model.weight, parameter)
        self.assertIs(optimizer.param_groups[0]["params"][0], parameter)
        torch.testing.assert_close(parameter, payload["weight"])

    def test_checkpoint_get_uses_native_hyper_state_and_transport_views(self):
        """Hyper still owns state collection, not a reimplemented state algorithm."""
        model = torch.nn.Linear(2, 2)
        leaf, view, options = object(), object(), object()
        with (
            patch.object(
                fsdp, "hyper_model_state_dict", return_value={"weight": leaf}
            ) as native,
            patch.object(fsdp, "to_torch_view", return_value=view) as convert,
        ):
            result = compat.get_model_state_dict(model, options=options)
        native.assert_called_once_with(model, options=options)
        convert.assert_called_once_with(leaf)
        self.assertIs(result["weight"], view)

    def test_checkpoint_restores_sharded_parameter_identity_after_eval(self):
        """No-grad forward may leave full parameters installed in the module."""

        class Wrapped(torch.nn.Linear, HSDPModule):
            def reshard(self):
                self.weight = self.sharded_weight

        model = Wrapped(2, 2)
        original = model.weight
        object.__setattr__(model, "sharded_weight", original)
        model.weight = torch.nn.Parameter(original.detach().clone())

        def collect(target, **kwargs):
            self.assertIs(target.weight, original)
            return {"weight": original}

        with (
            patch.object(fsdp, "hyper_model_state_dict", side_effect=collect),
            patch.object(fsdp, "to_torch_view", side_effect=lambda value: value),
        ):
            compat.get_model_state_dict(model)
        self.assertIs(model.weight, original)

    def test_checkpoint_restore_rejects_unsupported_requests_before_writing(self):
        """Invalid requests fail without partially updating model storage."""
        model = torch.nn.Linear(2, 2)
        parameter = dict(model.named_parameters())["weight"]
        before = parameter.detach().clone()
        for option in ("full_state_dict", "broadcast_from_rank0"):
            with (
                self.subTest(option=option),
                self.assertRaisesRegex(NotImplementedError, "sharded DCP"),
            ):
                compat.set_model_state_dict(
                    model, {}, options=SimpleNamespace(**{option: True})
                )
        with self.assertRaisesRegex(TypeError, "unexpected keyword"):
            # Invalid keywords deliberately exercise rejection before writes.
            compat.set_model_state_dict(model, {}, strict=False)  # pylint: disable=unexpected-keyword-arg
        torch.testing.assert_close(parameter, before)

    def test_checkpoint_options_fail_before_reshard_or_state_collection(self):
        """Reject unsupported name/transport requests before model-side work."""
        model = torch.nn.Linear(2, 2)
        for options in (
            StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
            StateDictOptions(dsd_fqn_modifiers="custom_fqn"),
        ):
            for restore in (False, True):
                with self.subTest(options=options, restore=restore):
                    with (
                        patch.object(fsdp, "_reshard_for_checkpoint") as reshard,
                        patch.object(fsdp, "hyper_model_state_dict") as collect,
                        self.assertRaises(NotImplementedError),
                    ):
                        if restore:
                            compat.set_model_state_dict(model, {}, options=options)
                        else:
                            compat.get_model_state_dict(model, options=options)
                    reshard.assert_not_called()
                    collect.assert_not_called()
        with (
            patch.object(fsdp, "_reshard_for_checkpoint") as reshard,
            self.assertRaises(TypeError),
        ):
            # Invalid keywords deliberately exercise rejection before reshard.
            compat.get_model_state_dict(model, unsupported_option=True)  # pylint: disable=unexpected-keyword-arg
        reshard.assert_not_called()

    def test_checkpoint_full_export_and_non_strict_restore_remain_supported(self):
        """Do not reject supported native gathering or non-strict sharded load."""
        model = torch.nn.Linear(2, 2)
        state = compat.get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        torch.testing.assert_close(state["weight"], model.weight)
        compat.set_model_state_dict(
            model,
            {"weight": torch.ones_like(model.weight)},
            options=StateDictOptions(strict=False),
        )
        torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
