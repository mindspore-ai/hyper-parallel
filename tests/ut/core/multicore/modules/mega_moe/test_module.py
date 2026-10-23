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
"""Unit tests for the model-facing MegaMoe module."""

import unittest
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.core.multicore import _automatic, _collective
from hyper_parallel.core.multicore.modules.mega_moe import module as mega_moe_module
from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from tests.common.mark_utils import arg_mark


class TestMegaMoeExperts(unittest.TestCase):
    """Validate the public API and execution-resource lifecycle."""

    # pylint: disable=protected-access

    def test_dynamic_constructor_retains_capacity_and_accepts_real_shapes(self) -> None:
        """Arbitrary including empty inputs do not change the reserved resource specification."""
        experts = MegaMoeExperts(max_local_num_tokens=257, hidden_size=16, intermediate_size=8,
                                 num_experts=4, top_k=2)
        self.addCleanup(experts.close)
        self.assertIsNone(experts.local_num_tokens)
        self.assertEqual(experts._resource_group.specification["local_num_tokens"], 384)
        with patch.object(experts, "_validate_tensors"):
            for tokens in (0, 1, 127, 128, 129, 257):
                with self.subTest(tokens=tokens):
                    result = experts._validate_forward_inputs(
                        torch.empty(1, tokens, 16), torch.empty(tokens, 2, dtype=torch.int32),
                        torch.empty(tokens, 2), None, (experts.gate_up_weight, experts.down_weight))
                    self.assertEqual(tuple(result.shape), (tokens, 16))
        for kwargs in ({}, {"local_num_tokens": 128, "max_local_num_tokens": 256},
                       {"max_local_num_tokens": 0}, {"max_local_num_tokens": True}, {"max_local_num_tokens": 2**31}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2, **kwargs)

    def test_plan_cache_evicts_without_mutating_retained_plans_or_capacity(self) -> None:
        """A deferred backward retains its old plan after cache eviction and a new shape."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(mega_moe_module._MegaMoeExecutionResources)
        resources.spec = MegaMoeSpec(4096, 16, 8, 4, 2, None, 8192, 1, None, 0, 24,
                                    max_local_num_tokens=4096)
        resources._plans = OrderedDict()
        with patch.object(mega_moe_module, "build_mega_moe_plan",
                          side_effect=lambda spec, device: SimpleNamespace(spec=spec)) as build:
            retained = resources.plan_for_tokens(128, "cpu")
            self.assertIs(retained, resources.plan_for_tokens(128, "cpu"))
            for tokens in (256, 384, 512, 4096):
                resources.plan_for_tokens(tokens, "cpu")
            self.assertEqual(len(resources._plans), 4)
            self.assertNotIn(128, resources._plans)
            self.assertEqual(retained.spec.plan_tokens, 128)
            self.assertEqual(retained.spec.routed_slots, 8192)
            self.assertEqual(retained.spec.receive_capacity, 8192)
            self.assertEqual(resources.spec, replace(retained.spec, schedule_tokens=None))
            self.assertEqual(build.call_count, 5)

    def test_resource_manifest_rejects_rank_configuration_mismatch(self) -> None:
        """Static and dynamic ranks enter the same check before symmetric allocation."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(mega_moe_module._MegaMoeExecutionResources)
        resources.spec = SimpleNamespace(ep_size=2, ep_group=object())

        def gather(output: list, value: Any, **kwargs: Any) -> None:
            """Simulate a peer using another resource mode or capacity."""
            self.assertIs(kwargs["group"], resources.spec.ep_group)
            output[:] = [value, ("different",)]

        with patch.object(mega_moe_module.dist, "all_gather_object", side_effect=gather):
            with self.assertRaisesRegex(ValueError, "configurations differ"):
                resources._validate_distributed_configuration(({"local_num_tokens": 128},))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_defaults_to_growing_push(self, mock_create_parameters: Mock) -> None:
        """Feature: Push capacity defaults.
        Description: Construct a layer without explicit capacity factors.
        Expectation: Resolve both factors to 1.25 and create the expected local weights.
        """
        mock_create_parameters.return_value = (object(), object())
        experts = MegaMoeExperts(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                                 num_experts=4, top_k=2, ep_size=2)
        self.addCleanup(experts.close)
        self.assertEqual((experts.dispatch_mode, experts.initial_capacity_factor, experts.capacity_growth_factor),
                         ("push", 1.25, 1.25))
        mock_create_parameters.assert_called_once_with(2, 16, 8)
        self.assertIsNone(experts.swiglu_limit)

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_rejects_invalid_static_values_before_allocation(
        self,
        mock_create_parameters: Mock,
    ) -> None:
        """Reject invalid capacity and token split during construction."""
        for overrides, message in (
            ({"local_num_tokens": 129}, "divisible"),
            ({"initial_capacity_factor": 0.999}, "initial_capacity_factor"),
            ({"num_experts": 35}, "divisible"),
            ({"dispatch_mode": "invalid"}, "dispatch_mode"),
            ({"capacity_growth_factor": 0.999}, "capacity_growth_factor"),
            ({"initial_capacity_factor": float("nan")}, "initial_capacity_factor"),
            ({"capacity_growth_factor": float("inf")}, "capacity_growth_factor"),
            ({"initial_capacity_factor": True}, "initial_capacity_factor"),
            ({"capacity_growth_factor": 1.5, "dispatch_mode": "pull"}, "push"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, message),
            ):
                MegaMoeExperts(
                    local_num_tokens=overrides.get("local_num_tokens", 128),
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=overrides.get("num_experts", 4),
                    dispatch_mode=overrides.get("dispatch_mode", "push"),
                    capacity_growth_factor=overrides.get("capacity_growth_factor"),
                    top_k=2,
                    initial_capacity_factor=overrides.get("initial_capacity_factor"),
                    ep_size=2,
                )

        mock_create_parameters.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_validates_and_records_swiglu_limit(
        self,
        mock_create_parameters: Mock,
    ) -> None:
        """Feature: validate the model-facing SwiGLU clamp option.

        Description: Construct experts with one valid limit and several invalid
            or non-float32-representable values.
        Expectation: The valid limit is retained and invalid limits fail before
            parameter allocation.
        """
        mock_create_parameters.return_value = (object(), object())
        experts = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            swiglu_limit=10,
            ep_size=2,
        )
        try:
            self.assertEqual(experts.swiglu_limit, 10.0)
            self.assertEqual(
                experts._resource_group.specification["swiglu_limit"],
                10.0,
            )
        finally:
            experts.close()

        for invalid_limit in (
            0,
            -1,
            1e-50,
            1e39,
            float("nan"),
            float("inf"),
            True,
            "10",
        ):
            with (
                self.subTest(swiglu_limit=invalid_limit),
                self.assertRaisesRegex(ValueError, "swiglu_limit"),
            ):
                MegaMoeExperts(
                    local_num_tokens=128,
                    hidden_size=16,
                    intermediate_size=8,
                    num_experts=4,
                    top_k=2,
                    swiglu_limit=invalid_limit,
                    ep_size=2,
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_passes_router_inputs_and_restores_shape(self) -> None:
        """Feature: forward passes router inputs and restores shape.
        Description: Execute forward with router metadata and caller-shaped inputs.
        Expectation: Preserve Router inputs, expert parameters and the caller's shape.
        """
        experts = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            ep_size=2,
        ).to(dtype=torch.bfloat16)
        self.addCleanup(experts.close)
        hidden_states = torch.arange(2048, dtype=torch.bfloat16).reshape(2, 64, 16)
        topk_ids = torch.zeros((128, 2), dtype=torch.int32)
        topk_weights = torch.full((128, 2), 0.5)
        tokens_per_expert = torch.tensor([256, 0, 0, 0], dtype=torch.int32)
        expected = hidden_states.reshape(128, 16) + 1
        resources = SimpleNamespace(spec=object(), plan=object(), workspace=object(), ensure_capacity=Mock())
        route = SimpleNamespace(
            routed_tokens=object(), metadata=object(), unpermute_mapping=object(), maximum_received_slots=512
        )
        expert_output = object()

        with (
            patch.object(
                torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True
            ),
            patch.object(torch.npu, "is_current_stream_capturing", return_value=False),
            patch.object(experts, "_get_execution_resources", return_value=resources),
            patch.object(
                mega_moe_module, "prepare_topk_route", return_value=route
            ) as mock_prepare,
            patch.object(
                mega_moe_module, "execute_mega_moe_with_permutation", return_value=expert_output
            ) as mock_execute,
            patch.object(
                mega_moe_module, "restore_topk_output", return_value=expected
            ) as mock_restore,
        ):
            actual = experts(
                hidden_states,
                topk_ids,
                topk_weights,
                tokens_per_expert=tokens_per_expert,
            )

        torch.testing.assert_close(actual, expected.reshape(hidden_states.shape))
        hidden_flat = mock_prepare.call_args.args[0]
        torch.testing.assert_close(hidden_flat, hidden_states.reshape(128, 16))
        mock_prepare.assert_called_once_with(
            hidden_flat, topk_ids, topk_weights, resources.spec, tokens_per_expert,
            workspace=resources.workspace,
        )
        resources.ensure_capacity.assert_called_once_with(512)
        mock_execute.assert_called_once_with(
            hidden_flat,
            topk_ids,
            experts.gate_up_weight,
            experts.down_weight,
            route,
            resources.plan,
            resources.workspace,
            topk_weights=None,
            workspace_claimed=False,
        )
        mock_restore.assert_called_once_with(
            expert_output, route.unpermute_mapping, topk_weights
        )

    def test_forward_rejects_invalid_weights_before_resource_creation(self) -> None:
        """Reject invalid expert weights before initializing native resources."""
        experts = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            ep_size=2,
        ).to(dtype=torch.bfloat16)
        self.addCleanup(experts.close)
        experts.down_weight = torch.nn.Parameter(
            torch.zeros((2, 16, 8), dtype=torch.bfloat16)
        )

        with (
            patch.object(
                torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True
            ),
            patch.object(
                experts, "_create_execution_resources"
            ) as mock_create_resources,
            self.assertRaisesRegex(ValueError, "down_weight must have shape"),
        ):
            experts(
                torch.zeros((128, 16), dtype=torch.bfloat16),
                torch.zeros((128, 2), dtype=torch.int32),
                torch.full((128, 2), 0.5),
            )
        experts.gate_up_weight = experts.down_weight = None
        with (patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True),
              patch.object(experts, "_create_execution_resources", mock_create_resources)):
            for weights, error in ((None, ValueError),
                                   ((torch.zeros(2, 16, 16), torch.zeros(2, 8, 16)), TypeError)):
                with self.assertRaises(error):
                    experts(torch.zeros(128, 16, dtype=torch.bfloat16), torch.zeros(128, 2),
                            torch.ones(128, 2), expert_weights=weights)
        mock_create_resources.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    @patch.object(MegaMoeExperts, "_register_runtime_release", return_value=None)
    def test_shared_layers_create_once_and_close_after_last_owner(
        self,
        mock_register_runtime_release: Mock,
        mock_create_parameters: Mock,
    ) -> None:
        """Keep parameters independent while one serial resource group is shared."""
        mock_create_parameters.return_value = (object(), object())
        layers = [
            MegaMoeExperts(
                local_num_tokens=128,
                hidden_size=16,
                intermediate_size=8,
                num_experts=4,
                top_k=2,
                ep_size=2,
            )
            for _ in range(4)
        ]
        resources = Mock()
        input_tensor = SimpleNamespace(device="npu:0", dtype="bfloat16")

        other = MegaMoeExperts(local_num_tokens=128, hidden_size=16, intermediate_size=8,
                               num_experts=4, top_k=2, ep_size=2, dispatch_mode="pull")
        self.addCleanup(other.close)
        with self.assertRaises(ValueError):
            MegaMoeExperts.share_execution_resources([layers[0], other])
        other.close()
        MegaMoeExperts.share_execution_resources(layers)
        shared_group = layers[0]._resource_group
        with patch.object(
            MegaMoeExperts,
            "_create_execution_resources",
            return_value=resources,
        ) as mock_create_resources:
            resolved = [
                layer._get_execution_resources(input_tensor) for layer in layers
            ]
        mock_register_runtime_release.assert_called_once()
        self.assertIs(mock_register_runtime_release.call_args.args[-1], resources)

        self.assertTrue(shared_group.shared)
        self.assertEqual(len(shared_group.members), len(layers))
        self.assertTrue(all(layer._resource_group is shared_group for layer in layers))
        self.assertTrue(all(value is resources for value in resolved))
        mock_create_resources.assert_called_once()
        self.assertTrue(mock_create_resources.call_args.kwargs["shared"])
        self.assertEqual(
            len(mock_create_resources.call_args.kwargs["active_specifications"]),
            1,
        )

        for layer in layers[:-1]:
            layer.close()
        resources.close.assert_not_called()
        layers[-1].close()
        layers[-1].close()
        resources.close.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_execution_resource_uses_manager_owned_runtime(self) -> None:
        """Feature: execution resource uses manager-owned Runtime.
        Description: Construct and close one execution resource twice.
        Expectation: Reserve and attach through the manager while close only removes the workspace.
        """
        root_group = object()
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128,
                                     dispatch_mode="push", max_local_num_tokens=None)
        workspace = Mock()
        manager = Mock(root_group=root_group, heap_bytes=1024, epoch=0, workspaces=[], growth_records=[])
        manager.reserve.return_value = ({},)
        manager.attach_runtime.return_value = True
        manager.bind.return_value = 128
        close_order = []
        manager.close_workspace.side_effect = lambda _workspace: close_order.append("workspace")

        with (
            patch.object(
                mega_moe_module,
                "bind_mega_moe_spec",
                return_value=bound_spec,
            ),
            patch.object(mega_moe_module, "_get_heap_manager", return_value=manager),
            patch.object(
                mega_moe_module, "build_mega_moe_plan", return_value=object()
            ) as mock_build_plan,
            patch.object(mega_moe_module, "MegaMoeWorkspace", return_value=workspace),
            patch.object(_automatic._CLEANUP, "root_group", object()),
            patch.object(_collective, "exchange", wraps=_collective.exchange) as exchange,
        ):
            resources = mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                {},
                SimpleNamespace(device="npu:0", dtype=torch.bfloat16, element_size=lambda: 2),
                shared=False,
                active_specifications=(),
            )
            manager.reserve.assert_called_once()
            manager.attach_runtime.assert_called_once_with()
            exchange.assert_called_once()
            self.assertIs(exchange.call_args.kwargs["group"], root_group)
            mock_build_plan.assert_called_once_with(bound_spec, "npu:0")
            resources.close()
            resources.close()

        manager.close_workspace.assert_called_once_with(workspace)
        manager.detach_runtime.assert_not_called()
        self.assertEqual(close_order, ["workspace"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_execution_resource_construction_failure_rolls_back_manager_state(self) -> None:
        """Feature: construction failure rolls back manager state.
        Description: Fail plan creation after attaching the sole Runtime reference.
        Expectation: Remove new reservations and detach a newly acquired unused Runtime.
        """
        root_group = object()
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128,
                                     dispatch_mode="push", max_local_num_tokens=None)

        manager = Mock(heap_bytes=1024, epoch=0, workspaces=[])
        manager.reserve.return_value = ({},)
        manager.attach_runtime.return_value = True
        with (
            patch.object(
                mega_moe_module,
                "bind_mega_moe_spec",
                return_value=bound_spec,
            ),
            patch.object(mega_moe_module, "_get_heap_manager", return_value=manager),
            patch.object(
                mega_moe_module,
                "build_mega_moe_plan",
                side_effect=RuntimeError("plan failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "plan failed"),
        ):
            mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                {},
                SimpleNamespace(device="npu:0", dtype=torch.bfloat16, element_size=lambda: 2),
                shared=False,
                active_specifications=(),
            )

        manager.rollback_reservations.assert_called_once()
        manager.detach_runtime.assert_called_once_with()

    def test_construction_failure_after_restart_preserves_committed_heap(self) -> None:
        """Roll back only the new reservation after a successful Heap restart."""
        root_group = object()
        specification = {"name": "new resource"}
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128,
                                     dispatch_mode="push", max_local_num_tokens=None)
        growth_record = {"epoch": 1, "heap_bytes": 4096}
        manager = Mock(heap_bytes=2048, epoch=0, workspaces=[object()], growth_records=[])
        manager.attach_runtime.return_value = False

        def reserve(*_args: object, **_kwargs: object) -> tuple[dict]:
            """Simulate a committed restart while adding the new reservation."""
            manager.heap_bytes = 4096
            manager.epoch = 1
            manager.growth_records.append(growth_record)
            return (specification,)

        manager.reserve.side_effect = reserve
        with (
            patch.object(mega_moe_module, "bind_mega_moe_spec", return_value=bound_spec),
            patch.object(mega_moe_module, "_workspace_layout", return_value=object()),
            patch.object(mega_moe_module, "_get_heap_manager", return_value=manager),
            patch.object(mega_moe_module, "build_mega_moe_plan", side_effect=RuntimeError("plan failed")),
            self.assertRaisesRegex(RuntimeError, "plan failed"),
        ):
            mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                specification,
                SimpleNamespace(device="npu:0", dtype=torch.bfloat16, element_size=lambda: 2),
                shared=False,
                active_specifications=(specification,),
            )

        manager.rollback_reservations.assert_called_once_with((specification,), None)
        manager.detach_runtime.assert_not_called()
        self.assertEqual((manager.heap_bytes, manager.epoch), (4096, 1))
        self.assertEqual(manager.growth_records, [growth_record])

    def test_workspace_close_failure_is_retryable(self) -> None:
        """Keep the resource open until manager-owned workspace close succeeds."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(  # pylint: disable=protected-access
            mega_moe_module._MegaMoeExecutionResources  # pylint: disable=protected-access
        )
        resources.workspace = Mock()
        resources._workspace_closed = False  # pylint: disable=protected-access
        resources.heap_manager = Mock()
        resources._plans = OrderedDict()
        resources._closed = False  # pylint: disable=protected-access

        resources.heap_manager.close_workspace.side_effect = RuntimeError("close failed")
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            resources.close()

        self.assertFalse(resources._closed)  # pylint: disable=protected-access

        resources.heap_manager.close_workspace.side_effect = None
        resources.close()
        self.assertEqual(resources.heap_manager.close_workspace.call_count, 2)


if __name__ == "__main__":
    unittest.main()
