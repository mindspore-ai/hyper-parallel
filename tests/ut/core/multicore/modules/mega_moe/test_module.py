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

from hyper_parallel.core.multicore.modules.mega_moe import module as mega_moe_module
from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from tests.common.mark_utils import arg_mark


class TestMegaMoeExperts(unittest.TestCase):
    """Validate the public API and execution-resource lifecycle."""

    # pylint: disable=protected-access

    @staticmethod
    def _cache_resources(capacity: int, *, automatic: bool = False, mode: str = "push") -> Any:
        """Create a lazy resource with an explicit committed source reservation."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(mega_moe_module._MegaMoeExecutionResources)
        resources.spec = MegaMoeSpec(capacity, 16, 8, 4, 2, None, capacity * 2, 1, None, 0, 24,
                                    max_local_num_tokens=None if automatic else capacity,
                                    auto_tokens=automatic, dispatch_mode=mode)
        resources._token_capacity = capacity
        resources._plans = {}
        return resources

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dynamic_constructor_retains_capacity_and_accepts_real_shapes(self) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Arbitrary including empty inputs do not change the reserved resource specification.
        Expectation: Valid real shapes preserve the reservation and invalid token bounds are rejected.
        """
        experts = MegaMoeExperts(max_local_num_tokens=257, hidden_size=16, intermediate_size=8,
                                 num_experts=4, top_k=2)
        self.addCleanup(experts.close)
        self.assertIsNone(experts.local_num_tokens)
        self.assertEqual(experts._resource_group.specification["local_num_tokens"], 128)
        with patch.object(experts, "_validate_tensors"):
            for tokens in (0, 1, 127, 128, 129, 257):
                with self.subTest(tokens=tokens):
                    result = experts._validate_forward_inputs(
                        torch.empty(1, tokens, 16), torch.empty(tokens, 2, dtype=torch.int32),
                        torch.empty(tokens, 2), None, (experts.gate_up_weight, experts.down_weight))
                    self.assertEqual(tuple(result.shape), (tokens, 16))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_omitted_token_arguments_enable_automatic_push_and_pull(self) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Unbounded input lengths use a small lazy reservation in either transport.
        Expectation: Both transports start with a lazy 128-row source reservation and accept varying shapes.
        """
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                experts = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
                                         dispatch_mode=mode)
                self.addCleanup(experts.close)
                self.assertIsNone(experts.local_num_tokens)
                self.assertIsNone(experts.max_local_num_tokens)
                self.assertTrue(experts._resource_group.specification["auto_tokens"])
                self.assertEqual(experts._resource_group.specification["local_num_tokens"], 128)
                self.assertIsNone(experts._resource_group.resources)
                with patch.object(experts, "_validate_tensors"):
                    for tokens in (0, 1, 129, 4097):
                        result = experts._validate_forward_inputs(
                            torch.empty(tokens, 16), torch.zeros(tokens, 2, dtype=torch.int32),
                            torch.ones(tokens, 2), None, (experts.gate_up_weight, experts.down_weight))
                        self.assertEqual(tuple(result.shape), (tokens, 16))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_plan_cache_retains_more_than_four_buckets(self) -> None:
        """Feature: geometric token plan retention.

        Description: Repeat seven length buckets through several cycles.
        Expectation: All seven plans are reused and retained plans keep their original specification.
        """
        resources = self._cache_resources(8192)
        with patch.object(mega_moe_module, "build_mega_moe_plan",
                          side_effect=lambda spec, device: SimpleNamespace(spec=spec)) as build:
            retained = resources.plan_for_tokens(128, "cpu")
            for tokens in (128, 256, 512, 1024, 2048, 4096, 8192) * 3:
                resources.plan_for_tokens(tokens, "cpu")
            self.assertEqual(len(resources._plans), 7)
            self.assertIs(retained, resources.plan_for_tokens(128, "cpu"))
            self.assertEqual(retained.spec.plan_tokens, 128)
            self.assertEqual(retained.spec.routed_slots, 16384)
            self.assertEqual(retained.spec.receive_capacity, 16384)
            self.assertEqual(resources.spec, replace(retained.spec, schedule_tokens=None))
            self.assertEqual(build.call_count, 7)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_nearby_and_wide_dynamic_cycles_stop_building_after_warmup(self) -> None:
        """Feature: dynamic token schedule reuse.

        Description: Replay nearby and wide length sequences that previously thrashed the plan cache.
        Expectation: Only the initial covering plans are constructed.
        """
        for lengths, expected_builds in (
            ((3200, 3712, 3328, 3968, 3456, 4096, 3584, 3840), 1),
            ((8192, 8064, 7936, 7808, 7680), 1),
            ((128, 1152, 2176, 3200, 4224), 4),
        ):
            with self.subTest(lengths=lengths):
                resources = self._cache_resources(8192)
                with patch.object(mega_moe_module, "build_mega_moe_plan",
                                  side_effect=lambda spec, device: SimpleNamespace(spec=spec)) as build:
                    first = [resources.plan_for_tokens(tokens, "cpu") for tokens in lengths]
                    for _ in range(3):
                        for tokens, retained in zip(lengths, first):
                            self.assertIs(resources.plan_for_tokens(tokens, "cpu"), retained)
                    self.assertEqual(build.call_count, expected_builds)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_partial_and_small_buckets_cover_inputs_without_expanding_storage(self) -> None:
        """Feature: bounded token schedule coverage.

        Description: Exercise zero, tiny inputs and non-power-of-two public token limits.
        Expectation: Every plan covers its input within the original reservation and tiny inputs use 128 rows.
        """
        for public_bound, reserved in ((1, 128), (129, 256), (257, 384), (8000, 8064), (8193, 8320)):
            with self.subTest(public_bound=public_bound):
                resources = self._cache_resources(reserved)
                resources.spec = replace(resources.spec, max_local_num_tokens=public_bound)
                with patch.object(mega_moe_module, "build_mega_moe_plan",
                                  side_effect=lambda spec, device: SimpleNamespace(spec=spec)):
                    for requested in (0, 128, reserved):
                        plan = resources.plan_for_tokens(requested, "cpu")
                        self.assertGreaterEqual(plan.spec.plan_tokens, max(128, requested))
                        self.assertLessEqual(plan.spec.plan_tokens, reserved)
                        self.assertEqual(plan.spec.plan_tokens % 128, 0)
                        self.assertEqual(plan.spec.routed_slots, reserved * 2)
                    self.assertEqual(resources.plan_for_tokens(1, "cpu").spec.plan_tokens, 128)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_plan_uses_committed_capacity_in_both_transports(self) -> None:
        """Feature: automatic token schedule coverage.

        Description: Grow a 128-row bootstrap reservation to 4096 rows in push and pull.
        Expectation: The plan covers 4096 rows while the original resource specification remains unchanged.
        """
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                resources = self._cache_resources(128, automatic=True, mode=mode)
                resources.workspace = Mock()
                resources.heap_manager = Mock()
                physical = replace(resources.spec, local_num_tokens=4096, receive_capacity=32768)
                resources.heap_manager.ensure_token_capacity.return_value = physical
                tensor = torch.empty(1, 16, dtype=torch.bfloat16)
                resources.prepare_auto_workspace(4096, 32768, tensor)
                with patch.object(mega_moe_module, "build_mega_moe_plan",
                                  side_effect=lambda spec, device: SimpleNamespace(spec=spec)):
                    self.assertEqual(resources.plan_for_tokens(4096, "cpu").spec.plan_tokens, 4096)
                resources.workspace.ensure.assert_called_once_with(physical, tensor.dtype, tensor.device)
                self.assertEqual(resources.workspace.claim.call_count, int(mode == "pull"))
                self.assertEqual(resources.spec.local_num_tokens, 128)
                self.assertEqual(resources._token_capacity, 4096)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_same_bucket_growth_replaces_only_the_cached_plan(self) -> None:
        """Feature: immutable partial bucket replacement.

        Description: Increase the committed capacity within a bucket while retaining its previous plan.
        Expectation: The cache adopts the larger plan and the retained backward plan remains unchanged.
        """
        resources = self._cache_resources(128, automatic=True)
        resources._token_capacity = 384
        with patch.object(mega_moe_module, "build_mega_moe_plan",
                          side_effect=lambda spec, device: SimpleNamespace(spec=spec)):
            retained = resources.plan_for_tokens(384, "cpu")
            resources._token_capacity = 512
            expanded = resources.plan_for_tokens(512, "cpu")
            self.assertIsNot(expanded, retained)
            self.assertEqual((retained.spec.plan_tokens, expanded.spec.plan_tokens), (384, 512))
            self.assertIs(resources.plan_for_tokens(384, "cpu"), expanded)
            self.assertEqual(len(resources._plans), 1)
            self.assertEqual(resources.spec.local_num_tokens, 128)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_failed_bucket_upgrade_preserves_the_previous_plan(self) -> None:
        """Feature: retryable token plan construction.

        Description: Fail construction of a larger covering plan in an existing partial bucket.
        Expectation: The previous plan remains cached and unchanged.
        """
        resources = self._cache_resources(384, automatic=True)
        with patch.object(mega_moe_module, "build_mega_moe_plan",
                          side_effect=lambda spec, device: SimpleNamespace(spec=spec)):
            retained = resources.plan_for_tokens(384, "cpu")
        resources._token_capacity = 512
        with (patch.object(mega_moe_module, "build_mega_moe_plan", side_effect=RuntimeError("build failed")),
              self.assertRaisesRegex(RuntimeError, "build failed")):
            resources.plan_for_tokens(512, "cpu")
        self.assertIs(resources._plans[512], retained)
        self.assertEqual(retained.spec.plan_tokens, 384)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unprepared_capacity_is_rejected_before_building(self) -> None:
        """Feature: committed token capacity enforcement.

        Description: Fail workspace preparation before requesting a larger token schedule.
        Expectation: The previous capacity is retained and no undersized plan is constructed.
        """
        resources = self._cache_resources(128, automatic=True)
        resources.workspace = Mock()
        resources.workspace.ensure.side_effect = RuntimeError("workspace failed")
        resources.heap_manager = Mock()
        resources.heap_manager.ensure_token_capacity.return_value = replace(resources.spec, local_num_tokens=4096)
        with self.assertRaisesRegex(RuntimeError, "workspace failed"):
            resources.prepare_auto_workspace(4096, 8192, torch.empty(1, 16))
        self.assertEqual(resources._token_capacity, 128)
        with (patch.object(mega_moe_module, "build_mega_moe_plan") as build,
              self.assertRaisesRegex(RuntimeError, "requested=4096, capacity=128")):
            resources.plan_for_tokens(4096, "cpu")
        build.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_native_int32_limit_bounds_the_number_of_cached_plans(self) -> None:
        """Feature: bounded geometric plan cache.

        Description: Replay every geometric level within the native INT32 source row limit.
        Expectation: Each level is constructed once and no cached plan exceeds the reservation.
        """
        capacity = ((2**31 - 1) // 16 // 128) * 128
        resources = self._cache_resources(capacity)
        lengths = [2**power for power in range(7, 27)] + [capacity]
        with patch.object(mega_moe_module, "build_mega_moe_plan",
                          side_effect=lambda spec, device: SimpleNamespace(spec=spec)) as build:
            for tokens in lengths * 3:
                resources.plan_for_tokens(tokens, "cpu")
            self.assertEqual(build.call_count, 21)
            self.assertEqual(len(resources._plans), 21)
            self.assertLessEqual(max(plan.spec.plan_tokens for plan in resources._plans.values()), capacity)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_resource_manifest_rejects_rank_configuration_mismatch(self) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Static and dynamic ranks enter the same check before symmetric allocation.
        Expectation: Inconsistent EP configurations fail before symmetric allocations are acquired.
        """
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(mega_moe_module._MegaMoeExecutionResources)
        resources.spec = SimpleNamespace(ep_size=2, ep_group=object())

        def gather(output: list, value: Any, **kwargs: Any) -> None:
            """Simulate a peer using another resource mode or capacity."""
            self.assertIs(kwargs["group"], resources.spec.ep_group)
            output[:] = [value, ("different",)]

        with (patch.object(mega_moe_module.dist, "all_gather_object", side_effect=gather),
              self.assertRaisesRegex(ValueError, "configurations differ")):
            resources._validate_distributed_configuration(({"local_num_tokens": 128},))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_compact_count_protocol_rejects_legacy_peer(self) -> None:
        """Feature: automatic count wire compatibility.

        Description: A peer has identical capacities but still exchanges INT64 automatic counts.
        Expectation: Version agreement fails before either rank can allocate or exchange counts.
        """
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(mega_moe_module._MegaMoeExecutionResources)
        resources.spec = SimpleNamespace(ep_size=2, ep_group=object())

        def gather(output: list, value: Any, **kwargs: Any) -> None:
            """Retain every configuration field while changing only the wire version."""
            self.assertIs(kwargs["group"], resources.spec.ep_group)
            output[:] = [value, ("dynamic_tokens_v1", *value[1:])]

        with (patch.object(mega_moe_module.dist, "all_gather_object", side_effect=gather),
              self.assertRaisesRegex(ValueError, "configurations differ")):
            resources._validate_distributed_configuration(({"local_num_tokens": 128, "auto_tokens": True},))

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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_rejects_invalid_static_values_before_allocation(
        self,
        mock_create_parameters: Mock,
    ) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Reject invalid capacity and token split during construction.
        Expectation: Invalid topology, alignment and capacity factors fail before parameter allocation.
        """
        for overrides, message in (
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
        resources = SimpleNamespace(spec=SimpleNamespace(auto_tokens=False, dynamic_tokens=False),
                                    plan=object(), workspace=object(), ensure_capacity=Mock(),
                                    plan_for_tokens=Mock())
        route = SimpleNamespace(
            routed_tokens=object(), metadata=SimpleNamespace(plan_tokens=128),
            unpermute_mapping=object(), maximum_received_slots=512
        )
        resources.plan_for_tokens.return_value = resources.plan
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
            workspace=resources.workspace, autograd_mask=13,
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_rejects_invalid_weights_before_resource_creation(self) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Reject invalid expert weights before initializing native resources.
        Expectation: Invalid external expert tensors fail before any execution resources are created.
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    @patch.object(MegaMoeExperts, "_register_runtime_release", return_value=None)
    def test_shared_layers_create_once_and_close_after_last_owner(
        self,
        mock_register_runtime_release: Mock,
        mock_create_parameters: Mock,
    ) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Keep parameters independent while one serial resource group is shared.
        Expectation: Serial owners share one resource while retaining their independent Parameters.
        """
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
                                     dispatch_mode="push", max_local_num_tokens=None,
                                     dynamic_tokens=False, auto_tokens=False)
        workspace = Mock()
        manager = Mock(heap_bytes=1024, epoch=0, workspaces=[], growth_records=[])
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
        ):
            resources = mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                {},
                SimpleNamespace(device="npu:0", dtype=torch.bfloat16, element_size=lambda: 2),
                shared=False,
                active_specifications=(),
            )
            manager.reserve.assert_called_once()
            manager.attach_runtime.assert_called_once_with()
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
                                     dispatch_mode="push", max_local_num_tokens=None,
                                     dynamic_tokens=False, auto_tokens=False)

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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_construction_failure_after_restart_preserves_committed_heap(self) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Roll back only the new reservation after a successful Heap restart.
        Expectation: A failed lazy bind preserves the previously committed heap and its existing Runtime user.
        """
        root_group = object()
        specification = {"name": "new resource"}
        bound_spec = SimpleNamespace(ep_group=root_group, ep_size=1, local_num_tokens=128,
                                     dispatch_mode="push", max_local_num_tokens=None,
                                     dynamic_tokens=False, auto_tokens=False)
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_workspace_close_failure_is_retryable(self) -> None:
        """Feature: MegaMoe public API and resource lifecycle.

        Description: Keep the resource open until manager-owned workspace close succeeds.
        Expectation: A failed close remains open and succeeds on a later retry.
        """
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


class TestAutomaticTokenInterface(unittest.TestCase):
    """Keep one lazy automatic path while accepting historical token keywords."""

    # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_legacy_token_keywords_do_not_reserve_or_bound_automatic_capacity(self) -> None:
        """Feature: Automatic token compatibility.

        Description: Construct with either or both legacy keywords, including oversized or obsolete bounds.
        Expectation: Every constructor retains the same lazy minimum reservation and permits varying lengths.
        """
        for options in ({}, {"local_num_tokens": 4096}, {"max_local_num_tokens": 8192},
                        {"local_num_tokens": 128, "max_local_num_tokens": 256},
                        {"local_num_tokens": -1, "max_local_num_tokens": 0}):
            with self.subTest(options=options):
                layer = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2, **options)
                self.addCleanup(layer.close)
                specification = layer._resource_group.specification
                self.assertTrue(specification["auto_tokens"])
                self.assertEqual(specification["local_num_tokens"], 128)
                self.assertIsNone(layer._resource_group.resources)
                for rows in (0, 129, 4097):
                    hidden = torch.empty(rows, 16, dtype=torch.bfloat16)
                    ids = torch.zeros(rows, 2, dtype=torch.int32)
                    weights = torch.ones(rows, 2)
                    with patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True):
                        result = layer._validate_forward_inputs(
                            hidden, ids, weights, None,
                            (layer.gate_up_weight.bfloat16(), layer.down_weight.bfloat16()))
                    self.assertEqual(result.shape[0], rows)


if __name__ == "__main__":
    unittest.main()
