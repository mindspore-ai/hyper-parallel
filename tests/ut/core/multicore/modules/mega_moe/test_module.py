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
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import module as mega_moe_module
from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts


class TestMegaMoeExperts(unittest.TestCase):
    """Validate the public API and execution-resource lifecycle."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_external_weights_preserve_ownership_and_gradients(self) -> None:
        """Feature: Externally owned expert weights.

        Description: Replace external weights between consecutive forwards.
        Expectation: Each current weight receives gradients without module registration or caching.
        """
        hidden = torch.ones(128, 16, dtype=torch.bfloat16)
        ids = torch.zeros(128, 2, dtype=torch.int32)
        probabilities = torch.full((128, 2), 0.5)
        resources = SimpleNamespace(spec=object(), plan=object(), workspace=object())
        route = SimpleNamespace(unpermute_mapping=object())

        def _execute(states, _ids, gate_up, down, *_args, **_kwargs):
            """Model a differentiable expert bridge without native resources."""
            return states * (gate_up.sum() + down.sum())

        experts = MegaMoeExperts(
            local_num_tokens=128, hidden_size=16, intermediate_size=8,
            num_experts=4, top_k=2, ep_size=2,
            create_parameters=False,
        )
        self.addCleanup(experts.close)
        self.assertEqual(list(experts.parameters()), [])
        self.assertEqual(dict(experts.state_dict()), {})
        previous_weights = None
        with (
            patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True),
            patch.object(experts, "_get_execution_resources", return_value=resources),
            patch.object(mega_moe_module, "prepare_topk_route", return_value=route),
            patch.object(mega_moe_module, "execute_mega_moe_with_permutation", side_effect=_execute) as bridge,
            patch.object(mega_moe_module, "restore_topk_output", side_effect=lambda output, *_args: output),
        ):
            for value in (1.0, 2.0):
                weights = (
                    torch.full((2, 16, 16), value, dtype=torch.bfloat16, requires_grad=True),
                    torch.full((2, 8, 16), value, dtype=torch.bfloat16, requires_grad=True),
                )
                output = experts(hidden, ids, probabilities, expert_weights=weights)
                output.sum().backward()
                self.assertIs(bridge.call_args.args[2], weights[0])
                self.assertIs(bridge.call_args.args[3], weights[1])
                for weight in weights:
                    torch.testing.assert_close(weight.grad, torch.full_like(weight, hidden.numel()))
                if previous_weights is not None:
                    for weight in previous_weights:
                        torch.testing.assert_close(weight.grad, torch.full_like(weight, hidden.numel()))
                previous_weights = weights
        self.assertIsNone(experts.gate_up_weight)
        self.assertIsNone(experts.down_weight)
        self.assertEqual(dict(experts.state_dict()), {})
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_external_weights_validate_before_resource_allocation(self) -> None:
        """Feature: External expert weight validation.

        Description: Supply missing, incorrectly shaped or incorrectly typed external weights.
        Expectation: Reject invalid inputs before acquiring native execution resources.
        """
        experts = MegaMoeExperts(
            local_num_tokens=128, hidden_size=16, intermediate_size=8,
            num_experts=4, top_k=2, ep_size=2, create_parameters=False,
        )
        self.addCleanup(experts.close)
        hidden = torch.ones(128, 16, dtype=torch.bfloat16)
        ids = torch.zeros(128, 2, dtype=torch.int32)
        probabilities = torch.full((128, 2), 0.5)
        down = torch.ones(2, 8, 16, dtype=torch.bfloat16)
        cases = (
            (None, ValueError, "explicit expert_weights"),
            ((torch.ones(2, 16, 8, dtype=torch.bfloat16), down), ValueError, "gate_up_weight must have shape"),
            ((torch.ones(2, 16, 16), down), TypeError, "BF16"),
        )
        with (
            patch.object(torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True),
            patch.object(experts, "_get_execution_resources") as acquire,
        ):
            for weights, error, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(error, message):
                    experts(hidden, ids, probabilities, expert_weights=weights)
        acquire.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_defaults_to_lossless_capacity(
        self,
        mock_create_parameters: Mock,
    ) -> None:
        """Expose local-token topology with a lossless default capacity."""
        mock_create_parameters.return_value = (object(), object())

        experts = MegaMoeExperts(
            local_num_tokens=128,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            ep_size=2,
        )
        try:
            self.assertEqual(experts.local_num_tokens, 128)
            self.assertIsNone(experts.expert_capacity_factor)
            self.assertIsNone(experts.swiglu_limit)
            self.assertEqual(
                experts._resource_group.specification,
                {
                    "local_num_tokens": 128,
                    "hidden_size": 16,
                    "intermediate_size": 8,
                    "num_experts": 4,
                    "top_k": 2,
                    "expert_capacity_factor": None,
                    "swiglu_limit": None,
                    "ep_size": 2,
                    "ep_group": None,
                },
            )
            mock_create_parameters.assert_called_once_with(2, 16, 8)
        finally:
            experts.close()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_constructor_rejects_invalid_static_values_before_allocation(
        self,
        mock_create_parameters: Mock,
    ) -> None:
        """Reject invalid capacity and token split during construction."""
        for overrides, message in (
            ({"local_num_tokens": 129}, "divisible"),
            ({"expert_capacity_factor": 0.999}, "expert_capacity_factor"),
            ({"num_experts": 34}, "device scratch capacity"),
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
                    top_k=2,
                    expert_capacity_factor=overrides.get("expert_capacity_factor"),
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

    def test_forward_passes_router_inputs_and_restores_shape(self) -> None:
        """Preserve Router inputs, expert parameters and the caller's shape."""
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
        resources = SimpleNamespace(spec=object(), plan=object(), workspace=object())
        route = SimpleNamespace(
            routed_tokens=object(), metadata=object(), unpermute_mapping=object()
        )
        expert_output = object()

        with (
            patch.object(
                torch.Tensor, "is_npu", new_callable=PropertyMock, return_value=True
            ),
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
        mock_execute.assert_called_once_with(
            hidden_flat,
            topk_ids,
            experts.gate_up_weight,
            experts.down_weight,
            route,
            resources.plan,
            resources.workspace,
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
        mock_create_resources.assert_not_called()

    @patch.object(mega_moe_module, "_create_mega_moe_parameters")
    def test_shared_layers_create_once_and_close_after_last_owner(
        self,
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
        resources.close.assert_called_once_with()

    def test_execution_resource_pairs_shmem_acquire_and_release(self) -> None:
        """Pair one SHMEM reference with one execution-resource lifetime."""
        root_group = object()
        bound_spec = SimpleNamespace(ep_group=root_group)
        workspace = Mock()

        with (
            patch.object(
                mega_moe_module,
                "bind_mega_moe_spec",
                return_value=bound_spec,
            ),
            patch.object(mega_moe_module, "configure_symmetric_heap"),
            patch.object(mega_moe_module.shmem, "acquire") as mock_acquire,
            patch.object(mega_moe_module.shmem, "release") as mock_release,
            patch.object(mega_moe_module, "build_mega_moe_plan", return_value=object()),
            patch.object(mega_moe_module, "MegaMoeWorkspace", return_value=workspace),
        ):
            resources = mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                {},
                SimpleNamespace(device="npu:0"),
                shared=False,
                active_specifications=(),
            )
            mock_acquire.assert_called_once_with(root_group)
            mock_release.assert_not_called()
            resources.close()
            resources.close()

        workspace.close.assert_called_once_with()
        mock_release.assert_called_once_with()

    def test_execution_resource_construction_failure_releases_shmem(self) -> None:
        """Release the acquired SHMEM reference when resource construction fails."""
        root_group = object()
        bound_spec = SimpleNamespace(ep_group=root_group)

        with (
            patch.object(
                mega_moe_module,
                "bind_mega_moe_spec",
                return_value=bound_spec,
            ),
            patch.object(mega_moe_module, "configure_symmetric_heap"),
            patch.object(mega_moe_module.shmem, "acquire") as mock_acquire,
            patch.object(mega_moe_module.shmem, "release") as mock_release,
            patch.object(
                mega_moe_module,
                "build_mega_moe_plan",
                side_effect=RuntimeError("plan failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "plan failed"),
        ):
            mega_moe_module._MegaMoeExecutionResources(  # pylint: disable=protected-access
                {},
                SimpleNamespace(device="npu:0"),
                shared=False,
                active_specifications=(),
            )

        mock_acquire.assert_called_once_with(root_group)
        mock_release.assert_called_once_with()

    def test_workspace_close_failure_keeps_shmem_user(self) -> None:
        """Do not leave SHMEM when a workspace cannot release its resources."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(  # pylint: disable=protected-access
            mega_moe_module._MegaMoeExecutionResources  # pylint: disable=protected-access
        )
        resources.workspace = Mock()
        resources.workspace.close.side_effect = RuntimeError("workspace busy")
        resources._closed = False  # pylint: disable=protected-access

        with (
            patch.object(mega_moe_module.shmem, "release") as mock_release,
            self.assertRaisesRegex(RuntimeError, "workspace busy"),
        ):
            resources.close()

        mock_release.assert_not_called()
        self.assertFalse(resources._closed)  # pylint: disable=protected-access

    def test_shmem_release_failure_keeps_execution_resource_open(self) -> None:
        """Keep the resource open when its SHMEM reference cannot be released."""
        resources = mega_moe_module._MegaMoeExecutionResources.__new__(  # pylint: disable=protected-access
            mega_moe_module._MegaMoeExecutionResources  # pylint: disable=protected-access
        )
        resources.workspace = Mock()
        resources._closed = False  # pylint: disable=protected-access

        with (
            patch.object(
                mega_moe_module.shmem,
                "release",
                side_effect=RuntimeError("release failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "release failed"),
        ):
            resources.close()

        resources.workspace.close.assert_called_once_with()
        self.assertFalse(resources._closed)  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
