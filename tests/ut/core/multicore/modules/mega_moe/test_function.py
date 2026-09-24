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
"""CPU storage-lifetime tests for the MegaMoe autograd bridge."""

import unittest
import weakref
from itertools import product
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import function as function_module


class TestMegaMoeFunction(unittest.TestCase):
    """Exercise real autograd contexts with mocked communication and kernels."""

    def test_deferred_backward_keeps_local_owned_storage_after_workspace_reuse(self) -> None:
        """Retain each route's data and capacity across grow/shrink and reverse backward."""
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                self._check_deferred_backward(permuted=False, reuse=reuse)

    def test_permutation_gradient_consumes_workspace_before_release(self) -> None:
        """Keep token gradients intact when release immediately poisons shared rows."""
        for reuse, pull in product((False, True), repeat=2):
            with self.subTest(reuse=reuse, pull=pull):
                self._check_deferred_backward(permuted=True, reuse=reuse, pull=pull)

    def test_frozen_tokens_do_not_launch_a_permutation_gradient(self) -> None:
        """Compute expert weight gradients without an unused token reduction."""
        self._check_deferred_backward(permuted=True, input_grad=False, reuse=True)

    def test_pull_dispatch_skips_only_the_exact_source_self_copy(self) -> None:
        """Use pre-permuted SHMEM directly while preserving ordinary-source staging."""
        spec = SimpleNamespace(dispatch_mode="pull", hidden_size=4)
        source = torch.empty(4, 4)
        workspace = SimpleNamespace(source_buffer=source)
        with patch.object(torch.Tensor, "copy_", autospec=True) as copy:
            dispatch, actual_source = function_module._dispatch_and_source(spec, workspace, source, 3)
            copy.assert_not_called()
        self.assertIs(actual_source, source)
        self.assertEqual(tuple(dispatch.shape), (3, 4))
        rows = torch.ones(4, 4)
        function_module._dispatch_and_source(spec, workspace, rows, 3)
        torch.testing.assert_close(source, rows)

    def _check_deferred_backward(
        self, *, permuted: bool, input_grad: bool = True, reuse: bool = False, pull: bool = False,
    ) -> None:
        """Exercise delayed backward with an optional input permutation boundary."""
        spec = SimpleNamespace(hidden_size=4, intermediate_size=2, rank_id=0,
                               ep_size=2, num_experts=4, local_num_tokens=2, top_k=2,
                               dispatch_mode="pull" if pull else "push")
        plan = SimpleNamespace(spec=spec, reuse_backward_dispatch=reuse)
        for name in ("up_proj", "swiglu", "down_proj", "act_grad", "gate_grad",
                     "w1_grad", "w2_grad", "swiglu_grad"):
            setattr(plan, f"{name}_tiling", None)
        runtime = SimpleNamespace(normal_tensor=None)
        plan.fwd_runtime = runtime
        plan.bwd_runtime = runtime
        workspace = Mock(
            in_use=False,
            expert_capacity=128,
            source_buffer=torch.empty(4, 4),
            expert_buffer=torch.empty(128, 4),
            routed_buffer=torch.empty(4, 4),
            forward_event_counters=torch.empty(16),
            backward_event_counters=torch.empty(16),
            gmm_workspace=torch.empty(1),
            swiglu_grad_workspace=torch.empty(1),
        )
        weight1 = torch.ones(2, 4, 4, requires_grad=True)
        weight2 = torch.ones(2, 2, 4, requires_grad=True)
        down_pointers = []
        dispatch_pointers = []
        backward_capacities = []
        scratch_references = []
        restore_gradient = function_module._restore_input_gradient  # pylint: disable=protected-access

        def restore_without_scratch(*args: Any) -> Any:
            """Require ordinary scratch and the optional SHMEM view to be released."""
            self.assertTrue(scratch_references)
            self.assertTrue(all(reference() is None for reference in scratch_references))
            return restore_gradient(*args)

        def release_workspace() -> None:
            """Make accidental reads after the lease immediately observable."""
            self.assertTrue(workspace.in_use)
            workspace.in_use = False
            workspace.source_buffer.fill_(float("nan"))
            workspace.expert_buffer.fill_(float("nan"))
            workspace.routed_buffer.fill_(float("nan"))

        def claim_workspace() -> None:
            """Reject nested claims while exercising the real autograd bridge."""
            self.assertFalse(workspace.in_use)
            workspace.in_use = True

        workspace.claim.side_effect = claim_workspace
        workspace.release.side_effect = release_workspace

        def permutation_gradient(grad: Any, mapping: Any, token_rows: int, top_k: int) -> Any:
            """Require direct consumption of shared rows and return owned token rows."""
            self.assertEqual(grad.data_ptr(), workspace.routed_buffer.data_ptr())
            self.assertEqual((token_rows, top_k), (2, 2))
            self.assertEqual(mapping.tolist(), [0, 2, 1, 3])
            self.assertTrue(torch.isfinite(grad).all())
            return grad.reshape(token_rows, top_k, 4).sum(dim=1)

        def forward_kernel(*args: Any) -> None:
            """Fill every output with a call-specific sentinel."""
            dispatch, source = args[0], args[2]
            up_proj, activation, down_proj, combine = args[7], args[8], args[11], args[12]
            tag = source[0, 0].item()
            capacity = up_proj.shape[0]
            self.assertEqual(tuple(down_proj.shape), (capacity, 4))
            self.assertEqual(tuple(activation.shape), (capacity, 2))
            self.assertEqual(tuple(dispatch.shape), (capacity if pull else 128, 4))
            dispatch.fill_(tag)
            up_proj.fill_(tag + 10)
            activation.fill_(tag + 20)
            down_proj.fill_(tag + 30)
            combine.fill_(tag + 40)
            down_pointers.append(down_proj.data_ptr())
            dispatch_pointers.append(dispatch.data_ptr())

        def backward_kernel(*args: Any) -> None:
            """Check saved values before emulating the in-place gradient outputs."""
            saved_dispatch = args[17]
            capacity = saved_dispatch.shape[0]
            backward_capacities.append(capacity)
            tag = saved_dispatch[0, 0].item()
            self.assertTrue(torch.equal(saved_dispatch, torch.full_like(saved_dispatch, tag)))
            self.assertTrue(torch.equal(args[9], torch.full_like(args[9], tag + 10)))
            self.assertTrue(torch.equal(args[5], torch.full_like(args[5], tag + 20)))
            for tensor, width in ((args[8], 2), (args[10], 4), (args[12], 4)):
                self.assertEqual(tuple(tensor.shape), (capacity, width))
            self.assertEqual(tuple(args[13].shape), (4, 4))
            self.assertEqual(args[12].data_ptr() == args[0].data_ptr(), reuse)
            self.assertNotEqual(saved_dispatch.data_ptr(), args[0].data_ptr())
            scratch_references[:] = [weakref.ref(args[index]) for index in (8, 10, 12)]
            self.assertEqual(torch.count_nonzero(args[6]).item(), 0)
            self.assertEqual(torch.count_nonzero(args[18]).item(), 0)
            args[13].fill_(tag)
            args[6].fill_(tag)
            args[18].fill_(tag)

        def _forward(source: torch.Tensor, route: Any) -> torch.Tensor:
            """Borrow the caller's lease when routing directly into pull source."""
            if not permuted:
                return function_module.execute_mega_moe(source, weight1, weight2, route, plan, workspace)
            ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
            prepared = SimpleNamespace(
                routed_tokens=source.detach().repeat_interleave(2, dim=0), metadata=route,
                unpermute_mapping=torch.tensor([0, 2, 1, 3], dtype=torch.int32),
            )
            if pull:
                workspace.claim()
                workspace.source_buffer.copy_(prepared.routed_tokens)
                prepared.routed_tokens = workspace.source_buffer
            output = function_module.execute_mega_moe_with_permutation(
                source, ids, weight1, weight2, prepared, plan, workspace, workspace_claimed=pull,
            )
            if pull:
                self.assertTrue(workspace.in_use)
                workspace.release()
            return output

        capacities = (7, 2, 1, 5)
        outputs = []
        sources = []
        with (
            patch.object(
                function_module.multicore_ops,
                "mega_moe_with_profile_buffer",
                side_effect=forward_kernel,
            ),
            patch.object(
                function_module.multicore_ops,
                "mega_moe_grad_with_profile_buffer",
                new=backward_kernel,
            ),
            patch.object(function_module, "_restore_input_gradient", new=restore_without_scratch),
            patch.object(function_module.multicore_ops, "moe_token_permute_grad",
                         side_effect=permutation_gradient) as mock_permutation,
        ):
            for tag, capacity in enumerate(capacities, start=1):
                source = torch.full((2 if permuted else 4, 4), float(tag), requires_grad=input_grad)
                route = SimpleNamespace(expert_capacity=capacity, group_list=torch.tensor([0, capacity]))
                for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size",
                             "combine_src_off", "combine_target_off", "combine_size"):
                    setattr(route, name, torch.zeros(4, dtype=torch.int64))
                output = _forward(source, route)
                saved = output.grad_fn.saved_tensors
                self.assertEqual(len(saved), 13 if permuted and input_grad else 12)
                self.assertTrue(all(t.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
                                    for t in saved))
                self.assertEqual(saved[0].data_ptr(), (dispatch_pointers if pull else down_pointers)[-1])
                self.assertNotEqual(saved[0].data_ptr(), workspace.expert_buffer.data_ptr())
                self.assertEqual(saved[0].untyped_storage().nbytes(), capacity * 4 * source.element_size())
                self.assertEqual(tuple(output.shape), (4, 4))
                outputs.append(output)
                sources.append(source)
            for tag in range(len(outputs), 0, -1):
                self.assertTrue(torch.equal(outputs[tag - 1], torch.full((4, 4), float(tag + 40))))
                outputs[tag - 1].sum().backward()
                source = sources[tag - 1]
                expected = torch.full_like(source, float(tag * (2 if permuted else 1)))
                if input_grad:
                    self.assertTrue(torch.equal(source.grad, expected))
                else:
                    self.assertIsNone(source.grad)
        self.assertTrue(torch.equal(weight1.grad, torch.full_like(weight1, 10.0)))
        self.assertTrue(torch.equal(weight2.grad, torch.full_like(weight2, 10.0)))
        self.assertEqual(mock_permutation.call_count, len(outputs) if permuted and input_grad else 0)
        self.assertEqual(backward_capacities, list(reversed(capacities)))
        self.assertEqual(workspace.claim.call_count, 8)
        self.assertEqual(workspace.release.call_count, 8)


if __name__ == "__main__":
    unittest.main()
