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
"""Tests for the Torch-native selective checkpoint compile adapter."""
import os
import unittest
from unittest.mock import MagicMock

import torch
import torch.utils.checkpoint as torch_checkpoint

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, checkpoint
from hyper_parallel.platform.torch.activation_checkpoint.native_compile import (
    _to_torch_checkpoint_policy,
    _torch_policy_adapter,
    create_native_selective_checkpoint_contexts,
)


class TestNativeCompilePolicyAdapter(unittest.TestCase):
    """Validate the deliberately narrow compile policy compatibility layer."""

    def test_converts_supported_hyper_parallel_policies_by_name(self):
        for policy_name in (
                "MUST_SAVE", "PREFER_SAVE", "MUST_RECOMPUTE", "PREFER_RECOMPUTE"):
            with self.subTest(policy_name=policy_name):
                result = _to_torch_checkpoint_policy(getattr(CheckpointPolicy, policy_name))
                self.assertIs(result, getattr(torch_checkpoint.CheckpointPolicy, policy_name))

    def test_preserves_supported_native_torch_policies(self):
        for policy_name in (
                "MUST_SAVE", "PREFER_SAVE", "MUST_RECOMPUTE", "PREFER_RECOMPUTE"):
            with self.subTest(policy_name=policy_name):
                policy = getattr(torch_checkpoint.CheckpointPolicy, policy_name)
                self.assertIs(_to_torch_checkpoint_policy(policy), policy)

    def test_rejects_bool_and_unknown_return_types(self):
        for policy in (True, False, 0, "MUST_SAVE", None):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(TypeError, "must return"):
                    _to_torch_checkpoint_policy(policy)

    def test_rejects_hyper_parallel_swap(self):
        with self.assertRaisesRegex(ValueError, "MUST_SWAP"):
            _to_torch_checkpoint_policy(CheckpointPolicy.MUST_SWAP)

    def test_rejects_torch_offload_policies_when_present(self):
        for policy_name in ("MUST_CPU_OFFLOAD", "PREFER_CPU_OFFLOAD"):
            policy = getattr(torch_checkpoint.CheckpointPolicy, policy_name, None)
            if policy is None:
                continue
            with self.subTest(policy_name=policy_name):
                with self.assertRaisesRegex(ValueError, policy_name):
                    _to_torch_checkpoint_policy(policy)

    def test_passes_native_context_and_operator_to_user_policy(self):
        native_context = object()
        op = object()
        policy_fn = MagicMock(return_value=CheckpointPolicy.MUST_SAVE)

        result = _torch_policy_adapter(policy_fn, native_context, op, 1, flag=True)

        self.assertIs(result, torch_checkpoint.CheckpointPolicy.MUST_SAVE)
        policy_fn.assert_called_once_with(native_context, op, 1, flag=True)

    def test_creates_torch_native_selective_contexts(self):
        policy_fn = MagicMock(return_value=CheckpointPolicy.PREFER_RECOMPUTE)

        forward_mode, recompute_mode = create_native_selective_checkpoint_contexts(policy_fn)

        self.assertIsInstance(forward_mode, torch.utils._python_dispatch.TorchDispatchMode)
        self.assertIsInstance(recompute_mode, torch.utils._python_dispatch.TorchDispatchMode)

    def test_requires_callable_policy(self):
        with self.assertRaisesRegex(TypeError, "callable"):
            create_native_selective_checkpoint_contexts(CheckpointPolicy.MUST_SAVE)

    def test_selective_checkpoint_compiles_with_aot_autograd(self):
        """Native SAC should survive Dynamo capture and AOTAutograd backward tracing."""
        received_contexts = []

        def policy_fn(context, op, *args, **kwargs):  # pylint: disable=W0613
            received_contexts.append(context)
            if op == torch.ops.aten.mm.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(8, 16)
                self.linear2 = torch.nn.Linear(16, 8)

            def checkpointed_body(self, value):
                return self.linear2(torch.nn.functional.gelu(self.linear1(value)))

            def forward(self, value):
                return checkpoint(self.checkpointed_body, value, policy_fn=policy_fn)

        model = Block()
        compiled_model = torch.compile(model, backend="aot_eager", fullgraph=True)
        inputs = torch.randn(4, 8, requires_grad=True)

        compiled_model(inputs).square().mean().backward()

        self.assertTrue(received_contexts)
        self.assertTrue(all(isinstance(context, torch_checkpoint.SelectiveCheckpointContext)
                            for context in received_contexts))
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
