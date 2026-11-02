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
"""Check registered NPU operator policies without executing a device kernel."""
# pylint: disable=protected-access

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.distributed import activation_checkpoint as checkpointing


class SelectiveNpuOperatorPolicyTest(unittest.TestCase):
    """Optional operator registration is sufficient; no accelerator is required."""

    def _registered(self, *names: str) -> list[tuple[str, object]]:
        operators = [(name, checkpointing._resolve_torch_op(name)) for name in names]
        available = [(name, op) for name, op in operators if op is not None]
        if not available:
            self.skipTest("Optional NPU operators are not registered in this environment")
        return available

    def test_fusion_attention_overloads_are_saved(self) -> None:
        """Legacy and v3 attention follow the expensive-operator policy."""
        operators = self._registered("npu.npu_fusion_attention", "npu.npu_fusion_attention_v3")
        for name, operator in operators:
            with self.subTest(operator=name):
                eager = checkpointing._make_selective_checkpoint_policy_fn()
                for is_recompute in (False, True):
                    context = SimpleNamespace(is_recompute=is_recompute)
                    self.assertEqual(eager(context, operator), checkpointing.CheckpointPolicy.MUST_SAVE)
                    self.assertEqual(eager(context, operator), checkpointing.CheckpointPolicy.MUST_SAVE)
                    self.assertEqual(checkpointing.compile_selective_checkpoint_policy(context, operator),
                                     checkpointing.CheckpointPolicy.MUST_SAVE)

    def test_grouped_matmul_overloads_share_the_region_counter(self) -> None:
        """Tensor/list group descriptors keep matmul cadence and phase isolation."""
        operators = self._registered("npu.npu_grouped_matmul", "npu.npu_grouped_matmul.List")
        for name, operator in operators:
            with self.subTest(operator=name):
                eager = checkpointing._make_selective_checkpoint_policy_fn()
                for is_recompute in (False, True):
                    context = SimpleNamespace(is_recompute=is_recompute)
                    self.assertEqual(eager(context, operator), checkpointing.CheckpointPolicy.MUST_SAVE)
                    self.assertEqual(eager(context, torch.ops.aten.mm.default),
                                     checkpointing.CheckpointPolicy.MUST_RECOMPUTE)
                    self.assertEqual(eager(context, operator), checkpointing.CheckpointPolicy.MUST_SAVE)
                    self.assertEqual(checkpointing.compile_selective_checkpoint_policy(context, operator),
                                     checkpointing.CheckpointPolicy.MUST_SAVE)
                fresh = checkpointing._make_selective_checkpoint_policy_fn()
                self.assertEqual(fresh(SimpleNamespace(is_recompute=False), operator),
                                 checkpointing.CheckpointPolicy.MUST_SAVE)


if __name__ == "__main__":
    unittest.main()
