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
"""Returned PCP probabilities preserve native token selection and the CP1 formula."""

from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rl.unified_execution.runtimes.vllm.model import compute_pcp_token_log_probs, install_pcp_log_probs


class TestPCPLogProbs(unittest.TestCase):
    """Check the numerical boundary and the narrowly scoped native binding."""

    def test_scores_native_ids_without_mutating_logits_or_selection(self) -> None:
        """Selected and top-k IDs retain order, duplicates and the existing FP32 semantics."""
        generator = torch.Generator().manual_seed(51)
        for dtype in (torch.bfloat16, torch.float32):
            logits = torch.randn(3, 131, generator=generator).to(dtype)[:, ::2]
            ids = torch.tensor([[4, 0, 4], [65, 3, 1], [2, 17, 6]], dtype=torch.int32)
            before_logits, before_ids = logits.clone(), ids.clone()
            actual = compute_pcp_token_log_probs(logits, ids)
            expected = torch.stack([row.float().log_softmax(-1)[selected.long()]
                                    for row, selected in zip(logits, ids)])
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
            self.assertTrue(torch.equal(logits, before_logits))
            self.assertTrue(torch.equal(ids, before_ids))

    def test_empty_request_batch(self) -> None:
        """An inactive worker may receive no selected requests."""
        actual = compute_pcp_token_log_probs(torch.empty(0, 67), torch.empty(0, 1, dtype=torch.int32))
        self.assertEqual(tuple(actual.shape), (0, 1))

    def test_native_binding_is_idempotent_and_rejects_foreign_override(self) -> None:
        """Do not silently overwrite another extension's probability implementation."""
        module = SimpleNamespace(__name__="vllm_ascend.worker.v2.sample.logprob")
        package = ModuleType("vllm_ascend.worker.v2.sample")
        package.logprob = module
        worker_package = ModuleType("vllm.v1.worker.gpu.sample")
        worker_module = SimpleNamespace()
        worker_package.logprob = worker_module

        def native(*_args):
            raise AssertionError("Native arithmetic must be replaced for the approved PCP path")

        native.__module__ = "vllm_ascend.worker.v2.sample.logprob"
        module.compute_token_logprobs = native
        worker_module.compute_token_logprobs = native
        with patch.dict("sys.modules", {"vllm_ascend.worker.v2.sample": package,
                                        "vllm.v1.worker.gpu.sample": worker_package}):
            install_pcp_log_probs()
            install_pcp_log_probs()
            self.assertIs(module.compute_token_logprobs, compute_pcp_token_log_probs)
            self.assertIs(worker_module.compute_token_logprobs, compute_pcp_token_log_probs)
            module.compute_token_logprobs = native
            worker_module.compute_token_logprobs = lambda *_: None
            with self.assertRaisesRegex(RuntimeError, "unexpected MRV2"):
                install_pcp_log_probs()
            self.assertIs(module.compute_token_logprobs, native)
