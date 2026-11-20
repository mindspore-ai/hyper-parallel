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
"""CPU-only contracts for the CSA2 Indexer Q/K precision boundary."""

import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.quantization.config import CSA2IndexerQKFormat
from hyper_parallel.components.quantization.functional.indexer_qk import create_indexer_qk_policy


class TestIndexerQKPolicy(unittest.TestCase):
    """The independent Q/K policy reshapes per head and preserves autograd."""

    def test_policy_requires_a_configured_format(self):
        """An absent Q/K format is handled by the Indexer, not an identity policy."""
        with self.assertRaisesRegex(TypeError, "expects a CSA2IndexerQKFormat"):
            create_indexer_qk_policy(None)

    def test_query_qdq_and_key_single_packed_write_preserve_ste(self):
        """Full scores and Reindex cache share one Key quantization."""
        policy = create_indexer_qk_policy(CSA2IndexerQKFormat(act_format="mxfp4"))
        query = torch.randn(2, 3, 4, 32, requires_grad=True)
        cache = Mock()
        cache.dequantize.return_value = query + (query.round() - query).detach()
        with patch(
            "hyper_parallel.components.quantization.functional.indexer_qk._native_qdq",
            side_effect=lambda values: values.round(),
        ) as qdq, patch(
            "hyper_parallel.components.quantization.functional.indexer_qk.pack_csa2_cache",
            return_value=cache,
        ) as pack:
            quantized_query = policy.apply_query(query)
            quantized_key, shared_cache = policy.encode_owned_key(query)
        qdq.assert_called_once()
        pack.assert_called_once_with(query, "indexer_k")
        cache.dequantize.assert_called_once_with()
        self.assertIs(shared_cache, cache)
        self.assertEqual(qdq.call_args.args[0].shape, (24, 32))
        self.assertTrue(torch.equal(quantized_query, query.round()))
        self.assertTrue(torch.equal(quantized_key, quantized_query))
        (quantized_query + quantized_key).sum().backward()
        self.assertTrue(torch.equal(query.grad, torch.full_like(query, 2)))

    def test_invalid_width_fails_before_operator_call(self):
        """A partial head must never borrow values from the next row."""
        policy = create_indexer_qk_policy(CSA2IndexerQKFormat(act_format="mxfp4"))
        with patch("hyper_parallel.components.quantization.functional.indexer_qk._native_qdq") as qdq:
            with self.assertRaisesRegex(ValueError, "not divisible"):
                policy.apply_query(torch.randn(1, 2, 3, 33))
        qdq.assert_not_called()

    def test_empty_local_query_skips_native_operator(self):
        """An empty CP shard must not invoke a kernel with zero rows."""
        policy = create_indexer_qk_policy(CSA2IndexerQKFormat(act_format="mxfp4"))
        query = torch.empty(1, 0, 2, 32)
        with patch("hyper_parallel.components.quantization.functional.indexer_qk._native_qdq") as qdq:
            output = policy.apply_query(query)
        self.assertEqual(output.shape, query.shape)
        qdq.assert_not_called()


if __name__ == "__main__":
    unittest.main()
