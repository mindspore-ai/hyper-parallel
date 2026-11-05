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
"""CPU coverage for the DSA CANN Indexer interface."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional.dsa_indexer import dsa_indexer


class TestDSAIndexer(unittest.TestCase):
    """Check the package call receives the expected TND contract."""

    def test_cann_indexer_builds_tnd_metadata(self):
        """
        Feature: CANN DSA indexer metadata.
        Description: Invoke the indexer with variable-length packed sequences.
        Expectation: Build TND metadata and pass cumulative sequence lengths.
        """
        query = torch.randn(1, 3, 2, 4)
        key = torch.randn(1, 4, 1, 4)
        weights = torch.randn(1, 3, 2)
        indices = torch.zeros(3, 1, 2, dtype=torch.int32)
        ops = ModuleType("cann_ops_transformer.ops")
        ops.lightning_indexer = Mock(return_value=(indices, None))
        metadata = torch.zeros(1024, dtype=torch.int32)
        ops.lightning_indexer_metadata = Mock(return_value=metadata)
        package = ModuleType("cann_ops_transformer")
        package.ops = ops
        with patch.dict(sys.modules, {"cann_ops_transformer": package, "cann_ops_transformer.ops": ops}):
            actual = dsa_indexer(
                query,
                key,
                weights,
                torch.tensor([2, 3], dtype=torch.int32),
                torch.tensor([2, 4], dtype=torch.int32),
                2,
            )

        self.assertIs(actual[0], indices)
        self.assertEqual([tensor.shape for tensor in actual[1:]], [(3, 2, 4), (4, 1, 4), (3, 2)])
        kwargs = ops.lightning_indexer.call_args.kwargs
        self.assertEqual(kwargs["layout_q"], "TND")
        self.assertEqual(kwargs["mask_mode"], 3)
        self.assertIs(kwargs["metadata"], metadata)
        options = ops.lightning_indexer_metadata.call_args.kwargs
        self.assertEqual(options["max_seqlen_q"], 2)
        self.assertEqual(options["max_seqlen_k"], 2)
        torch.testing.assert_close(kwargs["cu_seqlens_q"], torch.tensor([0, 2, 3], dtype=torch.int32))
        torch.testing.assert_close(kwargs["cu_seqlens_k"], torch.tensor([0, 2, 4], dtype=torch.int32))
