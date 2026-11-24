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

"""Integer return restoration and explicit backend selection without NPU."""

import unittest

import torch

from hyper_parallel.components.ops import ops, prepare_indexer_topk_restore
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestIndexerTopKRestore(unittest.TestCase):
    """Check exact IDs and preserve the early Torch local bypass by default."""

    def test_native_integer_routes(self) -> None:
        """Validate the independent row oracle and input immutability."""
        for dtype in (torch.int32, torch.int64):
            for case in precision_cases("indexer_topk_restore", dtype):
                with self.subTest(case=case.name):
                    check_case(case)

    def test_schedule_selection(self) -> None:
        """CPU auto retains Torch and an explicit incompatible backend fails."""
        previous = ops.get_backends()
        selected = torch.zeros((3, 17), dtype=torch.int32)
        try:
            ops.set_backends({})
            self.assertIsNone(prepare_indexer_topk_restore(selected))
            ops.set_backends({"indexer_topk_restore": "triton_ascend"})
            with self.assertRaisesRegex(RuntimeError, "not registered"):
                prepare_indexer_topk_restore(selected)
        finally:
            ops.set_backends(previous)
