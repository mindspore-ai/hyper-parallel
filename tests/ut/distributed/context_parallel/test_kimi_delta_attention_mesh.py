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
"""Reject invalid KDA boundary contracts before creating transport work."""
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.distributed.context_parallel import kimi_delta_attention_mesh as cp
from tests.common.mark_utils import arg_mark


class TestKDABoundaryContracts(unittest.TestCase):
    """Keep protocol guards local; real ordering/lifetimes are covered by ST."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_contracts_rejected_before_transport(self):
        """Feature: KDA boundary validation.

        Description: Pass incompatible shapes, dtypes, rank orders and options.
        Expectation: Invalid calls fail before transport is invoked.
        """
        state = torch.zeros(1, 2, 128, 128)
        protocols = [cp.AllGatherBoundary(None, 0, 4),
                     cp.GroupedAllGatherBoundary(None, None, 0, (0, 2, 4, 6), 2)]
        with patch.object(cp, "dist") as transport:
            for protocol in protocols:
                with self.assertRaises(ValueError):
                    protocol.forward(state, state.double())
                with self.assertRaises(ValueError):
                    protocol.backward(state, state[..., :64])
            transport.all_gather_into_tensor.assert_not_called()
            transport.irecv.assert_not_called()
        with self.assertRaisesRegex(ValueError, "ordered"):
            cp.GroupedAllGatherBoundary(None, None, 0, (2, 0, 6, 4), 2)
        with self.assertRaisesRegex(ValueError, "only valid"):
            cp.build_kda_boundary(None, "allgather", 2)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            cp.build_kda_boundary(None, "unknown", 1)
