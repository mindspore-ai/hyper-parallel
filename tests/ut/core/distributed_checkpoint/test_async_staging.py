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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.async_staging`."""
# pylint: disable=wrong-import-position
import importlib
import os
import unittest

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import hyper_parallel.platform.platform as _platform_mod

_platform_mod.platform = None

import hyper_parallel.core.distributed_checkpoint.async_staging as staging_mod

importlib.reload(staging_mod)

from hyper_parallel.core.distributed_checkpoint.async_staging import build_staged_state_dict


class TestAsyncStaging(unittest.TestCase):
    """Tests for async checkpoint staging helpers."""

    def setUp(self) -> None:
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(staging_mod)

    def test_build_staged_state_dict_copies_tensors_to_cpu(self):
        """
        Feature: build_staged_state_dict host staging.
        Description: Stage a nested state dict with a GPU-resident tensor (if available).
        Expectation: Staged copy is on CPU, equal in value, and independent object.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight = torch.nn.Parameter(torch.ones(2, 3, device=device) * 3.0)
        original = {"model": {"weight": weight}}
        staged = build_staged_state_dict(original)
        staged_weight = staged["model"]["weight"]
        self.assertIsNot(staged_weight, weight)
        self.assertFalse(staged_weight.is_cuda)
        torch.testing.assert_close(staged_weight, weight.cpu())

    def test_build_staged_state_dict_deep_copies_bytes(self):
        """
        Feature: build_staged_state_dict bytes isolation.
        Description: Stage state dict containing a bytearray leaf.
        Expectation: Staged bytes are equal but not the same mutable buffer object.
        """
        buf = bytearray(b"checkpoint-meta")
        staged = build_staged_state_dict({"meta": buf})
        self.assertEqual(staged["meta"], bytes(buf))
        self.assertIsInstance(staged["meta"], bytes)
        self.assertIsNot(staged["meta"], buf)

    def test_build_staged_state_dict_preserves_nested_structure(self):
        """
        Feature: build_staged_state_dict structural round-trip.
        Description: Nested dict/list optimizer-style state with tensor leaves.
        Expectation: Staged dict mirrors keys and nesting of the input.
        """
        state = {
            "model": {"w": torch.zeros(2)},
            "optim": [{"step": 1, "exp_avg": torch.zeros(2)}],
        }
        staged = build_staged_state_dict(state)
        self.assertIn("w", staged["model"])
        self.assertEqual(staged["optim"][0]["step"], 1)
        self.assertEqual(tuple(staged["optim"][0]["exp_avg"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
