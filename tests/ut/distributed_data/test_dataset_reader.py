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
"""Tests for raw payload memory pinning during planned reads."""

import unittest
from unittest.mock import patch

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.metadata import PlannedSampleLoader
from hyper_parallel.distributed_data.schema import SampleKey


class _PinnablePayload:
    def __init__(self, value: int) -> None:
        """Store a value and expose whether the walker visited it."""
        self.value = value
        self.pin_calls = 0

    def pin_memory(self) -> str:
        """Record recursive pinning without requiring accelerator hardware."""
        self.pin_calls += 1
        return f"pinned-{self.value}"


class TestPlannedPayload(unittest.TestCase):
    """Verify raw payloads retain native DataLoader memory-pinning behavior."""

    def test_planned_reads_pin_nested_payloads(self) -> None:
        """Feature: Raw payload memory pinning.
        Description: Fetch a nested custom pinnable object through the native DataLoader.
        Expectation: The requested key maps to the recursively pinned payload.
        """
        pinnable = _PinnablePayload(9)
        loader = PlannedSampleLoader(
            [{"nested": [pinnable]}], num_workers=0, pin_memory=True,
            prefetch_factor=None, persistent_workers=False, seed=17,
        )
        key = SampleKey(0, 0, 17)
        with patch("torch.accelerator.is_available", return_value=True), patch(
            "torch.accelerator.current_accelerator", return_value=torch.device("cuda"),
        ):
            payloads = loader.fetch_samples([key])

        self.assertEqual(payloads, {key: {"nested": ["pinned-9"]}})
        self.assertEqual(pinnable.pin_calls, 1)
