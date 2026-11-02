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
"""CPU checks for forward metadata alias validation and repeated query costs."""

from __future__ import annotations

import unittest
from typing import Any

import torch

from hyper_parallel.core.multicore.modules.routed_experts.forward import _check_tensor, _check_writers
from tests.common.mark_utils import arg_mark


class _CountedTensor:
    """Count scalar metadata queries while returning a real CPU tensor's storage."""

    def __init__(self, value: torch.Tensor) -> None:
        """Wrap one tensor and initialize its per-method query counters."""
        self.value = value
        self.calls: dict[str, int] = {}

    def __getattr__(self, name: str):
        value = getattr(self.value, name)
        if name not in ("numel", "data_ptr", "untyped_storage", "element_size"):
            return value

        def counted(*args: Any, **kwargs: Any) -> Any:
            """Count a metadata query before forwarding it to the real tensor."""
            self.calls[name] = self.calls.get(name, 0) + 1
            return value(*args, **kwargs)

        return counted


class _UntouchableReader:
    """Reject metadata access when there is no nonempty writer to validate."""

    def numel(self) -> int:
        """Fail if validation inspects this unnecessary reader."""
        raise AssertionError("empty writers must not inspect readers")


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestForwardMetadata(unittest.TestCase):
    """Preserve strict writer ownership without repeated storage queries."""

    def test_same_storage_disjoint_views_are_rejected(self) -> None:
        """Separate byte intervals still violate the shared-storage contract."""
        base = torch.empty(64, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "alias"):
            _check_writers([base[:16]], [base[32:48]])
        with self.assertRaisesRegex(ValueError, "alias"):
            _check_writers([base[:16], base[32:48]], [])

    def test_independent_storage_objects_still_check_overlapping_bytes(self) -> None:
        """Different storage wrappers may cover overlapping external buffer ranges."""
        backing = bytearray(64)
        writer = torch.frombuffer(backing, dtype=torch.uint8, count=32)
        reader = torch.frombuffer(backing, dtype=torch.uint8, count=16, offset=16)
        self.assertNotEqual(writer.untyped_storage().data_ptr(), reader.untyped_storage().data_ptr())
        with self.assertRaisesRegex(ValueError, "overlap"):
            _check_writers([writer], [reader])

    def test_adjacent_independent_storage_ranges_are_accepted(self) -> None:
        """Touching exclusive interval ends do not overlap."""
        backing = bytearray(64)
        writer = torch.frombuffer(backing, dtype=torch.uint8, count=32)
        reader = torch.frombuffer(backing, dtype=torch.uint8, count=16, offset=32)
        _check_writers([writer], [reader])

    def test_empty_writers_skip_readers_and_readonly_aliases_remain_valid(self) -> None:
        """Empty operands and aliases among readers introduce no writer conflict."""
        empty = torch.empty(0)
        _check_writers([], [_UntouchableReader()])
        _check_writers([empty], [_UntouchableReader()])
        reader = torch.empty(32)
        _check_writers([torch.empty(16), empty], [reader, reader[:8], reader[:0]])

    def test_each_operand_metadata_is_queried_once(self) -> None:
        """One invocation snapshots each operand without retaining it across frames."""
        values = [_CountedTensor(torch.empty(16)) for _ in range(8)]
        _check_writers(values[:3], values[3:])
        for value in values:
            self.assertEqual(value.calls, {"numel": 1, "untyped_storage": 1, "data_ptr": 1, "element_size": 1})
        _check_writers(values[:3], values[3:])
        for value in values:
            self.assertEqual(value.calls, {"numel": 2, "untyped_storage": 2, "data_ptr": 2, "element_size": 2})

    def test_tensor_device_contract_is_checked_before_alias_validation(self) -> None:
        """An invalid device is rejected without requiring accelerator hardware."""
        value = torch.empty((2, 2))
        with self.assertRaisesRegex(ValueError, "device"):
            _check_tensor(value, (2, 2), value.dtype, torch.device("meta"), "wrong device")
