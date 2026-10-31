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
"""CPU contracts for native MoonEP allocator loading and device UB forwarding."""
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import unittest
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.expert_parallel.moonep import allocator
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_allocator_cache_preserves_native_entry_and_forwards_ub() -> None:
    """Repeated imports must retain the native entry and pass the queried device UB."""
    native = ModuleType("test_native_moonep_allocator")
    result = [torch.empty(0)]
    original_allocate = Mock(return_value=result)
    native.allocate = original_allocate
    counts = Mock(spec=torch.Tensor)
    counts.device.index = 3
    counts.device.type = "npu"
    allocator.load_allocator.cache_clear()
    try:
        with (patch.object(allocator.importlib, "import_module", return_value=native) as imported,
              patch.object(allocator, "_device_ub_bytes", return_value=191 * 1024) as ub_bytes):
            assert allocator.load_allocator() is native
            assert allocator.load_allocator() is native
            assert allocator.allocate_experts(counts, 128) is result
            assert allocator.allocate_experts(counts, 256) is result
            imported.assert_called_once_with(allocator.__package__ + "._a3_allocator")
            ub_bytes.assert_called_with(3)
            assert ub_bytes.call_count == 2
        assert getattr(native, "allocate") is original_allocate
        assert original_allocate.call_args_list[0].args == (counts, 128, 191 * 1024)
        assert original_allocate.call_args_list[1].args == (counts, 256, 191 * 1024)
    finally:
        allocator.load_allocator.cache_clear()


class TestMoonEPRuntimeQuery(unittest.TestCase):
    """Keep device metadata fast, cached and independent of compiler subprocesses."""

    def tearDown(self) -> None:
        """Leave no mocked library or hardware query cached for another case."""
        allocator._device_ub_bytes.cache_clear()
        allocator._platform_query.cache_clear()

    def test_runtime_metadata_is_queried_once_per_device(self) -> None:
        """Use the queried capacity and preserve the caller's device context."""
        def query(_info: int, value: Any, _size: int) -> int:
            """Write the runtime result into its caller-owned C buffer."""
            value.value = b"196608"
            return 0

        native_query = Mock(side_effect=query)
        library = SimpleNamespace(aclplatformGetDeviceInfo=native_query)
        device_context = Mock(side_effect=lambda _index: nullcontext())
        allocator._device_ub_bytes.cache_clear()
        allocator._platform_query.cache_clear()
        with (patch.object(allocator.ctypes, "CDLL", return_value=library) as load,
              patch.object(allocator.torch, "npu", SimpleNamespace(device=device_context), create=True)):
            self.assertEqual(allocator._device_ub_bytes(3), 196608)
            self.assertEqual(allocator._device_ub_bytes(3), 196608)
            self.assertEqual(allocator._device_ub_bytes(4), 196608)
        load.assert_called_once_with("libascendcl.so")
        self.assertEqual(native_query.call_count, 2)
        self.assertEqual(device_context.call_count, 2)

    def test_missing_or_invalid_runtime_values_do_not_use_a_fixed_capacity(self) -> None:
        """Fail before a planner launch instead of assuming another device's UB."""
        for value in (b"0", b"", b"invalid"):
            with self.subTest(value=value):
                def query(_info: int, buffer: Any, _size: int) -> int:
                    """Provide malformed metadata without a hardware dependency."""
                    buffer.value = value
                    return 0

                allocator._device_ub_bytes.cache_clear()
                with (patch.object(allocator, "_platform_query", return_value=query),
                      patch.object(allocator.torch, "npu", SimpleNamespace(device=lambda _i: nullcontext()),
                                   create=True), self.assertRaises(RuntimeError)):
                    allocator._device_ub_bytes(0)

    def test_cpu_counts_are_rejected_before_loading_native_kernels(self) -> None:
        """The NPU kernel must never consume an ordinary CPU data pointer."""
        with (patch.object(allocator, "load_allocator") as load,
              self.assertRaisesRegex(ValueError, "Ascend NPU")):
            allocator.allocate_experts(torch.zeros(2, 4, dtype=torch.int32), 128)
        load.assert_not_called()

    def test_missing_runtime_api_fails_without_loading_a_compiler(self) -> None:
        """An incompatible CANN runtime produces an explicit compatibility error."""
        allocator._platform_query.cache_clear()
        with (patch.object(allocator.ctypes, "CDLL", return_value=SimpleNamespace()),
              self.assertRaisesRegex(RuntimeError, "platform-query API")):
            allocator._platform_query()
