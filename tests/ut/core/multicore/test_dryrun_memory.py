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
"""Unit tests for dynamic mega-kernel dry-run memory measurement."""

import os
import sys
import types
import unittest
from unittest import mock

import hyper_parallel.core.multicore as multicore_api
from hyper_parallel.core.multicore import MegaKernelMemoryUsage
from hyper_parallel.platform.mindspore.multicore import MSMulticoreHandler
from hyper_parallel.platform.torch.multicore import TorchMulticoreHandler


class TestMegaKernelMemoryUsage(unittest.TestCase):
    """Test the backend-neutral dynamic measurement result."""

    def test_peak_adds_only_external_reservation_overhead(self) -> None:
        """Test external logical bytes are not counted twice."""
        usage = MegaKernelMemoryUsage(
            kernel_name="mega_moe",
            allocated_before_bytes=100,
            allocated_after_bytes=110,
            allocator_peak_bytes=150,
            external_reserved_bytes=1024,
            external_logical_bytes=256,
        )

        self.assertEqual(usage.external_reservation_overhead_bytes, 768)
        self.assertEqual(usage.peak_bytes, 918)
        self.assertAlmostEqual(usage.allocator_peak_mib, 150 / (1024 * 1024))
        self.assertAlmostEqual(usage.peak_mib, 918 / (1024 * 1024))

    def test_invalid_values_are_rejected(self) -> None:
        """Test names, byte counters, and external accounting constraints."""
        with self.assertRaisesRegex(ValueError, "kernel_name"):
            MegaKernelMemoryUsage("", 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "allocator_peak_bytes"):
            MegaKernelMemoryUsage("mega_moe", 0, 0, -1)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            MegaKernelMemoryUsage("mega_moe", 0, 0, 0, 10, 11)


class TestMegaKernelMemoryApi(unittest.TestCase):
    """Test generic, forward, and backward convenience APIs."""

    def test_generic_api_validates_and_delegates(self) -> None:
        """Test validation occurs before dispatching to the backend."""
        handler = mock.Mock()
        expected = MegaKernelMemoryUsage("future_kernel", 1, 2, 3)
        handler.measure_mega_kernel_memory.return_value = expected
        kernel_call = mock.Mock()

        with mock.patch.object(multicore_api, "_get_multicore_handler", return_value=handler):
            actual = multicore_api.measure_mega_kernel_memory("  future_kernel  ", kernel_call)

        self.assertIs(actual, expected)
        handler.measure_mega_kernel_memory.assert_called_once_with("future_kernel", kernel_call)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            multicore_api.measure_mega_kernel_memory(" ", kernel_call)
        with self.assertRaisesRegex(TypeError, "callable"):
            multicore_api.measure_mega_kernel_memory("mega_moe", None)

    def test_mega_moe_wrapper_forwards_arguments(self) -> None:
        """Test the convenience wrapper uses the generic extension point."""
        expected = MegaKernelMemoryUsage("mega_moe", 1, 2, 3)

        def fake_measure(kernel_name, kernel_call):
            self.assertEqual(kernel_name, "mega_moe")
            self.assertEqual(kernel_call(), "kernel-output")
            return expected

        with mock.patch.object(multicore_api, "mega_moe", return_value="kernel-output") as kernel_mock:
            with mock.patch.object(multicore_api, "measure_mega_kernel_memory", side_effect=fake_measure):
                actual = multicore_api.measure_mega_moe_memory("tensor", rank_id=1)

        self.assertIs(actual, expected)
        kernel_mock.assert_called_once_with("tensor", rank_id=1)


    def test_mega_moe_grad_wrapper_forwards_arguments(self) -> None:
        """Test the backward convenience wrapper uses the generic path."""
        expected = MegaKernelMemoryUsage("mega_moe_grad", 1, 2, 3)

        def fake_measure(kernel_name, kernel_call):
            self.assertEqual(kernel_name, "mega_moe_grad")
            self.assertEqual(kernel_call(), "kernel-output")
            return expected

        with mock.patch.object(
                multicore_api, "mega_moe_grad", return_value="kernel-output") as kernel_mock:
            with mock.patch.object(
                    multicore_api, "measure_mega_kernel_memory", side_effect=fake_measure):
                actual = multicore_api.measure_mega_moe_grad_memory("tensor", rank_id=1)

        self.assertIs(actual, expected)
        kernel_mock.assert_called_once_with("tensor", rank_id=1)


class TestMindSporeDryRunHandler(unittest.TestCase):
    """Test MindSpore measurement orchestration with mocked runtime state."""

    def _fake_modules(self, runtime, symmetric_stats=(1024, 256)):
        """Create import-compatible fake MindSpore and symmetric modules."""
        mindspore_module = types.ModuleType("mindspore")
        mindspore_module.runtime = runtime
        symmetric_module = types.ModuleType(
            "hyper_parallel.platform.mindspore.symmetric_memory"
        )
        symmetric_handler = mock.Mock()
        symmetric_handler.dryrun_memory_stats.return_value = symmetric_stats
        symmetric_module.MSSymmetricMemoryHandler = symmetric_handler
        return {
            "mindspore": mindspore_module,
            "hyper_parallel.platform.mindspore.symmetric_memory": symmetric_module,
        }

    def test_measurement_skips_launch_and_restores_environment(self) -> None:
        """Test counters, external memory, and process environment lifecycle."""
        runtime = mock.Mock()
        runtime.memory_allocated.side_effect = [100, 110]
        runtime.max_memory_allocated.return_value = 150
        observed_flags = []

        def kernel_call():
            observed_flags.append(os.environ.get("HP_MEGA_KERNEL_DRY_RUN"))

        with mock.patch.dict(sys.modules, self._fake_modules(runtime)):
            with mock.patch.dict(os.environ, {"HP_MEGA_KERNEL_DRY_RUN": "previous"}):
                usage = MSMulticoreHandler.measure_mega_kernel_memory("mega_moe", kernel_call)
                self.assertEqual(os.environ["HP_MEGA_KERNEL_DRY_RUN"], "previous")

        self.assertEqual(observed_flags, ["1"])
        runtime.reset_peak_memory_stats.assert_called_once_with()
        runtime.synchronize.assert_called_once_with()
        self.assertEqual(usage.allocated_before_bytes, 100)
        self.assertEqual(usage.allocated_after_bytes, 110)
        self.assertEqual(usage.allocator_peak_bytes, 150)
        self.assertEqual(usage.external_reserved_bytes, 1024)
        self.assertEqual(usage.external_logical_bytes, 256)

    def test_kernel_error_still_restores_environment(self) -> None:
        """Test the process-wide dry-run flag cannot leak after failure."""
        runtime = mock.Mock()
        runtime.memory_allocated.return_value = 100

        def kernel_call():
            raise RuntimeError("kernel planning failed")

        with mock.patch.dict(sys.modules, self._fake_modules(runtime)):
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "planning failed"):
                    MSMulticoreHandler.measure_mega_kernel_memory("mega_moe", kernel_call)
                self.assertNotIn("HP_MEGA_KERNEL_DRY_RUN", os.environ)

    def test_global_simulation_is_rejected(self) -> None:
        """Test compile simulation is rejected because it reports zero workspace."""
        runtime = mock.Mock()
        with mock.patch.dict(sys.modules, self._fake_modules(runtime)):
            with mock.patch.dict(os.environ, {"MS_SIMULATION_LEVEL": "1"}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "compile simulation"):
                    MSMulticoreHandler.measure_mega_kernel_memory("mega_moe", mock.Mock())

    def test_torch_backend_is_explicitly_unsupported(self) -> None:
        """Test unsupported backends fail without executing the callable."""
        kernel_call = mock.Mock()
        with self.assertRaisesRegex(NotImplementedError, "only supported by MindSpore"):
            TorchMulticoreHandler.measure_mega_kernel_memory("mega_moe", kernel_call)
        kernel_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()