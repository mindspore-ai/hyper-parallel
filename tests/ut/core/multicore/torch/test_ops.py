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

"""Unit tests for native Torch adapter registration diagnostics."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore._loader import NativeComponentUnavailableError
from hyper_parallel.core.multicore.torch import ops


class TestTorchOps(unittest.TestCase):
    """Native initialization is cached, but Torch itself is imported normally."""

    def tearDown(self) -> None:
        """Never retain a mocked native-registration cache between tests."""
        ops._load_native.cache_clear()

    @staticmethod
    def _backward_op(alias: str = "a") -> SimpleNamespace:
        """Build a real schema for the mutable receive and expert input-gradient slots."""
        middle = ", ".join(f"Tensor input_{index}" for index in range(1, 12))
        schema = ops.torch._C.parse_schema(
            f"mega_moe_grad(Tensor(a!) dispatch, {middle}, Tensor({alias}!) gate_dx) -> ()"
        )
        return SimpleNamespace(default=SimpleNamespace(_schema=schema))

    def test_adapter_failure_reports_original_cause(self):
        """Preserve the failed library and ABI error in the native diagnostic."""
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("bad.so"))),
            patch.object(ops, "preload_vendor_library"),
            patch.object(ops.torch.ops, "load_library", side_effect=OSError("bad ABI")),
            self.assertRaisesRegex(NativeComponentUnavailableError, "ADAPTER-LOAD-FAILED.*bad ABI"),
        ):
            ops._load_native()

    def test_successful_registration_is_cached(self):
        """Load the adapter only once across repeated operation calls."""
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("good.so"))),
            patch.object(ops, "preload_vendor_library") as preload,
            patch.object(ops.torch.ops, "load_library") as load,
            patch.object(ops.torch.ops.hyper_parallel, "mega_moe_grad", self._backward_op(), create=True),
        ):
            ops._load_native()
            ops._load_native()
        preload.assert_called_once_with(Path("vendor"))
        load.assert_called_once_with("good.so")

    def test_adapter_rejects_stale_backward_alias_schema(self) -> None:
        """Reject adapters with disjoint dY/dX alias sets before sharing receive storage."""
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("old.so"))),
            patch.object(ops, "preload_vendor_library"),
            patch.object(ops.torch.ops, "load_library"),
            patch.object(ops.torch.ops.hyper_parallel, "mega_moe_grad", self._backward_op("e"), create=True),
            self.assertRaisesRegex(NativeComponentUnavailableError, "backward dispatch storage reuse"),
        ):
            ops._load_native()
