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
"""Hardware-independent eligibility and optional-backend failure boundaries."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.expert_parallel import receive_permutation as receive


class TestReceivePermutation(unittest.TestCase):
    """Keep unsupported paths independent of Triton and surface execution bugs."""

    def tearDown(self) -> None:
        """Avoid caching mocked optional dependency discovery across tests."""
        receive._get_direct_builder.cache_clear()

    def test_cpu_never_imports_backend(self):
        """Feature: CPU receive-map fallback.

        Description: Pass a large CPU map while observing the optional loader.
        Expectation: Return no candidate and never import the NPU backend.
        """
        counts = torch.tensor([8192], dtype=torch.int64)
        with patch.object(receive, "_get_direct_builder") as loader:
            result = receive.try_generate_receive_indices(counts, counts, 8192)
        self.assertIsNone(result)
        loader.assert_not_called()

    def test_small_unknown_and_unsupported_metadata(self):
        """Feature: Receive-map metadata eligibility.

        Description: Exercise small and unknown sizes and unsupported metadata.
        Expectation: All unsupported inputs bypass optional backend discovery.
        """
        for rows, dtype, contiguous in ((None, torch.int64, True), (8191, torch.int64, True),
                                        (8192, torch.float32, True), (8192, torch.int64, False)):
            with self.subTest(rows=rows, dtype=dtype, contiguous=contiguous):
                tensor = MagicMock(spec=torch.Tensor)
                tensor.device = SimpleNamespace(type="npu")
                tensor.dtype = dtype
                tensor.is_contiguous.return_value = contiguous
                with patch.object(receive, "_get_direct_builder") as loader:
                    result = receive.try_generate_receive_indices(tensor, tensor, rows)
                self.assertIsNone(result)
                loader.assert_not_called()

    def test_missing_optional_backend(self):
        """Feature: Optional receive backend discovery.

        Description: Simulate absent Triton, absent Ascend and backend import failure.
        Expectation: Discovery returns no builder without executing a kernel.
        """
        for discovery, error in ((None, None), (ModuleNotFoundError("triton"), None),
                                  (object(), ImportError("optional backend unavailable"))):
            receive._get_direct_builder.cache_clear()
            with self.subTest(discovery=discovery, error=error):
                with patch.object(receive.importlib.util, "find_spec") as find, \
                        patch.object(receive.importlib, "import_module") as load:
                    if isinstance(discovery, Exception):
                        find.side_effect = discovery
                    else:
                        find.return_value = discovery
                    load.side_effect = error
                    self.assertIsNone(receive._get_direct_builder())
                    if error is None:
                        load.assert_not_called()

    def test_kernel_failure_is_not_silently_replaced(self):
        """Feature: Receive-kernel failure propagation.

        Description: Make an eligible kernel raise an execution error.
        Expectation: The original error propagates instead of silently falling back.
        """
        tensor = MagicMock(spec=torch.Tensor)
        tensor.device = SimpleNamespace(type="npu")
        tensor.dtype = torch.int64
        tensor.is_contiguous.return_value = True
        builder = MagicMock(side_effect=RuntimeError("kernel failure"))
        with patch.object(receive, "_get_direct_builder", return_value=builder):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                receive.try_generate_receive_indices(tensor, tensor, 8192)
        builder.assert_called_once_with(tensor, tensor, 8192)
