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
"""CPU contracts for the optional symmetric Triton product."""

import importlib
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.functional import symmetric_matmul


_MODULE = importlib.import_module("hyper_parallel.components.functional._symmetric_matmul")


def _reference_launch(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Replace only device execution while testing the public buffer contract."""
    product = (a.float() @ b.float()).to(out.dtype)
    lower = torch.tril(product)
    out.copy_(lower + torch.tril(product, diagonal=-1).mT)


class TestSymmetricMatmul(unittest.TestCase):
    """Keep metadata, allocation and dependency tests independent of NPU hardware."""

    def setUp(self) -> None:
        """Start each test with an uncached optional dependency loader."""
        _MODULE._triton_backend.cache_clear()

    def tearDown(self) -> None:
        """Do not leak a mocked backend into another test."""
        _MODULE._triton_backend.cache_clear()

    def test_public_import_does_not_load_device_dependencies(self) -> None:
        """CPU consumers can import the public function without either NPU package."""
        script = """
import importlib.abc
import os
import sys
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
class BlockDevicePackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('triton', 'torch_npu'):
            raise RuntimeError('unexpected device dependency: ' + fullname)
sys.meta_path.insert(0, BlockDevicePackages())
from hyper_parallel.components.functional import symmetric_matmul
assert callable(symmetric_matmul)
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    def test_forward_preserves_inputs_and_reuses_out(self) -> None:
        """The wrapper keeps transposed operands and supports FP32 destinations."""
        for shape in ((7, 19), (2, 7, 19)):
            for dtype in (torch.float16, torch.bfloat16):
                for output_dtype in (dtype, torch.float32):
                    with self.subTest(shape=shape, dtype=dtype, output_dtype=output_dtype):
                        a = torch.randn(shape, dtype=dtype)
                        original = a.clone()
                        b = a.mT
                        out = torch.empty((*shape[:-2], shape[-2], shape[-2]), dtype=output_dtype)
                        launch = Mock(side_effect=_reference_launch)
                        with patch.object(_MODULE, "_require_npu"), patch.object(
                            _MODULE, "_triton_backend", return_value=SimpleNamespace(launch_symmetric_matmul=launch),
                        ):
                            actual = symmetric_matmul(a, b, out=out, out_dtype=output_dtype)
                        self.assertIs(actual, out)
                        self.assertIs(launch.call_args.args[0], a)
                        self.assertIs(launch.call_args.args[1], b)
                        self.assertIs(launch.call_args.args[2], out)
                        torch.testing.assert_close(a, original, rtol=0, atol=0)
                        reference = (a.float() @ b.float()).to(output_dtype)
                        lower = torch.ones(shape[-2], shape[-2], dtype=torch.bool).tril()
                        torch.testing.assert_close(actual[..., lower], reference[..., lower], rtol=0, atol=0)
                        torch.testing.assert_close(actual, actual.mT, rtol=0, atol=0)

    def test_new_output_has_requested_metadata(self) -> None:
        """A fresh destination is contiguous and never aliases either input."""
        a = torch.randn(2, 9, 13, dtype=torch.bfloat16).mT
        with patch.object(_MODULE, "_require_npu"), patch.object(
            _MODULE, "_triton_backend", return_value=SimpleNamespace(launch_symmetric_matmul=_reference_launch),
        ):
            result = symmetric_matmul(a, a.mT)
        self.assertEqual(result.shape, (2, 13, 13))
        self.assertEqual(result.dtype, a.dtype)
        self.assertTrue(result.is_contiguous())
        self.assertNotEqual(result.untyped_storage().data_ptr(), a.untyped_storage().data_ptr())

    def test_empty_and_zero_reductions_do_not_load_backend(self) -> None:
        """Empty results and zero-length reductions need no optional compiler."""
        for shape in ((0, 7), (0, 5, 7), (0, 65536, 65536), (2, 0, 7), (5, 0), (2, 5, 0)):
            with self.subTest(shape=shape):
                a = torch.empty(shape, dtype=torch.bfloat16)
                with patch.object(_MODULE, "_require_npu"), patch.object(_MODULE, "_triton_backend") as load:
                    result = symmetric_matmul(a, a.mT, out_dtype=torch.float32)
                load.assert_not_called()
                self.assertEqual(result.shape, (*shape[:-2], shape[-2], shape[-2]))
                torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)

    def test_bad_shapes_and_broadcasting_are_rejected(self) -> None:
        """Do not silently turn a symmetric product into a rectangular or broadcast GEMM."""
        for left, right in (((7,), (7,)), ((3, 7), (7, 4)), ((2, 3, 7), (1, 7, 3)),
                            ((2, 3, 7), (7, 3)), ((3, 7), (8, 3))):
            with self.subTest(left=left, right=right):
                with self.assertRaisesRegex(ValueError, "rank|Expected"):
                    symmetric_matmul(torch.empty(left, dtype=torch.float16),
                                     torch.empty(right, dtype=torch.float16))

    def test_dtype_device_and_layout_validation(self) -> None:
        """Reject unsupported precision, devices and non-dense matrix views."""
        a = torch.ones(7, 13, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "NPU"):
            symmetric_matmul(a, a.mT)
        with self.assertRaisesRegex(ValueError, "dtype"):
            symmetric_matmul(a.float(), a.float().mT)
        with self.assertRaisesRegex(ValueError, "dtype"):
            symmetric_matmul(a, a.mT.bfloat16())
        with self.assertRaisesRegex(ValueError, "same device"):
            symmetric_matmul(a, torch.empty(13, 7, dtype=a.dtype, device="meta"))
        with self.assertRaisesRegex(ValueError, "contiguous matrices"):
            symmetric_matmul(a[:, ::2], a[:, ::2].mT)
        with self.assertRaisesRegex(ValueError, "out_dtype"):
            symmetric_matmul(a, a.mT, out_dtype=torch.float64)

    def test_output_metadata_and_shared_storage_are_rejected(self) -> None:
        """Aliasing is rejected even when two views use disjoint offsets."""
        a = torch.ones(7, 7, dtype=torch.float16)
        for out in (torch.empty(7, 6, dtype=a.dtype), torch.empty(7, 7), torch.empty_like(a).mT):
            with self.subTest(shape=out.shape, dtype=out.dtype, stride=out.stride()):
                with self.assertRaisesRegex(ValueError, "out must be contiguous"):
                    symmetric_matmul(a, a, out=out)
        with self.assertRaisesRegex(ValueError, "share storage"):
            symmetric_matmul(a, a, out=a)
        backing = torch.empty(98, dtype=a.dtype)
        with self.assertRaisesRegex(ValueError, "share storage"):
            symmetric_matmul(backing[:49].view(7, 7), a, out=backing[49:].view(7, 7))

    def test_forward_only_allows_explicit_no_grad(self) -> None:
        """Do not silently detach a live autograd computation."""
        a = torch.ones(7, 7, dtype=torch.float16, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "forward-only"):
            symmetric_matmul(a, a)
        with torch.no_grad(), patch.object(_MODULE, "_require_npu"), patch.object(
            _MODULE, "_triton_backend", return_value=SimpleNamespace(launch_symmetric_matmul=_reference_launch),
        ):
            result = symmetric_matmul(a, a)
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)

    def test_index_range_is_checked_without_allocating_storage(self) -> None:
        """A square output can exceed indexing limits despite a small reduction."""
        a = torch.empty(65536, 1, dtype=torch.float16, device="meta")
        with self.assertRaisesRegex(ValueError, "indexing range"):
            symmetric_matmul(a, a.mT)

    def test_large_batch_can_reach_backend(self) -> None:
        """Total batch size can exceed int32 for both inputs and output."""
        value = torch.empty((129, 4096, 8192), dtype=torch.bfloat16, device="meta")
        transpose = value.mT
        launch = Mock()
        with patch.object(_MODULE, "_require_npu"), patch.object(
            _MODULE, "_triton_backend", return_value=SimpleNamespace(launch_symmetric_matmul=launch),
        ):
            result = symmetric_matmul(value, transpose)
        expected_shape = (129, 4096, 4096)
        self.assertEqual(result.shape, expected_shape, f"Expected shape {expected_shape}, got {result.shape}")
        launch.assert_called_once_with(value, transpose, result)

    def test_missing_backend_preserves_import_cause(self) -> None:
        """Explain missing dependencies without hiding installation failures."""
        for error in (ModuleNotFoundError("triton"), OSError("missing runtime")):
            with self.subTest(error=error), patch.object(_MODULE.importlib, "import_module", side_effect=error):
                with self.assertRaisesRegex(ImportError, "Triton-Ascend") as caught:
                    _MODULE._triton_backend()
                self.assertIs(caught.exception.__cause__, error)

    def test_execution_errors_propagate_without_fallback(self) -> None:
        """A broken kernel must never silently run a native full GEMM instead."""
        a = torch.ones(7, 7, dtype=torch.float16)
        launch = Mock(side_effect=RuntimeError("kernel failed"))
        with patch.object(_MODULE, "_require_npu"), patch.object(
            _MODULE, "_triton_backend", return_value=SimpleNamespace(launch_symmetric_matmul=launch),
        ):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                symmetric_matmul(a, a)
