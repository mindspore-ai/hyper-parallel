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
"""Unit tests for custom_ops module loading logic."""
import os
import sys
import importlib
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

_MODULE_NAME = "hyper_parallel_custom_ops_ms"
_PKG = "hyper_parallel.platform.mindspore.custom_ops.custom_op_impl"


def _force_reimport():
    """Remove cached module to allow clean re-import in each test."""
    for key in list(sys.modules):
        if key == _PKG or key.startswith(_PKG + "."):
            del sys.modules[key]
    sys.modules.pop(_MODULE_NAME, None)


class TestCustomOpsLoading(unittest.TestCase):
    """Tests for custom_ops module loading logic."""

    def setUp(self):
        _force_reimport()
        self._mock_ms = MagicMock()
        self._patcher = patch.dict(sys.modules, {"mindspore": self._mock_ms})
        self._patcher.start()
        super().setUp()

    def tearDown(self):
        self._patcher.stop()
        _force_reimport()
        super().tearDown()

    def test_imports_prebuilt_so_when_available(self):
        """Pre-built .so module is imported when it exists in sys.modules."""
        mock_mod = MagicMock()
        with patch.dict(sys.modules, {_MODULE_NAME: mock_mod}):
            from hyper_parallel.platform.mindspore.custom_ops import custom_op_impl

        self.assertIs(custom_op_impl._custom_ops, mock_mod)

    def test_build_lib_added_to_sys_path(self):
        """build/lib is added to sys.path before attempting import."""
        from hyper_parallel.platform.mindspore.custom_ops import custom_op_impl

        expected_suffix = os.path.join("custom_ops", "build", "lib")
        build_lib_paths = [p for p in sys.path if expected_suffix in p]
        self.assertTrue(build_lib_paths, f"build/lib not found in sys.path: {sys.path[:5]}")

    def test_kda_backward_token_major_boundary(self):
        """Transpose gate fields at the real bridge boundary and restore gradients."""
        class Tensor(np.ndarray):
            """Array supporting the bridge's contiguous tensor operation."""

            def contiguous(self):
                """Keep an already materialized test array."""
                return self

        extension = MagicMock()
        with patch.dict(sys.modules, {_MODULE_NAME: extension}):
            module = importlib.import_module(_PKG)
        self._mock_ms.ops.zeros_like.side_effect = np.zeros_like
        self._mock_ms.ops.transpose.side_effect = lambda value, axes: value.transpose(axes)
        for layout, shape in (("BNSD", (2, 3, 5, 128)), ("NTD", (3, 5, 128))):
            with self.subTest(layout=layout):
                q = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).view(Tensor)
                beta = q[..., 0]
                bias = np.zeros((3, 128), dtype=np.float32).view(Tensor)
                token_axes = (0, 2, 1, 3) if layout == "BNSD" else (1, 0, 2)
                beta_axes = (0, 2, 1) if layout == "BNSD" else (1, 0)

                def backward(*args, query=q, updates=beta, gate_bias=bias,
                             perm=token_axes, scalar_perm=beta_axes, output_layout=layout):
                    np.testing.assert_array_equal(args[3], updates.transpose(scalar_perm))
                    np.testing.assert_array_equal(args[12], query.transpose(perm))
                    np.testing.assert_array_equal(args[13], query.transpose(perm))
                    self.assertIs(args[4], query)
                    self.assertEqual(args[-1], output_layout)
                    return query, query, query, args[3] + 4, args[13] + 5, np.zeros(3), gate_bias

                extension.npu_chunk_kda_bwd.side_effect = backward
                result = module._run_chunk_kda_bwd(
                    q, q, q, beta, q, q, q, q, q, q, q, q, q, .1, 64,
                    layout=layout, raw_g=q, a_log=np.zeros(3), dt_bias=bias,
                    safe_gate=True, use_gate_in_kernel=True)
                np.testing.assert_array_equal(result[3], beta + 4)
                np.testing.assert_array_equal(result[4], q + 5)

    def test_kda_backward_rejects_unfused_gate(self):
        """Unsupported training contracts fail before invoking the extension."""
        extension = MagicMock()
        with patch.dict(sys.modules, {_MODULE_NAME: extension}):
            module = importlib.import_module(_PKG)
        with self.assertRaisesRegex(ValueError, "fused safe gate"):
            module._run_chunk_kda_bwd(*(None,) * 13, .1, 64)
        extension.npu_chunk_kda_bwd.assert_not_called()

    def test_kda_forward_flattens_bias_only_for_dispatch(self):
        """Keep the caller's head-shaped bias while adapting CANN's forward ABI."""
        extension = MagicMock()
        with patch.dict(sys.modules, {_MODULE_NAME: extension}):
            module = importlib.import_module(_PKG)
        q = np.zeros((2, 3, 5, 128))
        bias = np.zeros((3, 128))
        module._run_chunk_kda_fwd(q, q, q, q, q[..., 0], .1, 64, "BNSD",
                                  True, -5., True, np.zeros(3), bias, None, None)
        self.assertEqual(extension.npu_chunk_kda_fwd.call_args.args[13].shape, (384,))
        self.assertEqual(bias.shape, (3, 128))


if __name__ == "__main__":
    unittest.main()
