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
"""Unit tests for MindSpore activation checkpoint wrappers."""

import os
import unittest

import numpy as np
import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_default,
)

ensure_mindspore_platform_default()

import mindspore as ms  # noqa: E402
from mindspore import nn  # noqa: E402

from hyper_parallel.platform.mindspore.activation_checkpoint import checkpoint_wrapper  # noqa: E402


class TestActivationWrapper(unittest.TestCase):
    """Cover wrapper name transparency for MindSpore cell traversal."""

    def test_cells_and_names_strips_wrapped_module_prefix(self):
        """Cell traversal should expose the same names as the unwrapped model."""

        class _Sub(nn.Cell):
            def __init__(self):
                super().__init__()
                self.weight = ms.Parameter(ms.Tensor(np.ones((2, 2), np.float32)))

            def construct(self, x):
                return x

        class _Block(nn.Cell):
            def __init__(self):
                super().__init__()
                self.weight = ms.Parameter(ms.Tensor(np.ones((2, 2), np.float32)))
                self.sub = _Sub()

            def construct(self, x):
                return self.sub(x)

        class _Root(nn.Cell):
            def __init__(self):
                super().__init__()
                self.block = checkpoint_wrapper(_Block())

            def construct(self, x):
                return self.block(x)

        model = _Root()

        cell_names = [name for name, _ in model.cells_and_names()]
        param_names = [name for name, _ in model.parameters_and_names()]

        self.assertFalse(any("_ckpt_wrapped_module" in name for name in cell_names))
        self.assertNotIn("block._ckpt_wrapped_module", cell_names)
        self.assertNotIn("block._ckpt_wrapped_module.sub", cell_names)
        self.assertIn("block", cell_names)
        self.assertIn("block.sub", cell_names)
        self.assertEqual(param_names, ["block.weight", "block.sub.weight"])


if __name__ == "__main__":
    unittest.main()
