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
"""Saver and loader ut test."""
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import mindspore as ms
from mindspore import Tensor
from hyper_parallel.core.checkpoint.loader import load_checkpoint
from hyper_parallel.core.checkpoint.saver import save_checkpoint


class TestSaverLoader(unittest.TestCase):
    """Ut test for Saver and Loader."""

    def setUp(self):
        """Set up test case."""
        self.weight = ms.Parameter(Tensor(np.ones([32, 2]), ms.float32), name='weight', requires_grad=True)
        self.mock_cell = MagicMock(_params={"weight": self.weight})

    def test_save_checkpoint_success_use_str_file_name(self):
        file_path = "test_save_checkpoint_success_use_str_file_name.safetensors"
        save_checkpoint(self.mock_cell, file_path)
        self.assertTrue(os.path.exists(file_path))
        os.remove(file_path)

    def test_save_checkpoint_success_use_str_file_path(self):
        file_path = "str/path/test_save_checkpoint_success_use_str_file_path.safetensors"
        save_checkpoint(self.mock_cell, file_path)
        self.assertTrue(os.path.exists(file_path))
        os.remove(file_path)

    def test_save_checkpoint_raise_value_error_use_str_file_path(self):
        file_path = "."
        with self.assertRaises(ValueError) as exception:
            save_checkpoint(self.mock_cell, file_path)
        self.assertIn("Saver file_path should contains file name", str(exception.exception))

    def test_save_checkpoint_success_use_path_file_name(self):
        file_path = Path("test_save_checkpoint_success_use_path_file_name.safetensors")
        save_checkpoint(self.mock_cell, file_path)
        self.assertTrue(os.path.exists(file_path))
        os.remove(file_path)

    def test_save_checkpoint_success_use_path_file_path(self):
        file_path = Path("path/path/test_save_checkpoint_success_use_path_file_path.safetensors")
        save_checkpoint(self.mock_cell, file_path)
        self.assertTrue(os.path.exists(file_path))
        os.remove(file_path)

    def test_save_checkpoint_raise_value_error_use_path_file_path(self):
        file_path = Path("path/path/")
        with self.assertRaises(ValueError) as exception:
            save_checkpoint(self.mock_cell, file_path)
        self.assertIn("Saver file_path should contains file name", str(exception.exception))

    def test_load_checkpoint_success_use_str_file_name(self):
        file_path = "test_load_checkpoint_success_use_str_file_name.safetensors"
        save_checkpoint(self.mock_cell, file_path)
        param_dict = load_checkpoint(file_path)
        assert isinstance(param_dict, dict)
        os.remove(file_path)

    def test_load_checkpoint_success_use_path_file_path(self):
        file_path = Path("path/path/test_save_checkpoint_success_use_path_file_path.safetensors")
        save_checkpoint(self.mock_cell, file_path)
        param_dict = load_checkpoint(file_path)
        assert isinstance(param_dict, dict)
        os.remove(file_path)

    def test_load_checkpoint_raise_value_error_use_path_file_path(self):
        file_path = Path("path/path/")
        with self.assertRaises(ValueError) as exception:
            load_checkpoint(file_path)
        self.assertIn("Loader file_path should contains file name", str(exception.exception))


if __name__ == "__main__":
    unittest.main()
