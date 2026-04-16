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
# pylint: disable=wrong-import-position
import importlib
import os
import shutil
import unittest
from pathlib import Path

import pytest

pytest.importorskip("mindspore")

import numpy as np
import mindspore as ms
from mindspore import Tensor

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
import hyper_parallel.platform.platform as _platform_mod

_platform_mod.platform = None

import hyper_parallel.core.distributed_checkpoint.loader as loader_mod
import hyper_parallel.core.distributed_checkpoint.saver as saver_mod

importlib.reload(saver_mod)
importlib.reload(loader_mod)


def setUpModule() -> None:  # pylint: disable=invalid-name  # unittest module hook (name is fixed by API)
    """Ensure MindSpore platform backs saver/loader after other tests may have cached torch."""
    os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
    _platform_mod.platform = None
    importlib.reload(saver_mod)
    importlib.reload(loader_mod)


def tearDownModule() -> None:  # pylint: disable=invalid-name  # unittest module hook (name is fixed by API)
    """Remove dirs and files created by path-based save tests."""
    for name in ("str", "path"):
        p = Path(name)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    for p in Path(".").glob("test_*checkpoint*.safetensors"):
        p.unlink(missing_ok=True)


def _unlink_file(path: Path) -> None:
    if path.is_file():
        path.unlink(missing_ok=True)


class TestSaverLoader(unittest.TestCase):
    """Ut test for Saver and Loader."""

    def setUp(self) -> None:
        """Set up test case."""
        self.weight = ms.Parameter(
            Tensor(np.ones([32, 2]), ms.float32), name="weight", requires_grad=True
        )
        self.save_obj = {"weight": self.weight}

    def test_save_checkpoint_success_use_str_file_name(self):
        file_path = "test_save_checkpoint_success_use_str_file_name.safetensors"
        saver_mod.save_checkpoint(self.save_obj, file_path)
        self.assertTrue(os.path.exists(file_path))
        _unlink_file(Path(file_path))

    def test_save_checkpoint_success_use_str_file_path(self):
        file_path = "str/path/test_save_checkpoint_success_use_str_file_path.safetensors"
        saver_mod.save_checkpoint(self.save_obj, file_path)
        self.assertTrue(os.path.exists(file_path))
        _unlink_file(Path(file_path))

    def test_save_checkpoint_raise_value_error_use_str_file_path(self):
        file_path = "."
        with self.assertRaises(ValueError) as exception:
            saver_mod.save_checkpoint(self.save_obj, file_path)
        self.assertIn("Saver file_path should contains valid filename", str(exception.exception))

    def test_save_checkpoint_success_use_path_file_name(self):
        file_path = Path("test_save_checkpoint_success_use_path_file_name.safetensors")
        saver_mod.save_checkpoint(self.save_obj, file_path)
        self.assertTrue(file_path.exists())
        _unlink_file(file_path)

    def test_save_checkpoint_success_use_path_file_path(self):
        file_path = Path("path/path/test_save_checkpoint_success_use_path_file_path.safetensors")
        saver_mod.save_checkpoint(self.save_obj, file_path)
        self.assertTrue(file_path.exists())
        _unlink_file(file_path)

    def test_save_checkpoint_raise_value_error_use_path_file_path(self):
        file_path = Path("path/path/")
        with self.assertRaises(ValueError) as exception:
            saver_mod.save_checkpoint(self.save_obj, file_path)
        self.assertIn("Saver file_path should contains valid filename", str(exception.exception))

    def test_load_checkpoint_success_use_str_file_name(self):
        file_path = "test_load_checkpoint_success_use_str_file_name.safetensors"
        saver_mod.save_checkpoint(self.save_obj, file_path)
        param_dict = loader_mod.load_checkpoint(file_path)
        self.assertIsInstance(param_dict, dict)
        _unlink_file(Path(file_path))

    def test_load_checkpoint_success_use_path_file_path(self):
        file_path = Path("path/path/test_load_checkpoint_success_use_path_file_path.safetensors")
        saver_mod.save_checkpoint(self.save_obj, file_path)
        param_dict = loader_mod.load_checkpoint(file_path)
        self.assertIsInstance(param_dict, dict)
        _unlink_file(file_path)

    def test_load_checkpoint_raise_value_error_use_path_file_path(self):
        file_path = Path("path/path/")
        with self.assertRaises(ValueError) as exception:
            loader_mod.load_checkpoint(file_path)
        self.assertIn("Loader file_path should contains valid filename", str(exception.exception))


if __name__ == "__main__":
    unittest.main()
