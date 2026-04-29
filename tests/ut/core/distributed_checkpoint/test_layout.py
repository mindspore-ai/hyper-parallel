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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.layout`."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.common.mark_utils import arg_mark

from hyper_parallel.core.distributed_checkpoint.layout import (
    get_current_layout,
    load_layout,
    save_layout,
)


def _layout_mock_with_to_dict(layout_dict: dict):
    """Return a mock layout object whose to_dict() returns ``layout_dict``."""
    mock_layout = MagicMock()
    mock_layout.to_dict.return_value = layout_dict
    return mock_layout


def _make_param(name, layout, dtype="float32", shape=(2, 4)):
    param = MagicMock()
    param.name = name
    param.layout = layout
    param.dtype = dtype
    param.shape = shape
    return param


class TestLayout(unittest.TestCase):
    """Tests for :mod:`hyper_parallel.core.distributed_checkpoint.layout`."""

    def setUp(self):
        # ``parameters_dict`` may map to ``named_parameters`` (Torch) or ``parameters_and_names`` (MS);
        # mocks only define ``parameters_and_names``, so delegate for stable ``get_current_layout`` tests.
        self._params_dict_patcher = patch(
            "hyper_parallel.core.distributed_checkpoint.layout.platform.parameters_dict",
            side_effect=lambda cell: cell.parameters_and_names(),
        )
        self._params_dict_patcher.start()

    def tearDown(self):
        self._params_dict_patcher.stop()

    @patch("hyper_parallel.core.distributed_checkpoint.layout.platform.get_rank", return_value=0)
    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_get_current_layout_mesh_shape_becomes_device_matrix(self, mock_rank):
        """
        Feature: Distributed checkpoint layout export from a cell.
        Description: Parameters with layout.to_dict containing mesh_shape are serialized.
        Expectation: Output uses device_matrix instead of mesh_shape; source dicts are unchanged.
        """
        weight_src = {"mesh_shape": (2, 4), "tensor_map": (0, -1)}
        bias_src = {"mesh_shape": (2, 4), "tensor_map": (-1, 0)}
        weight_param = _make_param("weight", _layout_mock_with_to_dict(weight_src))
        bias_param = _make_param("bias", _layout_mock_with_to_dict(bias_src), shape=(4,))

        mock_cell = MagicMock()
        mock_cell.parameters_and_names.return_value = [
            ("weight", weight_param),
            ("bias", bias_param),
        ]

        layout_dict = get_current_layout(mock_cell)

        self.assertEqual(set(layout_dict.keys()), {"0"})
        rank_layout = layout_dict["0"]
        self.assertEqual(set(rank_layout.keys()), {"weight", "bias"})

        w = rank_layout["weight"]
        self.assertEqual(w["device_matrix"], (2, 4))
        self.assertEqual(w["tensor_map"], (0, -1))
        self.assertNotIn("mesh_shape", w)
        self.assertEqual(w["type"], "float32")
        self.assertEqual(w["full_shape"], (2, 4))

        b = rank_layout["bias"]
        self.assertEqual(b["device_matrix"], (2, 4))
        self.assertNotIn("mesh_shape", b)

        weight_param.layout.to_dict.assert_called()
        # Source dict from to_dict() must not be mutated (get_current_layout copies first).
        self.assertIn("mesh_shape", weight_src)
        self.assertIn("mesh_shape", bias_src)
        mock_rank.assert_called_once()

    @patch("hyper_parallel.core.distributed_checkpoint.layout.platform.get_rank", return_value=3)
    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_get_current_layout_uses_string_rank_as_key(self, mock_rank):
        """
        Feature: Rank-scoped layout dictionary keys.
        Description: get_rank is patched to return 3 while building the layout map.
        Expectation: The outer dict key is the string rank id and parameter entries are nested under it.
        """
        mock_cell = MagicMock()
        mock_cell.parameters_and_names.return_value = [
            ("w", _make_param("w", _layout_mock_with_to_dict({"mesh_shape": (1,)}))),
        ]
        layout_dict = get_current_layout(mock_cell)
        self.assertIn("3", layout_dict)
        self.assertIn("w", layout_dict["3"])
        mock_rank.assert_called_once()

    @patch("hyper_parallel.core.distributed_checkpoint.layout.platform.get_rank", return_value=0)
    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_get_current_layout_falsy_layout_records_type_and_full_shape(self, mock_rank):
        """
        Feature: Parameters with no usable layout still record dtype and global shape.
        Description: One parameter has a truthy layout and another has layout set to None.
        Expectation: Both names appear under the rank; the falsy-layout parameter maps to a dict
            with only type and full_shape (no shard metadata).
        """
        mock_cell = MagicMock()
        mock_cell.parameters_and_names.return_value = [
            ("weight", _make_param("weight", _layout_mock_with_to_dict({"mesh_shape": (2,)}))),
            ("ignored", _make_param("ignored", None)),
        ]
        layout_dict = get_current_layout(mock_cell)
        rank_layout = layout_dict["0"]
        self.assertEqual(set(rank_layout.keys()), {"weight", "ignored"})
        self.assertIsNotNone(rank_layout["weight"])
        self.assertEqual(
            rank_layout["ignored"],
            {"type": "float32", "full_shape": (2, 4)},
        )
        mock_rank.assert_called_once()

    @patch("hyper_parallel.core.distributed_checkpoint.layout.logger")
    @patch("hyper_parallel.core.distributed_checkpoint.layout.platform.get_rank", return_value=0)
    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_get_current_layout_logs_params_without_layout_attr(self, mock_rank, mock_logger):
        """
        Feature: Observability for parameters missing a layout attribute.
        Description: A parameter object without a layout attribute is included alongside normal params.
        Expectation: logger.info runs once with the parameter names; rank layout stores type/full_shape
            for that param (no layout shard keys).
        """
        no_layout_attr = SimpleNamespace(name="buf", dtype="float32", shape=())
        mock_cell = MagicMock()
        mock_cell.parameters_and_names.return_value = [
            ("weight", _make_param("weight", _layout_mock_with_to_dict({"mesh_shape": (2,)}))),
            ("buf", no_layout_attr),
        ]
        layout_dict = get_current_layout(mock_cell)
        self.assertEqual(layout_dict["0"]["buf"], {"type": "float32", "full_shape": ()})
        self.assertIsNotNone(layout_dict["0"]["weight"])
        mock_logger.info.assert_called_once()
        msg, names = mock_logger.info.call_args[0]
        self.assertIn("layout attribute", msg)
        self.assertEqual(names, ["buf"])
        mock_rank.assert_called_once()

    @patch("hyper_parallel.core.distributed_checkpoint.layout.logger")
    @patch("hyper_parallel.core.distributed_checkpoint.layout.platform.get_rank", return_value=0)
    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_get_current_layout_only_missing_layout_attr_type_and_full_shape(self, mock_rank, mock_logger):
        """
        Feature: Rank layout map when every parameter lacks a layout attribute.
        Description: Two SimpleNamespace params without ``layout`` are enumerated from the cell.
        Expectation: Each param name maps to type/full_shape only; logger.info lists both names once.
        """
        p1 = SimpleNamespace(name="a", dtype="float32", shape=(1,))
        p2 = SimpleNamespace(name="b", dtype="float32", shape=(2,))
        mock_cell = MagicMock()
        mock_cell.parameters_and_names.return_value = [("a", p1), ("b", p2)]
        layout_dict = get_current_layout(mock_cell)["0"]
        self.assertEqual(
            layout_dict,
            {
                "a": {"type": "float32", "full_shape": (1,)},
                "b": {"type": "float32", "full_shape": (2,)},
            },
        )
        mock_logger.info.assert_called_once()
        names = mock_logger.info.call_args[0][1]
        self.assertEqual(set(names), {"a", "b"})
        mock_rank.assert_called_once()

    @patch("hyper_parallel.core.distributed_checkpoint.layout.platform.get_rank", return_value=0)
    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_get_current_layout_no_rename_without_mesh_shape(self, mock_rank):
        """
        Feature: Layout export without mesh_shape in to_dict.
        Description: layout.to_dict omits mesh_shape but includes other keys.
        Expectation: device_matrix is not injected and tensor_map is preserved.
        """
        info = {"tensor_map": (0,), "alias_name": ("dp",)}
        mock_cell = MagicMock()
        mock_cell.parameters_and_names.return_value = [
            ("w", _make_param("w", _layout_mock_with_to_dict(info))),
        ]
        out = get_current_layout(mock_cell)["0"]["w"]
        self.assertNotIn("device_matrix", out)
        self.assertNotIn("mesh_shape", out)
        self.assertEqual(out["tensor_map"], (0,))
        mock_rank.assert_called_once()

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_save_layout_success_with_string_path(self):
        """
        Feature: save_layout with a string filesystem path.
        Description: Write a nested dict to a temporary JSON file using a str path.
        Expectation: File exists and json.load matches the original dict.
        """
        layout_dict = {
            "weight": {"shard": [1, 2], "device_mesh": [2, 2]},
            "bias": {"shard": [2, 1], "device_mesh": [2, 2]},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_layout.json")
            save_layout(layout_dict, file_path)
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
            self.assertEqual(loaded_data, layout_dict)

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_save_layout_success_with_path_object(self):
        """
        Feature: save_layout with pathlib.Path.
        Description: Write using a Path instance under a temporary directory.
        Expectation: File exists on disk and round-trips to the same JSON structure.
        """
        layout_dict = {"weight": {"shard": [1, 2]}}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_layout.json"
            save_layout(layout_dict, file_path)
            self.assertTrue(file_path.exists())
            with open(file_path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), layout_dict)

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_load_layout_success(self):
        """
        Feature: load_layout from a string path.
        Description: Persist a dict to a temp file then load_layout with its path.
        Expectation: Returned dict equals the serialized content.
        """
        expected_layout = {
            "weight": {"shard": [1, 2]},
            "bias": {"shard": [2, 1]},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(expected_layout, f, ensure_ascii=False)
            temp_path = f.name
        try:
            loaded_layout = load_layout(temp_path)
            self.assertEqual(loaded_layout, expected_layout)
        finally:
            os.unlink(temp_path)

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_load_layout_success_with_path_object(self):
        """
        Feature: load_layout from pathlib.Path.
        Description: Write JSON via tempfile then pass Path to load_layout.
        Expectation: Loaded dict matches what was written.
        """
        expected_layout = {"weight": {"shard": [1, 2]}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(expected_layout, f, ensure_ascii=False)
            temp_path = Path(f.name)
        try:
            self.assertEqual(load_layout(temp_path), expected_layout)
        finally:
            os.unlink(str(temp_path))

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_load_layout_missing_file_raises(self):
        """
        Feature: load_layout error handling for missing files.
        Description: Call load_layout with a path that does not exist on disk.
        Expectation: FileNotFoundError is raised and mentions the path.
        """
        missing = Path(tempfile.gettempdir()) / "nonexistent_layout_xyz123.layout"
        with self.assertRaises(FileNotFoundError) as ctx:
            load_layout(missing)
        self.assertIn(str(missing), str(ctx.exception))

    @arg_mark(
        plat_marks=["platform_ascend910b"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="unessential",
    )
    def test_save_load_layout_roundtrip_preserves_null_entries(self):
        """
        Feature: JSON persistence of rank layout maps containing null parameter entries.
        Description: save_layout writes a dict whose per-rank values include explicit JSON nulls.
        Expectation: load_layout returns the same structure with Python None for those entries.
        """
        layout_dict = {
            "0": {
                "weight": {"device_matrix": [2, 2], "type": "float32", "full_shape": [4, 8]},
                "bias": None,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.layout.json"
            save_layout(layout_dict, path)
            loaded = load_layout(path)
        self.assertEqual(loaded, layout_dict)
        self.assertIsNone(loaded["0"]["bias"])


if __name__ == "__main__":
    unittest.main()
