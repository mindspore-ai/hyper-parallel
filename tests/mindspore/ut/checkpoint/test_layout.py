# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test the layout module."""
import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock
from pathlib import Path

from hyper_parallel import Layout
from hyper_parallel.core.checkpoint.layout import (
    get_current_layout,
    save_layout,
    load_layout
)


def create_mock_layout_with_side_effect(layout_dict):
    """
    Create a mock layout with side_effect.
    """
    mock_layout = MagicMock(spec=Layout)

    def specific_side_effect():
        """
        Mock side effect.
        """
        return layout_dict

    mock_to_dict = MagicMock()
    mock_to_dict.side_effect = specific_side_effect

    mock_layout.to_dict = mock_to_dict

    return mock_layout


class TestLayout(unittest.TestCase):
    """Test the layout module."""

    def test_get_current_layout_success(self):
        """Test get_current_layout with valid parameters."""
        weight_layout_dict = create_mock_layout_with_side_effect({"mesh_shape": "mock_weight_mesh_shape"})
        bias_layout_dict = create_mock_layout_with_side_effect({"mesh_shape": "mock_bias_mesh_shape"})

        weight_param = MagicMock()
        weight_param.name = "weight"
        weight_param.layout = weight_layout_dict
        bias_param = MagicMock()
        bias_param.name = "bias"
        bias_param.layout = bias_layout_dict
        mock_cell = MagicMock()
        mock_cell.named_parameter.return_value = {
            "weight": weight_param,
            "bias": bias_param,
            "no_layout": MagicMock(layout=None)
        }
        layout_dict = get_current_layout(mock_cell)

        # Verify results
        self.assertEqual(len(layout_dict), 2)
        self.assertIn("weight", layout_dict)
        self.assertIn("bias", layout_dict)
        self.assertNotIn("no_layout", layout_dict)
        self.assertEqual(layout_dict["weight"], {"mesh_shape": "mock_weight_mesh_shape"})
        self.assertEqual(layout_dict["bias"], {"mesh_shape": "mock_bias_mesh_shape"})

    def test_save_layout_success_with_string_path(self):
        """Test save_layout successfully saves layout to file with string path."""
        layout_dict = {
            "weight": {"shard": [1, 2], "device_mesh": [2, 2]},
            "bias": {"shard": [2, 1], "device_mesh": [2, 2]}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_layout.json")

            # Call function
            save_layout(layout_dict, file_path)

            # Verify file exists
            self.assertTrue(os.path.exists(file_path))

            # Verify file content
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            self.assertEqual(loaded_data, layout_dict)

    def test_save_layout_success_with_path_object(self):
        """Test save_layout successfully saves layout with Path object."""
        layout_dict = {"weight": {"shard": [1, 2]}}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_layout.json"

            # Call function
            save_layout(layout_dict, file_path)

            # Verify file exists
            self.assertTrue(file_path.exists())

    def test_load_layout_success(self):
        """Test load_layout successfully loads layout from file."""
        expected_layout = {
            "weight": {"shard": [1, 2]},
            "bias": {"shard": [2, 1]}
        }

        # Create temporary file with layout data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(expected_layout, f, ensure_ascii=False)
            temp_path = f.name

        try:
            # Call function
            loaded_layout = load_layout(temp_path)

            # Verify results
            self.assertEqual(loaded_layout, expected_layout)
        finally:
            os.unlink(temp_path)

    def test_load_layout_success_with_path_object(self):
        """Test load_layout successfully loads layout with Path object."""
        expected_layout = {"weight": {"shard": [1, 2]}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(expected_layout, f, ensure_ascii=False)
            temp_path = Path(f.name)

        try:
            # Call function
            loaded_layout = load_layout(temp_path)

            # Verify results
            self.assertEqual(loaded_layout, expected_layout)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
