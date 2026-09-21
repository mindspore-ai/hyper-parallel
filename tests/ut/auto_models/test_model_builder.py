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
"""Unit tests for atomic auto-model construction helpers."""
# pylint: disable=wrong-import-position

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
from torch import nn

from hyper_parallel import DTensor, DeviceMesh, Replicate
from hyper_parallel.models._transformers.model_builder import (
    validate_model_init_dtype,
)
from tests.common.mark_utils import arg_mark


class TestValidateModelInitDtype(unittest.TestCase):
    """Tests for non-mutating model dtype validation."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    @patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0)
    def test_matching_dtype_preserves_dtensor_runtime_metadata(self, mock_get_rank) -> None:
        """
        Feature: model builder
        Description: Validation must retain optimizer metadata on a matching DTensor.
        Expectation: Matching dtype preserves dtensor runtime metadata.
        """
        del mock_get_rank
        mesh = DeviceMesh(
            "cpu",
            [0],
            mesh_dim_names=("dp",),
            _init_backend=False,
        )
        parameter = nn.Parameter(
            DTensor.from_local(
                torch.ones(2, dtype=torch.float32),
                mesh,
                (Replicate(),),
            )
        )
        parameter.model_name = "weight"
        parameter.main_param = parameter
        model = nn.Module()
        model.register_parameter("weight", parameter)

        validate_model_init_dtype(model, "float32")

        self.assertIs(model.weight, parameter)
        self.assertEqual(model.weight.model_name, "weight")
        self.assertIs(model.weight.main_param, parameter)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_mismatched_dtype_fails_without_converting_parameter(self) -> None:
        """
        Feature: model builder
        Description: Validation must reject rather than convert a mismatched parameter.
        Expectation: Mismatched dtype fails without converting parameter.
        """
        parameter = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        model = nn.Module()
        model.register_parameter("weight", parameter)

        with self.assertRaisesRegex(
                RuntimeError,
                "Model initialization dtype validation failed for: weight",
        ):
            validate_model_init_dtype(model, "float32")

        self.assertIs(model.weight, parameter)
        self.assertEqual(model.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
