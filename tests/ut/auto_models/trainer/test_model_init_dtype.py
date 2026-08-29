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
"""Unit tests for Trainer-owned model initialization dtype conversion."""
# pylint: disable=wrong-import-position

import os
import unittest
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch  # pylint: disable=wrong-import-position
from torch import nn  # pylint: disable=wrong-import-position

from hyper_parallel import DeviceMesh, DTensor, Replicate
from hyper_parallel.auto_models.trainer.model_init_dtype import apply_model_init_dtype
from tests.common.mark_utils import arg_mark


class _DtypeModel(nn.Module):  # pylint: disable=abstract-method
    """Small model with floating and non-floating persistent tensors."""

    def __init__(self, dtype: torch.dtype) -> None:
        """Register floating, integer, and boolean tensors."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, dtype=dtype))
        self.register_buffer("scale", torch.ones(2, dtype=dtype))
        self.register_buffer("indices", torch.tensor([1, 2], dtype=torch.int64))
        self.register_buffer("mask", torch.tensor([True, False]))


class TestModelInitDtype(unittest.TestCase):
    """Verify conversion direction, disabled behavior, and tensor identity."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_float32_to_bfloat16_preserves_identity_and_integer_buffers(self):
        """Convert a float32 model to bfloat16 after model finalization.

        Feature: Trainer-owned model initialization dtype.
        Description: Convert floating state while retaining non-floating buffers.
        Expectation: Dtypes change as configured and tensor identities remain stable.
        """
        model = _DtypeModel(torch.float32)
        identities = {
            "weight": id(model.weight),
            "indices": id(model.indices),
            "mask": id(model.mask),
        }

        apply_model_init_dtype(model, "bfloat16")

        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(model.scale.dtype, torch.bfloat16)
        self.assertEqual(model.indices.dtype, torch.int64)
        self.assertEqual(model.mask.dtype, torch.bool)
        self.assertEqual(id(model.weight), identities["weight"])
        self.assertEqual(id(model.indices), identities["indices"])
        self.assertEqual(id(model.mask), identities["mask"])

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_bfloat16_to_float32_and_none(self):
        """Promote bfloat16 state and preserve disabled behavior.

        Feature: Trainer-owned model initialization dtype.
        Description: Apply null conversion followed by float32 promotion.
        Expectation: Null is unchanged and float32 promotion succeeds.
        """
        model = _DtypeModel(torch.bfloat16)
        apply_model_init_dtype(model, None)
        self.assertEqual(model.weight.dtype, torch.bfloat16)

        apply_model_init_dtype(model, "float32")
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertEqual(model.scale.dtype, torch.float32)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_meta_parameter_conversion_to_float32_preserves_identity(self):
        """Convert a low-precision meta-initialized model to float32.

        Feature: Trainer-owned model initialization dtype.
        Description: Apply float32 dtype to a bfloat16 meta parameter.
        Expectation: The dtype promotes without replacing the Parameter object.
        """
        model = nn.Linear(
            2,
            2,
            bias=False,
            device="meta",
            dtype=torch.bfloat16,
        )
        parameter_id = id(model.weight)

        apply_model_init_dtype(model, "float32")

        self.assertEqual(id(model.weight), parameter_id)
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertTrue(model.weight.is_meta)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_invalid_dtype_fails(self):
        """Report an invalid final model initialization dtype.

        Feature: Model initialization dtype validation.
        Description: Request an unsupported dtype.
        Expectation: The invalid configuration produces an explicit error.
        """
        model = _DtypeModel(torch.float32)
        with self.assertRaisesRegex(ValueError, "model_init_dtype"):
            apply_model_init_dtype(model, "float64")  # type: ignore[arg-type]

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    @patch("hyper_parallel.core.dtensor.device_mesh.platform.get_rank", return_value=0)
    def test_dtensor_parameter_keeps_identity_and_layout(self, mock_get_rank):
        """Preserve distributed parameter identity and layout.

        Feature: Model initialization dtype conversion for distributed tensors.
        Description: Convert a distributed parameter from float32 to bfloat16.
        Expectation: The parameter, mesh, and placements remain identical.
        """
        del mock_get_rank
        mesh = DeviceMesh(
            "cpu",
            [0],
            mesh_dim_names=("dp",),
            _init_backend=False,
        )
        parameter = nn.Parameter(
            DTensor.from_local(torch.ones(2), mesh, (Replicate(),))
        )
        model = nn.Module()
        model.register_parameter("weight", parameter)
        parameter_id = id(parameter)
        placements = tuple(parameter.placements)

        apply_model_init_dtype(model, "bfloat16")

        self.assertEqual(id(model.weight), parameter_id)
        self.assertIs(model.weight.device_mesh, mesh)
        self.assertEqual(tuple(model.weight.placements), placements)
        self.assertEqual(model.weight.to_local().dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
