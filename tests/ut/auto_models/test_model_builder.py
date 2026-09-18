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
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
from torch import nn

from hyper_parallel import DTensor, DeviceMesh, Replicate
from hyper_parallel.models._transformers.model_builder import (
    validate_model_init_dtype,
)
from hyper_parallel.models._transformers.auto_model import _BaseHyperAutoModelClass
from tests.common.mark_utils import arg_mark


class TestValidateModelInitDtype(unittest.TestCase):
    """Tests for non-mutating model dtype validation."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    @patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0)
    def test_matching_dtype_preserves_dtensor_runtime_metadata(self, mock_get_rank) -> None:
        """Validation must retain optimizer metadata on a matching DTensor."""
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
        """Validation must reject rather than convert a mismatched parameter."""
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


class TestAutoModelBuildForwarding(unittest.TestCase):
    """Tests for options forwarded through the private build orchestrator."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_activation_checkpoint_selection_reaches_infrastructure(self) -> None:
        """Forward adapter-safe layer selection to activation checkpointing.

        Feature: Adapter-safe activation-checkpoint selection.
        Description: Build a model with an explicit layer selector.
        Expectation: Infrastructure receives the identical selection object.
        """
        model = nn.Linear(2, 2)
        selection = SimpleNamespace(
            source="model_adapter_safe_regions",
            layer_count=2,
            layer_indices=None,
        )
        module_path = "hyper_parallel.models._transformers.auto_model"
        with (
            patch(f"{module_path}._init_model", return_value=(None, model)),
            patch(
                f"{module_path}.apply_model_infrastructure",
                return_value=model,
            ) as apply_infrastructure,
            patch(f"{module_path}.torch.distributed.is_initialized", return_value=False),
            patch(f"{module_path}._current_device", return_value=torch.device("cpu")),
        ):
            result = _BaseHyperAutoModelClass._build_model(  # pylint: disable=protected-access
                None,
                is_hf_model=True,
                hf_config=SimpleNamespace(),
                mesh=None,
                sharding_planner=None,
                fsdp2_manager=None,
                backend=None,
                peft_config=None,
                torch_dtype=torch.float32,
                attn_implementation="eager",
                validate_placement=False,
                load_base_model=False,
                activation_checkpoint="selective",
                activation_checkpoint_selection=selection,
            )

        self.assertIs(result, model)
        self.assertIs(
            apply_infrastructure.call_args.kwargs["activation_checkpoint_selection"],
            selection,
        )


if __name__ == "__main__":
    unittest.main()
