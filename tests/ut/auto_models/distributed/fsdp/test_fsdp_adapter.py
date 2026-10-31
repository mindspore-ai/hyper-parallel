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
"""Trainer YAML to weakly keyed FSDP precision policies, without distributed setup."""

import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.weak import WeakIdKeyDictionary

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, clear_device_mesh_cache
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models._transformers.model_builder import _refresh_hsdp_precision_state
from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.common.mark_utils import arg_mark


class _DecoderModel(nn.Module):
    """Two discoverable decoder units plus root-owned parameters."""

    def __init__(self) -> None:
        """Build units discoverable by the manager's transformer-block policy."""
        super().__init__()
        self.gradient_checkpointing = False
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.head = nn.Linear(4, 4)


class TestFSDPCustomPrecisionConfig(unittest.TestCase):
    """Test config validation and the actual nested wrapping path."""

    def setUp(self) -> None:
        """Isolate mesh metadata between tests."""
        clear_device_mesh_cache()
        self.addCleanup(clear_device_mesh_cache)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_trainer_yaml_preserves_custom_params(self) -> None:
        """Feature: Trainer custom precision YAML.
        Description: Parse fsdp_config.mix_precision.custom_params and build the core policy.
        Expectation: Serialization preserves the subtree and names resolve to weak Parameter keys and Torch dtypes.
        """
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "train.yaml"
            config_path.write_text(
                """model:
  _target_: torch.nn.Linear
  in_features: 4
  out_features: 4
optimizer:
  _target_: torch.optim.SGD
  lr: 0.01
fsdp_config:
  mix_precision:
    param_dtype: bfloat16
    reduce_dtype: float32
    custom_params:
      weight:
        param_dtype: float32
        reduce_dtype: null
      bias:
        reduce_dtype: float16
""",
                encoding="utf-8",
            )

            config = parse_training_args([str(config_path)])

        self.assertEqual(config.fsdp_config.mix_precision.custom_params, {
            "weight": {"param_dtype": "float32", "reduce_dtype": None},
            "bias": {"reduce_dtype": "float16"},
        })
        self.assertEqual(config.to_dict()["fsdp_config"]["mix_precision"]["custom_params"],
                         config.fsdp_config.mix_precision.custom_params)
        model = nn.Linear(4, 4)
        manager = FSDP2Manager(config.fsdp_config, MeshContext())
        policy = manager._build_mixed_precision_policy(model)  # pylint: disable=protected-access
        self.assertEqual(policy.custom_params[model.weight], {"param_dtype": torch.float32, "reduce_dtype": None})
        self.assertEqual(policy.custom_params[model.bias], {"reduce_dtype": torch.float16})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_nested_units_capture_overrides_before_parameter_replacement(self) -> None:
        """Feature: Nested FSDP precision policy.
        Description: Wrap two child units and the root, then refresh precision through the model builder.
        Expectation: Units share one weak policy, release originals, and retain overrides through dtype refresh.
        """
        model = _DecoderModel()
        parameter_refs = [weakref.ref(model.layers[0].weight), weakref.ref(model.head.weight)]
        config = FSDP2Config(mix_precision=FSDP2MixedPrecisionConfig(
            param_dtype="bfloat16",
            reduce_dtype="bfloat16",
            custom_params={
                "layers.0.weight": {"param_dtype": "float32", "reduce_dtype": "float32"},
                "head.weight": {"param_dtype": "float32", "reduce_dtype": "float32"},
            },
        ))
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0), \
                patch.object(DeviceMesh, "get_group", return_value=None), \
                patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu")):
            mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("fsdp_shard",), _init_backend=False)
            manager = FSDP2Manager(config, MeshContext(fsdp_non_moe_mesh=mesh), fp32_main_params=True)

            manager.parallelize(model)

        _refresh_hsdp_precision_state(model)
        root_state = model.hsdp_scheduler.hsdp_state
        for module in (model.layers[0], model.layers[1], model):
            state = module.hsdp_scheduler.hsdp_state
            state._init_mp_dtypes()  # pylint: disable=protected-access
            self.assertIs(state.mp_policy, root_state.mp_policy)
            self.assertTrue(state.mp_policy.apply_grad_on_fp32_main_grad)
            expected_dtype = torch.bfloat16 if module is model.layers[1] else torch.float32
            self.assertEqual(state.hsdp_params[0].all_gather_inputs[0].dtype, expected_dtype)
            self.assertEqual(state.hsdp_params[0].reduce_comm_dtype(torch.ones(4, dtype=expected_dtype)), expected_dtype)
        self.assertTrue(all(parameter_ref() is None for parameter_ref in parameter_refs))
        self.assertIsInstance(root_state.mp_policy.custom_params, WeakIdKeyDictionary)
        self.assertEqual(len(root_state.mp_policy.custom_params), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unknown_parameter_fails_before_wrapping(self) -> None:
        """Feature: Parameter name validation.
        Description: Configure an override for a nonexistent parameter name.
        Expectation: The manager raises ValueError before replacing any parameter.
        """
        model = _DecoderModel()
        weight = model.layers[0].weight
        config = FSDP2Config(mix_precision=FSDP2MixedPrecisionConfig(
            custom_params={"layers.9.weight": {"param_dtype": "float32"}},
        ))
        manager = FSDP2Manager(config, MeshContext())

        with self.assertRaisesRegex(ValueError, "unknown parameter FQN: layers.9.weight"):
            manager.parallelize(model)

        self.assertIs(model.layers[0].weight, weight)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_expert_units_share_the_resolved_precision_policy(self) -> None:
        """Feature: Expert FSDP precision overrides.
        Description: Wrap dense and expert units on their respective meshes with an expert override.
        Expectation: Expert units reuse the shared policy and retain the FP32 override.
        """
        model = _DecoderModel()
        model.layers[0].mlp = nn.Module()
        model.layers[0].mlp.experts = nn.ModuleList([nn.Linear(4, 4)])
        config = FSDP2Config(mix_precision=FSDP2MixedPrecisionConfig(
            param_dtype="bfloat16",
            reduce_dtype="bfloat16",
            custom_params={
                "layers.0.mlp.experts.0.weight": {"param_dtype": "float32", "reduce_dtype": "float32"},
            },
        ))
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0), \
                patch.object(DeviceMesh, "get_group", return_value=None), \
                patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu")):
            ranks = torch.arange(4).reshape(2, 2)
            dense_mesh = DeviceMesh("cpu", ranks, mesh_dim_names=("fsdp_shard", "tp"), _init_backend=False)
            expert_mesh = DeviceMesh("cpu", ranks, mesh_dim_names=("edp_shard", "ep"), _init_backend=False)
            source_info = {
                parameter_fqn: ((Replicate(),), expert_mesh["ep"] if ".experts." in parameter_fqn else dense_mesh["tp"])
                for parameter_fqn, _ in model.named_parameters()
            }
            manager = FSDP2Manager(config, MeshContext(
                fsdp_non_moe_mesh=dense_mesh,
                fsdp_moe_mesh=expert_mesh,
            ))

            manager.parallelize(model, source_info)

        expert_state = model.layers[0].mlp.experts.hsdp_scheduler.hsdp_state
        expert_state._init_mp_dtypes()  # pylint: disable=protected-access
        self.assertIs(expert_state.mp_policy, model.hsdp_scheduler.hsdp_state.mp_policy)
        self.assertEqual(expert_state.hsdp_params[0].all_gather_inputs[0].dtype, torch.float32)
        self.assertEqual(expert_state.hsdp_params[0].reduce_comm_dtype(torch.ones(4)), torch.float32)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_tied_parameter_aliases_resolve_one_identity(self) -> None:
        """Feature: Shared parameter configuration.
        Description: Resolve equal and conflicting overrides under aliases of one Parameter.
        Expectation: Equal overrides coalesce; conflicts raise ValueError rather than depending on entry order.
        """
        model = nn.Linear(4, 4)
        model.register_parameter("tied_weight", model.weight)
        precision = FSDP2MixedPrecisionConfig(custom_params={
            "weight": {"param_dtype": "float32"},
            "tied_weight": {"param_dtype": "float32"},
        })
        manager = FSDP2Manager(FSDP2Config(mix_precision=precision), MeshContext())

        policy = manager._build_mixed_precision_policy(model)  # pylint: disable=protected-access

        self.assertEqual(len(policy.custom_params), 1)
        self.assertEqual(policy.custom_params[model.weight], {"param_dtype": torch.float32})
        precision.custom_params["tied_weight"] = {"param_dtype": "float16"}
        with self.assertRaisesRegex(ValueError, "conflicting overrides for tied_weight"):
            manager._build_mixed_precision_policy(model)  # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_yaml_overrides_are_rejected(self) -> None:
        """Feature: YAML precision validation.
        Description: Supply invalid parameter names, override shapes, fields, and dtype strings.
        Expectation: Configuration construction rejects malformed overrides before model wrapping.
        """
        for custom_params in (
            None,
            [],
            {1: {"param_dtype": "float32"}},
            {"": {"param_dtype": "float32"}},
            {"weight": ["float32", "float32"]},
            {"weight": {"output_dtype": "float32"}},
            {"weight": {"param_dtype": "float64"}},
        ):
            with self.subTest(custom_params=custom_params), self.assertRaisesRegex(ValueError, "custom_params"):
                FSDP2MixedPrecisionConfig(custom_params=custom_params)
