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
"""Stage-1 tests: build options surface, Trainer re-export identity, boundaries."""
# pylint: disable=wrong-import-position

import runpy
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from dataclasses import fields
from pathlib import Path

import torch

import hyper_parallel.models as auto_models
from hyper_parallel.components.quantization.config import LowPrecisionConfig
from hyper_parallel.trainer import config as trainer_config
from hyper_parallel.models import api, build_options
from hyper_parallel.models.build_options import (
    CompileConfig,
    FSDP2Config,
    FSDP2MixedPrecisionConfig,
    ModelBuildOptions,
    normalize_build_options,
)
from hyper_parallel.trainer.config.parallelism import ActivationCheckpointConfig


_REPO_ROOT = Path(__file__).resolve().parents[3]


class TestDeviceDiscovery(unittest.TestCase):
    """CUDA compatibility shims must not change the native accelerator type."""

    def test_cuda_availability_requires_compiled_backend(self) -> None:
        """An NPU shim may report CUDA availability on a CPU/NPU Torch build."""
        cases = ((False, True, False), (True, True, True), (True, False, False), (False, False, False))
        for compiled, available, expected in cases:
            with self.subTest(compiled=compiled, available=available), \
                    patch.object(torch.backends.cuda, "is_built", return_value=compiled), \
                    patch.object(torch.cuda, "is_available", return_value=available):
                actual = build_options._cuda_is_available()
                self.assertEqual(actual, expected, f"CUDA availability={actual}, expected={expected}")

    def test_selects_device_namespace_and_current_id(self) -> None:
        """Prefer CUDA, then NPU, with CPU as the fallback."""
        for cuda, npu, expected in ((True, True, "cuda"), (False, True, "npu"), (False, False, "cpu")):
            with self.subTest(device=expected), patch.multiple(
                build_options, IS_CUDA_AVAILABLE=cuda, IS_NPU_AVAILABLE=npu,
            ):
                self.assertEqual(build_options.get_device_type(), expected)
        device = Mock(current_device=Mock(return_value=3))
        with patch.object(build_options, "get_device_type", return_value="cpu"), patch.object(torch, "cpu", device):
            self.assertEqual(build_options.get_device_id(), 3)
        with patch.object(build_options, "get_device_type", return_value="missing_device"):
            self.assertIs(build_options.get_torch_device(), torch.cuda)

    def test_import_without_optional_npu_package(self) -> None:
        """Permit import when the NPU wheel is unavailable."""
        with patch.dict(sys.modules, {"torch_npu": None}):
            namespace = runpy.run_path(build_options.__file__)
        self.assertFalse(namespace["IS_NPU_AVAILABLE"])

    def test_registered_npu_disables_internal_format(self) -> None:
        """Initialize NPU config when the optional backend is registered."""
        npu = Mock(is_available=Mock(return_value=True))
        with patch.dict(sys.modules, {"torch_npu": Mock()}), patch.object(torch, "npu", npu, create=True):
            namespace = runpy.run_path(build_options.__file__)
        self.assertTrue(namespace["IS_NPU_AVAILABLE"])
        self.assertFalse(npu.config.allow_internal_format)


class TestBuildOptionsValidation(unittest.TestCase):
    """Reject malformed runtime options before building a model."""

    def test_validation_messages(self) -> None:
        """Identify invalid option fields and preserve valid mixed precision overrides."""
        cases = (
            (CompileConfig, {"dynamic_arg_dims": []}, ValueError, "dynamic_arg_dims"),
            (CompileConfig, {"backend": ""}, ValueError, "backend"),
            (CompileConfig, {"options": []}, TypeError, "options"),
            (CompileConfig, {"max_in_flight_gb": -1}, ValueError, "max_in_flight_gb"),
            (CompileConfig, {"dvm": True}, ValueError, "joint-graph"),
            (FSDP2Config, {"wrapper_class": ["not valid"]}, ValueError, "wrapper_class"),
            (FSDP2Config, {"reshard_after_forward": 1}, TypeError, "reshard_after_forward"),
            (FSDP2MixedPrecisionConfig, {"custom_params": []}, ValueError, "custom_params"),
            (FSDP2MixedPrecisionConfig, {"custom_params": {"": {}}}, ValueError, "keys"),
            (FSDP2MixedPrecisionConfig, {"custom_params": {"weight": []}}, ValueError, "values"),
            (FSDP2MixedPrecisionConfig, {"custom_params": {"weight": {"output_dtype": None}}}, ValueError, "supports"),
            (FSDP2MixedPrecisionConfig, {"custom_params": {"weight": {"param_dtype": "float64"}}}, ValueError, "dtype"),
            (ModelBuildOptions, {"device": 0}, TypeError, "device"),
            (ModelBuildOptions, {"low_precision": []}, TypeError, "low_precision"),
        )
        for config, options, error, message in cases:
            with self.subTest(options=options), self.assertRaisesRegex(error, message):
                config(**options)
        joint = CompileConfig(enabled=True, use_joint_graph=True, dvm=True)
        self.assertTrue(joint.dvm)
        with self.assertRaisesRegex(TypeError, "requires a mapping"):
            ModelBuildOptions.from_dict([])
        precision = FSDP2MixedPrecisionConfig(custom_params={"weight": {"param_dtype": "float32"}})
        self.assertEqual(precision.custom_params["weight"]["param_dtype"], "float32")


def _field_snapshot(cls):
    """Snapshot dataclass field names and defaults in declaration order."""
    return [(field.name, repr(field.default)) for field in fields(cls)]


class TestBuildOptionsFields(unittest.TestCase):
    """Field/default snapshots for the options owned by build_options."""

    def test_compile_config_fields(self):
        """CompileConfig field order and defaults stay unchanged after the move."""
        self.assertEqual(
            _field_snapshot(CompileConfig),
            [
                ("enabled", "False"),
                ("use_joint_graph", "False"),
                ("mode", "'default'"),
                ("fullgraph", "False"),
                ("dynamic", "False"),
                ("backend", "None"),
                ("options", "None"),
                ("dynamo_cache_size_limit", "256"),
                ("dynamic_arg_dims", "None"),
                ("fsdp_allgather_prefetch", "True"),
                ("fsdp_overlap_grad_reduce", "True"),
                ("max_in_flight_gb", "0.25"),
                ("offload_budget_gb", "64.0"),
                ("offload_prefetch_n_layers", "1"),
                ("dvm", "False"),
            ],
        )

    def test_fsdp2_config_fields(self):
        """FSDP2Config field order and defaults stay unchanged after the move."""
        self.assertEqual(
            [name for name, _ in _field_snapshot(FSDP2Config)],
            [
                "dp_shard_size", "edp_shard_size", "replicate_params",
                "mix_precision", "enable_offload", "reshard_after_forward",
                "reshard_after_backward", "requires_grad_sync",
                "backward_prefetch_depth", "forward_prefetch_depth",
                "comm_fusion", "comm_fusion_zero_copy",
                "wrapper_class",
            ],
        )
        config = FSDP2Config()
        self.assertEqual(config.dp_shard_size, 1)
        self.assertEqual(config.edp_shard_size, 1)
        self.assertIsInstance(config.mix_precision, FSDP2MixedPrecisionConfig)
        self.assertIsNone(config.mix_precision.param_dtype)
        self.assertIsNone(config.comm_fusion_zero_copy)
        self.assertTrue(config.reshard_after_forward)

    def test_fsdp2_config_validation(self):
        """FSDP2Config rejects non-positive topology sizes and negative depths."""
        with self.assertRaisesRegex(ValueError, "dp_shard_size"):
            FSDP2Config(dp_shard_size=0)
        with self.assertRaisesRegex(ValueError, "edp_shard_size"):
            FSDP2Config(edp_shard_size=0)
        with self.assertRaisesRegex(ValueError, "backward_prefetch_depth"):
            FSDP2Config(backward_prefetch_depth=-1)
        with self.assertRaisesRegex(ValueError, "forward_prefetch_depth"):
            FSDP2Config(forward_prefetch_depth=-1)

    def test_compile_config_validation(self):
        """CompileConfig rejects non-bool flags and contradictory options."""
        with self.assertRaisesRegex(TypeError, "compile.enabled must be a bool"):
            CompileConfig(enabled=1)
        with self.assertRaisesRegex(TypeError, "compile.use_joint_graph must be a bool"):
            CompileConfig(use_joint_graph=1)
        with self.assertRaisesRegex(TypeError, "compile.dvm must be a bool"):
            CompileConfig(dvm=1)
        with self.assertRaisesRegex(ValueError, "compile.mode"):
            CompileConfig(mode="  ")
        with self.assertRaisesRegex(ValueError, "compile.options"):
            CompileConfig(mode="reduce-overhead", options={"triton": True})
        with self.assertRaisesRegex(ValueError, "dynamo_cache_size_limit"):
            CompileConfig(dynamo_cache_size_limit=0)
        with self.assertRaisesRegex(ValueError, "offload_budget_gb"):
            CompileConfig(offload_budget_gb=-1)
        with self.assertRaisesRegex(ValueError, "offload_prefetch_n_layers"):
            CompileConfig(offload_prefetch_n_layers=-1)

    def test_use_joint_graph_selects_graph_compiler(self):
        """Graph-mode trainer selection is driven by compile.enabled + flag."""
        config = CompileConfig(
            enabled=True,
            use_joint_graph=True,
            mode="reduce-overhead",
            options={"triton": True},
        )

        self.assertTrue(config.selects_graph_compiler())
        self.assertEqual(config.mode, "reduce-overhead")
        self.assertEqual(config.options, {"triton": True})
        self.assertFalse(
            CompileConfig(
                enabled=True, use_joint_graph=False
            ).selects_graph_compiler()
        )

    def test_model_build_options_fields(self):
        """ModelBuildOptions covers device/dtype/activation/compile/validate/low-precision."""
        self.assertEqual(
            [name for name, _ in _field_snapshot(ModelBuildOptions)],
            [
                "device", "model_init_dtype", "activation_checkpoint",
                "compile", "validate_placement", "low_precision",
            ],
        )
        options = ModelBuildOptions()
        self.assertIsNone(options.device)
        self.assertIsNone(options.model_init_dtype)
        self.assertIsNone(options.activation_checkpoint)
        self.assertIsInstance(options.compile, CompileConfig)
        self.assertFalse(options.validate_placement)
        self.assertIsNone(options.low_precision)


class TestModelBuildOptionsNormalization(unittest.TestCase):
    """Dict normalization at the AutoModels boundary (never a Trainer DTO)."""

    def test_normalize_none_gives_defaults(self):
        """None normalizes to a default ModelBuildOptions."""
        options = normalize_build_options(None)
        self.assertIsInstance(options, ModelBuildOptions)

    def test_normalize_passthrough_identity(self):
        """An existing ModelBuildOptions is returned unchanged (identity)."""
        options = ModelBuildOptions()
        self.assertIs(normalize_build_options(options), options)

    def test_from_dict_normalizes_nested(self):
        """Nested compile/low_precision dicts and device strings are normalized."""
        activation_checkpoint = ActivationCheckpointConfig(mode="off")
        options = normalize_build_options({
            "device": "cpu",
            "model_init_dtype": "bfloat16",
            "activation_checkpoint": activation_checkpoint,
            "compile": {"enabled": True, "mode": "default"},
            "low_precision": {"enabled": True, "format": "hif8",
                              "scaling": "current"},
        })
        self.assertEqual(options.device, torch.device("cpu"))
        self.assertEqual(options.model_init_dtype, "bfloat16")
        self.assertIs(options.activation_checkpoint, activation_checkpoint)
        self.assertIsInstance(options.compile, CompileConfig)
        self.assertTrue(options.compile.enabled)
        self.assertIsInstance(options.low_precision, LowPrecisionConfig)
        self.assertTrue(options.low_precision.enabled)

    def test_invalid_values_fail_fast(self):
        """Bad enum values and types are rejected with clear messages."""
        with self.assertRaisesRegex(ValueError, "model_init_dtype"):
            ModelBuildOptions(model_init_dtype="float64")
        with self.assertRaisesRegex(ValueError, "activation_checkpoint"):
            ModelBuildOptions(activation_checkpoint="aggressive")
        with self.assertRaisesRegex(TypeError, "validate_placement must be a bool"):
            ModelBuildOptions(validate_placement="yes")
        with self.assertRaisesRegex(TypeError, "compile"):
            ModelBuildOptions(compile=42)
        with self.assertRaisesRegex(TypeError, "build options"):
            normalize_build_options(42)

    def test_from_dict_rejects_unknown_keys(self):
        """Unknown keys fail fast instead of being silently swallowed."""
        with self.assertRaises(TypeError):
            ModelBuildOptions.from_dict({"not_a_field": 1})
        with self.assertRaisesRegex(TypeError, "activation_swap"):
            ModelBuildOptions.from_dict({"activation_swap": "attention"})


class TestTrainerReexportIdentity(unittest.TestCase):
    """Trainer re-exports the very same class objects (no duplicate types)."""

    def test_trainer_config_reexport_identity(self):
        """trainer.config CompileConfig/FSDP2Config ARE the build_options classes."""
        self.assertIs(trainer_config.CompileConfig, CompileConfig)
        self.assertIs(trainer_config.FSDP2Config, FSDP2Config)

    def test_package_root_reexport_identity(self):
        """The auto_models package root re-exports the same class objects."""
        self.assertIs(auto_models.ModelBuildOptions, ModelBuildOptions)
        self.assertIs(auto_models.CompileConfig, CompileConfig)
        self.assertIs(auto_models.FSDP2Config, FSDP2Config)


class TestBuildOptionsBoundary(unittest.TestCase):
    """build_options/api must not import the Trainer layer (subprocess probe)."""

    def test_build_options_imports_no_trainer(self):
        """Importing build_options/api pulls in no trainer module."""
        probe = (
            "import sys\n"
            "import hyper_parallel.models.build_options\n"
            "import hyper_parallel.models.api\n"
            "import hyper_parallel.models\n"
            "offenders = [name for name in sys.modules\n"
            "             if name.startswith(('hyper_parallel.trainer',\n"
            "                                 'hyper_parallel.models.trainer',\n"
            "                                 'hyper_parallel.data',\n"
            "                                 'hyper_parallel.models.data'))]\n"
            "assert not offenders, offenders\n"
            "import torch.distributed as dist\n"
            "assert not dist.is_initialized()\n"
            "print('BUILD_OPTIONS_CLEAN')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            timeout=300,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BUILD_OPTIONS_CLEAN", result.stdout)

    def test_api_skeleton_surface(self):
        """api.py exposes the stable skeleton and delegates from_pretrained lazily."""
        self.assertEqual(
            api.__all__,
            [
                "CompileConfig", "FSDP2Config", "FSDP2MixedPrecisionConfig",
                "ModelBuildOptions", "from_pretrained", "normalize_options",
            ],
        )
        self.assertIs(api.ModelBuildOptions, ModelBuildOptions)
        with self.assertRaisesRegex(TypeError, "activation_swap"):
            api.normalize_options({"activation_swap": "attention"})


if __name__ == "__main__":
    unittest.main()
