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

import os
import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from hyper_parallel.models.build_options import (
    CompileConfig,
    FSDP2Config,
    FSDP2MixedPrecisionConfig,
    ModelBuildOptions,
    normalize_build_options,
)
from tests.common.mark_utils import arg_mark

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _field_snapshot(cls):
    """Snapshot dataclass field names and defaults in declaration order."""
    return [(field.name, repr(field.default)) for field in fields(cls)]


class TestBuildOptionsFields(unittest.TestCase):
    """Field/default snapshots for the options owned by build_options."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_compile_config_fields(self):
        """CompileConfig field order and defaults stay unchanged after the move."""
        self.assertEqual(
            _field_snapshot(CompileConfig),
            [
                ("enabled", "False"),
                ("mode", "'default'"),
                ("fullgraph", "False"),
                ("dynamic", "False"),
                ("backend", "None"),
                ("options", "None"),
                ("dynamo_cache_size_limit", "256"),
            ],
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
            ],
        )
        config = FSDP2Config()
        self.assertEqual(config.dp_shard_size, 1)
        self.assertEqual(config.edp_shard_size, 1)
        self.assertIsInstance(config.mix_precision, FSDP2MixedPrecisionConfig)
        self.assertIsNone(config.mix_precision.param_dtype)
        self.assertIsNone(config.comm_fusion_zero_copy)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_compile_config_validation(self):
        """CompileConfig rejects non-bool flags and contradictory options."""
        with self.assertRaisesRegex(TypeError, "compile.enabled must be a bool"):
            CompileConfig(enabled=1)
        with self.assertRaisesRegex(ValueError, "compile.mode"):
            CompileConfig(mode="  ")
        with self.assertRaisesRegex(ValueError, "compile.options"):
            CompileConfig(mode="reduce-overhead", options={"triton": True})
        with self.assertRaisesRegex(ValueError, "dynamo_cache_size_limit"):
            CompileConfig(dynamo_cache_size_limit=0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_model_build_options_fields(self):
        """ModelBuildOptions covers device/dtype/activation/compile/validate/low-precision."""
        self.assertEqual(
            [name for name, _ in _field_snapshot(ModelBuildOptions)],
            [
                "device", "model_init_dtype", "activation_checkpoint",
                "activation_swap", "swap_inputs", "compile",
                "validate_placement", "low_precision",
            ],
        )
        options = ModelBuildOptions()
        self.assertIsNone(options.device)
        self.assertIsNone(options.model_init_dtype)
        self.assertIsNone(options.activation_checkpoint)
        self.assertEqual(options.activation_swap, "none")
        self.assertFalse(options.swap_inputs)
        self.assertIsInstance(options.compile, CompileConfig)
        self.assertFalse(options.validate_placement)
        self.assertIsNone(options.low_precision)


class TestModelBuildOptionsNormalization(unittest.TestCase):
    """Dict normalization at the AutoModels boundary (never a Trainer DTO)."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_normalize_none_gives_defaults(self):
        """None normalizes to a default ModelBuildOptions."""
        options = normalize_build_options(None)
        self.assertIsInstance(options, ModelBuildOptions)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_normalize_passthrough_identity(self):
        """An existing ModelBuildOptions is returned unchanged (identity)."""
        options = ModelBuildOptions(activation_swap="attention")
        self.assertIs(normalize_build_options(options), options)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_from_dict_normalizes_nested(self):
        """Nested compile/low_precision dicts and device strings are normalized."""
        options = normalize_build_options({
            "device": "cpu",
            "model_init_dtype": "bfloat16",
            "activation_checkpoint": "full",
            "activation_swap": "attention",
            "compile": {"enabled": True, "mode": "default"},
            "low_precision": {"enabled": True, "format": "hif8",
                              "scaling": "current"},
        })
        self.assertEqual(options.device, torch.device("cpu"))
        self.assertEqual(options.model_init_dtype, "bfloat16")
        self.assertEqual(options.activation_checkpoint, "full")
        self.assertEqual(options.activation_swap, "attention")
        self.assertIsInstance(options.compile, CompileConfig)
        self.assertTrue(options.compile.enabled)
        from hyper_parallel.components.quantization.config import (
            LowPrecisionConfig,
        )
        self.assertIsInstance(options.low_precision, LowPrecisionConfig)
        self.assertTrue(options.low_precision.enabled)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_values_fail_fast(self):
        """Bad enum values and types are rejected with clear messages."""
        with self.assertRaisesRegex(ValueError, "model_init_dtype"):
            ModelBuildOptions(model_init_dtype="float64")
        with self.assertRaisesRegex(ValueError, "activation_checkpoint"):
            ModelBuildOptions(activation_checkpoint="aggressive")
        with self.assertRaisesRegex(ValueError, "activation_swap"):
            ModelBuildOptions(activation_swap="all")
        with self.assertRaisesRegex(TypeError, "swap_inputs must be a bool"):
            ModelBuildOptions(swap_inputs=1)
        with self.assertRaisesRegex(TypeError, "validate_placement must be a bool"):
            ModelBuildOptions(validate_placement="yes")
        with self.assertRaisesRegex(TypeError, "compile"):
            ModelBuildOptions(compile=42)
        with self.assertRaisesRegex(TypeError, "build options"):
            normalize_build_options(42)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_from_dict_rejects_unknown_keys(self):
        """Unknown keys fail fast instead of being silently swallowed."""
        with self.assertRaises(TypeError):
            ModelBuildOptions.from_dict({"not_a_field": 1})


class TestTrainerReexportIdentity(unittest.TestCase):
    """Trainer re-exports the very same class objects (no duplicate types)."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_trainer_config_reexport_identity(self):
        """trainer.config CompileConfig/FSDP2Config ARE the build_options classes."""
        from hyper_parallel.trainer import config as trainer_config

        self.assertIs(trainer_config.CompileConfig, CompileConfig)
        self.assertIs(trainer_config.FSDP2Config, FSDP2Config)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_package_root_reexport_identity(self):
        """The auto_models package root re-exports the same class objects."""
        import hyper_parallel.models as auto_models

        self.assertIs(auto_models.ModelBuildOptions, ModelBuildOptions)
        self.assertIs(auto_models.CompileConfig, CompileConfig)
        self.assertIs(auto_models.FSDP2Config, FSDP2Config)


class TestBuildOptionsBoundary(unittest.TestCase):
    """build_options/api must not import the Trainer layer (subprocess probe)."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
            env={**os.environ, "HYPER_PARALLEL_PLATFORM": "torch"},
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BUILD_OPTIONS_CLEAN", result.stdout)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_api_skeleton_surface(self):
        """api.py exposes the stable skeleton and delegates from_pretrained lazily."""
        from hyper_parallel.models import api

        self.assertEqual(
            api.__all__,
            [
                "CompileConfig", "FSDP2Config", "FSDP2MixedPrecisionConfig",
                "ModelBuildOptions", "from_pretrained", "normalize_options",
            ],
        )
        self.assertIs(api.ModelBuildOptions, ModelBuildOptions)
        options = api.normalize_options({"activation_swap": "attention"})
        self.assertEqual(options.activation_swap, "attention")


if __name__ == "__main__":
    unittest.main()
