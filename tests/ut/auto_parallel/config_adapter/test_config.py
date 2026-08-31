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
"""Unit tests for auto parallel strategy search configuration interfaces."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

try:
    import yaml  # type: ignore[import-untyped]  # pylint: disable=C0415
except ImportError:
    yaml = None  # pragma: no cover

from hyper_parallel.auto_parallel.config_adapter._normalized_config import (
    NormalizedConfig,
    ValidationError,
)
from hyper_parallel.auto_parallel.config_adapter._config_loader import (
    read_search_config,
    read_hp_yaml_config,
)
from hyper_parallel.auto_parallel.config_adapter._constraint_checker import (
    validate,
    validate_strict,
)
from hyper_parallel.auto_parallel.config_adapter._strategy_output import (
    normalized_to_summary,
    write_ppb_config,
    write_resolved_strategy,
    write_resolved_yaml,
    write_strategy_config,
)


def _make_dense_model_spec(**overrides) -> Dict[str, Any]:
    """Create a default dense model spec dict for testing."""
    spec = {
        "name": "test-dense",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "local_batch_size": 1,
        "moe_enabled": False,
    }
    spec.update(overrides)
    return spec


def _make_cluster_spec(**overrides) -> Dict[str, Any]:
    """Create a default cluster spec dict for testing."""
    spec = {
        "num_nodes": 4,
        "cards_per_node": 8,
        "device_memory_gb": 64.0,
        "device_type": "ascend",
        "intra_node_bandwidth_gbps": 200.0,
        "inter_node_bandwidth_gbps": 100.0,
    }
    spec.update(overrides)
    return spec


def _make_search_space(**overrides) -> Dict[str, Any]:
    """Create a default search space dict for testing."""
    space = {
        "data_parallel_replicate_degree": [1, 2, 4],
        "data_parallel_shard_degree": [1],
        "tensor_parallel_degree": [1, 2, 4],
        "pipeline_parallel_degree": [1, 2],
        "context_parallel_degree": [1],
        "expert_parallel_degree": [1],
        "micro_batch_num": [1, 2, 4],
    }
    space.update(overrides)
    return space


def _make_constraint(**overrides) -> Dict[str, Any]:
    """Create a default constraint dict for testing."""
    const = {
        "global_batch_size": 128,
        "memory_limit_gb": 60.0,
        "fixed_dp_degree": None,
        "fixed_tp_degree": None,
        "fixed_pp_degree": None,
        "fixed_cp_degree": None,
        "fixed_ep_degree": None,
    }
    const.update(overrides)
    return const


def _make_estimator(**overrides) -> Dict[str, Any]:
    """Create a default estimator dict for testing."""
    est = {
        "type": "symbolic",
        "recompute_strategy": "none",
        "enable_profiling_calibration": False,
    }
    est.update(overrides)
    return est


def _make_pp_config(**overrides) -> Dict[str, Any]:
    """Create a default pp_config dict for testing."""
    pp = {
        "pp_degree": 2,
        "stage_partition_mode": "uniform",
        "stage_partition": [],
        "layer_offset_range": (0, 0),
        "layer_recompute_layers": [],
        "micro_batch_num": 2,
    }
    pp.update(overrides)
    return pp


def _make_full_config(**overrides) -> NormalizedConfig:
    """Create a fully populated NormalizedConfig for testing."""
    return NormalizedConfig(
        model_spec=_make_dense_model_spec(),
        cluster_spec=_make_cluster_spec(),
        search_space=_make_search_space(),
        constraint=_make_constraint(),
        estimator=_make_estimator(),
        pp_config=_make_pp_config(),
        **overrides,
    )


def _write_yaml(path: str, data: Any) -> None:
    """Write a YAML file from a dict or string."""
    with open(path, "w", encoding="utf-8") as fh:
        if isinstance(data, str):
            fh.write(data)
        else:
            yaml.dump(data, fh, default_flow_style=False)


def _dense_search_yaml_content() -> str:
    """Return a minimal valid Search Config YAML string for a dense model."""
    return """
model:
  name: "test-dense"
  num_hidden_layers: 32
  hidden_size: 4096
  intermediate_size: 11008
  num_attention_heads: 32
  num_key_value_heads: 8
  vocab_size: 128256
  max_position_embeddings: 8192
  local_batch_size: 1

cluster:
  num_nodes: 4
  cards_per_node: 8
  device_memory_gb: 64.0
  device_type: "ascend"

parallelism:
  dp: [1, 2, 4]
  fsdp: 1
  tp: [1, 2, 4]
  pp: [1, 2]
  cp: 1
  ep: 1
  micro_batch_num: [1, 2, 4]

constraint:
  global_batch_size: 128
  memory_limit_gb: 60.0

recompute: "selective"

pp_config:
  stage_partition_mode: "uniform"
  pp_interleave_num: 1
  pipeline_schedule: "1F1B"
"""


def _dense_hp_yaml_content() -> str:
    """Return a minimal HyperParallel train.yaml string for testing."""
    return """
model:
  name: "test-dense"
  config_overrides:
    num_hidden_layers: 32
    hidden_size: 4096
    num_attention_heads: 32
    vocab_size: 128256

train:
  global_batch_size: 128
  micro_batch_size: 1
  accelerator:
    dp_shard: 1
    dp_replicate: 2
    tp_degree: 4
    pipeline_parallel_degree: 2
  gradient_checkpointing:
    activation_checkpoint: "selective"

data:
  max_seq_len: 8192
"""


def _auto_models_hp_yaml_content() -> str:
    """Return a minimal current AutoModels Trainer YAML string."""
    return """
model:
  _target_: hyper_parallel.auto_models._transformers.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: local/model
  torch_dtype: bfloat16
  local_files_only: true
training:
  global_batch_size: 64
  micro_batch_size: 2
accelerator:
  tp_size: 4
  cp_size: 1
  ep_size: 2
  pp_size: 2
fsdp_config:
  dp_shard_size: 4
activation_checkpoint:
  mode: full
dataset:
  data_transform:
    max_seq_len: 2048
"""


class TestTypes(unittest.TestCase):
    """Unit tests for type definitions."""

    def test_validation_error_defaults(self) -> None:
        """ValidationError defaults to error severity."""
        err = ValidationError(field_path="x", message="bad")
        self.assertEqual(err.severity, "error",
                         f"Expected 'error', got {err.severity!r}")

    def test_validation_error_full(self) -> None:
        """ValidationError stores field_path, message, severity."""
        err = ValidationError(
            field_path="model.hidden_size",
            message="hidden_size must be divisible by tp",
            severity="error",
        )
        self.assertEqual(err.field_path, "model.hidden_size",
                         f"Expected 'model.hidden_size', got {err.field_path!r}")
        self.assertEqual(err.severity, "error",
                         f"Expected 'error', got {err.severity!r}")

    def test_normalized_config_to_dict(self) -> None:
        """to_dict includes all expected sections."""
        config = _make_full_config()
        d = config.to_dict()
        self.assertIn("model_spec", d, f"model_spec missing: {list(d.keys())}")
        self.assertIn("cluster_spec", d, f"cluster_spec missing: {list(d.keys())}")
        self.assertIn("search_space", d, f"search_space missing: {list(d.keys())}")

    def test_normalized_config_resolved_strategy(self) -> None:
        """to_dict includes resolved_strategy when set."""
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4, "tp": 4, "pp": 2}
        d = config.to_dict()
        self.assertIn("resolved_strategy", d,
                      f"resolved_strategy missing: {list(d.keys())}")
        self.assertEqual(d["resolved_strategy"]["dp"], 4,
                         f"Expected 4, got {d['resolved_strategy']['dp']}")


class TestSearchConfigReader(unittest.TestCase):
    """Unit tests for the Search Config YAML reader (read_search_config)."""

    def setUp(self) -> None:
        """Create a temporary directory for test files."""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        for root, _, files in os.walk(self.tmpdir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            os.rmdir(root)

    def test_read_complete_search_yaml(self) -> None:
        """Reader produces correct NormalizedConfig from a complete search YAML."""
        path = os.path.join(self.tmpdir, "search.yaml")
        _write_yaml(path, _dense_search_yaml_content())
        config = read_search_config(path)
        self.assertEqual(config.model_spec["hidden_size"], 4096,
                         f"Expected 4096, got {config.model_spec['hidden_size']}")
        self.assertEqual(config.model_spec["num_hidden_layers"], 32,
                         f"Expected 32, got {config.model_spec['num_hidden_layers']}")
        self.assertEqual(config.cluster_spec["num_nodes"], 4,
                         f"Expected 4, got {config.cluster_spec['num_nodes']}")
        self.assertIn("data_parallel_replicate_degree", config.search_space,
                      f"dp missing from search_space: {list(config.search_space.keys())}")
        self.assertEqual(config.constraint["global_batch_size"], 128,
                         f"Expected 128, got {config.constraint['global_batch_size']}")

    def test_scalar_parallelism_is_fixed(self) -> None:
        """Scalar parallelism value sets both constraint.fixed_* and search_space."""
        content = """
model:
  num_hidden_layers: 10
  hidden_size: 1024
  num_attention_heads: 8
  vocab_size: 32000
parallelism:
  tp: 4
  cp: 1
"""
        path = os.path.join(self.tmpdir, "fixed.yaml")
        _write_yaml(path, content)
        config = read_search_config(path)
        self.assertEqual(config.constraint["fixed_tp_degree"], 4,
                         f"Expected fixed_tp_degree=4, got {config.constraint.get('fixed_tp_degree')}")
        self.assertEqual(config.constraint["fixed_cp_degree"], 1,
                         f"Expected fixed_cp_degree=1, got {config.constraint.get('fixed_cp_degree')}")
        self.assertEqual(config.search_space["tensor_parallel_degree"], [4],
                         "scalar tp should produce single-element list in search_space")

    def test_list_parallelism_is_search(self) -> None:
        """List parallelism values populate search_space."""
        content = """
model:
  num_hidden_layers: 10
  hidden_size: 1024
  num_attention_heads: 8
  vocab_size: 32000
parallelism:
  tp: [1, 2, 4]
  dp: [1, 2]
"""
        path = os.path.join(self.tmpdir, "search_list.yaml")
        _write_yaml(path, content)
        config = read_search_config(path)
        self.assertEqual(config.search_space["tensor_parallel_degree"], [1, 2, 4],
                         f"Expected [1,2,4], got {config.search_space['tensor_parallel_degree']}")
        self.assertNotIn("fixed_tp_degree", config.constraint,
                         "list-style tp should not produce fixed_tp_degree")

    def test_auto_parallelism_skipped(self) -> None:
        """'auto' dimension produces neither constraint nor search_space entry."""
        content = """
model:
  num_hidden_layers: 10
  hidden_size: 1024
  num_attention_heads: 8
  vocab_size: 32000
parallelism:
  pp: auto
"""
        path = os.path.join(self.tmpdir, "auto.yaml")
        _write_yaml(path, content)
        config = read_search_config(path)
        self.assertNotIn("fixed_pp_degree", config.constraint,
                         "auto pp should not set fixed_pp_degree")
        self.assertNotIn("pipeline_parallel_degree", config.search_space,
                         "auto pp should not appear in search_space")

    def test_missing_optional_fields_default(self) -> None:
        """Missing optional fields get default values."""
        content = """
model:
  num_hidden_layers: 20
  hidden_size: 2048
  num_attention_heads: 16
  vocab_size: 50000
"""
        path = os.path.join(self.tmpdir, "minimal.yaml")
        _write_yaml(path, content)
        config = read_search_config(path)
        self.assertEqual(config.model_spec["num_hidden_layers"], 20,
                         f"Expected 20, got {config.model_spec['num_hidden_layers']}")
        self.assertEqual(config.cluster_spec.get("num_nodes", 1), 1,
                         "cluster_spec should default")
        self.assertEqual(config.pp_config["stage_partition_mode"], "uniform",
                         "pp_config should default")

    def test_nonexistent_file_raises(self) -> None:
        """Reader raises FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            read_search_config("/nonexistent/path.yaml")

    def test_invalid_extension_raises(self) -> None:
        """Reader rejects non-.yaml file extension."""
        path = os.path.join(self.tmpdir, "config.toml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("dummy")
        with self.assertRaises(ValueError):
            read_search_config(path)

    def test_invalid_yaml_syntax_raises(self) -> None:
        """Reader raises ValueError on malformed YAML."""
        path = os.path.join(self.tmpdir, "bad.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(": invalid yaml :")
        with self.assertRaises(ValueError):
            read_search_config(path)

    def test_recompute_field_parsed(self) -> None:
        """Top-level recompute field maps to estimator.recompute_strategy."""
        content = """
model:
  num_hidden_layers: 10
  hidden_size: 1024
  num_attention_heads: 8
  vocab_size: 32000
recompute: "full"
"""
        path = os.path.join(self.tmpdir, "recompute.yaml")
        _write_yaml(path, content)
        config = read_search_config(path)
        self.assertEqual(config.estimator["recompute_strategy"], "full")

    def test_fsdp_dimension_mapped(self) -> None:
        """fsdp short name maps to data_parallel_shard_degree."""
        content = """
model:
  num_hidden_layers: 10
  hidden_size: 1024
  num_attention_heads: 8
  vocab_size: 32000
parallelism:
  fsdp: [1, 2]
  dp: 4
"""
        path = os.path.join(self.tmpdir, "fsdp_test.yaml")
        _write_yaml(path, content)
        config = read_search_config(path)
        self.assertEqual(config.search_space["data_parallel_shard_degree"], [1, 2],
                         f"fsdp should set shard_degree, got {config.search_space}")
        self.assertEqual(config.constraint["fixed_dp_degree"], 4,
                         "dp scalar should set fixed_dp_degree")

    def test_train_yaml_reference(self) -> None:
        """train_yaml reference loads model params from external file."""
        train_yaml = """
model:
  name: "qwen3_5"
  config_overrides:
    hidden_size: 2048
    num_hidden_layers: 16
    num_attention_heads: 16
    vocab_size: 50000
train:
  global_batch_size: 64
  micro_batch_size: 1
  accelerator:
    tp_degree: 2
    pipeline_parallel_degree: 2
data:
  max_seq_len: 4096
"""
        train_path = os.path.join(self.tmpdir, "train.yaml")
        _write_yaml(train_path, train_yaml)

        search_data = {
            "train_yaml": train_path,
            "cluster": {"num_nodes": 2, "cards_per_node": 8},
            "parallelism": {"tp": [1, 2, 4]},
        }
        search_path = os.path.join(self.tmpdir, "search.yaml")
        _write_yaml(search_path, search_data)
        config = read_search_config(search_path)

        # Model params from train.yaml (normalized to internal short names)
        self.assertEqual(config.model_spec["num_hidden_layers"], 16,
                         f"Expected 16, got {config.model_spec.get('num_hidden_layers')}")
        self.assertEqual(config.model_spec["hidden_size"], 2048,
                         f"Expected 2048, got {config.model_spec.get('hidden_size')}")
        # Cluster from search.yaml
        self.assertEqual(config.cluster_spec["num_nodes"], 2)
        # TP explicitly declared as search dim
        self.assertEqual(config.search_space["tensor_parallel_degree"], [1, 2, 4])
        # PP inherited from train.yaml (not declared, so fixed)
        self.assertIn("pipeline_parallel_degree", config.search_space,
                       "pp should be inherited from train.yaml")
        self.assertEqual(config.search_space["pipeline_parallel_degree"], [2],
                         "inherited pp should be fixed as [2]")
        # GBS inherited from train.yaml
        self.assertEqual(config.constraint["global_batch_size"], 64)


class TestHpYamlReader(unittest.TestCase):
    """Unit tests for the HyperParallel train.yaml reader (read_hp_yaml_config)."""

    def setUp(self) -> None:
        """Create a temporary directory for test files."""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        for root, _, files in os.walk(self.tmpdir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            os.rmdir(root)

    def test_read_hp_yaml_basic(self) -> None:
        """read_hp_yaml_config extracts accelerator parallelism fields."""
        path = os.path.join(self.tmpdir, "train.yaml")
        _write_yaml(path, _dense_hp_yaml_content())
        config = read_hp_yaml_config(path)
        self.assertEqual(config.model_spec["num_hidden_layers"], 32)
        self.assertEqual(config.model_spec["hidden_size"], 4096)
        self.assertEqual(config.search_space["data_parallel_shard_degree"], [1])
        self.assertEqual(config.search_space["tensor_parallel_degree"], [4])
        self.assertEqual(config.search_space["pipeline_parallel_degree"], [2])
        self.assertEqual(config.estimator["recompute_strategy"], "selective")

    @patch(
        "hyper_parallel.auto_models._transformers.registry.get_hf_config"
    )
    def test_read_auto_models_yaml_basic(self, mock_get_hf_config) -> None:
        """read_hp_yaml_config extracts the current Trainer schema."""
        mock_get_hf_config.return_value = SimpleNamespace(
            model_type="qwen3_moe",
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=11008,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=128256,
            max_position_embeddings=8192,
            num_experts=1,
        )
        path = os.path.join(self.tmpdir, "auto_models.yaml")
        _write_yaml(path, _auto_models_hp_yaml_content())

        config = read_hp_yaml_config(path)

        mock_get_hf_config.assert_called_once_with(
            "local/model", "sdpa", "bfloat16", local_files_only=True,
        )
        self.assertEqual(config.model_spec["name"], "qwen3_moe")
        self.assertEqual(config.model_spec["num_hidden_layers"], 32)
        self.assertEqual(config.model_spec["max_position_embeddings"], 2048)
        self.assertEqual(config.model_spec["local_batch_size"], 2)
        self.assertEqual(config.search_space["data_parallel_shard_degree"], [4])
        self.assertEqual(config.search_space["tensor_parallel_degree"], [4])
        self.assertEqual(config.search_space["pipeline_parallel_degree"], [2])
        self.assertEqual(config.estimator["recompute_strategy"], "full")
        self.assertEqual(config.pp_config["micro_batch_num"], 8)

    def test_hp_yaml_empty_accelerator_defaults(self) -> None:
        """Empty accelerator section produces default search_space (single-element lists)."""
        content = """
model:
  name: "test"
train:
  global_batch_size: 8
  micro_batch_size: 1
data:
  max_seq_len: 4096
"""
        path = os.path.join(self.tmpdir, "empty_accel.yaml")
        _write_yaml(path, content)
        config = read_hp_yaml_config(path)
        self.assertEqual(config.search_space, {},
                         f"Expected empty search_space, got {config.search_space}")
        self.assertEqual(config.constraint["global_batch_size"], 8)

    def test_hp_yaml_nonexistent_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_hp_yaml_config("/nonexistent/train.yaml")


class TestValidator(unittest.TestCase):
    """Unit tests for configuration validator (AP-CFG-04 ~ AP-CFG-09, AP-CFG-11)."""

    def test_ap_cfg_04_dims_product_exceeds_devices(self) -> None:
        """AP-CFG-04: Parallel dimension product exceeds devices -> ERROR."""
        config = _make_full_config()
        config.search_space["data_parallel_replicate_degree"] = [32]
        config.search_space["tensor_parallel_degree"] = [16]
        config.cluster_spec = _make_cluster_spec(num_nodes=1, cards_per_node=8)
        errors = validate(config)
        has_product_error = any(
            "product" in e.message.lower() or "exceeds" in e.message.lower()
            for e in errors
        )
        self.assertTrue(has_product_error,
                        f"Expected product-exceeds error, got errors: {errors}")

    def test_ap_cfg_05_hidden_size_not_divisible_by_tp(self) -> None:
        """AP-CFG-05: hidden_size not divisible by tp_degree -> ERROR."""
        config = _make_full_config()
        config.model_spec["hidden_size"] = 4096
        config.search_space["tensor_parallel_degree"] = [3]
        errors = validate(config)
        has_div_error = any(
            "divisible" in e.message.lower()
            and "hidden_size" in e.field_path
            for e in errors
        )
        self.assertTrue(has_div_error,
                        f"Expected hidden_size divisibility error, got errors: {errors}")

    def test_ap_cfg_06_seq_length_not_divisible_by_cp(self) -> None:
        """AP-CFG-06: max_position_embeddings not divisible by cp_degree -> ERROR."""
        config = _make_full_config()
        config.model_spec["max_position_embeddings"] = 8191
        config.search_space["context_parallel_degree"] = [2]
        errors = validate(config)
        has_div_error = any(
            "divisible" in e.message.lower()
            and "max_position_embeddings" in e.field_path
            for e in errors
        )
        self.assertTrue(has_div_error,
                        f"Expected max_position_embeddings divisibility error, got errors: {errors}")

    def test_ap_cfg_07_num_experts_not_divisible_by_ep(self) -> None:
        """AP-CFG-07: num_experts not divisible by ep_degree -> ERROR."""
        config = _make_full_config()
        config.model_spec["moe_enabled"] = True
        config.model_spec["num_experts"] = 8
        config.search_space["expert_parallel_degree"] = [3]
        errors = validate(config)
        has_div_error = any(
            "divisible" in e.message.lower()
            and "num_experts" in e.field_path
            for e in errors
        )
        self.assertTrue(has_div_error,
                        f"Expected num_experts divisibility error, got errors: {errors}")

    def test_ap_cfg_08_pp_stage_partition_validation(self) -> None:
        """AP-CFG-08: PP stage_partition validation."""
        config = _make_full_config()
        config.model_spec["num_hidden_layers"] = 4
        config.pp_config["pp_degree"] = 2
        config.pp_config["stage_partition_mode"] = "manual"
        config.pp_config["stage_partition"] = [[0, 1], [2, 3]]
        errors = validate(config)
        stage_errors = [e for e in errors if "stage_partition" in e.field_path]
        self.assertEqual(len(stage_errors), 0,
                         f"Expected 0 stage_partition errors, got {stage_errors}")

    def test_ap_cfg_08_pp_stage_partition_list_pp_passes_pp1(self) -> None:
        """AP-CFG-08: stage_partition not checked when pp_degree list includes 1."""
        config = _make_full_config()
        config.model_spec["num_hidden_layers"] = 4
        config.pp_config["pp_degree"] = [1, 2]
        config.pp_config["stage_partition_mode"] = "manual"
        config.pp_config["stage_partition"] = [[0, 1], [2, 3]]
        errors = validate(config)
        stage_errors = [e for e in errors if "stage_partition" in e.field_path]
        self.assertEqual(len(stage_errors), 0,
                         f"Expected 0 stage_partition errors, got {stage_errors}")

    def test_ap_cfg_08_pp_stage_count_mismatch(self) -> None:
        """AP-CFG-08: stage_partition count mismatch with pp_degree."""
        config = _make_full_config()
        config.model_spec["num_hidden_layers"] = 6
        config.pp_config["pp_degree"] = 4
        config.pp_config["stage_partition_mode"] = "manual"
        config.pp_config["stage_partition"] = [[0, 1], [2, 3]]
        errors = validate(config)
        stage_errors = [e for e in errors if "stage_partition" in e.field_path]
        self.assertTrue(
            any("2 stages" in e.message for e in stage_errors),
            f"Expected stage count mismatch error, got {stage_errors}",
        )

    def test_ap_cfg_08_pp_stage_count_mismatch_list_pp(self) -> None:
        """AP-CFG-08: stage_partition mismatch when pp_degree is a list."""
        config = _make_full_config()
        config.model_spec["num_hidden_layers"] = 6
        config.pp_config["pp_degree"] = [2, 4]
        config.pp_config["stage_partition_mode"] = "manual"
        config.pp_config["stage_partition"] = [[0], [1], [2]]
        errors = validate(config)
        stage_errors = [e for e in errors if "stage_partition" in e.field_path]
        self.assertTrue(
            any("3 stages" in e.message for e in stage_errors),
            f"Expected stage count mismatch error for list pp_degree, "
            f"got {stage_errors}",
        )

    def test_ap_cfg_09_layer_offset_validation(self) -> None:
        """AP-CFG-09: layer_offset_range validation."""
        config = _make_full_config()
        config.pp_config["layer_offset_range"] = (50, 100)
        errors = validate(config)
        offset_errors = [e for e in errors if "layer_offset_range" in e.field_path]
        self.assertTrue(len(offset_errors) >= 1,
                        f"Expected offset error, got {offset_errors}")

    def test_ap_cfg_09_layer_recompute_validation(self) -> None:
        """AP-CFG-09: layer_recompute_layers validation."""
        config = _make_full_config()
        config.pp_config["layer_recompute_layers"] = [0, 1, 50]
        errors = validate(config)
        recompute_errors = [
            e for e in errors if "layer_recompute" in e.field_path
        ]
        self.assertTrue(len(recompute_errors) >= 1,
                        f"Expected recompute error, got {recompute_errors}")

    def test_ap_cfg_11_fixed_dim_vs_search_space(self) -> None:
        """AP-CFG-11: Fixed dimension not in search space -> ERROR."""
        config = _make_full_config()
        config.constraint["fixed_tp_degree"] = 3
        config.search_space["tensor_parallel_degree"] = [1, 2, 4, 8]
        errors = validate(config)
        has_fixed_error = any(
            "fixed" in e.field_path
            for e in errors
        )
        self.assertTrue(has_fixed_error,
                        f"Expected fixed-dim conflict error, got errors: {errors}")

    def test_required_fields_error(self) -> None:
        """Validator catches missing required fields."""
        config = NormalizedConfig()
        errors = validate(config)
        has_required = any("must be > 0" in e.message for e in errors)
        self.assertTrue(has_required,
                        f"Expected required-fields error, got errors: {errors}")

    def test_validate_strict_raises(self) -> None:
        """validate_strict raises ValueError on errors."""
        config = NormalizedConfig()
        config.model_spec = {
            "num_hidden_layers": 32, "hidden_size": 4096,
            "num_attention_heads": 32, "vocab_size": 32000,
        }
        config.cluster_spec = _make_cluster_spec(num_nodes=1, cards_per_node=8)
        config.search_space["data_parallel_replicate_degree"] = [32]
        config.search_space["tensor_parallel_degree"] = [16]
        with self.assertRaises(ValueError):
            validate_strict(config)

    def test_validate_strict_clean_passes(self) -> None:
        """validate_strict passes on clean config."""
        config = _make_full_config()
        validate_strict(config)

    def test_valid_config_no_errors(self) -> None:
        """A valid configuration produces no errors."""
        config = _make_full_config()
        errors = validate(config)
        self.assertEqual(len(errors), 0,
                         f"Expected 0 errors, got {len(errors)}: {errors}")

    def test_dense_model_ep_cp_warning(self) -> None:
        """Dense model with ep > 1 or cp > 1 emits warnings."""
        config = _make_full_config()
        config.model_spec["moe_enabled"] = False
        config.search_space["expert_parallel_degree"] = [1, 2]
        config.search_space["context_parallel_degree"] = [4]
        errors = validate(config)
        warnings = [e for e in errors if e.severity == "warning"]
        ep_warnings = [e for e in warnings if "expert_parallel" in e.field_path]
        cp_warnings = [e for e in warnings if "context_parallel" in e.field_path]
        self.assertTrue(len(ep_warnings) >= 1,
                        f"Expected EP warning on dense model, got {ep_warnings}")
        self.assertTrue(len(cp_warnings) >= 1,
                        f"Expected CP warning on dense model, got {cp_warnings}")

    def test_fsdp_hsdp_device_product_error(self) -> None:
        """FSDP/HSDP device product exceeds available devices."""
        config = _make_full_config()
        config.cluster_spec = _make_cluster_spec(num_nodes=1, cards_per_node=8)
        config.search_space["data_parallel_shard_degree"] = [8]
        config.search_space["data_parallel_replicate_degree"] = [4]
        config.search_space["tensor_parallel_degree"] = [4]
        config.search_space["pipeline_parallel_degree"] = [2]
        errors = validate(config)
        product_errors = [
            e for e in errors
            if e.severity == "error" and "FSDP" in e.message
        ]
        self.assertTrue(len(product_errors) >= 1,
                        f"Expected FSDP/HSDP product error, got {errors}")

    def test_batch_size_divisibility_errors(self) -> None:
        """Global batch size must be divisible by micro_batch_num and dp."""
        config = _make_full_config()
        config.constraint["global_batch_size"] = 7
        config.search_space["micro_batch_num"] = [3]
        config.search_space["data_parallel_replicate_degree"] = [3]
        errors = validate(config)
        has_mbn_error = any(
            "micro_batch_num" in e.message for e in errors
        )
        has_dp_error = any(
            "effective DP" in e.message for e in errors
        )
        self.assertTrue(has_mbn_error,
                        f"Expected gbs % mbn error, got {errors}")
        self.assertTrue(has_dp_error,
                        f"Expected gbs % dp error, got {errors}")

    def test_fsdp_batch_size_divisibility(self) -> None:
        """FSDP: global_batch_size must be divisible by shard * replicate."""
        config = _make_full_config()
        config.constraint["global_batch_size"] = 15
        config.search_space["data_parallel_replicate_degree"] = [2]
        config.search_space["data_parallel_shard_degree"] = [2]
        errors = validate(config)
        has_fsdp_error = any(
            "effective DP" in e.message for e in errors
        )
        self.assertTrue(has_fsdp_error,
                        f"Expected FSDP batch divisibility error, got {errors}")

    def test_pp_degree_exceeds_layers(self) -> None:
        """pp_degree must not exceed num_hidden_layers."""
        config = _make_full_config()
        config.model_spec["num_hidden_layers"] = 4
        config.pp_config["pp_degree"] = 8
        errors = validate(config)
        has_pp_error = any(
            "pp_degree" in e.field_path for e in errors
        )
        self.assertTrue(has_pp_error,
                        f"Expected pp_degree exceeds layers error, got {errors}")

    def test_stage_partition_extra_layers(self) -> None:
        """stage_partition must not reference non-existent layers."""
        config = _make_full_config()
        config.model_spec["num_hidden_layers"] = 4
        config.pp_config["pp_degree"] = 2
        config.pp_config["stage_partition_mode"] = "manual"
        config.pp_config["stage_partition"] = [[0, 1], [99]]
        errors = validate(config)
        has_extra_error = any(
            "non-existent" in e.message.lower() for e in errors
        )
        self.assertTrue(has_extra_error,
                        f"Expected extra layers error, got {errors}")

    def test_layer_offset_min_max(self) -> None:
        """layer_offset_range min must be <= max."""
        config = _make_full_config()
        config.pp_config["layer_offset_range"] = (5, 3)
        errors = validate(config)
        has_offset_error = any(
            "layer_offset_range" in e.field_path for e in errors
        )
        self.assertTrue(has_offset_error,
                        f"Expected offset min > max error, got {errors}")

    def test_memory_limit_warning(self) -> None:
        """memory_limit_gb exceeding device_memory_gb emits a warning."""
        config = _make_full_config()
        config.constraint["memory_limit_gb"] = 99.0
        config.cluster_spec["device_memory_gb"] = 64.0
        errors = validate(config)
        has_mem_warning = any(
            e.severity == "warning" and "memory_limit_gb" in e.field_path
            for e in errors
        )
        self.assertTrue(has_mem_warning,
                        f"Expected memory limit warning, got {errors}")


class TestWriter(unittest.TestCase):
    """Unit tests for configuration writer (AP-CFG-12)."""

    def setUp(self) -> None:
        """Create a temporary directory for test files."""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        for root, _, files in os.walk(self.tmpdir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            os.rmdir(root)

    def test_write_strategy_config_json(self) -> None:
        """Write strategy config to JSON file."""
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4, "tp": 4, "pp": 2}
        path = os.path.join(self.tmpdir, "strategy.json")
        write_strategy_config(config, path, fmt="json")
        self.assertTrue(os.path.isfile(path), f"Output file not created: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.assertEqual(raw["model_spec"]["num_hidden_layers"], 32,
                         f"Expected 32, got {raw['model_spec']['num_hidden_layers']}")

    def test_write_strategy_config_default_fmt_json(self) -> None:
        """write_strategy_config uses json by default."""
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4}
        path = os.path.join(self.tmpdir, "strategy.json")
        write_strategy_config(config, path)
        self.assertTrue(os.path.isfile(path))

    def test_write_strategy_config_creates_dir(self) -> None:
        """write_strategy_config creates parent directory."""
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4}
        path = os.path.join(self.tmpdir, "sub", "cfg.json")
        write_strategy_config(config, path)
        self.assertTrue(os.path.isfile(path))

    def test_write_strategy_config_invalid_format(self) -> None:
        """write_strategy_config raises ValueError for unsupported format."""
        config = _make_full_config()
        path = os.path.join(self.tmpdir, "cfg.yaml")
        with self.assertRaises(ValueError):
            write_strategy_config(config, path, fmt="yaml")

    def test_write_resolved_strategy(self) -> None:
        """Write resolved strategy to JSON."""
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4, "tp": 4, "pp": 2}
        path = os.path.join(self.tmpdir, "resolved.json")
        write_resolved_strategy(config, path)
        self.assertTrue(os.path.isfile(path), f"Output file not created: {path}")

    def test_write_resolved_strategy_default_fmt_json(self) -> None:
        """write_resolved_strategy uses json by default."""
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4}
        path = os.path.join(self.tmpdir, "resolved.json")
        write_resolved_strategy(config, path)
        self.assertTrue(os.path.isfile(path))

    def test_write_resolved_strategy_raises_when_none(self) -> None:
        """write_resolved_strategy raises ValueError when strategy is None."""
        config = _make_full_config()
        path = os.path.join(self.tmpdir, "resolved.json")
        with self.assertRaises(ValueError):
            write_resolved_strategy(config, path)

    def test_write_resolved_strategy_invalid_format(self) -> None:
        """write_resolved_strategy raises ValueError for unsupported format."""
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4}
        path = os.path.join(self.tmpdir, "resolved.yaml")
        with self.assertRaises(ValueError):
            write_resolved_strategy(config, path, fmt="yaml")

    def test_write_resolved_strategy_creates_dir(self) -> None:
        """write_resolved_strategy creates parent directory."""
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4}
        path = os.path.join(self.tmpdir, "sub", "resolved.json")
        write_resolved_strategy(config, path)
        self.assertTrue(os.path.isfile(path))

    def test_write_resolved_strategy_json_format(self) -> None:
        """write_resolved_strategy with explicit json format."""
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4}
        path = os.path.join(self.tmpdir, "resolved.json")
        write_resolved_strategy(config, path, fmt="json")
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.assertEqual(raw["resolved_strategy"]["dp"], 4)

    def test_write_ppb_config_stub(self) -> None:
        """write_ppb_config produces a valid JSON stub."""
        config = _make_full_config()
        path = os.path.join(self.tmpdir, "ppb_config.json")
        write_ppb_config(config, path)
        self.assertTrue(os.path.isfile(path), f"PPB config not created: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.assertIn("_hyper_model", raw, f"_hyper_model missing: {list(raw.keys())}")
        self.assertEqual(raw["_hyper_model"]["num_hidden_layers"], 32,
                         f"Expected 32, got {raw['_hyper_model']['num_hidden_layers']}")

    def test_write_ppb_config_creates_dir(self) -> None:
        """write_ppb_config creates parent directory."""
        config = _make_full_config()
        path = os.path.join(self.tmpdir, "sub", "ppb.json")
        write_ppb_config(config, path)
        self.assertTrue(os.path.isfile(path))

    def test_summary_generation(self) -> None:
        """normalized_to_summary produces a correct summary."""
        config = _make_full_config()
        summary = normalized_to_summary(config)
        self.assertEqual(summary["model"]["num_hidden_layers"], 32,
                         f"Expected 32, got {summary['model']['num_hidden_layers']}")
        self.assertEqual(summary["cluster"]["total_cards"], 32,
                         f"Expected 32, got {summary['cluster']['total_cards']}")
        self.assertIn("data_parallel_replicate_degree", summary["search_space"],
                      f"search_space summary missing dp key: {sorted(summary['search_space'].keys())}")

    def test_roundtrip_written_json(self) -> None:
        """Write then read back a JSON config roundtrip."""
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4, "tp": 4, "pp": 2}
        path = os.path.join(self.tmpdir, "roundtrip.json")
        write_strategy_config(config, path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["model_spec"]["num_hidden_layers"], 32,
                         f"Roundtrip failed: {data['model_spec']}")

    def test_resolve_pp_degree_list(self) -> None:
        """_resolve_pp_degree extracts first element from list."""
        config = _make_full_config()
        config.pp_config["pp_degree"] = [4]
        summary = normalized_to_summary(config)
        self.assertEqual(summary["pipeline"]["pp_degree"], 4)

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_write_resolved_yaml_basic(self) -> None:
        """write_resolved_yaml injects resolved strategy into a YAML file."""
        original_yaml_path = os.path.join(self.tmpdir, "original.yaml")
        original_content = {
            "model": {"name": "test"},
            "train": {
                "accelerator": {
                    "dp_shard": 1,
                    "tp_degree": 1,
                },
                "global_batch_size": 8,
            },
        }
        with open(original_yaml_path, "w", encoding="utf-8") as fh:
            yaml.dump(original_content, fh, default_flow_style=False)

        config = _make_full_config()
        config.resolved_strategy = {
            "dp_shard": 4,
            "tp_degree": 2,
            "pipeline_parallel_degree": 2,
            "global_batch_size": 64,
            "micro_batch_num": 4,
        }
        output_path = os.path.join(self.tmpdir, "resolved.yaml")
        write_resolved_yaml(config, original_yaml_path, output_path)

        self.assertTrue(os.path.isfile(output_path))
        with open(output_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        accel = data["train"]["accelerator"]
        self.assertEqual(accel["dp_shard"], 4)
        self.assertEqual(accel["tp_degree"], 2)
        self.assertEqual(accel["pipeline_parallel_degree"], 2)
        self.assertEqual(data["train"]["global_batch_size"], 64)
        self.assertEqual(data["train"]["micro_batch_num"], 4)

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_write_resolved_yaml_with_short_names(self) -> None:
        """write_resolved_yaml accepts short alias keys (dp, tp, pp)."""
        original_yaml_path = os.path.join(self.tmpdir, "original.yaml")
        original_content = {
            "model": {"name": "test"},
            "train": {
                "accelerator": {},
            },
        }
        with open(original_yaml_path, "w", encoding="utf-8") as fh:
            yaml.dump(original_content, fh, default_flow_style=False)

        config = _make_full_config()
        config.resolved_strategy = {
            "dp": 4,
            "tp": 2,
            "pp": 2,
            "cp": 1,
            "ep": 1,
        }
        output_path = os.path.join(self.tmpdir, "resolved_short.yaml")
        write_resolved_yaml(config, original_yaml_path, output_path)

        with open(output_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        accel = data["train"]["accelerator"]
        self.assertEqual(accel["dp_replicate"], 4)
        self.assertEqual(accel["tp_degree"], 2)
        self.assertEqual(accel["pipeline_parallel_degree"], 2)
        self.assertEqual(accel["context_parallel_degree"], 1)
        self.assertEqual(accel["expert_parallel_degree"], 1)

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_write_resolved_auto_models_yaml(self) -> None:
        """write_resolved_yaml updates current Trainer topology fields."""
        original_yaml_path = os.path.join(self.tmpdir, "auto_models.yaml")
        original_content = {
            "model": {"pretrained_model_name_or_path": "local/model"},
            "training": {"global_batch_size": 8},
            "accelerator": {"tp_size": 1, "pp_size": 1},
            "fsdp_config": {"dp_shard_size": 1},
        }
        with open(original_yaml_path, "w", encoding="utf-8") as fh:
            yaml.dump(original_content, fh, default_flow_style=False)

        config = _make_full_config()
        config.resolved_strategy = {
            "dp": 8,
            "dp_shard": 4,
            "tp": 2,
            "pp": 2,
            "cp": 1,
            "ep": 2,
            "global_batch_size": 64,
        }
        output_path = os.path.join(self.tmpdir, "resolved_auto_models.yaml")
        write_resolved_yaml(config, original_yaml_path, output_path)

        with open(output_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        self.assertEqual(data["fsdp_config"]["dp_shard_size"], 4)
        self.assertEqual(data["accelerator"]["tp_size"], 2)
        self.assertEqual(data["accelerator"]["pp_size"], 2)
        self.assertEqual(data["accelerator"]["cp_size"], 1)
        self.assertEqual(data["accelerator"]["ep_size"], 2)
        self.assertEqual(data["training"]["global_batch_size"], 64)
        self.assertNotIn("train", data)

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_write_resolved_yaml_raises_when_none(self) -> None:
        """write_resolved_yaml raises ValueError when strategy is None."""
        config = _make_full_config()
        out = os.path.join(self.tmpdir, "out.yaml")
        with self.assertRaises(ValueError):
            write_resolved_yaml(config, out, out)

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_write_resolved_yaml_creates_dir(self) -> None:
        """write_resolved_yaml creates parent directory."""
        original_yaml_path = os.path.join(self.tmpdir, "original.yaml")
        with open(original_yaml_path, "w", encoding="utf-8") as fh:
            yaml.dump({"train": {"accelerator": {}}}, fh)
        config = _make_full_config()
        config.resolved_strategy = {"dp": 4}
        output_path = os.path.join(self.tmpdir, "sub", "resolved.yaml")
        write_resolved_yaml(config, original_yaml_path, output_path)
        self.assertTrue(os.path.isfile(output_path))

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_write_resolved_yaml_overwrite(self) -> None:
        """write_resolved_yaml overwrites original file when requested."""
        original_yaml_path = os.path.join(self.tmpdir, "original.yaml")
        with open(original_yaml_path, "w", encoding="utf-8") as fh:
            yaml.dump({"train": {"accelerator": {}}}, fh)
        config = _make_full_config()
        config.resolved_strategy = {"dp_shard": 4}
        write_resolved_yaml(config, original_yaml_path, original_yaml_path,
                            overwrite=True)
        with open(original_yaml_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        self.assertEqual(data["train"]["accelerator"]["dp_shard"], 4)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests using the Search Config format."""

    def setUp(self) -> None:
        """Create a temporary directory for test files."""
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        for root, _, files in os.walk(self.tmpdir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            os.rmdir(root)

    def test_full_pipeline_read_validate_write(self) -> None:
        """Full pipeline: search YAML -> read -> validate -> write (JSON)."""
        yaml_path = os.path.join(self.tmpdir, "search.yaml")
        _write_yaml(yaml_path, _dense_search_yaml_content())

        config = read_search_config(yaml_path)
        errors = validate(config)
        error_errors = [e for e in errors if e.severity == "error"]
        self.assertEqual(
            len(error_errors), 0,
            f"Expected 0 validation errors, got {error_errors}",
        )

        config.resolved_strategy = {"dp": 4, "tp": 4, "pp": 2, "cp": 1, "ep": 1,
                                    "micro_batch_num": 2}
        out_path = os.path.join(self.tmpdir, "output.json")
        write_strategy_config(config, out_path)
        self.assertTrue(os.path.isfile(out_path),
                        f"Output file not created: {out_path}")

    def test_full_pipeline_read_validate_summary(self) -> None:
        """Full pipeline: search YAML -> read -> validate -> summary."""
        yaml_path = os.path.join(self.tmpdir, "search.yaml")
        _write_yaml(yaml_path, _dense_search_yaml_content())

        config = read_search_config(yaml_path)
        validate_strict(config)

        summary = normalized_to_summary(config)
        self.assertEqual(summary["model"]["num_hidden_layers"], 32)
        self.assertEqual(summary["cluster"]["total_cards"], 32)


if __name__ == "__main__":
    unittest.main()
