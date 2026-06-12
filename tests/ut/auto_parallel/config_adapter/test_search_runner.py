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
"""Unit tests for the ND search runner (_search_runner.py)."""
import os
import unittest
from unittest.mock import patch, MagicMock

from hyper_parallel.auto_parallel.config_adapter._normalized_config import (
    NormalizedConfig,
)


def _make_full_config(**overrides) -> NormalizedConfig:
    """Create a fully populated NormalizedConfig for testing."""
    spec = {
        "name": "test-dense",
        "n_layers": 32,
        "dim": 4096,
        "inter_dim": 11008,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "seq_len": 8192,
        "local_batch_size": 1,
        "compute_dtype": "bfloat16",
    }
    config = NormalizedConfig(
        model_spec=spec,
        cluster_spec={
            "num_nodes": 4,
            "cards_per_node": 8,
            "device_memory_gb": 64.0,
            "device_type": "ascend",
        },
        search_space={
            "data_parallel_replicate_degree": [1, 2, 4],
            "tensor_parallel_degree": [1, 2, 4],
            "pipeline_parallel_degree": [1, 2],
        },
        constraint={
            "global_batch_size": 128,
            "memory_limit_gb": 60.0,
        },
        estimator={
            "type": "symbolic",
            "recompute_strategy": "selective",
        },
        pp_config={
            "pp_degree": 2,
            "stage_partition_mode": "uniform",
        },
    )
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


# Shared mock dims — must be module-level so all test classes see the same objects.
_MOCK_DP = MagicMock(acronym="DP")
_MOCK_TP = MagicMock(acronym="MP")
_MOCK_PP = MagicMock(acronym="PP")
_MOCK_CP = MagicMock(acronym="CP")
_MOCK_EP = MagicMock(acronym="EP")
_MOCK_MBN = MagicMock(acronym="MB")
_MOCK_DIMS = MagicMock(
    DP=_MOCK_DP, TP=_MOCK_TP, PP=_MOCK_PP,
    CP=_MOCK_CP, EP=_MOCK_EP, MBN=_MOCK_MBN,
)


def _make_mock_dim_module():
    """Return a mock sapp_nd dimensions module with shared dim objects."""
    return _MOCK_DIMS


def _make_scored_entry(**overrides) -> tuple:
    """Create a mock scored_space entry (config, mem, score, values)."""
    dims = {
        _MOCK_DP: overrides.get("dp", 2),
        _MOCK_TP: overrides.get("tp", 2),
        _MOCK_PP: overrides.get("pp", 2),
        _MOCK_CP: overrides.get("cp", 1),
        _MOCK_EP: overrides.get("ep", 1),
        _MOCK_MBN: overrides.get("micro_batch_num", 2),
    }
    mock_dims = MagicMock()
    mock_dims.dims_val = dims
    mem = float(overrides.get("mem", 1024.0))
    score = float(overrides.get("score", 0.05))
    return (mock_dims, mem, score, [])


class TestValidateBeforeSearch(unittest.TestCase):
    """Tests for _validate_before_search."""

    def _get_runner(self):
        import hyper_parallel.auto_parallel.config_adapter._search_runner as sr  # pylint: disable=C0415
        return sr

    def test_valid_config_passes(self):
        """Valid config does not raise."""
        runner = self._get_runner()
        config = _make_full_config()
        runner._validate_before_search(config)

    def test_missing_dim_raises(self):
        """Missing 'dim' raises ValueError."""
        runner = self._get_runner()
        config = _make_full_config()
        config.model_spec["dim"] = 0
        with self.assertRaises(ValueError):
            runner._validate_before_search(config)

    def test_empty_cluster_raises(self):
        """Empty cluster_spec raises ValueError."""
        runner = self._get_runner()
        config = _make_full_config()
        config.cluster_spec = {}
        with self.assertRaises(ValueError):
            runner._validate_before_search(config)


class TestBuildHpYamlDict(unittest.TestCase):
    """Tests for _build_hp_yaml_dict."""

    def _get_runner(self):
        import hyper_parallel.auto_parallel.config_adapter._search_runner as sr  # pylint: disable=C0415
        return sr

    def test_basic_structure(self):
        """Output dict contains all required sections."""
        runner = self._get_runner()
        config = _make_full_config()
        result = runner._build_hp_yaml_dict(config)
        self.assertIn("model", result)
        self.assertIn("train", result)
        self.assertIn("data", result)
        self.assertIn("config_overrides", result["model"])
        self.assertIn("accelerator", result["train"])
        self.assertIn("global_batch_size", result["train"])

    def test_fixed_dim_in_accelerator(self):
        """Fixed dims are written into accelerator."""
        runner = self._get_runner()
        config = _make_full_config()
        config.constraint["fixed_tp_degree"] = 4
        result = runner._build_hp_yaml_dict(config)
        self.assertEqual(result["train"]["accelerator"]["tp_degree"], 4)

    def test_search_dim_first_candidate_as_placeholder(self):
        """Search dims use the first candidate as placeholder."""
        runner = self._get_runner()
        config = _make_full_config()
        config.search_space["tensor_parallel_degree"] = [1, 2, 4, 8]
        result = runner._build_hp_yaml_dict(config)
        self.assertEqual(result["train"]["accelerator"]["tp_degree"], 1)

    def test_recompute_mapped(self):
        """recompute_strategy maps to activation_checkpoint."""
        runner = self._get_runner()
        config = _make_full_config()
        config.estimator["recompute_strategy"] = "full"
        result = runner._build_hp_yaml_dict(config)
        self.assertEqual(
            result["train"]["gradient_checkpointing"]["activation_checkpoint"],
            "full",
        )


class TestResolveSearchDimensions(unittest.TestCase):
    """Tests for _resolve_search_dimensions."""

    def _get_runner(self):
        import hyper_parallel.auto_parallel.config_adapter._search_runner as sr  # pylint: disable=C0415
        return sr

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_list_values_returned(self, _):  # pylint: disable=W0613
        """Dimensions with >1 candidate are included."""
        runner = self._get_runner()
        config = _make_full_config()
        config.search_space["tensor_parallel_degree"] = [1, 2, 4]
        dims = runner._resolve_search_dimensions(config)
        dim_names = [d.acronym for d in dims]
        self.assertIn("MP", dim_names)
        self.assertIn("DP", dim_names)

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_no_dims_returns_empty(self, _):  # pylint: disable=W0613
        """When all dims are single-element, search list is empty."""
        runner = self._get_runner()
        config = _make_full_config()
        config.search_space = {
            "data_parallel_replicate_degree": [1],
            "tensor_parallel_degree": [1],
            "pipeline_parallel_degree": [1],
            "context_parallel_degree": [1],
            "expert_parallel_degree": [1],
            "micro_batch_num": [1],
        }
        dims = runner._resolve_search_dimensions(config)
        self.assertEqual(len(dims), 0)


class TestBuildMachine(unittest.TestCase):
    """Tests for _build_machine."""

    def _get_runner(self):
        import hyper_parallel.auto_parallel.config_adapter._search_runner as sr  # pylint: disable=C0415
        return sr

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_machine_mod"
    )
    def test_total_devices_computed(self, mock_get_hw):
        """Total devices = nodes * cards_per_node."""
        mock_hard = MagicMock()
        mock_machine = MagicMock()
        mock_machine._total_devices = 32
        mock_hard.Machine.return_value = mock_machine
        mock_get_hw.return_value = mock_hard

        runner = self._get_runner()
        config = _make_full_config()
        machine = runner._build_machine(config)
        mock_hard.Machine.assert_called_with(32, "A2")


class TestFormatResult(unittest.TestCase):
    """Tests for _format_result."""

    def _get_runner(self):
        import hyper_parallel.auto_parallel.config_adapter._search_runner as sr  # pylint: disable=C0415
        return sr

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_basic_format(self, _):  # pylint: disable=W0613
        """Result dict contains expected keys."""
        runner = self._get_runner()
        entry = _make_scored_entry()
        result = runner._format_result(entry)
        self.assertIn("dp", result)
        self.assertIn("tp", result)
        self.assertIn("memory_estimate_mb", result)
        self.assertIn("score", result)

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_dimension_values(self, _):  # pylint: disable=W0613
        """Dimension values match the entry."""
        runner = self._get_runner()
        entry = _make_scored_entry(tp=2, pp=4)
        result = runner._format_result(entry)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["pp"], 4)


class TestPostFilter(unittest.TestCase):
    """Tests for _post_filter."""

    def _get_runner(self):
        import hyper_parallel.auto_parallel.config_adapter._search_runner as sr  # pylint: disable=C0415
        return sr

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_all_matching_kept(self, _):  # pylint: disable=W0613
        """All entries within candidate list are kept."""
        runner = self._get_runner()
        config = _make_full_config()
        config.search_space["tensor_parallel_degree"] = [2, 4]
        entries = [_make_scored_entry(tp=2), _make_scored_entry(tp=4)]
        filtered = runner._post_filter(entries, config)
        self.assertEqual(len(filtered), 2)

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_non_matching_removed(self, _):  # pylint: disable=W0613
        """Entries outside candidate list are removed."""
        runner = self._get_runner()
        config = _make_full_config()
        config.search_space["tensor_parallel_degree"] = [2, 4]
        entries = [_make_scored_entry(tp=2), _make_scored_entry(tp=8)]
        filtered = runner._post_filter(entries, config)
        self.assertEqual(len(filtered), 1)


class TestWriteTempHpYaml(unittest.TestCase):
    """Tests for _write_temp_hp_yaml."""

    def _get_runner(self):
        import hyper_parallel.auto_parallel.config_adapter._search_runner as sr  # pylint: disable=C0415
        return sr

    def test_temp_file_created(self):
        """Temp file is created and contains valid YAML."""
        runner = self._get_runner()
        config = _make_full_config()
        path = runner._write_temp_hp_yaml(config)
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="utf-8") as fh:
            data = fh.read()
        self.assertIn("model:", data)
        self.assertIn("train:", data)
        os.remove(path)


class TestSearchStrategies(unittest.TestCase):
    """End-to-end tests for search_strategies with mocked ND."""

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    @patch("hyper_parallel.auto_parallel.sapp_nd.nd.parallelize.Parallelize")
    def test_search_strategies_returns_result(self, mock_parallelize_cls, mock_get_dim):
        """search_strategies returns a dict with expected keys."""
        import hyper_parallel.auto_parallel.config_adapter._search_runner as sr  # pylint: disable=C0415

        mock_dims = MagicMock()
        mock_dims.dims_val = {
            _MOCK_DP: 2, _MOCK_TP: 2, _MOCK_PP: 2,
            _MOCK_CP: 1, _MOCK_EP: 1, _MOCK_MBN: 2,
        }
        mock_entry = (mock_dims, 1024.0, 0.05, [])
        mock_runner = MagicMock()
        mock_runner.run_generation_to_ordering.return_value = [mock_entry]
        mock_parallelize_cls.return_value = mock_runner

        config = _make_full_config()
        config.search_space["tensor_parallel_degree"] = [1, 2, 4]
        result = sr.search_strategies(config)
        self.assertIn("tp", result)
        self.assertIn("dp", result)
        self.assertIn("memory_estimate_mb", result)
