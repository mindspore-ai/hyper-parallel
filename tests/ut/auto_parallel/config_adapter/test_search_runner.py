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
import tempfile
import unittest
from typing import Any, Optional
from unittest.mock import patch, MagicMock

import yaml

from hyper_parallel.auto_parallel.config_adapter._normalized_config import (
    NormalizedConfig,
)
from hyper_parallel.auto_parallel.config_adapter import _search_runner as sr
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_hyper import (
    CostModelParserHyperV2,
)


def _make_full_config(**overrides) -> NormalizedConfig:
    """Create a fully populated NormalizedConfig for testing."""
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
        config.model_spec["hidden_size"] = 0
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

    def test_cp_algo_propagated(self):
        """cp_algo in estimator is written to accelerator.context_parallel_algo."""
        runner = self._get_runner()
        config = _make_full_config()
        config.estimator["cp_algo"] = "ulysses_cp"
        result = runner._build_hp_yaml_dict(config)
        self.assertEqual(
            result["train"]["accelerator"]["context_parallel_algo"],
            "ulysses_cp",
        )

    def test_cp_algo_absent_omitted(self):
        """When cp_algo is absent, accelerator has no context_parallel_algo key."""
        runner = self._get_runner()
        config = _make_full_config()
        config.estimator.pop("cp_algo", None)
        result = runner._build_hp_yaml_dict(config)
        self.assertNotIn("context_parallel_algo", result["train"]["accelerator"])


class TestResolveSearchDimensions(unittest.TestCase):
    """Tests for _resolve_search_dimensions."""

    def _get_runner(self):
        return sr

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_list_values_returned(self, _):
        """Dimensions with >1 candidate are included."""
        runner = self._get_runner()
        config = _make_full_config()
        config.search_space["tensor_parallel_degree"] = [1, 2, 4]
        dims, candidate_dims = runner._resolve_search_dimensions(config)
        dim_names = [d.acronym for d in dims]
        self.assertIn("MP", dim_names)
        self.assertIn("DP", dim_names)
        self.assertIn(
            next(d for d in dims if d.acronym == "MP"), candidate_dims
        )

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_no_dims_returns_empty(self, _):
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
        dims, candidate_dims = runner._resolve_search_dimensions(config)
        self.assertEqual(len(dims), 0)
        self.assertEqual(len(candidate_dims), 0)


class TestBuildMachine(unittest.TestCase):
    """Tests for _build_machine."""

    def _get_runner(self):
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
        runner._build_machine(config)
        mock_hard.Machine.assert_called_with(32, "A2")


class TestFormatResult(unittest.TestCase):
    """Tests for _format_result."""

    def _get_runner(self):
        return sr

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_basic_format(self, _):
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
    def test_dimension_values(self, _):
        """Dimension values match the entry."""
        runner = self._get_runner()
        entry = _make_scored_entry(tp=2, pp=4)
        result = runner._format_result(entry)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["pp"], 4)


class TestPostFilter(unittest.TestCase):
    """Tests for _post_filter."""

    def _get_runner(self):
        return sr

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_all_matching_kept(self, _):
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
    def test_non_matching_removed(self, _):
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
    def test_search_strategies_returns_result(self, mock_parallelize_cls, mock_get_dim):  # pylint: disable=unused-argument
        """search_strategies returns a dict with expected keys."""
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


# ── Minimal HyperV2 yaml builder ──────────────────────────────────────────

def _write_minimal_hp_yaml(cp_algo=None, cp_degree=2):
    """Write a minimal HyperV2 train.yaml and return the file path."""
    accel = {
        "dp_shard": 2,
        "tp_degree": 2,
        "context_parallel_degree": cp_degree,
    }
    if cp_algo is not None:
        accel["context_parallel_algo"] = cp_algo

    content = {
        "model": {
            "name": "test-tiny",
            "config_overrides": {
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 512,
                "vocab_size": 1024,
            },
        },
        "train": {
            "global_batch_size": 4,
            "micro_batch_size": 1,
            "accelerator": accel,
            "gradient_checkpointing": {"activation_checkpoint": "none"},
        },
        "data": {"max_seq_len": 128},
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        yaml.dump(content, fh)
    return path


class _MinimalCcfg:
    """Minimal ccfg that satisfies CostModelParserHyperV2 without circular imports.

    Provides the attributes the parser writes to and the ``__getattr__`` fallback
    that ``CostModelConfig`` uses for unrecognised fields.
    """

    def __init__(self, config: Any) -> None:
        """Initialise with a Config object and sensible defaults."""
        self.config = config
        self.hooks_dict: dict = {}
        self.source_code: Optional[str] = None

    def __getattr__(self, attr):
        _ = attr
        return 0

    @staticmethod
    def fp_bytes(precision: Any) -> int:
        """Return bytes per element for the given precision string."""
        if "16" in str(precision):
            return 2
        if "32" in str(precision):
            return 4
        return 0


class TestCostModelParserCpAlgoReal(unittest.TestCase):
    """Real end-to-end tests: CostModelParserHyperV2 reads cp_algo from yaml.

    These tests instantiate the real parser with a lightweight ccfg object
    (no ``_CostModVar`` — avoids the circular-import chain in
    ``_cost_model_variables`` → ``generate_partitions``).  They verify
    that ``ccfg.cp_algo`` is set correctly after ``parse()``.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Bind real Config and CostModelParserHyperV2 once for all tests."""
        cls._Config = Config
        cls._Parser = CostModelParserHyperV2

    def _parse_yaml(self, yaml_path):
        """Parse a yaml with the real parser and return the ccfg."""
        ccfg = _MinimalCcfg(self._Config(yaml_path))
        self._Parser(ccfg).parse()
        return ccfg

    def test_ulysses_cp_from_yaml(self):
        """context_parallel_algo=ulysses_cp flows through to ccfg.cp_algo."""
        path = _write_minimal_hp_yaml(cp_algo="ulysses_cp")
        try:
            ccfg = self._parse_yaml(path)
            self.assertEqual(ccfg.cp_algo, "ulysses_cp")
        finally:
            os.unlink(path)

    def test_colossalai_cp_from_yaml(self):
        """context_parallel_algo=colossalai_cp flows through to ccfg.cp_algo."""
        path = _write_minimal_hp_yaml(cp_algo="colossalai_cp")
        try:
            ccfg = self._parse_yaml(path)
            self.assertEqual(ccfg.cp_algo, "colossalai_cp")
        finally:
            os.unlink(path)

    def test_default_cp_algo_when_absent(self):
        """When context_parallel_algo is absent, ccfg.cp_algo defaults to colossalai_cp."""
        path = _write_minimal_hp_yaml(cp_algo=None, cp_degree=2)
        try:
            ccfg = self._parse_yaml(path)
            self.assertEqual(ccfg.cp_algo, "colossalai_cp")
        finally:
            os.unlink(path)

    def test_warning_emitted_when_cp_gt_1_and_algo_absent(self):
        """When cp>1 and context_parallel_algo is absent, a warning is logged."""
        path = _write_minimal_hp_yaml(cp_algo=None, cp_degree=2)
        parser_logger_name = (
            "hyper_parallel.auto_parallel.sapp_nd.nd.common."
            "framework_parsers.cost_model_parser_hyper"
        )
        try:
            with self.assertLogs(parser_logger_name, level="WARNING") as log_ctx:
                ccfg = self._parse_yaml(path)
            self.assertEqual(ccfg.cp_algo, "colossalai_cp")
            warning_text = "\n".join(log_ctx.output)
            self.assertIn("context_parallel_algo not set", warning_text)
            self.assertIn("ulysses_cp", warning_text)
        finally:
            os.unlink(path)

    def test_no_warning_when_cp_is_1_and_algo_absent(self):
        """When cp=1 and algo absent, no warning is logged (algo is moot)."""
        path = _write_minimal_hp_yaml(cp_algo=None, cp_degree=1)
        parser_logger_name = (
            "hyper_parallel.auto_parallel.sapp_nd.nd.common."
            "framework_parsers.cost_model_parser_hyper"
        )
        try:
            # assertNoLogs requires Python 3.10+; use assertLogs with try/except fallback.
            with self.assertLogs(parser_logger_name, level="WARNING") as log_ctx:
                self._parse_yaml(path)
            # If we get here, a warning WAS logged — fail the test.
            self.fail(
                "Expected no warning when cp=1, but got: "
                + "\n".join(log_ctx.output)
            )
        except AssertionError as exc:
            # assertLogs raises AssertionError("no logs of level WARNING or higher")
            # when nothing is logged — that is the expected outcome here.
            if "no logs" not in str(exc).lower():
                raise
        finally:
            os.unlink(path)

    def test_hybrid_cp_from_yaml(self):
        """context_parallel_algo=hybrid_cp flows through to ccfg.cp_algo."""
        path = _write_minimal_hp_yaml(cp_algo="hybrid_cp")
        try:
            ccfg = self._parse_yaml(path)
            self.assertEqual(ccfg.cp_algo, "hybrid_cp")
        finally:
            os.unlink(path)

    def test_cp_degree_one_no_warning_on_absent_algo(self):
        """When cp=1 and no algo, default is still colossalai_cp (no warning path)."""
        path = _write_minimal_hp_yaml(cp_algo=None, cp_degree=1)
        try:
            ccfg = self._parse_yaml(path)
            self.assertEqual(ccfg.cp_algo, "colossalai_cp")
        finally:
            os.unlink(path)
