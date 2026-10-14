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
import dataclasses
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch, MagicMock

import yaml

from hyper_parallel.auto_parallel.config_adapter._normalized_config import (
    NormalizedConfig,
)
from hyper_parallel.auto_parallel.config_adapter import _search_runner as sr
from hyper_parallel.auto_parallel.config_adapter import read_hp_yaml_config
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import HYPER_SELECTIVE_REC_OP
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
_MOCK_OP = MagicMock(acronym="OP")
_MOCK_DIMS = MagicMock(
    DP=_MOCK_DP, TP=_MOCK_TP, PP=_MOCK_PP,
    CP=_MOCK_CP, EP=_MOCK_EP, MBN=_MOCK_MBN, OP=_MOCK_OP,
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
        _MOCK_OP: overrides.get("dp_shard", 1),
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
        self.assertIn("training", result)
        self.assertIn("accelerator", result)
        self.assertIn("fsdp_config", result)
        self.assertIn("dataset", result)
        self.assertIn("config_overrides", result["model"])
        self.assertIn("global_batch_size", result["training"])

    def test_fixed_dim_in_accelerator(self):
        """Fixed dims are written into accelerator."""
        runner = self._get_runner()
        config = _make_full_config()
        config.constraint["fixed_tp_degree"] = 4
        result = runner._build_hp_yaml_dict(config)
        self.assertEqual(result["accelerator"]["tp_size"], 4)

    def test_search_dim_first_candidate_as_placeholder(self):
        """Search dims use the first candidate as placeholder."""
        runner = self._get_runner()
        config = _make_full_config()
        config.search_space["tensor_parallel_degree"] = [1, 2, 4, 8]
        result = runner._build_hp_yaml_dict(config)
        self.assertEqual(result["accelerator"]["tp_size"], 1)

    def test_recompute_mapped(self):
        """recompute_strategy maps to activation_checkpoint."""
        runner = self._get_runner()
        config = _make_full_config()
        config.estimator["recompute_strategy"] = "full"
        result = runner._build_hp_yaml_dict(config)
        self.assertEqual(
            result["activation_checkpoint"]["mode"],
            "full",
        )

    def test_visual_seq_len_propagated(self):
        """A declared visual token count reaches the cost-model yaml."""
        runner = self._get_runner()
        config = _make_full_config()
        config.model_spec["visual_seq_len"] = 2304
        result = runner._build_hp_yaml_dict(config)
        self.assertEqual(result["context"]["visual_seq_len"], 2304)

    def test_visual_seq_len_absent_omitted(self):
        """Without one, the parser is left to derive it."""
        runner = self._get_runner()
        config = _make_full_config()
        config.model_spec.pop("visual_seq_len", None)
        result = runner._build_hp_yaml_dict(config)
        self.assertNotIn("visual_seq_len", result.get("context", {}))

    def test_cp_algo_propagated(self):
        """cp_algo in estimator is written to accelerator.context_parallel_algo."""
        runner = self._get_runner()
        config = _make_full_config()
        config.estimator["cp_algo"] = "ulysses_cp"
        result = runner._build_hp_yaml_dict(config)
        self.assertEqual(
            result["accelerator"]["context_parallel_algo"],
            "ulysses_cp",
        )

    def test_cp_algo_absent_omitted(self):
        """When cp_algo is absent, accelerator has no context_parallel_algo key."""
        runner = self._get_runner()
        config = _make_full_config()
        config.estimator.pop("cp_algo", None)
        result = runner._build_hp_yaml_dict(config)
        self.assertNotIn("context_parallel_algo", result["accelerator"])


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
            "data_parallel_shard_degree": [1],
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

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_machine_mod"
    )
    def test_device_names_map_to_their_devices(self, mock_get_hw):
        """Ascend's chip names map to their Atlas series, a code to itself."""
        mock_hard = MagicMock()
        mock_get_hw.return_value = mock_hard
        runner = self._get_runner()
        got = {}
        for name in ("ascend", "ascend910b", "Ascend910_93", "ascend910c", "A3", "a3", "V100"):
            config = _make_full_config()
            config.cluster_spec["device_type"] = name
            runner._build_machine(config)  # pylint: disable=protected-access
            got[name] = mock_hard.Machine.call_args[0][1]
        self.assertEqual(got, {
            "ascend": "A2", "ascend910b": "A2", "Ascend910_93": "A3", "ascend910c": "A3",
            "A3": "A3", "a3": "A3", "V100": "V100",
        })


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
        result = runner._format_result(entry, _make_full_config())
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
        result = runner._format_result(entry, _make_full_config())
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["pp"], 4)
        self.assertEqual(result["dp_shard"], 1)
        self.assertEqual(result["dp_replicate"], 2)

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_a_shard_spans_dp_and_cp(self, _):
        """
        Feature: _format_result under CP.
        Description: A strategy at DP 4 and CP 2 sharding over 8 ranks, then
            over 2.
        Expectation: The shard spans the DP x CP domain of 8 ranks the
            trainer shards over, where it was cut back to DP's 4, and the
            trainer replicates the rest: 1 and 4 replicas, where 2 was read.
        """
        got = []
        for shard in (8, 2):
            result = self._get_runner()._format_result(
                _make_scored_entry(dp=4, cp=2, dp_shard=shard), _make_full_config())
            got.append((result["dp_shard"], result["dp_replicate"]))
        self.assertEqual(got, [(8, 1), (2, 4)])


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


class TestMemoryBudget(unittest.TestCase):
    """Tests for _memory_budget_gb and _filter_by_memory."""

    def test_tighter_of_limit_and_device(self):
        """The user's memory_limit_gb wins when it is below the device size."""
        config = _make_full_config()
        self.assertEqual(sr._memory_budget_gb(config), 60.0)

    def test_device_used_when_no_limit(self):
        """An unset memory_limit_gb falls back to the device memory."""
        config = _make_full_config()
        config.constraint["memory_limit_gb"] = 0.0
        self.assertEqual(sr._memory_budget_gb(config), 64.0)

    def test_unconstrained_when_neither_set(self):
        """No budget at all reports 0.0, which disables the gate."""
        config = _make_full_config()
        config.constraint["memory_limit_gb"] = 0.0
        config.cluster_spec["device_memory_gb"] = 0.0
        self.assertEqual(sr._memory_budget_gb(config), 0.0)

    def test_over_budget_entries_dropped(self):
        """Entries above the budget are removed, entries below are kept."""
        entries = [_make_scored_entry(mem=50.0 * 1024), _make_scored_entry(mem=61.0 * 1024)]
        kept = sr._filter_by_memory(entries, 60.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0][1], 50.0 * 1024)

    def test_zero_budget_keeps_everything(self):
        """A zero budget disables the gate rather than rejecting everything."""
        entries = [_make_scored_entry(mem=99.0 * 1024)]
        self.assertEqual(len(sr._filter_by_memory(entries, 0.0)), 1)


class TestPostFilterMemory(unittest.TestCase):
    """Tests for the memory gate inside _post_filter."""

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_over_budget_removed(self, _):
        """A strategy above memory_limit_gb never reaches the caller."""
        config = _make_full_config()
        entries = [_make_scored_entry(mem=61.0 * 1024), _make_scored_entry(mem=50.0 * 1024)]
        filtered = sr._post_filter(entries, config)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][1], 50.0 * 1024)

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_no_candidate_match_returns_nothing(self, _):
        """A fitting strategy outside the candidate lists is not returned in their place."""
        config = _make_full_config()
        config.search_space["tensor_parallel_degree"] = [16, 32]
        entries = [
            _make_scored_entry(tp=2, mem=61.0 * 1024),
            _make_scored_entry(tp=4, mem=50.0 * 1024),
        ]
        filtered = sr._post_filter(entries, config)
        self.assertEqual(filtered, [], f"asked TP 16 or 32, got TP {[e[0].dims_val[_MOCK_TP] for e in filtered]}")

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    def test_all_over_budget_returns_empty(self, _):
        """Every entry over budget yields an empty list for the caller to reject."""
        config = _make_full_config()
        entries = [_make_scored_entry(mem=61.0 * 1024)]
        self.assertEqual(sr._post_filter(entries, config), [])


class TestModelSectionIsTheSpec(unittest.TestCase):
    """The model section a search hands ND is the model IR, serialised."""

    def _get_runner(self):
        return sr

    def test_config_is_a_mapping_not_a_path(self):
        """The search passes ND a config, not a temp file to parse back."""
        runner = self._get_runner()
        self.assertFalse(hasattr(runner, "_write_temp_hp_yaml"))
        self.assertFalse(hasattr(runner, "CONFIG_OVERRIDE_FIELDS"))
        data = runner._build_hp_yaml_dict(_make_full_config())
        self.assertIsInstance(data, dict)
        self.assertIn("model", data)
        self.assertIn("training", data)

    def test_overrides_come_from_the_spec(self):
        """Every declared field reaches ND without a hand-kept field list."""
        runner = self._get_runner()
        data = runner._build_hp_yaml_dict(_make_full_config())
        overrides = data["model"]["config_overrides"]
        self.assertEqual(overrides["hidden_size"], 4096)
        self.assertEqual(overrides["num_key_value_heads"], 8)
        self.assertNotIn("name", overrides)
        self.assertEqual(data["model"]["name"], "test-dense")

    def test_a_field_the_old_whitelist_omitted_now_survives(self):
        """shared_expert_intermediate_size is a spec field, so it carries.

        The hand-kept list did not name it, which is how an all-MoE model
        reached the cost model with its shared expert unsized.
        """
        runner = self._get_runner()
        config = _make_full_config()
        config.model_spec.update(
            num_experts=256,
            num_experts_per_tok=8,
            num_shared_experts=1,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
        )
        overrides = runner._build_hp_yaml_dict(config)["model"]["config_overrides"]
        self.assertEqual(overrides["shared_expert_intermediate_size"], 512)

    def test_an_incoherent_model_section_raises(self):
        """A model ND cannot cost is refused here, not priced as a zero."""
        runner = self._get_runner()
        config = _make_full_config()
        config.model_spec["num_key_value_heads"] = 7
        with self.assertRaises(ValueError):
            runner._build_hp_yaml_dict(config)

    def test_max_device_memory_follows_memory_limit(self):
        """The yaml carries the memory budget, so ND prunes against it."""
        runner = self._get_runner()
        config = _make_full_config()
        hp_yaml = runner._build_hp_yaml_dict(config)
        self.assertEqual(hp_yaml["context"]["max_device_memory"], "60.0GB")

    def test_auto_recompute_describes_the_model_fully_recomputed(self):
        """With recompute "auto" or "per_layer", candidates are kept fully recomputed, so the model is described so."""
        runner = self._get_runner()
        config = _make_full_config()
        for strategy in ("auto", "per_layer"):
            config.estimator["recompute_strategy"] = strategy
            self.assertEqual(runner._build_hp_yaml_dict(config)["activation_checkpoint"]["mode"], "full")


# The run a train.yaml states beyond the strategy: the model's dtype, FSDP's
# precision and resharding, an fp32 optimizer, clipping and Ulysses CP; and
# the strategy it states too, which the search replaces.
_STATED_RUN = {
    "model": {"torch_dtype": "bfloat16"},
    "model_init_dtype": "bfloat16",
    "accelerator": {"tp_degree": 8, "tp_size": 8, "context_parallel_algo": "ulysses_cp"},
    "fsdp_config": {
        "dp_shard_size": 8,
        "reshard_after_forward": False,
        "mix_precision": {"param_dtype": "bfloat16"},
    },
    "training": {"global_batch_size": 64, "micro_batch_size": 1, "max_grad_norm": 1.0},
    "optimizer": {"_target_": "hyper_parallel.optim.AdamW", "fp32_main_params": True},
}


def _search_yaml(config: NormalizedConfig) -> dict:
    """The yaml the search hands ND for *config*."""
    return sr._build_hp_yaml_dict(config)  # pylint: disable=protected-access


class TestTheStatedRun(unittest.TestCase):
    """The run the train.yaml states reaches ND beneath the searched strategy."""

    def test_the_run_stays_as_the_train_yaml_states_it(self):
        """What the search does not decide reaches ND as the train.yaml states it."""
        config = _make_full_config(run=_STATED_RUN)
        data = _search_yaml(config)
        self.assertEqual(data["model"]["torch_dtype"], "bfloat16")
        self.assertEqual(data["model"]["config_overrides"]["hidden_size"], 4096)
        self.assertEqual(data["model_init_dtype"], "bfloat16")
        self.assertEqual(data["optimizer"], _STATED_RUN["optimizer"])
        self.assertFalse(data["fsdp_config"]["reshard_after_forward"])
        self.assertEqual(data["fsdp_config"]["mix_precision"], {"param_dtype": "bfloat16"})
        self.assertEqual(data["training"]["max_grad_norm"], 1.0)
        self.assertEqual(data["accelerator"]["context_parallel_algo"], "ulysses_cp")

    def test_the_searched_strategy_replaces_the_stated_one(self):
        """The train.yaml's degrees and batch give way, under either spelling."""
        config = _make_full_config(run=_STATED_RUN)
        data = _search_yaml(config)
        self.assertEqual(data["accelerator"]["tp_size"], 1)
        self.assertNotIn("tp_degree", data["accelerator"])
        self.assertEqual(data["fsdp_config"]["dp_shard_size"], 1)
        self.assertEqual(data["training"]["global_batch_size"], 128)
        self.assertEqual(_STATED_RUN["fsdp_config"]["dp_shard_size"], 8)

    def test_the_train_yamls_pricing_options_reach_nd(self):
        """
        Feature: the train.yaml's context in a search.
        Description: A train.yaml asking for a census and stating a vision
            tower's token count, searched on a cluster that states its
            devices, with a search config that states the token count too.
        Expectation: The census reaches ND; the search's device count and
            the search config's token count win over the train.yaml's.
        """
        config = _make_full_config(run=dict(_STATED_RUN, context={"census": True, "visual_seq_len": 1024}))
        config.model_spec["visual_seq_len"] = 2304
        context = _search_yaml(config)["context"]
        self.assertIs(context["census"], True)
        self.assertEqual(context["visual_seq_len"], 2304)
        self.assertEqual(context["device_num"], _search_yaml(_make_full_config())["context"]["device_num"])
        self.assertEqual(config.run["context"], {"census": True, "visual_seq_len": 1024})

    def test_no_stated_run_changes_nothing(self):
        """A config read from no train.yaml builds the sections it always did."""
        data = _search_yaml(_make_full_config())
        self.assertEqual(
            set(data), {"model", "training", "accelerator", "fsdp_config", "activation_checkpoint", "dataset",
                        "context"},
        )
        self.assertNotIn("torch_dtype", data["model"])


class TestCensusInASearch(unittest.TestCase):
    """A train.yaml's census reaches the estimates of the search it drives."""

    def test_the_spec_carries_the_census_to_nd(self):
        """
        Feature: context.census through the search runner.
        Description: A train.yaml of a two-layer hybrid Qwen3.5-MoE model
            asking for a census, read as a search reads it, and the yaml the
            search hands ND, which states the model as overrides only.
        Expectation: The reader measures each layer kind and the output
            layer; the overrides carry the records, and ND prices with them
            without a checkpoint to read, and without a warning.
        """
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # pylint: disable=C0415
            Qwen3_5MoeTextConfig,
        )
        text = Qwen3_5MoeTextConfig.from_dict({
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 32, "linear_num_key_heads": 2, "linear_key_head_dim": 16,
            "linear_num_value_heads": 4, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
            "vocab_size": 128, "max_position_embeddings": 256,
            "layer_types": ["linear_attention", "full_attention"],
        })
        train = {
            "model": {"pretrained_model_name_or_path": "local/qwen3_5_moe", "torch_dtype": "bfloat16"},
            "training": {"global_batch_size": 4, "micro_batch_size": 1},
            "dataset": {"data_transform": {"max_seq_len": 64}},
            "context": {"census": True},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "train.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(train, handle)
            with patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config", return_value=text):
                config = read_hp_yaml_config(path)
            self.assertEqual(sorted(config.model_spec["activations"]), ["full_attention", "linear_attention"])
            data = _search_yaml(config)
            self.assertNotIn("pretrained_model_name_or_path", data["model"])
            self.assertIn("output_activations", data["model"]["config_overrides"])
            search_path = os.path.join(folder, "search.yaml")
            with open(search_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle)
            with self.assertNoLogs("hyper_parallel.auto_parallel._hf_model_spec", "WARNING"):
                ccfg = EvaluatorV2(search_path, framework="hyper_v2", log_level=0).ccfg
        self.assertEqual(sorted(ccfg.census), ["full_attention", "linear_attention"])
        self.assertEqual(ccfg.output_census.seq_length, 64)

    def test_the_census_spec_reaches_nd(self):
        """
        Feature: context.census_spec through the search runner.
        Description: A train.yaml of a Mixtral of 4 experts asking for the
            census's spec, whose layers run two norms where the family's
            profile states five, read as a search reads it, and the yaml
            the search hands ND.
        Expectation: The reader states the census's op counts, the option
            rides in the run, and ND prices the layers' two norms.
        """
        from transformers import MixtralConfig  # pylint: disable=C0415
        text = MixtralConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             intermediate_size=128, num_local_experts=4, vocab_size=128)
        train = {
            "model": {"pretrained_model_name_or_path": "local/mixtral", "torch_dtype": "bfloat16"},
            "training": {"global_batch_size": 4, "micro_batch_size": 1},
            "dataset": {"data_transform": {"max_seq_len": 64}},
            "context": {"census_spec": True},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "train.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(train, handle)
            with patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config", return_value=text):
                config = read_hp_yaml_config(path)
            self.assertEqual(config.model_spec["ops"]["decoder"]["normOp"], 2)
            self.assertEqual(config.run["context"], {"census_spec": True})
            search_path = os.path.join(folder, "search.yaml")
            with open(search_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(_search_yaml(config), handle)
            ccfg = EvaluatorV2(search_path, framework="hyper_v2", log_level=0).ccfg
        self.assertEqual(ccfg.n_normOp, 2)


class TestSearchStrategies(unittest.TestCase):
    """End-to-end tests for search_strategies with mocked ND."""

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    @patch("hyper_parallel.auto_parallel.sapp_nd.nd.parallelize.Parallelize")
    def test_search_strategies_returns_result(
        self, mock_parallelize_cls, mock_get_dim,
    ):  # pylint: disable=unused-argument
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
        # Without recompute "auto" every candidate is priced fully recomputed,
        # whatever the search yaml says, and the result states it.
        self.assertEqual(config.estimator["recompute_strategy"], "selective")
        self.assertEqual(result["activation_checkpoint"], "full")

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    @patch("hyper_parallel.auto_parallel.sapp_nd.nd.parallelize.Parallelize")
    def test_search_strategies_chooses_the_trainer_mode(
        self, mock_parallelize_cls, mock_get_dim,
    ):  # pylint: disable=unused-argument
        """With recompute "auto" the search chooses among the trainer's modes and reports each layer's own."""
        # pylint: disable=import-outside-toplevel,unused-import
        import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2  # noqa: F401
        from hyper_parallel.auto_parallel.sapp_nd.recompute.candidate import LayerRange, RecomputeChoice
        from hyper_parallel.auto_parallel.sapp_nd.recompute.front import LayerOption
        mock_dims = MagicMock()
        mock_dims.dims_val = {
            _MOCK_DP: 2, _MOCK_TP: 2, _MOCK_PP: 2,
            _MOCK_CP: 1, _MOCK_EP: 1, _MOCK_MBN: 2,
        }
        mock_runner = MagicMock()
        mock_runner.run_generation_to_ordering.return_value = [(mock_dims, 1024.0, 0.05, [])]
        plain = LayerOption(recompute=frozenset(), memory_per_micro_batch=1.0, memory_once=0.0,
                            forward_time=1.0, backward_time=2.0)
        per_layer = RecomputeChoice(ranges=(LayerRange(0, 4, None, plain),), stage_memory=(900.0, 950.0),
                                    stage_savings=(1.0, 1.0))
        mock_runner.recompute_choices = {mock_dims: SimpleNamespace(mode="off")}
        mock_runner.recompute_per_layer.return_value = (per_layer, 0.04)
        mock_parallelize_cls.return_value = mock_runner

        config = _make_full_config()
        config.estimator["recompute_strategy"] = "auto"
        result = sr.search_strategies(config)
        kwargs = mock_parallelize_cls.call_args.kwargs
        self.assertTrue(kwargs["auto_recompute"])
        self.assertEqual(kwargs["recompute_modes"], sr.TRAINER_RECOMPUTE_MODES)
        self.assertEqual(result["activation_checkpoint"], "off")
        self.assertEqual(result["recompute_per_layer"], {
            "score": 0.04, "memory_estimate_mb": 950.0,
            "ranges": [{"first": 0, "count": 4, "kind": None, "recompute": "none"}],
        })

        mock_runner.recompute_choices = {}
        mock_runner.recompute_per_layer.return_value = (None, None)
        result = sr.search_strategies(config)
        self.assertEqual(result["activation_checkpoint"], "full")
        self.assertNotIn("recompute_per_layer", result)

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    @patch("hyper_parallel.auto_parallel.sapp_nd.nd.parallelize.Parallelize")
    def test_search_strategies_offloads_in_a_mode_per_layer(
        self, mock_parallelize_cls, mock_get_dim,
    ):  # pylint: disable=unused-argument
        """
        Feature: search_strategies with offload.
        Description: A mode per layer searched with offload figures, the best
            configuration offloading its first two layers; then one mode for
            every layer searched with offload.
        Expectation: The search offloads over the device's link with the
            stated figures replacing its own, and the result names the
            offloaded layers, with the GiB each moves and the share of what
            it keeps, beside a plan stating them off; the second is refused.
        """
        # pylint: disable=import-outside-toplevel,unused-import
        import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2  # noqa: F401
        from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware as hw
        from hyper_parallel.auto_parallel.sapp_nd.recompute.candidate import LayerRange, RecomputeChoice
        from hyper_parallel.auto_parallel.sapp_nd.recompute.front import LayerOption
        mock_dims = MagicMock()
        mock_dims.dims_val = {
            _MOCK_DP: 2, _MOCK_TP: 2, _MOCK_PP: 2,
            _MOCK_CP: 1, _MOCK_EP: 1, _MOCK_MBN: 2,
        }
        mock_runner = MagicMock()
        mock_runner.run_generation_to_ordering.return_value = [(mock_dims, 1024.0, 0.05, [])]
        plain = LayerOption(recompute=frozenset(), memory_per_micro_batch=4.0 * 2 ** 30, memory_once=0.0,
                            forward_time=1.0, backward_time=2.0)
        offloaded = dataclasses.replace(plain, memory_per_micro_batch=2 ** 30, link_bandwidth=3.0 * 2 ** 30)
        full = LayerOption(recompute=None, memory_per_micro_batch=0.1, memory_once=0.0,
                           forward_time=1.0, backward_time=3.0)
        choice = RecomputeChoice(ranges=(LayerRange(0, 2, None, offloaded, "off"), LayerRange(2, 2, None, full, "full")),
                                 stage_memory=(900.0,), stage_savings=(1.0,))
        mock_runner.recompute_choices = {mock_dims: choice}
        mock_runner.recompute_per_layer.return_value = (None, None)
        mock_parallelize_cls.return_value = mock_runner

        config = _make_full_config()
        config.estimator["recompute_strategy"] = "per_layer"
        config.estimator["recompute_modes"] = ["off", "full"]
        figures = {"gib_per_s": None, "sustained_tflops": None, "overlap": 1.0, "ms_per_unit": 6.4e-11}
        result = sr.search_strategies(config, offload=figures)
        kwargs = mock_parallelize_cls.call_args.kwargs
        self.assertTrue(kwargs["auto_offload"])
        device = sr._build_machine(config).device  # pylint: disable=protected-access
        self.assertEqual(kwargs["host_link"], hw.HostLink.of(device.host_link, figures))
        self.assertEqual(result["offloaded_layers"], ["0-1"])
        self.assertEqual(result["offloaded_gib"], {"0-1": 3.0})
        self.assertEqual(result["offloaded_share"], {"0-1": 0.75})
        self.assertEqual(result["activation_checkpoint"], "full")
        self.assertEqual(result["activation_checkpoint_layers"], {"0-1": "off"})

        config.estimator["recompute_strategy"] = "auto"
        with self.assertRaises(ValueError):
            sr.search_strategies(config, offload=figures)

        # A train.yaml that states a census prices the trainer's selective
        # mode, its policy, as well.
        census = _make_full_config(run={"context": {"census": True}})
        census.estimator["recompute_strategy"] = "auto"
        mock_runner.recompute_choices = {mock_dims: SimpleNamespace(mode="selective")}
        result = sr.search_strategies(census)
        kwargs = mock_parallelize_cls.call_args.kwargs
        self.assertEqual(kwargs["recompute_modes"], sr.TRAINER_CENSUS_MODES)
        self.assertEqual(kwargs["recompute_selective"], HYPER_SELECTIVE_REC_OP)
        self.assertFalse(kwargs["recompute_mode_per_layer"])
        self.assertEqual(result["activation_checkpoint"], "selective")
        self.assertNotIn("activation_checkpoint_layers", result)

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    @patch("hyper_parallel.auto_parallel.sapp_nd.nd.parallelize.Parallelize")
    def test_search_strategies_chooses_a_mode_per_layer(
        self, mock_parallelize_cls, mock_get_dim,
    ):  # pylint: disable=unused-argument
        """
        Feature: search_strategies, recompute "per_layer".
        Description: A search asking for a mode per layer, its best
            candidate running full on its first three layers and off on the
            last five; then the same search with the modes narrowed, and a
            census stated.
        Expectation: The search chooses among the trainer's modes for each
            layer, and the result states the plan as the trainer reads it:
            its mode, and the layers that run another. The narrowed modes
            reach the search whatever the census.
        """
        # pylint: disable=import-outside-toplevel,unused-import
        import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2  # noqa: F401
        from hyper_parallel.auto_parallel.sapp_nd.recompute.candidate import LayerRange, RecomputeChoice
        from hyper_parallel.auto_parallel.sapp_nd.recompute.front import LayerOption
        mock_dims = MagicMock()
        mock_dims.dims_val = {
            _MOCK_DP: 2, _MOCK_TP: 2, _MOCK_PP: 2,
            _MOCK_CP: 1, _MOCK_EP: 1, _MOCK_MBN: 2,
        }
        mock_runner = MagicMock()
        mock_runner.run_generation_to_ordering.return_value = [(mock_dims, 1024.0, 0.05, [])]

        def option(recompute: Optional[frozenset]) -> LayerOption:
            """An option recomputing *recompute*, costs aside."""
            return LayerOption(recompute=recompute, memory_per_micro_batch=1.0, memory_once=0.0,
                               forward_time=1.0, backward_time=2.0)

        plan = RecomputeChoice(
            ranges=(LayerRange(0, 3, None, option(None), "full"), LayerRange(3, 5, None, option(frozenset()), "off")),
            stage_memory=(900.0,), stage_savings=(1.0,))
        mock_runner.recompute_choices = {mock_dims: plan}
        mock_runner.recompute_per_layer.return_value = (None, None)
        mock_parallelize_cls.return_value = mock_runner

        config = _make_full_config()
        config.estimator["recompute_strategy"] = "per_layer"
        result = sr.search_strategies(config)
        kwargs = mock_parallelize_cls.call_args.kwargs
        self.assertTrue(kwargs["auto_recompute"])
        self.assertTrue(kwargs["recompute_mode_per_layer"])
        self.assertEqual(kwargs["recompute_modes"], sr.TRAINER_RECOMPUTE_MODES)
        self.assertEqual(result["activation_checkpoint"], "full")
        self.assertEqual(result["activation_checkpoint_layers"], {"3-7": "off"})

        census = _make_full_config(run={"context": {"census": True}})
        census.estimator.update(recompute_strategy="per_layer", recompute_modes=("off", "full"))
        sr.search_strategies(census)
        self.assertEqual(mock_parallelize_cls.call_args.kwargs["recompute_modes"], ("off", "full"))

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    @patch("hyper_parallel.auto_parallel.sapp_nd.nd.parallelize.Parallelize")
    def test_search_strategies_rejects_over_budget(
        self, mock_parallelize_cls, mock_get_dim,
    ):  # pylint: disable=unused-argument
        """A search whose strategies all exceed the budget raises, never returns one."""
        mock_dims = MagicMock()
        mock_dims.dims_val = {
            _MOCK_DP: 2, _MOCK_TP: 2, _MOCK_PP: 2,
            _MOCK_CP: 1, _MOCK_EP: 1, _MOCK_MBN: 2,
        }
        mock_entry = (mock_dims, 70.0 * 1024, 0.05, [])
        mock_runner = MagicMock()
        mock_runner.run_generation_to_ordering.return_value = [mock_entry]
        mock_parallelize_cls.return_value = mock_runner

        config = _make_full_config()
        with self.assertRaises(ValueError) as ctx:
            sr.search_strategies(config)
        self.assertIn("memory budget", str(ctx.exception))
        self.assertNotIn("candidate lists", str(ctx.exception))

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    @patch("hyper_parallel.auto_parallel.sapp_nd.nd.parallelize.Parallelize")
    def test_search_strategies_refuses_lists_nothing_matches(
        self, mock_parallelize_cls, mock_get_dim,
    ):  # pylint: disable=unused-argument
        """Asked for degrees no fitting strategy has, the search refuses and names what it found."""
        mock_runner = MagicMock()
        mock_runner.run_generation_to_ordering.return_value = [
            _make_scored_entry(tp=2, mem=40.0 * 1024),
            _make_scored_entry(tp=4, mem=30.0 * 1024),
        ]
        mock_parallelize_cls.return_value = mock_runner

        config = _make_full_config()
        config.search_space["tensor_parallel_degree"] = [16, 32]
        with self.assertRaises(ValueError) as ctx:
            sr.search_strategies(config)
        message = str(ctx.exception)
        self.assertIn("candidate lists", message, f"the refusal should name the lists: {message}")
        self.assertIn("tensor_parallel_degree asks [16, 32]", message, f"got: {message}")
        self.assertIn("[2, 4]", message, f"the refusal should give the degrees found: {message}")
        self.assertNotIn("pipeline_parallel_degree", message, f"a list some strategy meets is not named: {message}")

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    @patch("hyper_parallel.auto_parallel.sapp_nd.nd.parallelize.Parallelize")
    def test_a_ratios_file_reaches_the_search(
        self, mock_parallelize_cls, mock_get_dim,
    ):  # pylint: disable=unused-argument
        """estimator["ratios_file"] is the ratios file ND scores with; without it ND scores raw."""
        mock_runner = MagicMock()
        mock_runner.run_generation_to_ordering.return_value = [_make_scored_entry(tp=2)]
        mock_parallelize_cls.return_value = mock_runner

        for stated in (None, "ratios.json"):
            config = _make_full_config()
            if stated:
                config.estimator["ratios_file"] = stated
            sr.search_strategies(config)
            passed = mock_runner.run_generation_to_ordering.call_args.kwargs.get("cache_file")
            self.assertEqual(passed, stated, f"stated {stated}, the search got {passed}")

    @patch(
        "hyper_parallel.auto_parallel.config_adapter._search_runner._get_dim_module",
        return_value=_make_mock_dim_module(),
    )
    @patch("hyper_parallel.auto_parallel.sapp_nd.nd.parallelize.Parallelize")
    def test_search_strategies_refuses_lists_met_only_apart(
        self, mock_parallelize_cls, mock_get_dim,
    ):  # pylint: disable=unused-argument
        """Lists each met by some fitting strategy, never by one, are refused as such."""
        mock_runner = MagicMock()
        mock_runner.run_generation_to_ordering.return_value = [
            _make_scored_entry(tp=2, dp=4, mem=40.0 * 1024),
            _make_scored_entry(tp=8, dp=2, mem=30.0 * 1024),
        ]
        mock_parallelize_cls.return_value = mock_runner

        config = _make_full_config()
        config.search_space["tensor_parallel_degree"] = [2, 4]
        config.search_space["data_parallel_replicate_degree"] = [1, 2]
        with self.assertRaises(ValueError) as ctx:
            sr.search_strategies(config)
        self.assertIn("never all at once", str(ctx.exception), f"got: {ctx.exception}")


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
            # Stated, so the parser has no dtype to warn about.
            "param_init_type": "float32",
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


class TestTheParserReadsTheStatedRun(unittest.TestCase):
    """The real parser prices the run the train.yaml states, on the search's yaml."""

    @staticmethod
    def _parse(config):
        """Parse the search's yaml for *config* with the real parser and return the ccfg."""
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.dump(_search_yaml(config), fh)
        try:
            ccfg = _MinimalCcfg(Config(path))
            CostModelParserHyperV2(ccfg).parse()
        finally:
            os.unlink(path)
        return ccfg

    def test_the_stated_run_is_priced(self):
        """A bf16 model under an fp32 optimizer, gathered layers, clipping and Ulysses CP.

        Without the stated run the search priced fp32 parameters, no fp32
        copy, resharding, no clipping and ring CP.
        """
        ccfg = self._parse(_make_full_config(run=_STATED_RUN))
        self.assertEqual(ccfg.bytes_p, 2)
        self.assertEqual(ccfg.optimizer_state_bytes, 4)
        self.assertEqual(ccfg.main_param_bytes, 4)
        self.assertFalse(ccfg.reshard_params)
        self.assertTrue(ccfg.has_clip)
        self.assertEqual(ccfg.cp_algo, "ulysses_cp")
        self.assertEqual(ccfg.optimizer, "hyper_parallel.optim.AdamW")

    def test_no_stated_run_takes_the_defaults(self):
        """A config read from no train.yaml keeps the parser's defaults.

        It states no dtype and no checkpoint does, so its weights, and the
        optimizer states that follow them, are bfloat16.
        """
        ccfg = self._parse(_make_full_config())
        self.assertEqual(ccfg.bytes_p, 2)
        self.assertEqual(ccfg.optimizer_state_bytes, 2)
        self.assertEqual(ccfg.main_param_bytes, 0)
        self.assertEqual(ccfg.cp_algo, "colossalai_cp")
