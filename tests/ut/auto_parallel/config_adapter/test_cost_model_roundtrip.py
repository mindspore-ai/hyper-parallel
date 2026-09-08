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
"""Round-trip tests for the generated cost-model YAML.

``_build_hp_yaml_dict`` writes the temporary YAML that actually drives the ND
search, and ``CostModelParserHyperV2`` reads it back. Neither side is useful
alone, so these tests pin the contract between them.

How to run this:
    pytest tests/ut/auto_parallel/config_adapter/test_cost_model_roundtrip.py -v
"""
import unittest
from typing import Any
from unittest.mock import patch

from hyper_parallel.auto_parallel.config_adapter._normalized_config import NormalizedConfig
from hyper_parallel.auto_parallel.config_adapter._search_runner import _build_hp_yaml_dict
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_hyper import (
    CostModelParserHyperV2,
)


class _ParserCostModelConfig:
    """Minimal cost-model object for parser tests.

    Mirrors the helper in ``test_cost_model_parser_hyper.py``: a permissive
    ``__getattr__`` returns 0 for anything the parser has not set yet, which
    is how ``_CostModVar`` behaves.
    """

    def __init__(self, input_config: Any = None) -> None:
        """Initialize with an optional config dict."""
        self.config = Config(input_config or {})
        self.hooks_dict = {}
        self.source_code = None

    def __getattr__(self, attr: str) -> int:
        _ = attr
        return 0

    @staticmethod
    def fp_bytes(precision: str) -> int:
        """Return bytes encoded in a precision string."""
        if "16" in precision:
            return 2
        if "32" in precision:
            return 4
        return 0


def _normalized_config() -> NormalizedConfig:
    """Return a fully populated NormalizedConfig for a MoE model."""
    return NormalizedConfig(
        model_spec={
            "name": "qwen3_moe", "hidden_size": 4096, "num_hidden_layers": 32,
            "num_attention_heads": 32, "num_key_value_heads": 8,
            "head_dim": 128, "intermediate_size": 11008, "vocab_size": 128256,
            "max_position_embeddings": 4096, "local_batch_size": 1,
            "compute_dtype": "bfloat16", "num_experts": 64,
            "num_experts_per_tok": 8, "moe_intermediate_size": 1408,
        },
        cluster_spec={"device_memory_gb": 64, "num_nodes": 8, "cards_per_node": 8},
        search_space={
            "data_parallel_shard_degree": [8],
            "tensor_parallel_degree": [2],
            "pipeline_parallel_degree": [2],
        },
        constraint={"global_batch_size": 64, "memory_limit_gb": 0.0},
        estimator={"type": "symbolic", "recompute_strategy": "full"},
        pp_config={
            "pp_degree": 2, "stage_partition_mode": "uniform",
            "micro_batch_num": 8,
        },
    )


def _parse(yaml_dict: dict) -> _ParserCostModelConfig:
    """Run the ND parser over a generated cost-model YAML dict."""
    ccfg = _ParserCostModelConfig(yaml_dict)
    CostModelParserHyperV2(ccfg).parse()
    return ccfg


class TestCostModelRoundTrip(unittest.TestCase):
    """The generated YAML must parse back into the intended cost model."""

    def test_generated_yaml_parses_into_ccfg(self) -> None:
        """
        Feature: search-runner to parser round trip.
        Description: Build the temporary YAML from a NormalizedConfig and
            parse it with the ND parser.
        Expectation: Model, topology, batch and recompute fields survive.
        """
        config = _normalized_config()
        ccfg = _parse(_build_hp_yaml_dict(config))

        self.assertEqual(ccfg.h, 4096)
        self.assertEqual(ccfg.n_lay, 32)
        self.assertEqual(ccfg.a, 32)
        self.assertEqual(ccfg.n_kv, 8)
        self.assertEqual(ccfg.dh, 128)
        self.assertEqual(ccfg.v, 128256)
        self.assertEqual(ccfg.s, 4096)
        self.assertEqual(ccfg.n_exp, 64)
        self.assertEqual(ccfg.hff_exp, 1408)
        self.assertEqual(ccfg.t, 2)
        self.assertEqual(ccfg.p, 2)
        self.assertTrue(ccfg.full_rec)
        self.assertEqual(ccfg.device_capacity.to_gb().size, 64)

    def test_head_dim_survives_the_round_trip(self) -> None:
        """
        Feature: head_dim propagation.
        Description: head_dim is resolved on the loader side and must reach
            the parser through config_overrides.
        Expectation: ccfg.dh is the declared head_dim, not hidden/heads.
        """
        config = _normalized_config()
        yaml_dict = _build_hp_yaml_dict(config)
        self.assertEqual(yaml_dict["model"]["config_overrides"]["head_dim"], 128)
        self.assertEqual(_parse(yaml_dict).dh, 128)

    def test_device_num_reaches_the_parser(self) -> None:
        """
        Feature: cluster size propagation.
        Description: The cluster spec knows the world size; the parser needs
            it to derive the data-parallel degree.
        Expectation: context.device_num is emitted and drives ccfg.d.
        """
        config = _normalized_config()
        yaml_dict = _build_hp_yaml_dict(config)
        self.assertEqual(yaml_dict["context"]["device_num"], 64)
        # 64 devices over t=2, p=2, cp=1.
        self.assertEqual(_parse(yaml_dict).d, 16)

    def test_generated_yaml_never_hits_transformers(self) -> None:
        """
        Feature: offline search.
        Description: The generated YAML carries explicit config_overrides, so
            the parser must never reach the Transformers hub for it.
        Expectation: Parsing succeeds with the HF entry point disabled.
        """
        config = _normalized_config()
        yaml_dict = _build_hp_yaml_dict(config)
        with patch(
            "hyper_parallel.auto_parallel._hf_model_spec._get_hf_config",
            side_effect=AssertionError("the search must not reach the network"),
        ):
            ccfg = _parse(yaml_dict)
        self.assertEqual(ccfg.n_lay, 32)

    def test_recompute_slice_lands_in_the_recompute_section(self) -> None:
        """
        Feature: recompute_slice_activation placement.
        Description: FSDP2Config declares no such field, so it belongs with
            the recompute settings.
        Expectation: The flag is written under activation_checkpoint and the
            parser honours it.
        """
        config = _normalized_config()
        config.model_spec["recompute_slice_activation"] = True
        yaml_dict = _build_hp_yaml_dict(config)
        self.assertNotIn("recompute_slice_activation", yaml_dict["fsdp_config"])
        self.assertTrue(
            yaml_dict["activation_checkpoint"]["recompute_slice_activation"]
        )
        ccfg = _parse(yaml_dict)
        self.assertEqual(ccfg.shard_recompute_input, ccfg.t)


if __name__ == "__main__":
    unittest.main()
