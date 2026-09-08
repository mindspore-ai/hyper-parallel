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
"""Unit tests for CostModelParserHyperV2.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_cost_model_parser_hyper.py -v
"""
import unittest
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_hyper import (
    CostModelParserHyperV2,
    custom_vision_tower_hook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ParserCostModelConfig:
    """Minimal cost-model object for parser unit tests.

    Mirrors the helper in ``test_run_nd.py``.  A permissive ``__getattr__``
    returns 0 for any attribute not explicitly set, matching
    ``_CostModVar``'s default behaviour.
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


def _make_ccfg(config_dict: Dict[str, Any]) -> _ParserCostModelConfig:
    """Build a parser target, run ``CostModelParserHyperV2.parse()``, return it."""
    ccfg = _ParserCostModelConfig(config_dict)
    parser = CostModelParserHyperV2(ccfg)
    parser.parse()
    return ccfg


def _dense_overrides(**kw: Any) -> Dict[str, Any]:
    """Default Dense-model config overrides (config_overrides path)."""
    base = {
        "model": {
            "name": "nonexistent",  # triggers fallback to config_overrides
            "config_overrides": {
                "hidden_size": 3584,
                "num_hidden_layers": 8,
                "num_attention_heads": 32,
                "intermediate_size": 18944,
                "vocab_size": 152064,
                "max_position_embeddings": 4096,
            },
        },
        "data": {"max_seq_len": 4096},
        "train": {
            "accelerator": {
                "dp_shard": 1,
                "dp_replicate": 1,
                "tp_degree": 2,
                "pipeline_parallel_degree": 4,
            },
            "micro_batch_size": 1,
            "micro_batch_num": 4,
            "gradient_checkpointing": {"activation_checkpoint": "full"},
            "optimizer": {"max_grad_norm": 1.0},
        },
        "context": {"max_device_memory": "64GB"},
    }
    _deep_update(base, kw)
    return base


def _moe_overrides(**kw: Any) -> Dict[str, Any]:
    """Default MoE-model config overrides."""
    base = {
        "model": {
            "name": "nonexistent",
            "config_overrides": {
                "hidden_size": 3584,
                "num_hidden_layers": 16,
                "num_attention_heads": 32,
                "intermediate_size": 18944,
                "vocab_size": 152064,
                "max_position_embeddings": 4096,
                "num_experts": 64,
                "num_experts_per_tok": 8,
                "num_shared_experts": 1,
                "moe_intermediate_size": 1408,
                "first_k_dense_replace": 2,
                "use_gmm": True,
            },
        },
        "data": {"max_seq_len": 4096},
        "train": {
            "accelerator": {
                "dp_shard": 4,
                "dp_replicate": 1,
                "tp_degree": 2,
                "expert_parallel_degree": 4,
            },
            "micro_batch_size": 1,
            "micro_batch_num": 4,
            "gradient_checkpointing": {"activation_checkpoint": "none"},
        },
        "context": {"max_device_memory": "64GB"},
    }
    _deep_update(base, kw)
    return base


def _mla_overrides(**kw: Any) -> Dict[str, Any]:
    """Default MLA (DeepSeek-V2) config overrides."""
    base = {
        "model": {
            "name": "nonexistent",
            "config_overrides": {
                "hidden_size": 5120,
                "num_hidden_layers": 24,
                "num_attention_heads": 64,
                "intermediate_size": 12288,
                "vocab_size": 102400,
                "max_position_embeddings": 8192,
                "kv_lora_rank": 512,
                "q_lora_rank": 1536,
                "qk_rope_head_dim": 64,
                "num_key_value_heads": 0,
            },
        },
        "data": {"max_seq_len": 8192},
        "train": {
            "accelerator": {
                "dp_shard": 1,
                "tp_degree": 1,
                "pipeline_parallel_degree": 1,
            },
            "micro_batch_size": 1,
            "micro_batch_num": 1,
            "gradient_checkpointing": {"activation_checkpoint": "none"},
        },
    }
    _deep_update(base, kw)
    return base


def _auto_models_config(**kw: Any) -> Dict[str, Any]:
    """Return a minimal current AutoModels Trainer configuration."""
    base = {
        "model": {
            "_target_": (
                "hyper_parallel.models._transformers."
                "HyperAutoModelForCausalLM.from_pretrained"
            ),
            "pretrained_model_name_or_path": "local/model",
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "local_files_only": True,
        },
        "training": {
            "global_batch_size": 64,
            "micro_batch_size": 2,
            "max_grad_norm": 1.0,
        },
        "accelerator": {
            "tp_size": 4,
            "cp_size": 1,
            "ep_size": 2,
            "pp_size": 2,
            "sequence_parallel": True,
        },
        "fsdp_config": {
            "dp_shard_size": 4,
            "mix_precision": {"param_dtype": "bfloat16"},
        },
        "activation_checkpoint": {"mode": "full"},
        "dataset": {"data_transform": {"max_seq_len": 2048}},
        "optimizer": {
            "_target_": "hyper_parallel.components.optim.AdamW",
        },
    }
    _deep_update(base, kw)
    return base


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Recursively update *base* in-place with *overrides*."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCostModelParserHyperV2(unittest.TestCase):
    """Unit tests for ``CostModelParserHyperV2``."""

    # ---- L0: Core (config_overrides) -------------------------------------

    def test_overrides_dense_basic(self):
        """
        Feature: CostModelParserHyperV2 config_overrides path.
        Description: Parse a Dense-model config via config_overrides.
        Expectation: All basic model fields are populated correctly.
        """
        ccfg = _make_ccfg(_dense_overrides())
        self.assertEqual(ccfg.h, 3584)
        self.assertEqual(ccfg.n_lay, 8)
        self.assertEqual(ccfg.a, 32)
        self.assertEqual(ccfg.hff, 18944)
        self.assertEqual(ccfg.v, 152064)
        self.assertEqual(ccfg.s, 4096)
        self.assertEqual(ccfg.n_kv, 32)  # falls back to a
        self.assertEqual(ccfg.dh, 3584 / 32)
        self.assertEqual(ccfg.dc_kv, 0)
        self.assertEqual(ccfg.dc_q, 0)
        self.assertEqual(ccfg.dhr, 0)
        self.assertEqual(ccfg.multiple_of, 256)
        self.assertEqual(ccfg.fdm, 1.0)
        self.assertFalse(ccfg.multimodal)
        self.assertEqual(ccfg.model_name, "nonexistent")

    def test_overrides_moe_model(self):
        """
        Feature: CostModelParserHyperV2 MoE fields.
        Description: Parse an MoE-model config via config_overrides.
        Expectation: MoE-related fields are populated; Dense defaults overridden.
        """
        ccfg = _make_ccfg(_moe_overrides())
        self.assertEqual(ccfg.n_exp, 64)
        self.assertEqual(ccfg.n_chosen_exp, 8)
        self.assertEqual(ccfg.n_shared_exp, 1)
        self.assertEqual(ccfg.hff_exp, 1408)
        self.assertTrue(ccfg.gmm)
        self.assertEqual(ccfg.k_1st_dense, 2)
        self.assertEqual(ccfg.cap_fact, 1)

    def test_overrides_mla_model(self):
        """
        Feature: CostModelParserHyperV2 MLA fields.
        Description: Parse an MLA (DeepSeek-V2) config.
        Expectation: KV/Q compression and rope head dim populated.
        """
        ccfg = _make_ccfg(_mla_overrides())
        self.assertEqual(ccfg.dc_kv, 512)
        self.assertEqual(ccfg.dc_q, 1536)
        self.assertEqual(ccfg.dhr, 64)
        # When num_key_value_heads=0, fallback sets n_kv = a = 64
        self.assertEqual(ccfg.n_kv, 64)
        self.assertEqual(ccfg.dh, 5120 / 64)

    def test_overrides_mtp_depth(self):
        """
        Feature: CostModelParserHyperV2 MTP.
        Description: MTP depth > 0.
        Expectation: n_mtp set, is_mtp_in_offset True.
        """
        ccfg = _make_ccfg(_dense_overrides(
            model={"config_overrides": {"num_hidden_layers": 8, "mtp_depth": 1}},
        ))
        self.assertEqual(ccfg.n_mtp, 1)
        self.assertTrue(ccfg.is_mtp_in_offset)
        # layer_custom_config includes MTP layers
        self.assertEqual(ccfg.layer_custom_config, [(9, None)])

    def test_overrides_seq_len_priority(self):
        """
        Feature: CostModelParserHyperV2 seq_len resolution.
        Description: data.max_seq_len > overrides.max_position_embeddings > overrides.seq_length > 4096.
        Expectation: Highest-priority source wins.
        """
        cases = [
            ({"data": {"max_seq_len": 2048}}, 2048),
            ({}, 4096),  # default
        ]
        for extra, expected in cases:
            cfg = _dense_overrides()
            _deep_update(cfg, extra)
            ccfg = _make_ccfg(cfg)
            self.assertEqual(ccfg.s, expected)

    def test_overrides_missing_kv_heads_fallback(self):
        """
        Feature: CostModelParserHyperV2 KV-head fallback.
        Description: When num_key_value_heads is absent, n_kv = a.
        Expectation: n_kv equals num_attention_heads.
        """
        cfg = _dense_overrides()
        cfg["model"]["config_overrides"].pop("num_key_value_heads", None)
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.n_kv, ccfg.a)

    # ---- L0: Parallelism -------------------------------------------------

    def test_parallelism_basic(self):
        """
        Feature: _parse_parallelism.
        Description: Basic DP, TP, PP with seq_parallel.
        Expectation: d, t, p, cp, ep, sp, vp set correctly.
        """
        ccfg = _make_ccfg(_dense_overrides())
        self.assertEqual(ccfg.d, 1)
        self.assertEqual(ccfg.t, 2)
        self.assertEqual(ccfg.p, 4)
        self.assertEqual(ccfg.cp, 1)
        self.assertEqual(ccfg.ep, 1)
        self.assertEqual(ccfg.sp, 1)  # use_seq_parallel not set -> sp=1
        self.assertEqual(ccfg.vp, 1)
        self.assertEqual(ccfg.pp_sched, "1f1b")
        self.assertFalse(ccfg.has_grad_shard)

    def test_parallelism_fsdp(self):
        """
        Feature: _parse_parallelism — FSDP.
        Description: dp_shard * dp_replicate = d.
        Expectation: d = 8 = 4 * 2.
        """
        cfg = _dense_overrides(train={
            "accelerator": {"dp_shard": 4, "dp_replicate": 2},
        })
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.d, 8)

    def test_parallelism_seq_parallel(self):
        """
        Feature: _parse_parallelism — seq parallel.
        Description: use_seq_parallel=True makes sp = t; False makes sp = 1.
        Expectation: sp matches expectation.
        """
        for use_sp, expected in [(True, 2), (False, 1)]:
            cfg = _dense_overrides(train={
                "accelerator": {
                    "tp_degree": 2,
                    "use_seq_parallel": use_sp,
                },
            })
            ccfg = _make_ccfg(cfg)
            self.assertEqual(ccfg.sp, expected, f"use_seq_parallel={use_sp}")

    def test_parallelism_interleave(self):
        """
        Feature: _parse_parallelism — interleave.
        Description: pp_interleave_num maps to vp.
        Expectation: vp = 3.
        """
        cfg = _dense_overrides(train={
            "accelerator": {"pp_interleave_num": 3},
        })
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.vp, 3)

    def test_parallelism_optimizer_shard(self):
        """
        Feature: _parse_parallelism — optimizer shard.
        Description: enable_parallel_optimizer and optimizer_weight_shard_size.
        Expectation: has_op, op_weight_shard, os_max_shard match inputs.
        """
        cfg = _dense_overrides(train={
            "accelerator": {
                "enable_parallel_optimizer": True,
                "optimizer_weight_shard_size": 4,
            },
        })
        ccfg = _make_ccfg(cfg)
        self.assertTrue(ccfg.has_op)
        self.assertEqual(ccfg.op_weight_shard, 4)
        self.assertEqual(ccfg.os_max_shard, 4)

        cfg2 = _dense_overrides(train={
            "accelerator": {
                "enable_parallel_optimizer": False,
                "optimizer_weight_shard_size": 0,
            },
        })
        ccfg2 = _make_ccfg(cfg2)
        self.assertFalse(ccfg2.has_op)
        self.assertEqual(ccfg2.os_max_shard, ccfg2.d * ccfg2.t)

    def test_parallelism_grad_accum_shard(self):
        """
        Feature: _parse_parallelism — gradient accumulation shard.
        Description: gradient_accumulation_shard flag.
        Expectation: has_grad_shard matches input.
        """
        cfg = _dense_overrides(train={
            "accelerator": {"gradient_accumulation_shard": True},
        })
        ccfg = _make_ccfg(cfg)
        self.assertTrue(ccfg.has_grad_shard)

    def test_parallelism_etp_zero_default(self):
        """
        Feature: Regression — etp default is 0 (not 1).
        Description: When etp is absent, it defaults to 0 so that
            ``config_dp_tp_exp`` does not enter the ``if ccfg.etp`` branch
            and correctly sets ``t_exp = t``.
        Expectation: etp=0, t_exp=t=2.
        """
        cfg = _dense_overrides(train={
            "accelerator": {
                "tp_degree": 2,
                "expert_parallel_degree": 1,
            },
        })
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.etp, 0)
        self.assertEqual(ccfg.t_exp, ccfg.t)

    # ---- L0: Batch -------------------------------------------------------

    def test_batch_explicit(self):
        """
        Feature: _parse_batch.
        Description: micro_batch_size, micro_batch_num, global_batch_size all explicit.
        Expectation: b, m, gbs match inputs.
        """
        cfg = _dense_overrides(train={
            "micro_batch_size": 2,
            "micro_batch_num": 8,
            "global_batch_size": 64,
        })
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.b, 2)
        self.assertEqual(ccfg.m, 8)
        self.assertEqual(ccfg.gbs, 64)

    def test_batch_default_micro_batch_num(self):
        """
        Feature: _parse_batch — default m.
        Description: When micro_batch_num is 0, m defaults to p.
        Expectation: m = p = 4; gbs = b * d * m.
        """
        cfg = _dense_overrides(train={
            "micro_batch_size": 1,
            "micro_batch_num": 0,
            "global_batch_size": 0,
        })
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.m, ccfg.p)
        self.assertEqual(ccfg.gbs, ccfg.b * ccfg.d * ccfg.m)

    # ---- L0: Recompute ---------------------------------------------------

    def test_recompute_modes(self):
        """
        Feature: _parse_recompute.
        Description: Three activation_checkpoint modes.
        Expectation: Correct full_rec / sel_rec / rec_op values.
        """
        cases = [
            ("full", True, False),
            ("selective", False, True),
            ("none", False, False),
        ]
        for ac_mode, expect_full, expect_sel in cases:
            cfg = _dense_overrides(train={
                "gradient_checkpointing": {"activation_checkpoint": ac_mode},
            })
            ccfg = _make_ccfg(cfg)
            self.assertEqual(ccfg.full_rec, expect_full, f"mode={ac_mode}")
            self.assertEqual(ccfg.sel_rec, expect_sel, f"mode={ac_mode}")
            if ac_mode != "none":
                self.assertIsNotNone(ccfg.rec_op)

    # ---- L0: Feature flags -----------------------------------------------

    def test_feature_flags_defaults(self):
        """
        Feature: _parse_feature_flags.
        Description: Default feature-flag values.
        Expectation: has_fa=True, vocab_emb_dp=True, tie_emb_out=False,
            freeze=False, cp_algo='colossalai_cp'.
        """
        ccfg = _make_ccfg(_dense_overrides())
        self.assertTrue(ccfg.has_fa)
        self.assertTrue(ccfg.vocab_emb_dp)
        self.assertFalse(ccfg.tie_emb_out)
        self.assertFalse(ccfg.freeze)
        self.assertEqual(ccfg.cp_algo, "colossalai_cp")

    def test_feature_flags_clip(self):
        """
        Feature: _parse_feature_flags — clip.
        Description: max_grad_norm > 0 => has_clip=True.
        Expectation: has_clip matches.
        """
        cfg_noclip = _dense_overrides(train={
            "optimizer": {"max_grad_norm": 0.0},
        })
        self.assertFalse(_make_ccfg(cfg_noclip).has_clip)

        cfg_clip = _dense_overrides(train={
            "optimizer": {"max_grad_norm": 1.0},
        })
        self.assertTrue(_make_ccfg(cfg_clip).has_clip)

    # ---- L0: Bytes -------------------------------------------------------

    def test_init_bytes(self):
        """
        Feature: _init_bytes.
        Description: Bytes from dtype fields in model section.
        Expectation: bytes_p=4 (float32), bytes_compute=2 (bfloat16),
            bytes_softmax=4 (float32), bytes_grad=4, bytes_os=4, bytes_norm=4.
        """
        cfg = _dense_overrides(model={
            "param_init_type": "float32",
            "compute_dtype": "bfloat16",
            "softmax_compute_type": "float32",
        })
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.bytes_p, 4)
        self.assertEqual(ccfg.bytes_compute, 2)
        self.assertEqual(ccfg.bytes_softmax, 4)
        self.assertEqual(ccfg.bytes_grad, 4)
        self.assertEqual(ccfg.bytes_os, 4)
        self.assertEqual(ccfg.bytes_norm, 4)

    # ---- L0: Shard -------------------------------------------------------

    def test_init_shard_values(self):
        """
        Feature: Regression — _init_shard.
        Description: shard_output_activ defaults to 1 (matching MF parser);
            shard_recompute_input = t if recompute_slice_activation else 1.
        Expectation: shard_embed = t * d, shard_output_activ = 1,
            shard_recompute_input = 1 (no recompute_slice_activation),
            is_shard_mtp_param = True.
        """
        ccfg = _make_ccfg(_dense_overrides())
        self.assertEqual(ccfg.shard_embed, ccfg.t * ccfg.d)
        self.assertEqual(ccfg.shard_output_activ, 1)
        self.assertEqual(ccfg.shard_recompute_input, 1)
        self.assertTrue(ccfg.is_shard_mtp_param)

    # ---- L0: Device capacity ---------------------------------------------

    def test_device_capacity_from_config(self):
        """
        Feature: _resolve_device_capacity.
        Description: context.max_device_memory present.
        Expectation: device_capacity = 54GB.
        """
        cfg = _dense_overrides(context={"max_device_memory": "54GB"})
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.device_capacity.size, Memory.from_string("54GB").size)

    def test_device_capacity_default(self):
        """
        Feature: _resolve_device_capacity — default.
        Description: context section absent.
        Expectation: device_capacity = 64GB.
        """
        cfg = _dense_overrides()
        cfg.pop("context", None)
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.device_capacity.size, Memory.from_string("64GB").size)

    # ---- L0: Flash attention factor --------------------------------------

    def test_flash_attention_factor(self):
        """
        Feature: Flash attention s_fa.
        Description: When has_fa and a>0, s_fa = s/a; else s_fa = s.
        Expectation: s_fa computed correctly.
        """
        ccfg = _make_ccfg(_dense_overrides())
        self.assertTrue(ccfg.has_fa)
        self.assertAlmostEqual(ccfg.s_fa, ccfg.s / ccfg.a)

    # ---- L0: Layer custom config / offset --------------------------------

    def test_layer_custom_config(self):
        """
        Feature: Post-parse layer_custom_config.
        Description: Set to [(n_lay + n_mtp, None)]; offset defaults to
            [0]*pp (uniform balancing) via _init_offset.
        Expectation: Values correct.
        """
        ccfg = _make_ccfg(_dense_overrides())
        expected = [(ccfg.n_lay + ccfg.n_mtp, None)]
        self.assertEqual(ccfg.layer_custom_config, expected)
        self.assertEqual(ccfg.offset, [0] * ccfg.p)

    # ---- L0: config_format -----------------------------------------------

    def test_config_format(self):
        """
        Feature: parse sets config_format.
        Description: parse() sets config_format to "yaml".
        Expectation: config_format == "yaml".
        """
        ccfg = _make_ccfg(_dense_overrides())
        self.assertEqual(ccfg.config_format, "yaml")

    # ---- L1: Additional logic branches -----------------------------------

    def test_parallelism_ep_min_one(self):
        """
        Feature: _parse_parallelism — ep floor.
        Description: ep=0 is clamped to 1.
        Expectation: ep=1.
        """
        cfg = _dense_overrides(train={
            "accelerator": {"expert_parallel_degree": 0},
        })
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.ep, 1)

    def test_batch_global_batch_size_computed(self):
        """
        Feature: _parse_batch — computed gbs.
        Description: When global_batch_size is 0, gbs = b * d * m.
        Expectation: gbs = b * d * m.
        """
        cfg = _dense_overrides(train={
            "micro_batch_size": 2,
            "micro_batch_num": 4,
            "global_batch_size": 0,
        })
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.gbs, ccfg.b * ccfg.d * ccfg.m)

    def test_overrides_mla_zero_kv_heads_fallback(self):
        """
        Feature: CostModelParserHyperV2 — n_kv=0 fallback.
        Description: When num_key_value_heads=0, parser falls back to
            num_attention_heads (a).
        Expectation: n_kv = a = 64, dh = h / a.
        """
        ccfg = _make_ccfg(_mla_overrides())
        self.assertEqual(ccfg.n_kv, ccfg.a)
        self.assertEqual(ccfg.dh, ccfg.h / ccfg.a)

    def test_overrides_moe_without_explicit_hff_exp(self):
        """
        Feature: CostModelParserHyperV2 — MoE without moe_intermediate_size.
        Description: When moe_intermediate_size is absent, hff_exp = hff.
        Expectation: hff_exp = intermediate_size.
        """
        cfg = _moe_overrides()
        cfg["model"]["config_overrides"].pop("moe_intermediate_size", None)
        ccfg = _make_ccfg(cfg)
        self.assertEqual(ccfg.hff_exp, ccfg.hff)

    def test_feature_flags_vp_less_mem(self):
        """
        Feature: _parse_feature_flags — vp_less_mem.
        Description: vp_less_mem is always False.
        Expectation: vp_less_mem is False.
        """
        ccfg = _make_ccfg(_dense_overrides())
        self.assertFalse(ccfg.vp_less_mem)

    def test_bytes_dtype_edge_cases(self):
        """
        Feature: _bytes_from_dtype edge cases.
        Description: Non-standard or absent dtype strings.
        Expectation: float8 -> 1, unknown -> 4.
        """
        parser = CostModelParserHyperV2(_ParserCostModelConfig({}))
        # _bytes_from_dtype regex extracts digits, max(1, digit//8)
        self.assertEqual(parser._bytes_from_dtype("float8"), 1)   # max(1, 8//8)
        self.assertEqual(parser._bytes_from_dtype("bfloat16"), 2)
        self.assertEqual(parser._bytes_from_dtype("float32"), 4)
        self.assertEqual(parser._bytes_from_dtype("float64"), 8)
        self.assertEqual(parser._bytes_from_dtype("unknown"), 4)

    def test_get_cfg_attr(self):
        """
        Feature: _get_cfg_attr static helper.
        Description: Handles Config, YamlObject, and plain objects.
        Expectation: Returns correct value or default.
        """
        parser = CostModelParserHyperV2(_ParserCostModelConfig({}))

        # Config
        c = Config({"a": 1, "b": {"c": 2}})
        self.assertEqual(parser._get_cfg_attr(c, "a", 0), 1)
        self.assertEqual(parser._get_cfg_attr(c, "missing", 99), 99)

        # plain object
        obj = SimpleNamespace(x=10, y=20)
        self.assertEqual(parser._get_cfg_attr(obj, "x", 0), 10)
        self.assertEqual(parser._get_cfg_attr(obj, "z", -1), -1)

        # None input
        self.assertIsNone(parser._get_cfg_attr(None, "anything", None))

    def test_config_to_flat_dict(self):
        """
        Feature: _config_to_flat_dict.
        Description: Recursively flatten Config objects to plain dicts.
        Expectation: Output is a nested dict with no Config/YamlObject instances.
        """
        c = Config({
            "a": 1,
            "b": Config({"c": 2, "d": [3, Config({"e": 4})]}),
            "f": "hello",
        })
        flat = CostModelParserHyperV2._config_to_flat_dict(c)
        self.assertEqual(flat["a"], 1)
        self.assertEqual(flat["b"]["c"], 2)
        self.assertEqual(flat["b"]["d"][1]["e"], 4)
        self.assertEqual(flat["f"], "hello")

    # ---- L2: AutoModels config path (mock) -------------------------------

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_auto_models_config_happy_path(self, mock_get_hf_config):
        """
        Feature: AutoModels Trainer configuration parsing.
        Description: Resolve model metadata through the shared Transformers
            config path and read the new root-level Trainer sections.
        Expectation: Model, topology, batch, recompute, and dtype fields map
            to the cost model without importing the removed Trainer.
        """
        mock_get_hf_config.return_value = SimpleNamespace(
            model_type="qwen3_moe",
            hidden_size=8192,
            num_hidden_layers=80,
            num_attention_heads=64,
            intermediate_size=29568,
            vocab_size=152064,
            max_position_embeddings=8192,
            num_key_value_heads=8,
            kv_lora_rank=0,
            q_lora_rank=0,
            qk_rope_head_dim=0,
            num_experts=1,
            mtp_depth=0,
            multiple_of=256,
            ffn_dim_multiplier=1.0,
            first_k_dense_replace=0,
            moe_intermediate_size=0,
        )

        ccfg = _make_ccfg(_auto_models_config())

        mock_get_hf_config.assert_called_once()
        self.assertEqual(ccfg.model_name, "qwen3_moe")
        self.assertEqual(ccfg.h, 8192)
        self.assertEqual(ccfg.n_lay, 80)
        self.assertEqual(ccfg.a, 64)
        self.assertEqual(ccfg.hff, 29568)
        self.assertEqual(ccfg.v, 152064)
        self.assertEqual(ccfg.s, 2048)
        self.assertEqual(ccfg.d, 4)
        self.assertEqual(ccfg.t, 4)
        self.assertEqual(ccfg.p, 2)
        self.assertEqual(ccfg.ep, 2)
        self.assertEqual(ccfg.sp, 4)
        self.assertEqual(ccfg.b, 2)
        self.assertEqual(ccfg.m, 8)
        self.assertEqual(ccfg.gbs, 64)
        self.assertTrue(ccfg.full_rec)
        self.assertEqual(ccfg.bytes_p, 2)
        self.assertEqual(ccfg.bytes_compute, 2)
        self.assertTrue(ccfg.has_clip)
        self.assertIn("AdamW", ccfg.optimizer)

    def test_auto_models_config_requires_model_metadata(self):
        """
        Feature: AutoModels Trainer configuration validation.
        Description: Parse a model target without a pretrained path or
            explicit config overrides.
        Expectation: A clear ValueError is raised instead of a zero-sized
            cost model.
        """
        config = _auto_models_config(model={
            "pretrained_model_name_or_path": None,
        })
        with self.assertRaisesRegex(ValueError, "model.config_overrides"):
            _make_ccfg(config)


    # ---- L2b: AutoModels field mapping -----------------------------------

    @staticmethod
    def _hf_config(**kw):
        """Return a fake Transformers config with sane dense defaults."""
        base = {
            "model_type": "qwen3_moe", "hidden_size": 2048,
            "num_hidden_layers": 48, "num_attention_heads": 32,
            "num_key_value_heads": 4, "intermediate_size": 6144,
            "vocab_size": 151936, "max_position_embeddings": 40960,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_head_dim_honoured(self, mock_hf):
        """
        Feature: attention head dimension.
        Description: Transformers exposes head_dim explicitly.
        Expectation: ccfg.dh uses it rather than hidden_size / heads.
        """
        mock_hf.return_value = self._hf_config(head_dim=128)
        ccfg = _make_ccfg(_auto_models_config())
        self.assertEqual(ccfg.dh, 128)

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_head_dim_defaults_to_hidden_over_heads(self, mock_hf):
        """
        Feature: attention head dimension fallback.
        Description: A config that declares no head_dim.
        Expectation: ccfg.dh falls back to hidden_size / num_attention_heads.
        """
        mock_hf.return_value = self._hf_config()
        ccfg = _make_ccfg(_auto_models_config())
        self.assertEqual(ccfg.dh, 2048 / 32)

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_mtp_depth_alias(self, mock_hf):
        """
        Feature: MTP depth resolution.
        Description: Transformers spells MTP depth num_nextn_predict_layers.
        Expectation: n_mtp picks up the alias and enters offset balancing.
        """
        mock_hf.return_value = self._hf_config(num_nextn_predict_layers=1)
        ccfg = _make_ccfg(_auto_models_config())
        self.assertEqual(ccfg.n_mtp, 1)
        self.assertTrue(ccfg.is_mtp_in_offset)

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_mtp_depth_prefers_internal_name(self, mock_hf):
        """
        Feature: MTP depth resolution.
        Description: Both the internal and the Transformers spelling present.
        Expectation: The internal mtp_depth wins.
        """
        mock_hf.return_value = self._hf_config(mtp_depth=3, num_nextn_predict_layers=1)
        ccfg = _make_ccfg(_auto_models_config())
        self.assertEqual(ccfg.n_mtp, 3)

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_missing_field_falls_back_to_overrides(self, mock_hf):
        """
        Feature: AutoModels resolution failure.
        Description: A config object missing a required field raises
            AttributeError inside the resolver.
        Expectation: The parser falls back to config_overrides rather than
            propagating the error.
        """
        mock_hf.side_effect = AttributeError("intermediate_size")
        config = _auto_models_config(model={"config_overrides": {
            "hidden_size": 1024, "num_hidden_layers": 4,
            "num_attention_heads": 8, "intermediate_size": 2048,
            "vocab_size": 1000,
        }})
        ccfg = _make_ccfg(config)
        self.assertEqual(ccfg.h, 1024)
        self.assertEqual(ccfg.n_lay, 4)

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_missing_transformers_falls_back_to_overrides(self, mock_hf):
        """
        Feature: optional Transformers dependency.
        Description: transformers is not in requirements.txt, so importing it
            can raise ModuleNotFoundError on a lean install.
        Expectation: The parser falls back to config_overrides rather than
            propagating the ImportError.
        """
        mock_hf.side_effect = ModuleNotFoundError("No module named 'transformers'")
        config = _auto_models_config(model={"config_overrides": {
            "hidden_size": 512, "num_hidden_layers": 2,
            "num_attention_heads": 8, "intermediate_size": 1024,
            "vocab_size": 100,
        }})
        ccfg = _make_ccfg(config)
        self.assertEqual(ccfg.h, 512)
        self.assertEqual(ccfg.n_lay, 2)

    # ---- L2c: vision-language models --------------------------------------

    @staticmethod
    def _vl_config():
        """Return a fake composite vision-language Transformers config."""
        return SimpleNamespace(
            model_type="qwen3_vl_moe",
            text_config=SimpleNamespace(
                hidden_size=2048, num_hidden_layers=24, num_attention_heads=16,
                num_key_value_heads=16, intermediate_size=5632,
                vocab_size=151936, max_position_embeddings=128000,
                head_dim=128, num_experts=60, num_experts_per_tok=4,
                moe_intermediate_size=1408,
            ),
            vision_config=SimpleNamespace(
                hidden_size=1152, depth=27, num_heads=16,
                intermediate_size=4304, out_hidden_size=3584,
                patch_size=16, spatial_merge_size=2,
                num_position_embeddings=2304,
            ),
        )

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_vl_config_builds_submodules(self, mock_hf):
        """
        Feature: multimodal cost model.
        Description: A composite config keeps its language tower under
            text_config and exposes none of its fields at the top level.
        Expectation: The parser reads the text tower, and registers a vision
            submodule alongside it instead of raising AttributeError.
        """
        mock_hf.return_value = self._vl_config()
        ccfg = _make_ccfg(_auto_models_config())

        self.assertTrue(ccfg.multimodal)
        self.assertEqual(ccfg.mm_order, ["vision", "text"])
        self.assertEqual(ccfg.mm_main, "text")
        self.assertEqual(ccfg.n_lay, 0)
        self.assertEqual(set(ccfg.hooks_dict), {"vision", "text"})

        text = ccfg.mm_ccfgs["text"]
        self.assertEqual(text.h, 2048)
        self.assertEqual(text.n_lay, 24)
        self.assertEqual(text.dh, 128)
        self.assertEqual(text.n_exp, 60)
        self.assertEqual(text.s, 2048)

        vision = ccfg.mm_ccfgs["vision"]
        self.assertEqual(vision.h, 1152)
        self.assertEqual(vision.n_lay, 27)
        self.assertEqual(vision.a, 16)
        self.assertEqual(vision.v, 0)
        self.assertEqual(vision.n_exp, 1)
        self.assertEqual(vision.s, 2304 // 4)

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_vl_visual_seq_len_override(self, mock_hf):
        """
        Feature: visual sequence length.
        Description: The true visual token count depends on the dataset, so
            context.visual_seq_len overrides the derived bound.
        Expectation: The vision submodule adopts the override.
        """
        mock_hf.return_value = self._vl_config()
        ccfg = _make_ccfg(_auto_models_config(context={"visual_seq_len": 2304}))
        self.assertEqual(ccfg.mm_ccfgs["vision"].s, 2304)

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_vl_submodules_share_the_strategy(self, mock_hf):
        """
        Feature: multimodal placement.
        Description: combine_partition_multimodal requires both submodules to
            share the pipeline degree.
        Expectation: Vision and text agree on p/vp, and the vision tower is
            placed entirely on the first stage.
        """
        mock_hf.return_value = self._vl_config()
        ccfg = _make_ccfg(_auto_models_config())
        vision, text = ccfg.mm_ccfgs["vision"], ccfg.mm_ccfgs["text"]
        self.assertEqual(vision.p, text.p)
        self.assertEqual(vision.vp, text.vp)
        per_stage = vision.n_lay // vision.p // vision.vp
        self.assertEqual(vision.offset[0], vision.n_lay - per_stage)
        self.assertEqual(vision.offset[1:], [-per_stage] * (vision.p - 1))

    def test_vision_hook_wraps_a_bare_config(self):
        """
        Feature: vision-tower arch hook.
        Description: The hook is handed an evaluator during estimation, but a
            bare cost config when applied directly.
        Expectation: A config without set_ccfg is wrapped, and the tower's
            two-matmul MLP profile is applied either way.
        """
        bare = SimpleNamespace(has_op=False, p=2)
        custom_vision_tower_hook(bare)
        self.assertEqual(bare.n_ffMM, 2)
        self.assertEqual(bare.n_normOp, 2)
        # A ViT block has no gated triple, unlike the language model.
        self.assertEqual(bare.n_ffParamCast, 2)

    def test_capacity_factor_override(self):
        """
        Feature: MoE capacity factor.
        Description: Standalone search configs may pin a capacity factor.
        Expectation: ccfg.cap_fact takes the declared value.
        """
        config = _auto_models_config(model={
            "pretrained_model_name_or_path": None,
            "config_overrides": {
                "hidden_size": 1024, "num_hidden_layers": 4,
                "num_attention_heads": 8, "intermediate_size": 2048,
                "vocab_size": 100, "num_experts": 8,
                "moe_intermediate_size": 512, "capacity_factor": 1.5,
            },
        })
        ccfg = _make_ccfg(config)
        self.assertEqual(ccfg.n_exp, 8)
        self.assertEqual(ccfg.cap_fact, 1.5)

    # ---- L2d: device count -------------------------------------------------

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_device_num_drives_dp(self, mock_hf):
        """
        Feature: cluster size.
        Description: context.device_num states the world size that an
            AutoModels train.yaml cannot express.
        Expectation: d is derived as device_num / (t * p * cp).
        """
        mock_hf.return_value = self._hf_config()
        ccfg = _make_ccfg(_auto_models_config(context={"device_num": 32}))
        self.assertEqual(ccfg.t, 4)
        self.assertEqual(ccfg.p, 2)
        self.assertEqual(ccfg.d, 4)

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_device_num_indivisible_raises(self, mock_hf):
        """
        Feature: cluster size validation.
        Description: A device count not divisible by t * p * cp.
        Expectation: ValueError instead of a silently truncated degree.
        """
        mock_hf.return_value = self._hf_config()
        with self.assertRaisesRegex(ValueError, "not divisible"):
            _make_ccfg(_auto_models_config(context={"device_num": 30}))

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_missing_device_num_assumes_no_replicate(self, mock_hf):
        """
        Feature: cluster size fallback.
        Description: No context.device_num on an AutoModels config.
        Expectation: d falls back to the FSDP shard degree, with a warning
            that the replicate factor is assumed to be one.
        """
        mock_hf.return_value = self._hf_config()
        with self.assertLogs(
            "hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers."
            "cost_model_parser_hyper", level="WARNING",
        ) as captured:
            ccfg = _make_ccfg(_auto_models_config())
        self.assertEqual(ccfg.d, 4)
        self.assertTrue(any("device_num" in m for m in captured.output))

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_vl_offsets_are_two_dimensional_when_interleaved(self, mock_hf):
        """
        Feature: multimodal placement under virtual pipelining.
        Description: With pp_interleave_num > 1 the offset must be shaped
            [vp][p], which is what is_consistent_pp_config accepts.
        Expectation: Both submodules carry per-chunk offset rows, and the
            vision tower still lands on the first stage of the first chunk.
        """
        mock_hf.return_value = self._vl_config()
        ccfg = _make_ccfg(_auto_models_config(
            accelerator={"pp_size": 4, "pp_interleave_num": 2},
        ))
        vision, text = ccfg.mm_ccfgs["vision"], ccfg.mm_ccfgs["text"]
        self.assertEqual(ccfg.vp, 2)

        self.assertEqual(len(text.offset), 2)
        self.assertTrue(all(row == [0, 0, 0, 0] for row in text.offset))

        self.assertEqual(len(vision.offset), 2)
        per_stage = vision.n_lay // 4 // 2
        self.assertEqual(vision.offset[0][0], vision.n_lay - per_stage)
        self.assertEqual(vision.offset[0][1:], [-per_stage] * 3)
        self.assertEqual(vision.offset[1], [-per_stage] * 4)

    # ---- L2e: recompute-slice placement ------------------------------------

    @patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config")
    def test_recompute_slice_activation_precedence(self, mock_hf):
        """
        Feature: recompute_slice_activation lookup.
        Description: The flag belongs to the recompute section; fsdp_config
            and the legacy path remain accepted.
        Expectation: activation_checkpoint wins over both.
        """
        mock_hf.return_value = self._hf_config()
        config = _auto_models_config(
            activation_checkpoint={"mode": "full", "recompute_slice_activation": True},
            fsdp_config={"recompute_slice_activation": False},
        )
        ccfg = _make_ccfg(config)
        self.assertEqual(ccfg.shard_recompute_input, ccfg.t)

    # ---- L3: E2E integration ---------------------------------------------

    def test_full_parse_dense_e2e(self):
        """
        Feature: End-to-end parse — Dense model.
        Description: Full parse() with config_overrides, covering all
            sub-methods.
        Expectation: Key ccfg fields are internally consistent.
        """
        ccfg = _make_ccfg(_dense_overrides(train={
            "accelerator": {
                "dp_shard": 2,
                "dp_replicate": 2,
                "tp_degree": 4,
                "pipeline_parallel_degree": 2,
                "use_seq_parallel": True,
            },
            "micro_batch_size": 2,
            "micro_batch_num": 4,
            "global_batch_size": 64,
        }))

        # model
        self.assertEqual(ccfg.h, 3584)
        self.assertEqual(ccfg.n_lay, 8)
        self.assertEqual(ccfg.a, 32)
        self.assertEqual(ccfg.hff, 18944)
        self.assertEqual(ccfg.v, 152064)
        self.assertEqual(ccfg.s, 4096)
        self.assertEqual(ccfg.n_kv, 32)

        # parallelism (dp_shard=2, dp_replicate=2 => d=4)
        self.assertEqual(ccfg.d, 4)
        self.assertEqual(ccfg.t, 4)
        self.assertEqual(ccfg.p, 2)
        self.assertEqual(ccfg.sp, 4)  # use_seq_parallel=True

        # batch
        self.assertEqual(ccfg.b, 2)
        self.assertEqual(ccfg.m, 4)
        self.assertEqual(ccfg.gbs, 64)

        # recompute
        self.assertTrue(ccfg.full_rec)

        # shard
        self.assertEqual(ccfg.shard_embed, 16)  # t * d = 4 * 4
        self.assertEqual(ccfg.shard_output_activ, 1)
        self.assertEqual(ccfg.shard_recompute_input, 1)

        # bytes
        self.assertEqual(ccfg.bytes_p, 4)
        self.assertEqual(ccfg.bytes_compute, 2)
        self.assertEqual(ccfg.bytes_softmax, 4)

        # MoE defaults (Dense)
        self.assertEqual(ccfg.n_exp, 1)
        self.assertFalse(ccfg.gmm)

        # etp regression
        self.assertEqual(ccfg.etp, 0)
        self.assertEqual(ccfg.t_exp, 4)

        # device
        self.assertEqual(ccfg.device_capacity.size, Memory.from_string("64GB").size)
        self.assertEqual(ccfg.config_format, "yaml")

    def test_full_parse_moe_e2e(self):
        """
        Feature: End-to-end parse — MoE model.
        Description: Full parse() with MoE config_overrides.
        Expectation: MoE fields + t_exp/d_exp calculated.
        """
        ccfg = _make_ccfg(_moe_overrides())

        # MoE
        self.assertEqual(ccfg.n_exp, 64)
        self.assertEqual(ccfg.n_chosen_exp, 8)
        self.assertEqual(ccfg.n_shared_exp, 1)
        self.assertEqual(ccfg.hff_exp, 1408)
        self.assertTrue(ccfg.gmm)
        self.assertEqual(ccfg.k_1st_dense, 2)
        self.assertEqual(ccfg.cap_fact, 1)

        # d_exp, t_exp via config_dp_tp_exp
        # d=4, t=2, ep=4, etp=1 (MoE default)
        # Upstream EP PR changed `if ccfg.etp:` to `if ccfg.etp > 1:`,
        # so etp=1 now falls into the else branch:
        #   t_exp = t = 2,  d_exp = d // ep = 4 // 4 = 1
        self.assertEqual(ccfg.d_exp, 1)
        self.assertEqual(ccfg.t_exp, 2)

    def test_full_parse_mla_e2e(self):
        """
        Feature: End-to-end parse — MLA model.
        Description: Full parse() with MLA config_overrides.
        Expectation: MLA fields + all standard fields.
        """
        ccfg = _make_ccfg(_mla_overrides())

        self.assertEqual(ccfg.dc_kv, 512)
        self.assertEqual(ccfg.dc_q, 1536)
        self.assertEqual(ccfg.dhr, 64)
        self.assertEqual(ccfg.dh, 5120 / 64)
        self.assertEqual(ccfg.h, 5120)
        self.assertEqual(ccfg.n_lay, 24)
        self.assertEqual(ccfg.a, 64)
        self.assertEqual(ccfg.v, 102400)
        self.assertEqual(ccfg.s, 8192)

    # ---- Helper method unit tests ----------------------------------------

    def test_bytes_from_dtype_static(self):
        """
        Feature: _bytes_from_dtype.
        Description: Direct static-method coverage.
        Expectation: Correct byte sizes.
        """
        parser = CostModelParserHyperV2(_ParserCostModelConfig({}))
        self.assertEqual(parser._bytes_from_dtype("float32"), 4)
        self.assertEqual(parser._bytes_from_dtype("bfloat16"), 2)
        self.assertEqual(parser._bytes_from_dtype("float16"), 2)
        # float8 -> regex extracts 8 -> max(1, 8//8) = 1
        self.assertEqual(parser._bytes_from_dtype("float8"), 1)
        self.assertEqual(parser._bytes_from_dtype("float64"), 8)
        self.assertEqual(parser._bytes_from_dtype(""), 4)


if __name__ == "__main__":
    unittest.main()
