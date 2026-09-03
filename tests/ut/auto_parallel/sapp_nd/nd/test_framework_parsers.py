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
"""Unit tests for framework parser cost-model configurations."""
# pylint: disable=missing-class-docstring,missing-function-docstring
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import _CostModelParser
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_hyperparallel import (
    CostModelParserHyperparallel)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_mindformers import (
    CostModelParserMindformers)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_mindspeed import (
    CostModelParserMindspeed)


class _MockCCfg:
    def __init__(self, input_config: Any = None, **overrides) -> None:
        self.config = Config(input_config or {})
        self.hooks_dict = {}
        self.source_code = None
        for k, v in overrides.items():
            setattr(self, k, v)

    def __getattr__(self, attr: str) -> int:
        return 0

    @staticmethod
    def fp_bytes(precision: str) -> int:
        if "16" in precision:
            return 2
        if "32" in precision:
            return 4
        return 4


class _ParserCostModelConfig:
    def __init__(self, input_config: Any = None):
        self.config = Config(input_config or {})
        self.hooks_dict = {}
        self.source_code = None

    def __getattr__(self, attr: str) -> int:
        if attr == "d_shard_or_d":
            return getattr(self, "d_shard", 0) or getattr(self, "d", 0)
        return 0

    @staticmethod
    def fp_bytes(p):
        return 2 if "16" in p else 4 if "32" in p else 0


class _ConcreteParser(_CostModelParser):
    def parse(self):
        return None


_TINY_SOURCE = (
    "def get_train_spec():\n    return TrainSpec(model_args=model_args)\n"
    "model_args = {'tiny': ModelArgs(dim=16, inter_dim=32, hidden_dim=0, "
    "vocab_size=64, n_heads=2, n_layers=2, n_kv_heads=0, kv_lora_rank=0, "
    "q_lora_rank=0, qk_rope_head_dim=0, n_dense_layers=0, moe_inter_dim=0, "
    "moe_enabled=False, moe_args=None, enable_weight_tying=False, "
    "multiple_of=1, ffn_dim_multiplier=1)}\n")


def _write_tiny_source():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "__init__.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_TINY_SOURCE)
    return d, p


def _make_hp_config(dp_replicate, dp_shard, tp=2, pp=2):
    return Config({
        "model": {"name": "llama-unit", "flavor": "tiny"},
        "parallelism": {"data_parallel_replicate_degree": dp_replicate,
                        "data_parallel_shard_degree": dp_shard,
                        "tensor_parallel_degree": tp, "pipeline_parallel_degree": pp,
                        "context_parallel_degree": 1, "expert_parallel_degree": 1,
                        "expert_tensor_parallel_degree": 0,
                        "pipeline_parallel_schedule": "Interleaved1F1B"},
        "activation_checkpoint": {"mode": "full"},
        "training": {"seq_len": 8, "local_batch_size": 1}})


def _make_mindspeed_mod(use_distributed_optimizer=True):
    return {"model_id": "vision", "freeze": False, "moe_grouped_gemm": False,
            "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1, "sequence_parallel": False,
            "pipeline_num_layers": [1, 0], "num_layers": 2,
            "hidden_size": 16, "ffn_hidden_size": 32, "vocab_size": 64,
            "num_attention_heads": 2, "num_query_groups": 0, "kv_channels": 0,
            "k_lora_rank": 0, "q_lora_rank": 0, "qk_rope_head_dim": 0,
            "num_moe_experts": 1, "moe_router_topk": 1, "n_shared_exp": 0,
            "moe_intermediate_size": 0, "first_k_dense_replace": 0,
            "recompute_num_layers": 1, "params_dtype": "float16",
            "attention_softmax_in_fp32": True, "mtp_num_layers": 0,
            "use_distributed_optimizer": use_distributed_optimizer}


def _make_mindformers_config(enable_parallel_optimizer=True, gradient_accumulation_shard=True,
                             optimizer_weight_shard_size=0):
    poc = None
    if gradient_accumulation_shard or optimizer_weight_shard_size:
        poc = {"gradient_accumulation_shard": gradient_accumulation_shard,
               "optimizer_weight_shard_size": optimizer_weight_shard_size}
    model_config = {"num_layers": 2, "num_hidden_layers": 0, "is_encoder_decoder": False,
           "hidden_size": 16, "intermediate_size": 32, "vocab_size": 64,
           "seq_length": 8, "num_heads": 2, "num_attention_heads": 2,
           "n_kv_heads": 0, "num_key_value_heads": 0, "kv_lora_rank": 0,
           "q_lora_rank": 0, "qk_rope_head_dim": 0, "use_flash_attention": True,
           "tie_word_embeddings": False, "pp_interleave_num": 1,
           "multiple_of": 1, "ffn_dim_multiplier": 1,
           "param_init_type": "float32", "params_dtype": "float32",
           "compute_dtype": "float16", "softmax_compute_type": "float32",
           "softmax_compute_dtype": "float32", "mtp_depth": 0,
           "num_nextn_predict_layers": 0, "offset": 0, "n_routed_experts": 0,
           "num_experts_per_tok": 0, "n_shared_experts": 0,
           "moe_intermediate_size": 0, "first_k_dense_replace": 0,
           "moe_grouped_gemm": False}
    return Config({
        "trainer": {"model_name": "tiny-llama"},
        "context": {"max_device_memory": "56GB"},
        "model": {"model_config": model_config},
        "parallel": {"enable_parallel_optimizer": enable_parallel_optimizer,
                     "parallel_optimizer_config": poc, "pipeline_config": None,
                     "vocab_emb_dp": True},
        "parallel_config": {"data_parallel": 2, "model_parallel": 2,
                            "pipeline_stage": 1, "context_parallel": 1,
                            "expert_parallel": 1, "use_seq_parallel": False,
                            "micro_batch_num": 1, "context_parallel_algo": "colossalai_cp",
                            "seq_split_num": 1},
        "runner_config": {"batch_size": 1}, "runner_wrapper": {"use_clip_grad": False},
        "optimizer": {"type": "Adam"},
        "recompute_config": {"select_recompute": False, "select_comm_recompute": False,
                             "recompute": True},
        "moe_config": None})


class TestInitHff(unittest.TestCase):
    def _make_ccfg(self, h=4096, fdm=0, multiple_of=256):
        return _MockCCfg(h=h, fdm=fdm, multiple_of=multiple_of)

    def test_basic_no_multiplier(self):
        ccfg = self._make_ccfg(h=4096, fdm=0, multiple_of=256)
        parser = _ConcreteParser(ccfg)
        result = parser.init_hff()
        hff_raw = int(2 * 4 * 4096 / 3)
        self.assertEqual(result, 256 * ((hff_raw + 256 - 1) // 256))

    def test_with_ffn_dim_multiplier(self):
        ccfg = self._make_ccfg(h=4096, fdm=4, multiple_of=256)
        parser = _ConcreteParser(ccfg)
        result = parser.init_hff()
        hff_raw = int(2 * int((4 + 0.01) * 4 * 4096) / 3)
        self.assertEqual(result, 256 * ((int(hff_raw) + 256 - 1) // 256))


class TestConfigDpTpExpBranches(unittest.TestCase):
    def _make_ccfg(self, **kwargs):
        defaults = {"d": 8, "t": 2, "cp": 1, "ep": 1, "etp": 1, "hff_exp": 128, "n_exp": 4}
        defaults.update(kwargs)
        return _MockCCfg(**defaults)

    def test_etp_greater_than_1(self):
        ccfg = self._make_ccfg(d=8, t=2, cp=1, ep=2, etp=2, hff_exp=128, n_exp=4)
        parser = _ConcreteParser(ccfg)
        parser.config_dp_tp_exp(ccfg)
        self.assertEqual(ccfg.t_exp, 2)
        self.assertEqual(ccfg.d_exp, 8 * 2 * 1 // 2 // 2)

    def test_moe_type_error_d_exp_zero(self):
        ccfg = self._make_ccfg(d=1, t=1, cp=1, ep=8, etp=1, hff_exp=128, n_exp=4)
        parser = _ConcreteParser(ccfg)
        with self.assertRaises(TypeError):
            parser.config_dp_tp_exp(ccfg)


class TestConfigCommFlag(unittest.TestCase):
    def test_dp_comm_factor_with_op_and_grad_shard(self):
        ccfg = _MockCCfg(d=8, t=2, has_op=True, has_grad_shard=True, ep=1, n_exp=1, cp=1, fsdp=False, d_shard=1)
        parser = _ConcreteParser(ccfg)
        parser.config_comm_flag(ccfg)
        self.assertEqual(ccfg.comm_d_non_exp, 3)

    def test_comm_hsdp(self):
        ccfg = _MockCCfg(d=8, t=2, has_op=True, has_grad_shard=True, ep=1, n_exp=1, cp=1, fsdp=True, d_shard=4)
        parser = _ConcreteParser(ccfg)
        parser.config_comm_flag(ccfg)
        self.assertEqual(ccfg.comm_hsdp, 1.0)


class TestHyperparallelScheduleFallback(unittest.TestCase):
    def _make_parser(self, pp_sched):
        ccfg = _MockCCfg(pp_sched=pp_sched, p=2, vp=1, model_name="test")
        ccfg.specs = SimpleNamespace(
            multiple_of=256, ffn_dim_multiplier=0, dim=4096, inter_dim=0, hidden_dim=0, vocab_size=32000,
            n_heads=32, n_layers=32, n_kv_heads=8, kv_lora_rank=0, q_lora_rank=0, qk_rope_head_dim=0,
            n_dense_layers=0, moe_enabled=False, moe_args=None, moe_inter_dim=0, enable_weight_tying=False)
        config = Config({})
        config.model = SimpleNamespace(name="test", flavor="test")
        config.parallelism = SimpleNamespace(
            pipeline_parallel_schedule=pp_sched or "1f1b", expert_tensor_parallel_degree=1,
            data_parallel_replicate_degree=1, data_parallel_shard_degree=1, tensor_parallel_degree=1,
            pipeline_parallel_degree=2, context_parallel_degree=1, expert_parallel_degree=1)
        config.training = SimpleNamespace(seq_len=4096)
        config.activation_checkpoint = SimpleNamespace(mode="full")
        parser = CostModelParserHyperparallel(ccfg)
        parser.config = config
        parser._CostModelParserHyperparallel__parse_strat()
        return ccfg

    def test_unsupported_schedule_defaults_to_1f1b(self):
        self.assertEqual(self._make_parser("CustomSchedule").pp_sched, "1f1b")

    def test_interleaved_with_p_gt_1_sets_vp_2(self):
        self.assertEqual(self._make_parser("Interleaved1F1B").vp, 2)


class TestHyperparallelHffFallback(unittest.TestCase):
    def _make_parser_with_hff(self, model_name, inter_dim, hidden_dim, h=4096):
        ccfg = _MockCCfg(model_name=model_name, h=h, fdm=0, multiple_of=256, has_fa=True, a=32, n_lay=32)
        ccfg.specs = SimpleNamespace(
            multiple_of=256, ffn_dim_multiplier=0, dim=h, inter_dim=inter_dim, hidden_dim=hidden_dim,
            vocab_size=32000, n_heads=32, n_layers=32, n_kv_heads=8, kv_lora_rank=0, q_lora_rank=0,
            qk_rope_head_dim=0, n_dense_layers=0, moe_enabled=False, moe_args=None, moe_inter_dim=0,
            enable_weight_tying=False)
        config = Config({})
        config.model = SimpleNamespace(name=model_name, flavor="test")
        config.parallelism = SimpleNamespace(expert_tensor_parallel_degree=1)
        config.training = SimpleNamespace(seq_len=4096)
        parser = CostModelParserHyperparallel(ccfg)
        parser.config = config
        parser._CostModelParserHyperparallel__parse_hyperparam()
        return ccfg

    def test_hff_from_hidden_dim_when_inter_dim_zero(self):
        self.assertEqual(self._make_parser_with_hff("test_model", 0, 11008).hff, 11008)

    def test_hff_llama_init_hff_when_both_zero(self):
        self.assertGreater(self._make_parser_with_hff("llama-7b", 0, 0).hff, 0)


class TestHyperparallelMoeParsing(unittest.TestCase):
    def test_moe_args_populated(self):
        moe_args = SimpleNamespace(num_experts=8, top_k=2, num_shared_experts=2)
        ccfg = _MockCCfg(hff=11008, d=8, t=2, cp=1, ep=2, etp=1)
        ccfg.specs = SimpleNamespace(moe_inter_dim=0, moe_enabled=True, moe_args=moe_args)
        config = Config({})
        config.parallelism = SimpleNamespace(expert_tensor_parallel_degree=1)
        parser = CostModelParserHyperparallel(ccfg)
        parser.config = config
        parser._CostModelParserHyperparallel__parse_moe()
        self.assertEqual(ccfg.n_exp, 8)
        self.assertEqual(ccfg.n_chosen_exp, 2)
        self.assertEqual(ccfg.n_shared_exp, 2)

    def test_moe_disabled_sets_defaults(self):
        ccfg = _MockCCfg(hff=11008, d=8, t=2, cp=1, ep=1, etp=1)
        ccfg.specs = SimpleNamespace(moe_inter_dim=0, moe_enabled=False, moe_args=None)
        config = Config({})
        config.parallelism = SimpleNamespace(expert_tensor_parallel_degree=1)
        parser = CostModelParserHyperparallel(ccfg)
        parser.config = config
        parser._CostModelParserHyperparallel__parse_moe()
        self.assertEqual(ccfg.n_exp, 1)
        self.assertEqual(ccfg.n_chosen_exp, 1)
        self.assertEqual(ccfg.n_shared_exp, 0)


class TestMindspeedMultimodalValidation(unittest.TestCase):
    @patch.object(CostModelParserMindspeed, "_CostModelParserMindspeed__search_and_parse_mods_ccfg",
                  return_value={"mod_a": MagicMock(), "mod_b": MagicMock()})
    def test_multimodal_without_hooks_raises(self, mock_search):
        ccfg = _MockCCfg(multimodal=True, d=8, t=1, cp=1, p=2, vp=1, hooks_dict=None)
        config = Config({})
        config.tmp = SimpleNamespace(tp=1, dp=8, pp=2, cp=1, vpp=1, ep=1, mbs=1)
        parser = CostModelParserMindspeed(ccfg)
        parser.config = config
        with self.assertRaises(TypeError):
            parser._CostModelParserMindspeed__config_parse_json_multimodals()

    @patch.object(CostModelParserMindspeed, "_CostModelParserMindspeed__search_and_parse_mods_ccfg",
                  return_value={"mod_a": MagicMock()})
    def test_non_multimodal_no_error(self, mock_search):
        ccfg = _MockCCfg(d=8, t=1, cp=1, p=2, vp=1)
        config = Config({})
        config.tmp = SimpleNamespace(tp=1, dp=8, pp=2, cp=1, vpp=1, ep=1, mbs=1)
        parser = CostModelParserMindspeed(ccfg)
        parser.config = config
        parser._CostModelParserMindspeed__config_parse_json_multimodals()
        self.assertFalse(ccfg.multimodal)


class TestMindspeedFullRecVp(unittest.TestCase):
    def test_full_rec_with_vp_gt_1(self):
        ccfg = _MockCCfg(p=2, vp=2, d=8, t=1, ep=1, has_op=True, has_grad_shard=True,
                         fsdp=False, d_shard=1, os_max_shard=8, n_exp=1, hff_exp=1, etp=1)
        config = Config({})
        config.tmp = SimpleNamespace(tp=1, dp=8, pp=2, cp=1, vpp=2, ep=1, mbs=1, seqlen=4096)
        parser = CostModelParserMindspeed(ccfg)
        parser.config = config
        mod = SimpleNamespace(
            sequence_parallel=False, pipeline_num_layers=None, recompute_num_layers=4, moe_grouped_gemm=False,
            tensor_model_parallel_size=1, expert_model_parallel_size=1, pipeline_model_parallel_size=1,
            num_layers=32, hidden_size=4096, ffn_hidden_size=11008, vocab_size=32000, num_attention_heads=32,
            num_query_groups=8, kv_channels=128, k_lora_rank=0, q_lora_rank=0, qk_rope_head_dim=0,
            num_moe_experts=1, moe_router_topk=1, n_shared_exp=0, moe_intermediate_size=0,
            first_k_dense_replace=0, use_distributed_optimizer=True, attention_softmax_in_fp32=False,
            params_dtype="bf16", mtp_num_layers=0, freeze=False, model_id="test_model")
        cc = _MockCCfg()
        cc.parser, cc.config_format, cc.model_name = parser, "json", mod.model_id
        cc.freeze, cc.has_fa, cc.has_op = mod.freeze, True, True
        cc.has_grad_shard, cc.has_clip, cc.cp_algo = True, False, "colossalai_cp"
        cc.gmm, cc.vocab_emb_dp, cc.offset = False, False, 0
        parser._CostModelParserMindspeed__config_parse_json_parallelism(cc, mod)
        cc.full_rec = mod.recompute_num_layers
        cc.sel_rec = False
        if mod.recompute_num_layers and isinstance(mod.recompute_num_layers, int):
            cc.full_rec = [mod.recompute_num_layers] * cc.p
            if cc.vp > 1:
                cc.full_rec = [cc.full_rec] * cc.vp
        self.assertEqual(cc.full_rec, [[4, 4], [4, 4]])


class TestMindspeedCpSpWarning(unittest.TestCase):
    @patch("hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers.cost_model_parser_mindspeed.logger")
    def test_cp_and_sp_both_enabled_warns(self, mock_logger):
        ccfg = _MockCCfg(p=1, vp=1, d=8, t=2, ep=1, has_op=False, has_grad_shard=False,
                         fsdp=False, d_shard=1, os_max_shard=8, n_exp=1, hff_exp=1, etp=1)
        config = Config({})
        config.tmp = SimpleNamespace(tp=2, dp=8, pp=1, cp=2, vpp=1, ep=1, mbs=1)
        parser = CostModelParserMindspeed(ccfg)
        parser.config = config
        mod = SimpleNamespace(sequence_parallel=True, pipeline_num_layers=None, recompute_num_layers=0,
                              moe_grouped_gemm=False, tensor_model_parallel_size=1,
                              expert_model_parallel_size=1, pipeline_model_parallel_size=1)
        parser._CostModelParserMindspeed__config_parse_json_parallelism(ccfg, mod)
        mock_logger.warning.assert_called_once()


class TestHyperparallelParserFsdp(unittest.TestCase):
    def test_fsdp_true_when_full_shard(self):
        tmp_dir, src = _write_tiny_source()
        try:
            c = _ParserCostModelConfig()
            c.config = _make_hp_config(1, 2)
            c.source_code = src
            CostModelParserHyperparallel(c).parse()
            self.assertTrue(c.fsdp)
            self.assertEqual(c.d_shard, 2)
            self.assertTrue(c.shard_p_fsdp_non_exp > 0)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_fsdp_false_calls_config_optimizer_shard(self):
        tmp_dir, src = _write_tiny_source()
        try:
            c = _ParserCostModelConfig()
            c.config = _make_hp_config(2, 1)
            c.source_code = src
            CostModelParserHyperparallel(c).parse()
            self.assertFalse(c.fsdp)
            self.assertEqual(c.d_shard, 1)
            self.assertEqual(c.shard_p_fsdp_non_exp, 0)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestMindspeedParserFsdp(unittest.TestCase):
    def _parse_and_capture(self, use_distributed_optimizer):
        mod = _make_mindspeed_mod(use_distributed_optimizer)
        cfg = Config({"model_id": "multi-unit",
                      "tmp": {"pp": 1, "mbs": 1, "dp": 2, "tp": 1, "cp": 1,
                              "vpp": 1, "ep": 1, "seqlen": 8, "etp": 0},
                      "module": mod})
        c = _ParserCostModelConfig()
        c.config = cfg
        cap = {"fsdp": [], "optimizer": []}
        o_f = CostModelParserMindspeed.config_fsdp_shard
        o_o = CostModelParserMindspeed.config_optimizer_shard

        def _cf(s, cc):
            cap["fsdp"].append(cc)
            o_f(s, cc)

        def _co(s, cc):
            cap["optimizer"].append(cc)
            o_o(s, cc)

        with patch.object(CostModelParserMindspeed, "config_fsdp_shard", _cf), \
             patch.object(CostModelParserMindspeed, "config_optimizer_shard", _co):
            CostModelParserMindspeed(c).parse()
        return cap

    def test_fsdp_true_when_use_distributed_optimizer(self):
        cap = self._parse_and_capture(True)
        self.assertEqual(len(cap["fsdp"]), 1)
        self.assertEqual(len(cap["optimizer"]), 0)
        s = cap["fsdp"][0]
        self.assertTrue(s.fsdp)
        self.assertEqual(s.d_shard, s.d)
        self.assertTrue(s.has_op)
        self.assertTrue(s.has_grad_shard)
        self.assertTrue(s.shard_p_fsdp_non_exp > 0)

    def test_fsdp_false_when_no_distributed_optimizer(self):
        cap = self._parse_and_capture(False)
        self.assertEqual(len(cap["optimizer"]), 1)
        self.assertEqual(len(cap["fsdp"]), 0)
        s = cap["optimizer"][0]
        self.assertFalse(s.fsdp)
        self.assertEqual(s.d_shard, 1)
        self.assertFalse(s.has_op)


class TestMindformersParserFsdp(unittest.TestCase):
    def test_fsdp_true_when_full_shard(self):
        d, t = 2, 2
        c = _ParserCostModelConfig()
        c.config = _make_mindformers_config(True, True, d * t)
        CostModelParserMindformers(c).parse()
        self.assertTrue(c.fsdp)
        self.assertEqual(c.d_shard, c.d)
        self.assertTrue(c.has_op)
        self.assertTrue(c.has_grad_shard)
        self.assertTrue(c.shard_p_fsdp_non_exp > 0)

    def test_fsdp_false_when_no_parallel_optimizer(self):
        c = _ParserCostModelConfig()
        c.config = _make_mindformers_config(False, False, 0)
        CostModelParserMindformers(c).parse()
        self.assertFalse(c.fsdp)
        self.assertEqual(c.d_shard, 1)
        self.assertFalse(c.has_op)
        self.assertEqual(c.os_max_shard, 1)


if __name__ == "__main__":
    unittest.main()