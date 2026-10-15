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
"""Tests for the MindSpeed parser's modules.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_cost_model_parser_mindspeed.py -v
"""
import unittest
from typing import Any, Dict

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hook_base import MemEvalHook, hook_runner
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig

_GLOBAL = {"pp": 2, "mbs": 1, "dp": 2, "tp": 1, "cp": 1, "vpp": 1, "ep": 2, "seqlen": 512, "etp": 0}
_TEXT = {"pipeline_num_layers": [1, 3], "num_moe_experts": 8, "moe_router_topk": 2, "n_shared_exp": 1,
         "moe_intermediate_size": 128, "first_k_dense_replace": 1}


def _module(model_id: str, layers: int, **extra: Any) -> Dict[str, Any]:
    """One MindSpeed submodule."""
    module = {
        "model_id": model_id, "freeze": False, "moe_grouped_gemm": False,
        "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1, "sequence_parallel": False,
        "num_layers": layers, "hidden_size": 256, "ffn_hidden_size": 1024, "vocab_size": 4096,
        "num_attention_heads": 4, "num_query_groups": 0, "kv_channels": 0, "k_lora_rank": 0,
        "q_lora_rank": 0, "qk_rope_head_dim": 0, "num_moe_experts": 1, "moe_router_topk": 1,
        "n_shared_exp": 0, "moe_intermediate_size": 0, "first_k_dense_replace": 0,
        "recompute_num_layers": 1, "params_dtype": "bfloat16", "attention_softmax_in_fp32": True,
        "mtp_num_layers": 0,
    }
    module.update(extra)
    return module


def _vision_language(vit: Dict[str, Any], text: Dict[str, Any]) -> CostModelConfig:
    """A MindSpeed vision tower and text model, with hooks that change nothing."""
    old_registry = MemEvalHook.hook_registry.copy()
    try:
        MemEvalHook.hook_registry = {}

        class _Vit(MemEvalHook):
            """The tower's hook."""

            @staticmethod
            @hook_runner("vit")
            def run_hooks(e: Any) -> None:
                """No change."""
                del e

        class _Text(MemEvalHook):
            """The text model's hook."""

            @staticmethod
            @hook_runner("deepseek_v3")
            def run_hooks(e: Any) -> None:
                """No change."""
                del e

        class _Both(_Vit, _Text):
            """Both submodules."""

        # The config reads its hooks from the registry when it is built.
        return CostModelConfig(Config({"model_id": "multi", "tmp": _GLOBAL, "image_encoder": vit,
                                       "text_decoder": text}), hook_cls=_Both(), framework="mindspeed")
    finally:
        MemEvalHook.hook_registry = old_registry


class TestMindSpeedModules(unittest.TestCase):
    """Each MindSpeed module is priced as its own section states it."""

    def test_a_module_runs_its_own_degrees(self):
        """
        Feature: the MindSpeed parser's parallel dimensions.
        Description: A text module stating TP 4 and EP 4 under global TP 1
            and EP 2.
        Expectation: The module runs TP 4 and EP 4.
        """
        ccfg = _vision_language(_module("vit", 2, pipeline_num_layers=[2, 0]),
                                _module("deepseek_v3", 4, tensor_model_parallel_size=4,
                                        expert_model_parallel_size=4, **_TEXT))
        text = ccfg.mm_ccfgs["deepseek_v3"]
        self.assertEqual((text.t, text.ep), (4, 4))

    def test_one_module_is_the_model(self):
        """
        Feature: a MindSpeed config of one module.
        Description: The DeepSeek-shaped text model alone.
        Expectation: The config is that model, and its memory is priced.
        """
        evaluator = EvaluatorV2(Config({"model_id": "multi", "tmp": _GLOBAL,
                                        "text_decoder": _module("deepseek_v3", 4, **_TEXT)}),
                                framework="mindspeed", log_level=0)
        got = (evaluator.ccfg.multimodal, evaluator.ccfg.n_lay, evaluator.ccfg.n_exp, evaluator.ccfg.model_name)
        self.assertEqual(got, (False, 4, 8, "deepseek_v3"))
        self.assertGreater(evaluator.estimate_peak(), 0)


if __name__ == "__main__":
    unittest.main()
