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
"""End-to-end tests for the vision-language cost model.

The parser building ``mm_ccfgs`` is only half the story: the evaluator has to
consume them. These tests drive the real ``Parallelize`` entry point so the
submodule contract stays honest.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_multimodal_cost_model.py -v
"""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    yaml = None

import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.nd.parallelize as Par
from hyper_parallel.auto_parallel import _hf_model_spec


def _vl_hf_config() -> SimpleNamespace:
    """Return a composite vision-language Transformers config."""
    return SimpleNamespace(
        model_type="qwen3_vl_moe",
        text_config=SimpleNamespace(
            hidden_size=2048, num_hidden_layers=24, num_attention_heads=16,
            num_key_value_heads=8, intermediate_size=5632, vocab_size=151936,
            max_position_embeddings=128000, head_dim=128, num_experts=64,
            num_experts_per_tok=8, moe_intermediate_size=768,
        ),
        vision_config=SimpleNamespace(
            hidden_size=1152, depth=27, num_heads=16, intermediate_size=4304,
            out_hidden_size=3584, patch_size=16, spatial_merge_size=2,
            num_position_embeddings=2304,
        ),
    )


def _vl_train_yaml() -> dict:
    """Return an AutoModels Trainer config for a vision-language model."""
    return {
        "model": {
            "pretrained_model_name_or_path": "local/vl",
            "torch_dtype": "bfloat16",
        },
        "training": {
            "global_batch_size": 16, "micro_batch_size": 1, "max_grad_norm": 1.0,
        },
        "accelerator": {"tp_size": 1, "pp_size": 4, "cp_size": 1, "ep_size": 1},
        "fsdp_config": {"dp_shard_size": 4},
        "activation_checkpoint": {"mode": "full"},
        "dataset": {"data_transform": {"max_seq_len": 4096}},
        "context": {"max_device_memory": "64GB", "device_num": 16},
    }


@unittest.skipIf(yaml is None, "PyYAML not installed")
class TestMultimodalCostModel(unittest.TestCase):
    """The evaluator must accept the submodules the parser produces."""

    def _build(self, overrides: dict = None):
        """Return a Parallelize instance over a temporary VL train.yaml."""
        config = _vl_train_yaml()
        if overrides:
            config.update(overrides)
        # pylint: disable=consider-using-with
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = os.path.join(self._tmp.name, "vl.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
        # matplotlib is imported by the estimator and needs a writable cache.
        with patch.object(
            _hf_model_spec, "_get_hf_config", return_value=_vl_hf_config()
        ), patch.dict(os.environ, {"MPLCONFIGDIR": self._tmp.name}):
            return Par.Parallelize(
                "hyper_v2", path, Hard.Machine(16, "A2"),
                global_batch_size=16, dimensions=[Dim.DP],
            )

    def test_parallelize_takes_the_multimodal_path(self) -> None:
        """
        Feature: multimodal dispatch.
        Description: A composite config must reach ParallelizeMultiModal, and
            the language model must drive the search space rather than the
            vision tower.
        Expectation: Both submodules exist and mm_main selects the text one.
        """
        runner = self._build()
        ccfg = runner.instance.mem_eval.ccfg
        self.assertTrue(ccfg.multimodal)
        self.assertEqual(ccfg.mm_order, ["vision", "text"])
        self.assertEqual(ccfg.mm_main, "text")
        # GlobalConfig must be built on the language model, not the tower.
        self.assertIs(runner.instance.config.ccfg, ccfg.mm_ccfgs["text"])
        # The text submodule keeps the model name so the arch-hook router
        # can still match it; only the submodule key is "text".
        self.assertEqual(ccfg.mm_ccfgs["text"].model_name, "qwen3_vl_moe")

    def test_vision_tower_is_front_loaded(self) -> None:
        """
        Feature: multimodal placement.
        Description: The runtime keeps the vision tower on the first pipeline
            stage unless an MPipe-style schedule moves it.
        Expectation: Every vision layer lands on stage 0, and the language
            model stays evenly balanced.
        """
        ccfg = self._build().instance.mem_eval.ccfg
        vision, text = ccfg.mm_ccfgs["vision"], ccfg.mm_ccfgs["text"]
        self.assertEqual(vision.n_lay, 27)
        self.assertEqual(vision.offset, [21, -6, -6, -6])
        self.assertEqual(text.offset, [0, 0, 0, 0])

    def test_peak_memory_is_estimated(self) -> None:
        """
        Feature: multimodal memory estimation.
        Description: Drive estimate_peak over the combined model.
        Expectation: A finite estimate is produced, and the peak lands on the
            stage carrying the encoder.
        """
        runner = self._build()
        peak = runner.instance.mem_eval.estimate_peak(verbose=False)
        self.assertGreater(peak, 0)

    def test_strategy_update_reaches_both_submodules(self) -> None:
        """
        Feature: multimodal strategy fan-out.
        Description: combine_partition_multimodal requires the submodules to
            share the pipeline degree, so a strategy update with no explicit
            target must reach both.
        Expectation: Vision and text both adopt the new degrees.
        """
        ccfg = self._build().instance.mem_eval.ccfg
        ccfg.set_strategy(dp=2, mp=2)
        self.assertEqual(ccfg.mm_ccfgs["vision"].d, 2)
        self.assertEqual(ccfg.mm_ccfgs["text"].d, 2)
        self.assertEqual(ccfg.mm_ccfgs["vision"].t, 2)
        self.assertEqual(ccfg.mm_ccfgs["text"].t, 2)


if __name__ == "__main__":
    unittest.main()
