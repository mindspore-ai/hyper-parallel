# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0
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
"""``build_meta_model`` must build remote-code checkpoints the native way.

A checkpoint that ships an ``auto_map`` (the moonshot repos do) must not drag
remote code into the codegen meta-model build unless the YAML explicitly opts
in via ``trust_remote_code: true``.  Kimi K2.6 is the canonical case:
``kimi_k25`` resolves natively in AutoConfig but has no
``AutoModelForCausalLM`` mapping entry (VLM architecture), so the build must
dispatch to ``AutoModelForImageTextToText`` — never to
``auto_map["AutoModelForCausalLM"]``, whose remote code imports CUDA-only
fla-core and APIs absent from the installed transformers (the four-layer
blast on 910B3 hosts).
"""
from __future__ import annotations

import json
import os
import unittest

from hyper_parallel.codegen.plan.derive import build_meta_model
from hyper_parallel.codegen.spec.types import (
    CodegenSpec,
    ParallelDimsSpec,
    SourceSpec,
)

_REMOTE_CODE = 'raise RuntimeError("remote code must not be imported")\n'


def _make_kimi_k26_ckpt(root: str) -> str:
    """Write a cropped K2.6 checkpoint that also ships remote code."""
    from transformers.models.kimi_k25 import Kimi_K25Config

    text = dict(
        model_type="deepseek_v3", architectures=["DeepseekV3ForCausalLM"],
        hidden_size=128, intermediate_size=256, moe_intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=8,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=1,
        first_k_dense_replace=1, norm_topk_prob=False, scoring_func="sigmoid",
        hidden_act="silu", max_position_embeddings=4096, rms_norm_eps=1e-5,
        attention_bias=False, num_nextn_predict_layers=0,
        q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=16,
        qk_rope_head_dim=8, v_head_dim=16, vocab_size=512,
    )
    vision = dict(
        model_type="kimi_k25_vision", hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=128, image_size=56,
        patch_size=14, num_channels=3,
    )
    cfg = Kimi_K25Config(text_config=text, vision_config=vision)
    cfg.architectures = ["KimiK25ForConditionalGeneration"]

    ckpt = os.path.join(root, "kimi_k26_ckpt")
    os.makedirs(ckpt, exist_ok=True)
    cfg.save_pretrained(ckpt)
    with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    raw["auto_map"] = {
        "AutoConfig": "configuration_kimi_k25.Kimi_K25Config",
        "AutoModelForCausalLM": "modeling_kimi_k25.Kimi_K25ForConditionalGeneration",
        "AutoModel": "modeling_kimi_k25.Kimi_K25Model",
    }
    with open(os.path.join(ckpt, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2, default=str)
    for name in ("configuration_kimi_k25.py", "modeling_kimi_k25.py"):
        with open(os.path.join(ckpt, name), "w", encoding="utf-8") as fh:
            fh.write(_REMOTE_CODE)
    return ckpt


def _make_remote_only_ckpt(root: str) -> str:
    """Write a remote-code-only checkpoint (``kimi_linear`` shape)."""
    ckpt = os.path.join(root, "kimi_linear_ckpt")
    os.makedirs(ckpt, exist_ok=True)
    with open(os.path.join(ckpt, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "architectures": ["KimiLinearForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_kimi.KimiLinearConfig",
                    "AutoModelForCausalLM": "modeling_kimi.KimiLinearForCausalLM",
                },
                "model_type": "kimi_linear",
                "num_hidden_layers": 2, "hidden_size": 64,
                "num_attention_heads": 4, "vocab_size": 256,
                "intermediate_size": 128,
            },
            fh,
        )
    for name in ("configuration_kimi.py", "modeling_kimi.py"):
        with open(os.path.join(ckpt, name), "w", encoding="utf-8") as fh:
            fh.write(_REMOTE_CODE)
    return ckpt


def _spec(ckpt: str, architecture: str, model_type: str) -> CodegenSpec:
    return CodegenSpec(
        source=SourceSpec(
            model_name_or_path=ckpt,
            architecture=architecture,
            model_type=model_type,
        ),
        parallel_dims=ParallelDimsSpec(),
    )


class TestBuildMetaModelNativeFirst(unittest.TestCase):
    """The meta-model build prefers native code unless the YAML opts in."""

    def test_kimi_k26_dispatches_to_vlm_class_without_remote_code(self):
        """K2.6 with an auto_map checkpoint builds natively on the meta device.

        The checkpoint ships moonshot remote code and an ``auto_map`` pointing
        at it; the default (``trust_remote_code=False``) build must resolve
        the native config and dispatch the VLM mapping entry, never executing
        the remote files (they raise on import in this test).
        """
        import tempfile

        with tempfile.TemporaryDirectory(prefix="kimi_k26_") as root:
            ckpt = _make_kimi_k26_ckpt(root)
            model = build_meta_model(
                _spec(ckpt, "KimiK25ForConditionalGeneration", "kimi_k25")
            )
            self.assertEqual(
                type(model).__module__,
                "transformers.models.kimi_k25.modeling_kimi_k25",
            )
            self.assertEqual(
                type(model).__name__, "Kimi_K25ForConditionalGeneration"
            )
            self.assertGreater(sum(1 for _ in model.parameters()), 0)

    def test_remote_only_model_fails_fast_without_trust_remote_code(self):
        """A remote-code-only checkpoint fails fast, never imports remote code.

        ``kimi_linear`` has no native transformers registration, so the
        default build must fail instead of importing the remote files (which
        raise on import in this test).  The failure surfaces as
        transformers' trust-remote-code ``ValueError``; when an earlier test
        in the same process already touched the auto_map machinery,
        transformers' dynamic-module loader instead trips its own PyO3
        re-init guard (``ImportError``) — either way the remote code never
        successfully loads, which is the contract under test.
        """
        import tempfile

        with tempfile.TemporaryDirectory(prefix="kimi_lin_") as root:
            ckpt = _make_remote_only_ckpt(root)
            with self.assertRaises((ValueError, ImportError)) as ctx:
                build_meta_model(
                    _spec(ckpt, "KimiLinearForCausalLM", "kimi_linear")
                )
            self.assertNotIsInstance(ctx.exception, RuntimeError)


if __name__ == "__main__":
    unittest.main()
