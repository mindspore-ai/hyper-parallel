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
"""Component tests for the trainable DeepSeek-V4.1 DSpark drafter."""

from __future__ import annotations

import functools
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel import init_empty_weights
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models.deepseek_v41.dspark import DeepseekV41DSpark
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM


def _small_config() -> SimpleNamespace:
    """Return a CPU-sized DSpark configuration."""
    return SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        head_dim=16,
        partial_rotary_factor=0.5,
        q_lora_rank=24,
        o_groups=2,
        o_lora_rank=8,
        rms_norm_eps=1.0e-6,
        rope_theta=10000.0,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        hc_eps=1.0e-6,
        vocab_size=64,
        initializer_range=0.02,
        v41_dspark_depth=1,
        v41_dspark_block_size=3,
        v41_dspark_window=4,
        v41_dspark_markov_rank=8,
        v41_dspark_noise_token_id=63,
        v41_dspark_target_layer_ids=(1, 2),
        v41_dspark_confidence_coeff=0.1,
        v41_dspark_loss_chunk_size=4,
    )


class _TinyMlp(nn.Module):
    """Minimal MoE stand-in matching the stage FFN calling convention."""

    def __init__(self, hidden: int) -> None:
        """Create the single projection used as the stage FFN."""
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_ids: torch.Tensor | None = None,
            image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project the flattened draft stream like the crop MoE forward."""
        del input_ids, image_mask
        return self.proj(hidden_states)


def _build(config: SimpleNamespace) -> tuple[DeepseekV41DSpark, nn.Linear]:
    """Construct the drafter plus the tied backbone head surface."""
    torch.manual_seed(7)
    dspark = DeepseekV41DSpark(config, mlp_factory=lambda: _TinyMlp(config.hidden_size))
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    return dspark, head


def _batch(config: SimpleNamespace, batch: int = 2, seq: int = 10):
    """Return deterministic inputs with an ignored prefix."""
    generator = torch.Generator().manual_seed(11)
    input_ids = torch.randint(1, config.vocab_size - 1, (batch, seq), generator=generator)
    labels = input_ids.clone()
    labels[:, :3] = IGNORE_INDEX
    targets = [torch.randn(batch, seq, config.hidden_size) for _ in config.v41_dspark_target_layer_ids]
    inputs_embeds = torch.randn(batch, seq, config.hidden_size)
    return input_ids, labels, targets, inputs_embeds


class DSparkComponentTest(unittest.TestCase):
    """Validate structure, gradient isolation, and target alignment."""

    def test_forward_produces_finite_losses_and_metrics(self):
        """A full forward returns finite loss values and metric entries."""
        config = _small_config()
        dspark, head = _build(config)
        input_ids, labels, targets, embeds = _batch(config)
        loss, metrics = dspark(targets, input_ids, labels, embeds, head)
        self.assertTrue(torch.isfinite(loss).item(), "DSpark loss must be finite")
        for name in ("dspark_ce", "dspark_confidence_bce", "dspark_draft_accuracy",
                     "dspark_supervised_tokens"):
            self.assertIn(name, metrics, f"missing metric {name}")
            self.assertTrue(torch.isfinite(metrics[name].float()).item(), f"{name} must be finite")

    def test_requires_detached_target_hiddens(self):
        """Grad-carrying backbone hiddens are rejected up front."""
        config = _small_config()
        dspark, head = _build(config)
        input_ids, labels, targets, embeds = _batch(config)
        targets[0] = targets[0].clone().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "detached"):
            dspark(targets, input_ids, labels, embeds, head)

    def test_gradients_stay_inside_the_drafter(self):
        """The DSpark objective must not reach tied backbone surfaces."""
        config = _small_config()
        dspark, head = _build(config)
        input_ids, labels, targets, embeds = _batch(config)
        loss, _ = dspark(targets, input_ids, labels, embeds, head)
        loss.backward()
        self.assertIsNone(head.weight.grad, "tied head must stay frozen under DSpark loss")
        touched = [name for name, parameter in dspark.named_parameters()
                   if parameter.grad is not None and parameter.grad.abs().sum() > 0]
        self.assertTrue(any(name.startswith("stages.0.self_attn") for name in touched))
        self.assertTrue(any(name.startswith("main_proj") for name in touched))
        self.assertTrue(any(name.startswith("markov_head") for name in touched))
        self.assertTrue(any(name.startswith("confidence_head") for name in touched))
        self.assertTrue(any(name == "noise_embedding" for name in touched))

    def test_supervised_targets_align_with_future_tokens(self):
        """Draft offset j at position t must supervise the token at t+1+j."""
        config = _small_config()
        config.v41_dspark_block_size = 2
        dspark, head = _build(config)
        batch, seq = 1, 6
        input_ids = torch.arange(1, seq + 1).unsqueeze(0)
        labels = input_ids.clone()
        targets = [torch.zeros(batch, seq, config.hidden_size) for _ in config.v41_dspark_target_layer_ids]
        embeds = torch.zeros(batch, seq, config.hidden_size)
        _, metrics = dspark(targets, input_ids, labels, embeds, head)
        # Offsets j=0 supervises t+1 (five valid positions) and j=1 supervises
        # t+2 (four valid positions); the padded tail carries IGNORE_INDEX.
        self.assertEqual(int(metrics["dspark_supervised_tokens"]), (seq - 1) + (seq - 2))

    def test_shifted_labels_keep_targets_from_input_ids(self):
        """Pre-shifted trainer labels must only mask, never shift the targets.

        The trainer hands labels over already shifted (labels[t] supervises
        input_ids[t + 1]); gathering them as target values would move the
        draft objective to t + 2 + j and the drafter would train against the
        wrong token. Capture the cross-entropy targets and pin them to
        input_ids[t + 1 + j] with the label mask aligned via labels[t + j].
        """
        config = _small_config()
        config.v41_dspark_block_size = 2
        dspark, head = _build(config)
        batch, seq = 1, 6
        input_ids = torch.arange(1, seq + 1).unsqueeze(0)
        labels = torch.cat(
            [input_ids[:, 1:], torch.full((batch, 1), IGNORE_INDEX)], dim=1)
        hiddens = [torch.zeros(batch, seq, config.hidden_size)
                   for _ in config.v41_dspark_target_layer_ids]
        embeds = torch.zeros(batch, seq, config.hidden_size)
        original = torch.nn.functional.cross_entropy
        captured = []

        def _spy(logits, target, **kwargs):
            captured.append(target.detach().clone())
            return original(logits, target, **kwargs)

        with patch("hyper_parallel.models.deepseek_v41.dspark.functional.cross_entropy",
                   side_effect=_spy):
            _, metrics = dspark(hiddens, input_ids, labels, embeds, head)
        got = torch.cat(captured).view(batch, seq, 2)
        expected = torch.tensor([[[2, 3], [3, 4], [4, 5], [5, 6],
                                  [6, IGNORE_INDEX],
                                  [IGNORE_INDEX, IGNORE_INDEX]]])
        self.assertTrue(torch.equal(got, expected),
                        msg=f"draft targets misaligned: {got.tolist()}")
        self.assertEqual(int(metrics["dspark_supervised_tokens"]), 9)

    def test_markov_head_biases_the_draft_logits(self):
        """Zeroing the Markov head must change the draft cross-entropy."""
        config = _small_config()
        dspark, head = _build(config)
        input_ids, labels, targets, embeds = _batch(config)
        with torch.no_grad():
            baseline, _ = dspark(targets, input_ids, labels, embeds, head)
            dspark.markov_head.head.weight.mul_(0.0)
            ablated, _ = dspark(targets, input_ids, labels, embeds, head)
        self.assertNotAlmostEqual(float(baseline), float(ablated), places=6,
                                  msg="Markov bias must influence the draft objective")

    def test_none_labels_supervise_from_input_ids(self):
        """Without labels the drafter supervises all next tokens."""
        config = _small_config()
        config.v41_dspark_block_size = 2
        dspark, head = _build(config)
        batch, seq = 1, 6
        input_ids = torch.arange(1, seq + 1).unsqueeze(0)
        targets = [torch.zeros(batch, seq, config.hidden_size) for _ in config.v41_dspark_target_layer_ids]
        embeds = torch.zeros(batch, seq, config.hidden_size)
        _, metrics = dspark(targets, input_ids, None, embeds, head)
        self.assertEqual(int(metrics["dspark_supervised_tokens"]), (seq - 1) + (seq - 2))

    def test_zero_depth_configuration_is_rejected(self):
        """Constructing with no stages/targets is a configuration error."""
        config = _small_config()
        config.v41_dspark_block_size = 0
        with self.assertRaises(ValueError):
            DeepseekV41DSpark(config, mlp_factory=lambda: _TinyMlp(config.hidden_size))


def _write_engram_assets(
        directory: str,
        num_hidden_layers: int = 4,
        layer_ids: tuple[int, ...] = (1,),
        head_dim: int = 4,
) -> Path:
    """Write a minimal but internally consistent scaled Engram asset."""
    available_primes = (
        [[17, 19], [23, 29]],
        [[31, 37], [41, 43]],
    )
    if len(layer_ids) > len(available_primes):
        raise ValueError("the test asset helper supports at most two Engram layers")
    primes = list(available_primes[:len(layer_ids)])
    assets = {
        "source_model_type": "deepseek_v41",
        "num_hidden_layers": num_hidden_layers,
        "layer_ids": list(layer_ids),
        "bucket_base": 16,
        "max_ngram_size": 3,
        "num_heads": 2,
        "head_dim": head_dim,
        "primes": primes,
        "num_embeddings": [
            sum(value for row in layer_primes for value in row)
            for layer_primes in primes
        ],
        "multipliers": [[101, 103, 107] for _ in layer_ids],
        "token_map": list(range(64)),
        "pad_token_id": 0,
    }
    path = Path(directory) / "engram.json"
    path.write_text(json.dumps(assets), encoding="utf-8")
    return path


def _write_released_config(directory: str) -> Path:
    """Write a release-shaped config whose dimensions have exact crop ratios."""
    text_config = {
        "vocab_size": 64,
        "hidden_size": 256,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 40,
        # Drafter hyper-parameters as published in DeepSeek-V4.1-Flash.
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 63,
        "dspark_target_layer_ids": [37, 38, 39],
        "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128,
        "dspark_num_experts_per_tok": 3,
        "num_attention_heads": 32,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "qk_rope_head_dim": 32,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "o_groups": 8,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        "compress_rope_theta": 160000.0,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32,
        "index_head_dim": 64,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1.0e-6,
        "engram_layer_ids": [1, 14],
        "engram_head_dim": 32,
    }
    source = {
        "model_type": "deepseek_v41",
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "image_token_id": 3,
        "text_config": text_config,
        "vision_config": {
            "num_hidden_layers": 32,
            "hidden_size": 128,
            "num_attention_heads": 16,
            "intermediate_size": 192,
            "patch_size": 14,
            "rope_theta": 10000.0,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        },
    }
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


# Leaf parameter names of the released DeepSeek-V4.1-Flash drafter, taken from
# the mtp.* entries of model.safetensors.index.json (2401 entries, three
# stages). Stage and expert indices are collapsed to "N".
_RELEASED_DRAFTER_LEAVES = {
    "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
    "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
    "attn_norm.weight", "ffn_norm.weight",
    "attn.attn_sink", "attn.kv_norm.weight", "attn.q_norm.weight",
    "attn.wkv.weight", "attn.wo_a.weight", "attn.wo_b.weight",
    "attn.wq_a.weight", "attn.wq_b.weight",
    "ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl",
    "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.weight",
    "ffn.experts.N.w1.weight", "ffn.experts.N.w2.weight", "ffn.experts.N.w3.weight",
    "main_norm.weight", "main_proj.weight", "norm.weight",
    "markov_head.embed.weight", "markov_head.head.weight",
    "confidence_head.proj.weight",
}

# Released leaf -> our leaf. Only unambiguous correspondences are listed; the
# rest are recorded as known differences below.
_DRAFTER_RENAMES = {
    "hc_attn_base": "stages.N.attn_hc.base",
    "hc_attn_fn": "stages.N.attn_hc.fn",
    "hc_attn_scale": "stages.N.attn_hc.scale",
    "hc_ffn_base": "stages.N.ffn_hc.base",
    "hc_ffn_fn": "stages.N.ffn_hc.fn",
    "hc_ffn_scale": "stages.N.ffn_hc.scale",
    "attn.attn_sink": "stages.N.self_attn.attn_sink",
    "attn.kv_norm.weight": "stages.N.self_attn.kv_norm.weight",
    "attn.q_norm.weight": "stages.N.self_attn.q_norm.weight",
    "attn.wkv.weight": "stages.N.self_attn.wkv.weight",
    "attn.wo_a.weight": "stages.N.self_attn.wo_a.weight",
    "attn.wo_b.weight": "stages.N.self_attn.wo_b.weight",
    "attn.wq_a.weight": "stages.N.self_attn.wq_a.weight",
    "attn.wq_b.weight": "stages.N.self_attn.wq_b.weight",
    "ffn.gate.weight": "stages.N.mlp.gate.weight",
    "ffn.gate.bias": "stages.N.mlp.gate.bias",
    "ffn.gate.bias_vl": "stages.N.mlp.gate.bias_vl",
    "ffn.shared_experts.w1.weight": "stages.N.mlp.shared_experts.gate_proj.weight",
    "ffn.shared_experts.w2.weight": "stages.N.mlp.shared_experts.down_proj.weight",
    "ffn.shared_experts.w3.weight": "stages.N.mlp.shared_experts.up_proj.weight",
    "main_norm.weight": "main_norm.weight",
    "main_proj.weight": "main_proj.weight",
    "norm.weight": "norm.weight",
    "markov_head.embed.weight": "markov_head.embed.weight",
    "markov_head.head.weight": "markov_head.head.weight",
    "confidence_head.proj.weight": "confidence_head.proj.weight",
}

# Released names with no 1:1 counterpart here, and why.
_RELEASED_ONLY = {
    # The release stores 128 experts per stage separately; the crop MoE packs
    # them into two grouped tensors.
    "ffn.experts.N.w1.weight", "ffn.experts.N.w2.weight", "ffn.experts.N.w3.weight",
    # One RMSNorm per sublayer here lives inside the hyper-connection module,
    # so the stage exposes two norms the release keeps flat.
    "attn_norm.weight", "ffn_norm.weight",
}

# Ours with no counterpart in the released weights, and why.
_OURS_ONLY = {
    # Grouped expert tensors, see above.
    "stages.N.mlp.experts.gate_up_proj", "stages.N.mlp.experts.down_proj",
    # The hyper-connection sublayer norms and the stage norms.
    "stages.N.attn_hc.input_norm.weight", "stages.N.ffn_hc.input_norm.weight",
    "stages.N.input_layernorm.weight", "stages.N.post_attention_layernorm.weight",
    # The release fills draft slots from the embedding of dspark_noise_token_id;
    # this implementation learns a dedicated vector instead.
    "noise_embedding",
}


class TestDrafterMatchesReleasedLayout(unittest.TestCase):
    """The drafter's parameters correspond to the released mtp.* layout.

    FP8 ``.scale`` companions are out of scope for bf16 training and are
    excluded; every other released name must map onto a parameter here, and
    every parameter here must be accounted for.
    """

    @staticmethod
    def _drafter_leaves() -> set:
        """Build the drafter on meta and collapse index positions to "N"."""
        from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (  # pylint: disable=C0415
            build_deepseek_v41_validation_config,
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_released_config(directory)
            assets = _write_engram_assets(directory, num_hidden_layers=40, head_dim=8)
            config = build_deepseek_v41_validation_config(
                str(config_path.parent), str(assets), dspark_depth=3, enable_vision=True)
            with init_empty_weights():
                model = DeepseekV41ForCausalLM(config)
            return {re.sub(r"\.\d+\.", ".N.", name)
                    for name, _ in model.dspark.named_parameters()}

    def test_every_released_name_is_accounted_for(self):
        """No released parameter family is silently missing."""
        ours = self._drafter_leaves()
        for released in sorted(_RELEASED_DRAFTER_LEAVES):
            with self.subTest(released=released):
                if released in _RELEASED_ONLY:
                    continue
                self.assertIn(_DRAFTER_RENAMES[released], ours)

    def test_no_undocumented_parameters_here(self):
        """Anything we add beyond the released layout stays documented."""
        ours = self._drafter_leaves()
        mapped = set(_DRAFTER_RENAMES.values())
        self.assertEqual(sorted(ours - mapped - _OURS_ONLY), [])

    def test_hyperparameters_match_the_release(self):
        """Depth, block width, expert counts and the noise token come from the release."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_released_config(directory)
            released = json.loads(config_path.read_text(encoding="utf-8"))["text_config"]
            assets = _write_engram_assets(directory, num_hidden_layers=40, head_dim=8)
            from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (  # pylint: disable=C0415
                build_deepseek_v41_validation_config,
            )
            config = build_deepseek_v41_validation_config(
                str(config_path.parent), str(assets), dspark_depth=3)
        self.assertEqual(config.v41_dspark_block_size, released["dspark_block_size"])
        self.assertEqual(config.v41_dspark_markov_rank, released["dspark_markov_rank"])
        self.assertEqual(config.v41_dspark_noise_token_id, released["dspark_noise_token_id"])
        self.assertEqual(config.v41_dspark_target_layer_ids, released["dspark_target_layer_ids"])
        self.assertEqual(config.v41_dspark_n_routed_experts,
                         min(released["dspark_n_routed_experts"], 16))


class TestDSparkCropSwitches(unittest.TestCase):
    """The crop entry carries the DSpark drafter switches into the config."""

    @staticmethod
    def _build(**kwargs):
        """Build a validation config from the release-shaped fixtures."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_released_config(directory)
            assets = _write_engram_assets(directory, num_hidden_layers=40, head_dim=8)
            from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (  # pylint: disable=C0415
                build_deepseek_v41_validation_config,
            )
            return build_deepseek_v41_validation_config(
                str(config_path.parent), str(assets), **kwargs)

    def test_drafter_is_off_by_default(self):
        """No drafter parameters are requested unless depth is positive."""
        config = self._build()
        self.assertEqual(config.v41_dspark_depth, 0)
        self.assertEqual(config.v41_dspark_target_layer_ids, [])

    def test_depth_populates_the_released_layout(self):
        """A positive depth targets the last three layers, as released."""
        config = self._build(dspark_depth=1)
        self.assertEqual(config.v41_dspark_depth, 1)
        self.assertEqual(config.v41_dspark_target_layer_ids, [37, 38, 39])
        self.assertEqual(config.v41_dspark_window, 128)
        self.assertLessEqual(config.v41_dspark_top_k, config.v41_dspark_n_routed_experts)

    def test_model_builds_the_drafter_only_when_depth_is_positive(self):
        """The config field must actually reach the model, not just the config.

        A run with the drafter silently absent looks like a run with the
        drafter present but ineffective, so assert the object exists.
        """
        from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import (  # pylint: disable=C0415
            build_deepseek_v41_validation_config,
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = _write_released_config(directory)
            assets = _write_engram_assets(directory, num_hidden_layers=40, head_dim=8)
            build = functools.partial(
                build_deepseek_v41_validation_config, str(config_path.parent), str(assets))
            with init_empty_weights():
                off = DeepseekV41ForCausalLM(build())
                on = DeepseekV41ForCausalLM(build(dspark_depth=1))
        self.assertIsNone(off.dspark)
        self.assertIsInstance(on.dspark, DeepseekV41DSpark)
        self.assertGreater(on.dspark_loss_coeff, 0.0)

    def test_explicit_targets_override_the_default(self):
        """Callers can point the drafter at other backbone layers."""
        config = self._build(dspark_depth=1, dspark_target_layer_ids=[10, 11])
        self.assertEqual(config.v41_dspark_target_layer_ids, [10, 11])


if __name__ == "__main__":
    unittest.main()
