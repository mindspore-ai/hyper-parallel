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

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.components.modules.dspark import DeepseekV41DSpark
from hyper_parallel.data.constants import IGNORE_INDEX


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


if __name__ == "__main__":
    unittest.main()
