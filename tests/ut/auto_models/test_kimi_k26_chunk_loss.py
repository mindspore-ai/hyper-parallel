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
"""Unit tests for the Kimi-K2.6 model-integrated Chunk Loss adapter."""
# pylint: disable=not-callable

import os
import types
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # pylint: disable=wrong-import-position

from hyper_parallel.components.losses import ChunkedCausalLMLoss  # pylint: disable=wrong-import-position
from hyper_parallel.data.constants import IGNORE_INDEX  # pylint: disable=wrong-import-position
from hyper_parallel.models.kimi_k26.adapter.chunk_loss import (  # pylint: disable=wrong-import-position
    bind_chunk_loss,
)

VOCAB_SIZE = 64
SEQ_LEN = 12


def _tiny_model():
    """Return a tiny, randomly initialized Kimi-K2.6 model on CPU."""
    from transformers.models.kimi_k25 import (  # pylint: disable=import-outside-toplevel
        Kimi_K25Config,
        Kimi_K25ForConditionalGeneration,
    )

    text_config = {
        "model_type": "deepseek_v3",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 32,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "greedy",
        "first_k_dense_replace": 1,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "v_head_dim": 8,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }
    vision_config = {
        "num_hidden_layers": 1,
        "hidden_size": 32,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "patch_size": 14,
        "merge_kernel_size": (2, 2),
        "pos_emb_height": 8,
        "pos_emb_width": 8,
        "pos_emb_time": 1,
    }
    config = Kimi_K25Config(  # pylint: disable=unexpected-keyword-arg
        text_config=text_config,
        vision_config=vision_config,
        projection_hidden_size=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return Kimi_K25ForConditionalGeneration(config)


def _single_rank_setup():
    """Return a minimal single-rank distributed setup for loss binding."""
    mesh_context = types.SimpleNamespace(tp_size=1, pp_size=1, loss_parallel=False)
    return types.SimpleNamespace(mesh_context=mesh_context)


class TestKimiK25ChunkLoss(unittest.TestCase):
    """The Chunk Loss adapter must reproduce the full-logits objective."""

    def _inputs(self):
        generator = torch.Generator().manual_seed(11)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), generator=generator)
        labels = input_ids.clone()
        labels[0, 3] = IGNORE_INDEX
        labels[0, SEQ_LEN - 1] = IGNORE_INDEX
        return input_ids, labels

    def test_chunked_loss_matches_full_logits_loss_and_gradients(self) -> None:
        """Chunk Loss must match the eager full-logits loss and weight gradients."""
        input_ids, labels = self._inputs()

        reference = _tiny_model()
        reference_out = reference(input_ids=input_ids, labels=labels)
        reference_loss = reference_out.loss
        reference_loss.backward()
        reference_head_grad = reference.lm_head.weight.grad.detach().clone()

        model = _tiny_model()
        model.load_state_dict(reference.state_dict())
        loss_fn = ChunkedCausalLMLoss(chunk_size=5, ignore_index=IGNORE_INDEX)
        loss_fn.bind_model(model, _single_rank_setup())

        model_inputs = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }
        # OmniParallelBatch publishes pre-shifted targets (labels_are_shifted
        # defaults to True), aligned with every local hidden position.
        shift_labels = torch.full_like(labels, IGNORE_INDEX)
        shift_labels[:, :-1] = labels[:, 1:]
        loss_inputs = {
            "labels": labels,
            "shift_labels": shift_labels,
            "loss_mask": shift_labels.ne(IGNORE_INDEX),
        }
        prepared = loss_fn.prepare_model_inputs(model_inputs, loss_inputs)

        self.assertNotIn("labels", prepared)
        self.assertIn("chunk_loss_targets", prepared)
        model_output = model(**prepared, use_cache=False)
        self.assertIsNone(model_output.logits)

        loss = loss_fn(model_output=model_output, labels=labels)
        loss.backward()

        torch.testing.assert_close(loss, reference_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            model.lm_head.weight.grad,
            reference_head_grad,
            rtol=1e-3,
            atol=1e-6,
        )

    def test_bind_rejects_foreign_model_family(self) -> None:
        """Binding must refuse a model whose family has no Chunk Loss adapter."""
        model = _tiny_model()
        model.config.model_type = "not_kimi_k26"
        with self.assertRaises(TypeError):
            bind_chunk_loss(model)


if __name__ == "__main__":
    unittest.main()
