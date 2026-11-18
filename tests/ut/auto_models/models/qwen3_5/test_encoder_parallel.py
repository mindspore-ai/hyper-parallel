# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Regression coverage for Qwen's first Encoder Parallel batch."""

from types import SimpleNamespace
from unittest import TestCase

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP, init_device_mesh
from hyper_parallel.data.batching.get_batch import OmniParallelBatch
from hyper_parallel.data.parallel import DataLoaderParallelContext
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime
from hyper_parallel.distributed.encoder_parallel.transport import EncoderParallelBatchTransport
from hyper_parallel.models.qwen3_5_moe.adapter.encoder_parallel import (
    QwenEncoderAttention, get_encoder_parallel_spec,
)


def _tiny_model() -> Qwen3_5ForConditionalGeneration:
    """Build a test model without loading a checkpoint."""
    config = Qwen3_5Config.from_dict({
        "image_token_id": 15,
        "video_token_id": 16,
        "text_config": {
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
            "layer_types": ["full_attention"], "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1], "partial_rotary_factor": 1.0},
        },
        "vision_config": {
            "depth": 1, "hidden_size": 16, "out_hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
            "in_channels": 3, "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2,
            "num_position_embeddings": 16,
        },
    })
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        return Qwen3_5ForConditionalGeneration(config)


def _batch(with_images: bool) -> dict[str, torch.Tensor]:
    """Create a text batch or a batch with two differently shaped image grids."""
    ids = torch.tensor([[1, 15, 15, 2, 15, 15, 3, 4] if with_images else [1, 2, 3, 4]])
    batch = {"input_ids": ids, "labels": ids.masked_fill(ids == 15, -100)}
    if with_images:
        batch["pixel_values"] = torch.linspace(-1, 1, 16 * 24).reshape(16, 24)
        batch["image_grid_thw"] = torch.tensor([[1, 2, 4], [1, 4, 2]])
        batch["mm_token_type_ids"] = ids.eq(15).long()
    return batch


class TestQwenEncoderTransport(TestCase):
    """Drive the ordinary Omni-to-Qwen forward on a CPU-only mesh."""

    def tearDown(self) -> None:
        """Remove temporary meshes built with the backend disabled."""
        _DEVICE_MESH_MAP.clear()

    def test_first_batches_match_native_qwen(self) -> None:
        """Keep text-only batches inactive and match native image forward/backward."""
        model = _tiny_model()
        fqn = "model.visual.blocks.0.attn"
        block = model.model.visual.blocks[0]
        block.attn = QwenEncoderAttention(block.attn, fqn)
        model._hp_replaced_module_fqns = (fqn,)  # pylint: disable=protected-access
        mesh = init_device_mesh("cpu", (1, 1, 1), mesh_dim_names=("dp", "cp", "tp"),
                                rank_list=(0,), init_backend=False)
        context = SimpleNamespace(device_mesh=mesh, cp_size=1, tp_size=1, cp_rank=0, tp_rank=0,
                                  pp_size=1, sequence_parallel=False, loss_parallel=False)
        runtime = EncoderParallelRuntime(
            config=EncoderParallelConfig(enabled=True), mesh_context=context, model=model,
            spec=get_encoder_parallel_spec(), execution_options={"micro_batch_size": 1},
        )
        get_batch = OmniParallelBatch(mesh_context=context, device="cpu",
                                      dataloader_context=DataLoaderParallelContext())
        get_batch.batch_transport = EncoderParallelBatchTransport(runtime=runtime)
        projection = model.model.visual.patch_embed.proj.weight
        for with_images in (False, True):
            with self.subTest(with_images=with_images):
                batch = _batch(with_images)
                model_inputs, _ = get_batch(iter([batch]))
                self.assertNotIn("pixel_values", model_inputs)
                self.assertEqual(model_inputs["encoder_parallel_plan"].active, with_images)

                reference = model(**batch, use_cache=False)
                actual = model(**model_inputs, use_cache=False)
                torch.testing.assert_close(actual.logits, reference.logits)
                torch.testing.assert_close(actual.loss, reference.loss)
                if with_images:
                    reference.loss.backward()
                    reference_grad = projection.grad.clone()
                    model.zero_grad(set_to_none=True)
                    actual.loss.backward()
                    torch.testing.assert_close(projection.grad, reference_grad)
                else:
                    self.assertNotIn("encoder_parallel_inputs", model_inputs)
