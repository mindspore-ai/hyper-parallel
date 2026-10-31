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
"""Compare full HF image/text logits and gradients with decoder CP plus Encoder-DP."""

import copy
from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeVisionAttention,
)

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed.context_parallel.wrappers import sdpa_hf_cp_wrapper
from hyper_parallel.distributed.encoder_parallel.batch_input import EncoderParallelInputAdapter
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime
from hyper_parallel.models.qwen3_5_moe.adapter.encoder_parallel import (
    get_encoder_parallel_spec, replace_vision_attention,
)
from hyper_parallel.models.qwen3_5_moe.adapter.vlm import install_qwen_vlm_cp_forward
from hyper_parallel.models.qwen3_5_moe.vlm_cp import QwenVLMParallelBatch
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements,
)
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token


def _compare(encoder_parallel: bool) -> None:
    """Check cross-shard image spans, empty local supervision and all parameter gradients."""
    dist.init_process_group("gloo")
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = Qwen3_5MoeConfig(
            image_token_id=99, video_token_id=98, vision_start_token_id=97,
            text_config={"vocab_size": 128, "hidden_size": 32, "num_hidden_layers": 1,
                         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                         "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
                         "shared_expert_intermediate_size": 16, "layer_types": ["full_attention"],
                         "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                             "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]}},
            vision_config={"hidden_size": 16, "intermediate_size": 32, "num_heads": 4, "depth": 2,
                           "out_hidden_size": 32, "patch_size": 2, "spatial_merge_size": 2,
                           "temporal_patch_size": 2, "num_position_embeddings": 16},
        )
        # Exercise the shared internal integration contract; no public equivalent exists.
        config._attn_implementation = "sdpa"  # pylint: disable=protected-access
        model = Qwen3_5MoeForConditionalGeneration(config)
        oracle = copy.deepcopy(model)
        mesh = init_device_mesh("cpu", (1, 2, 1), mesh_dim_names=("dp", "cp", "tp"))
        context = SimpleNamespace(device_mesh=mesh, cp_size=2, tp_size=1, pp_size=1, dp_size=1, dp_rank=0,
                                  cp_rank=dist.get_rank(), tp_rank=0, sequence_parallel=False, loss_parallel=False)
        attention = model.model.language_model.layers[0].self_attn
        attention.forward = sdpa_hf_cp_wrapper(attention, mesh, mesh["tp"], mesh["cp"], None)
        batcher = QwenVLMParallelBatch(mesh_context=context, device=torch.device("cpu"))
        install_qwen_vlm_cp_forward(model)
        batcher.bind_model(model)
        if encoder_parallel:
            rule = ModuleReplacementSpec(("model.visual.blocks.*.attn",), replace_vision_attention,
                                         Qwen3_5MoeVisionAttention)
            apply_module_replacements(model, compile_module_replacements(model, (rule,)))
            runtime = EncoderParallelRuntime(config=EncoderParallelConfig(True, 1), mesh_context=context,
                                             model=model, spec=get_encoder_parallel_spec(),
                                             execution_options={"micro_batch_size": 1})
            runtime.bridge.install_forward()
            batcher.runtime_input_adapter = EncoderParallelInputAdapter(runtime=runtime, inner=None)
        for image_start, supervised_start in ((62, 2), (5, 65)):
            model.zero_grad(set_to_none=True)
            oracle.zero_grad(set_to_none=True)
            ids = torch.randint(1, 80, (1, 100))
            ids[:, image_start:image_start + 4] = 99
            labels = ids.clone()
            labels[:, :supervised_start] = -100
            labels[ids == 99] = -100
            batch = {"input_ids": ids, "labels": labels, "mm_token_type_ids": ids.eq(99).long(),
                     "pixel_values": torch.randn(16, 24), "image_grid_thw": torch.tensor([[1, 4, 4]])}
            # Exercise the shared internal integration contract; no public equivalent exists.
            full = batcher._pad(batch)  # pylint: disable=protected-access
            full.pop("loss_mask")
            expected = oracle(**full, use_cache=False)
            inputs, loss_inputs = batcher(None, external_batch=batch)
            actual = model(**inputs, use_cache=False)
            geometry = inputs["vlm_cp_inputs"]
            torch.testing.assert_close(actual.logits[:, :max(0, min(100, geometry.stop) - geometry.start)],
                                       expected.logits[:, geometry.start:min(100, geometry.stop)],
                                       rtol=2e-4, atol=2e-5)
            local_count = count_loss_token(loss_inputs)["foundation_tokens"]
            global_count = full["labels"][:, 1:].ne(-100).sum()
            loss = actual.loss * local_count / global_count
            reduced = loss.detach().clone()
            dist.all_reduce(reduced)
            torch.testing.assert_close(reduced, expected.loss, rtol=1e-5, atol=1e-6)
            loss.backward()
            expected.loss.backward()
            reference = dict(oracle.named_parameters())
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise AssertionError(f"Missing gradient: {name}")
                dist.all_reduce(parameter.grad)
                torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=3e-4, atol=3e-5,
                                           msg=lambda message, name=name: f"{name}: {message}")
    finally:
        dist.destroy_process_group()


def test_decoder_cp() -> None:
    """Compare replicated vision plus decoder CP=2 against full HF."""
    _compare(False)


def test_decoder_cp_encoder_dp() -> None:
    """Compare Encoder-DP=2 plus decoder CP=2 against full HF."""
    _compare(True)
