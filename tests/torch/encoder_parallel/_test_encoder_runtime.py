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
"""Gloo worker for E0 packed visual execution and feature exchange."""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime
from hyper_parallel.models.deepseek_v41.components.vision_transformer import (
    get_deepseek_v41_encoder_parallel_spec,
    replace_vision_attention,
)
from hyper_parallel.models.deepseek_v41.vision import (
    DeepseekV41VisionTower,
    DeepseekV41VisionAligner,
    DeepseekV41VisionAttention,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    compile_module_replacements,
    apply_module_replacements,
)


def _build_model_and_runtime(rank):
    """Create the tiny ViT+aligner model and the E0 encoder-parallel runtime."""
    torch.manual_seed(10)
    cfg = SimpleNamespace(
        v41_vision_hidden_size=8, v41_vision_patch_size=2, v41_vision_num_attention_heads=2,
        v41_vision_rope_theta=10000.0, v41_vision_intermediate_size=16,
        v41_vision_num_hidden_layers=2, v41_vision_downsample_ratio=3, hidden_size=8,
    )
    model = nn.Module()
    model.model = nn.Module()
    model.model.vision = DeepseekV41VisionTower(cfg)
    model.model.aligner = DeepseekV41VisionAligner(cfg)
    model.model.image_start = nn.Parameter(torch.zeros(8))
    model.model.image_end = nn.Parameter(torch.zeros(8))
    model.model.image_newline = nn.Parameter(torch.zeros(8))
    rule = ModuleReplacementSpec(
        match=("model.vision.blocks.*.attn",), factory=replace_vision_attention,
        module_type=DeepseekV41VisionAttention,
    )
    apply_module_replacements(model, compile_module_replacements(model, (rule,)), weights_mapping=[])
    mesh = init_device_mesh(
        "cpu", (1, 2, 4), mesh_dim_names=("dp", "cp", "tp"),
        rank_list=tuple(range(8)), init_backend=True,
    )
    context = SimpleNamespace(
        device_mesh=mesh, cp_size=2, tp_size=4, cp_rank=rank // 4, tp_rank=rank % 4,
        pp_size=1, sequence_parallel=True, loss_parallel=False,
    )
    runtime = EncoderParallelRuntime(
        config=EncoderParallelConfig(enabled=True, encoder_cp_size=2),
        mesh_context=context, model=model, spec=get_deepseek_v41_encoder_parallel_spec(),
        execution_options={"micro_batch_size": 1, "compile_enabled": False, "validate_placement": False},
    )
    return model, runtime


def _reference_aligner_output(model, batch, pixels, item_id):
    """Recompute one image's single-card vision+aligner features."""
    start = int(batch["image_patch_offsets"][item_id])
    stop = int(batch["image_patch_offsets"][item_id + 1])
    height, width = (int(value) for value in batch["image_vit_grid_hw"][item_id])
    return model.model.aligner(
        model.model.vision(pixels[start:stop], height, width), height, width,
    )


def _check_routed_records(rank, model, batch, pixels, records, expected):
    """Compare routed features against single-card references under no_grad."""
    with torch.no_grad():
        for item_id, record in zip(expected, records):
            reference = _reference_aligner_output(model, batch, pixels, item_id)
            candidate = record[1]
            assert torch.allclose(candidate, reference, atol=1e-5, rtol=1e-5), (
                f"rank={rank}; item={item_id}; max_abs={(candidate - reference).abs().max().item()}; "
                f"candidate_first={candidate[0]}; reference_first={reference[0]}"
            )


def _check_visual_gradients(rank, model, batch, pixels, records, anchor, expected):
    """Backprop routed features and compare visual gradients with references."""
    loss = sum((feature.sum() for _, feature in records), anchor)
    loss.backward()
    assert all(param.grad is not None for param in model.model.vision.parameters()), f"vision grad missing: rank={rank}"
    assert all(param.grad is not None for param in model.model.aligner.parameters()), (
        f"aligner grad missing: rank={rank}"
    )
    visual_parameters = tuple(model.model.vision.parameters()) + tuple(model.model.aligner.parameters())
    candidate_gradients = [parameter.grad.detach().clone() for parameter in visual_parameters]
    for parameter in visual_parameters:
        parameter.grad = None
    reference_losses = [
        _reference_aligner_output(model, batch, pixels, item_id).sum() for item_id in expected
    ]
    reference_loss = sum(reference_losses) if reference_losses else sum(
        parameter.sum() * 0.0 for parameter in visual_parameters
    )
    reference_loss.backward()
    for parameter, candidate_gradient in zip(visual_parameters, candidate_gradients):
        expected_gradient = parameter.grad.detach().clone()
        dist.all_reduce(candidate_gradient)
        dist.all_reduce(expected_gradient)
        assert torch.allclose(candidate_gradient, expected_gradient, atol=1e-4, rtol=1e-4), (
            f"rank={rank}; candidate_grad={candidate_gradient.flatten()[:4]}; "
            f"reference_grad={expected_gradient.flatten()[:4]}; "
            f"max_abs={(candidate_gradient - expected_gradient).abs().max().item()}"
        )


def test_e0_runtime():
    """Feature: E0 Encoder-DP/CP execution with an empty bucket.

    Description: Execute packed ViT, producer aligner, and parent A2A on eight ranks.
    Expectation: Exact destination records and non-missing visual gradients.
    """
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    model, runtime = _build_model_and_runtime(rank)
    torch.manual_seed(20)
    pixels = torch.randn(976, 3, 2, 2)
    batch = {
        "input_ids": torch.zeros(1, 128, dtype=torch.long),
        "token_types": torch.zeros(1, 128, dtype=torch.long),
        "pixel_values": pixels,
        "image_patch_offsets": torch.tensor([0, 576, 832, 976]),
        "image_vit_grid_hw": torch.tensor([[24, 24], [16, 16], [12, 12]]),
        "image_llm_grid_hw": torch.tensor([[8, 8], [6, 6], [4, 4]]),
        "image_batch_indices": torch.zeros(3, dtype=torch.long),
        "image_token_starts": torch.tensor([96, 32, 200]),
    }
    plan = runtime.finalize_plan(*runtime.preflight_batch(batch))
    records, anchor = plan.encode_and_route(batch)
    expected = ((), (1,), (1,), (0,), (0,), (0,), (2,), ())[rank]
    actual = tuple(item.decoder_token_start for item, _ in records)
    expected_starts = tuple((96, 32, 200)[item] for item in expected)
    assert actual == expected_starts, f"rank={rank}; actual={actual}; expected={expected_starts}"
    _check_routed_records(rank, model, batch, pixels, records, expected)
    _check_visual_gradients(rank, model, batch, pixels, records, anchor, expected)
    if rank == 0:
        print("E0 full runtime forward/backward passed", flush=True)
    dist.destroy_process_group()
