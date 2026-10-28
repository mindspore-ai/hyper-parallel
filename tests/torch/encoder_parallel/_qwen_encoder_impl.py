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
"""Two-process CPU oracle using the actual HF Qwen3.5-MoE vision model."""

import copy
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel, Qwen3_5MoeVisionAttention

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime
from hyper_parallel.models.qwen3_5_moe.adapter.encoder_parallel import (
    get_encoder_parallel_spec, replace_vision_attention,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec, apply_module_replacements, compile_module_replacements,
)


def _compare(encoder_cp_size: int) -> None:
    """Compare routed features and summed parameter gradients against native HF."""
    dist.init_process_group("gloo")
    try:
        torch.set_num_threads(1)
        torch.manual_seed(23)
        config = Qwen3_5MoeVisionConfig(hidden_size=16, intermediate_size=32, num_heads=4, depth=2,
                                       out_hidden_size=16, patch_size=2, spatial_merge_size=2,
                                       temporal_patch_size=2, num_position_embeddings=16)
        # Exercise the shared internal integration contract; no public equivalent exists.
        config._attn_implementation = "sdpa"  # pylint: disable=protected-access
        model = nn.Module()
        model.config = SimpleNamespace(model_type="qwen3_5_moe", image_token_id=99)
        model.model = nn.Module()
        model.model.visual = Qwen3_5MoeVisionModel(config)
        oracle = copy.deepcopy(model.model.visual)
        rule = ModuleReplacementSpec(("model.visual.blocks.*.attn",), replace_vision_attention,
                                     Qwen3_5MoeVisionAttention)
        apply_module_replacements(model, compile_module_replacements(model, (rule,)))
        mesh = init_device_mesh("cpu", (1, 1, 2), mesh_dim_names=("dp", "cp", "tp"))
        context = SimpleNamespace(device_mesh=mesh, cp_size=1, tp_size=2, pp_size=1,
                                  cp_rank=0, tp_rank=dist.get_rank(), sequence_parallel=False, loss_parallel=False)
        runtime = EncoderParallelRuntime(config=EncoderParallelConfig(True, encoder_cp_size), mesh_context=context,
                                         model=model, spec=get_encoder_parallel_spec(),
                                         execution_options={"micro_batch_size": 1})
        for grids in ([[1, 2, 4], [1, 4, 4]], [[1, 2, 2]]):
            model.zero_grad(set_to_none=True)
            oracle.zero_grad(set_to_none=True)
            pixels = torch.randn(sum(t * h * w for t, h, w in grids), 24)
            token_ids = [1]
            for temporal, height, width in grids:
                token_ids.extend([99] * (temporal * height * width // 4) + [2])
            batch = {"input_ids": torch.tensor([token_ids]), "pixel_values": pixels,
                     "image_grid_thw": torch.tensor(grids)}
            plan = runtime.finalize_plan(*runtime.preflight_batch(batch))
            records, anchor = plan.encode_and_route(batch)
            actual = torch.cat([feature for _, feature in records])
            expected = oracle(pixels, grid_thw=batch["image_grid_thw"]).pooler_output
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            (actual.square().sum() / dist.get_world_size() + anchor).backward()
            expected.square().sum().backward()
            reference = dict(oracle.named_parameters())
            for name, parameter in model.model.visual.named_parameters():
                if parameter.grad is None:
                    raise AssertionError(f"Missing gradient: {name}")
                dist.all_reduce(parameter.grad)
                torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=2e-4, atol=2e-5,
                                           msg=lambda message, name=name: f"{name}: {message}")
    finally:
        dist.destroy_process_group()


def test_encoder_dp() -> None:
    """Run Encoder-DP=2 / Encoder-CP=1, including an empty producer bucket."""
    _compare(1)


def test_encoder_cp() -> None:
    """Run Encoder-DP=1 / Encoder-CP=2 with native HF position encodings."""
    _compare(2)
