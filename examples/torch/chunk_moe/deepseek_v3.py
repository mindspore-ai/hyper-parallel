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
"""Train a local, randomly initialized DeepSeek-V3 model with Chunk MoE.

The default shape is a small integration example. --config reads a local HF
configuration only; this script never downloads weights, tokenizers or data.
"""
from __future__ import annotations

import argparse
import json
import os
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

from hyper_parallel import DeviceMesh, SkipDTensorDispatch, fully_shard
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.models.deepseek_v3.adapter.conversion.chunked_moe import DeepseekV3ChunkMoE
from hyper_parallel.models.replacement import (
    ModuleReplacementRequest, apply_module_replacements, compile_module_replacements,
)


def _config(path):
    if path:
        return DeepseekV3Config.from_json_file(path)
    return DeepseekV3Config.from_dict({
        "vocab_size": 256, "hidden_size": 256, "intermediate_size": 512, "moe_intermediate_size": 128,
        "num_hidden_layers": 3, "first_k_dense_replace": 1, "num_attention_heads": 4, "num_key_value_heads": 4,
        "q_lora_rank": 64, "kv_lora_rank": 32, "qk_nope_head_dim": 32, "qk_rope_head_dim": 32, "v_head_dim": 32,
        "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
        "norm_topk_prob": True, "routed_scaling_factor": 1.7, "use_cache": False, "attention_dropout": 0.0,
    })


def _replace(model, chunk_size):
    request = ModuleReplacementRequest(
        match=("model.layers.*.mlp",), target=DeepseekV3ChunkMoE, module_type=DeepseekV3MoE,
        kwargs={"chunk_size": chunk_size},
    )
    plan = compile_module_replacements(model, overrides=(request,))
    model, _ = apply_module_replacements(model, plan, weights_mapping=[])
    for layer in model.model.layers:
        layer.self_attn.forward = partial(checkpoint, layer.self_attn.forward, use_reentrant=False)
    return model


def _parallelize(model, mesh, fsdp_size):
    if mesh is not None:
        for layer in model.model.layers:
            if isinstance(layer.mlp, DeepseekV3ChunkMoE):
                ExpertParallel().apply(layer.mlp.experts, mesh["ep"])
    if fsdp_size > 1:
        # A parent boundary keeps all parameters alive through explicit chunk backward.
        # Do not independently fully_shard the bypassed experts child.
        fully_shard(model, mesh=mesh["fsdp"], reshard_after_forward=True,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                   cast_forward_inputs=False))
    return model


def _reduce_dense(model, mesh):
    if mesh is None:
        return
    with SkipDTensorDispatch():
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and ".mlp.experts." not in name:
                dist.all_reduce(parameter.grad, group=mesh["ep"].get_group(), async_op=True).wait()


def main() -> None:
    """Run a small real decoder/loss/optimizer example with explicit parallel ownership."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Local HF config JSON; parameters are randomly initialized")
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--fsdp-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=65)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.fsdp_size < 1 or world % args.fsdp_size or (world > 1 and args.device != "npu"):
        raise ValueError("Use NPU torchrun with world_size divisible by fsdp_size.")
    mesh = None
    rank = 0
    if args.device == "npu":
        # Optional backend; the CPU example does not require torch_npu.
        import torch_npu  # pylint: disable=import-outside-toplevel,unused-import
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world > 1:
        dist.init_process_group("hccl")
        rank = dist.get_rank()
        mesh = DeviceMesh("npu", torch.arange(world).reshape(args.fsdp_size, -1),
                          mesh_dim_names=("fsdp", "ep"))
    config = _config(args.config)
    if config.n_routed_experts % (world // args.fsdp_size):
        raise ValueError("The expert count must be divisible by EP size.")
    torch.manual_seed(831)
    model = _replace(DeepseekV3ForCausalLM(config), args.chunk_size)
    dtype = torch.float32 if args.device == "cpu" or args.fsdp_size > 1 else torch.bfloat16
    model.to(args.device, dtype=dtype).train()
    model = _parallelize(model, mesh, args.fsdp_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(901 + rank + step * world)
        inputs = torch.randint(0, config.vocab_size, (1, args.tokens), device=args.device)
        loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
        (loss / (world // args.fsdp_size)).backward()
        _reduce_dense(model, mesh)
        with SkipDTensorDispatch(no_skip={torch.zeros_like}):
            optimizer.step()
        if rank == 0:
            print(json.dumps({"step": step, "loss": loss.detach().item(), "ep": world // args.fsdp_size,
                              "fsdp": args.fsdp_size, "chunk": args.chunk_size}))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
