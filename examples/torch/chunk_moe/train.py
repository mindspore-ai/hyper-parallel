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
"""Train the existing common MoE with bounded source-token chunks."""
import argparse
import logging
import os

import torch
import torch.distributed as dist

from hyper_parallel import DeviceMesh, SkipDTensorDispatch
from hyper_parallel.components.modules.moe import MoE
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel

logger = logging.getLogger(__name__)


def _setup(device_type):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if device_type == "cpu":
        if world != 1:
            raise ValueError("This example uses CPU only as a single-process reference")
        return None, torch.float32
    # torch_npu is optional for the single-process CPU example.
    import torch_npu  # pylint: disable=C0415,unused-import

    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world == 1:
        return None, torch.bfloat16
    if 8 % world:
        raise ValueError("The example's eight experts must divide evenly across EP ranks")
    dist.init_process_group("hccl")
    return DeviceMesh("npu", list(range(world)), mesh_dim_names=("ep",)), torch.bfloat16


def _train(args, mesh, dtype):
    torch.manual_seed(23)
    model = MoE(128, 256, 8, top_k=2, score_before_experts=False,
                use_grouped_mm=args.device == "npu", chunk_size=args.chunk_size).to(args.device, dtype=dtype)
    if mesh is not None:
        ExpertParallel().apply(model.experts, mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    rank = dist.get_rank() if mesh is not None else 0
    world = dist.get_world_size() if mesh is not None else 1
    torch.manual_seed(101 + rank)
    inputs = torch.randn(1, args.tokens, 128, device=args.device, dtype=dtype)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        model.tokens_per_expert.zero_()
        loss = model(inputs).float().square().mean() / world
        loss.backward()
        with SkipDTensorDispatch(no_skip={torch.zeros_like}):
            if mesh is not None:
                # Independent source sequences share router weights; experts are EP shards, not replicas.
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not name.startswith("experts."):
                        dist.all_reduce(parameter.grad, async_op=True).wait()
            optimizer.step()
        if rank == 0:
            logger.info("step=%d local_loss=%g chunk_size=%d", step, loss.detach().item() * world, args.chunk_size)


def main() -> None:
    """Run a CPU example or a torchrun NPU job with one EP group."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if min(args.chunk_size, args.tokens, args.steps) < 1:
        parser.error("chunk-size, tokens and steps must be positive")
    logging.basicConfig(level=logging.INFO)
    mesh, dtype = _setup(args.device)
    try:
        _train(args, mesh, dtype)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
