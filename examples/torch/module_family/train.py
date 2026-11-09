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
"""Train the same model through public local or distributed construction APIs."""

from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.distributed as dist

from examples.torch.module_family import DecoderConfig, FamilyDecoder
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.build_options import FSDP2Config, get_device_type


logger = logging.getLogger(__name__)


def build_training_setup(cp_size: int, ep_size: int) -> DistributedSetup:
    """Configure the job; family selection remains inside the common builder.

    Args:
        cp_size: Context-parallel group size.
        ep_size: Expert-parallel group size within the same dense domain.
    """
    world = dist.get_world_size() if dist.is_initialized() else 1
    if min(cp_size, ep_size) < 1 or world % cp_size or world % ep_size or 4 % ep_size:
        raise ValueError("CP and EP must divide WORLD_SIZE; EP must divide the four experts")
    if world == 1:
        return DistributedSetup()
    device_type = get_device_type()
    if device_type == "cpu":
        raise ValueError("distributed training uses main HyperParallel FSDP2 and requires GPU/NPU")
    mesh = MeshContext(dp_size=world // cp_size, cp_size=cp_size, ep_size=ep_size,
                       dp_shard_size=world, edp_shard_size=world // ep_size)
    mesh.build_meshs(device_type, world)
    return DistributedSetup(mesh_context=mesh, strategy_config=FSDP2Config(
        dp_shard_size=world, edp_shard_size=world // ep_size))


def main() -> None:
    """Run ordinary AdamW training with import-time family registration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    logging.basicConfig(level=logging.INFO)
    device_type = get_device_type()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if device_type != "cpu":
        getattr(torch, device_type).set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world > 1:
        dist.init_process_group({"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}[device_type])
    try:
        setup = build_training_setup(args.cp_size, args.ep_size)
        config = DecoderConfig()
        model = HyperAutoModelForCausalLM.from_config(
            config, model_factory=FamilyDecoder, distributed_setup=setup, torch_dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        device = next(model.parameters()).device
        for _ in range(args.steps):
            # A real input pipeline selects this rank's contiguous sequence partition.
            hidden_states = torch.randn(2, 4, config.hidden_size, device=device)
            loss = model(hidden_states).square().mean()
            loss.backward()
            with SkipDTensorDispatch(no_skip={torch.zeros_like}):
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.info("Completed %s steps: %s / %s", args.steps,
                        type(model.attention).__name__, type(model.experts).__name__)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
