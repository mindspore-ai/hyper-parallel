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
"""FSDP demonstration with PyTorch SimpleTransformer.

This script shows how to use fully_shard for distributed training.
Run with: torchrun --nproc_per_node=8 fsdp_demo.py
"""
# pylint: disable=C0413
import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import nn
import torch.distributed as dist

from hyper_parallel import init_device_mesh, SkipDTensorDispatch
from hyper_parallel.core.fully_shard.api import fully_shard


def init_dist():
    """Initialize distributed training."""
    if not dist.is_initialized():
        dist.init_process_group()
    rank = dist.get_rank()
    torch.npu.set_device(rank)
    device_id = rank % 8
    torch.npu.set_device(device_id)
    return rank, device_id


class SimpleTransformer(nn.Module):
    """Simple transformer model for FSDP demonstration."""
    def __init__(self, num_layers=4, d_model=256, nhead=8, dim_ff=512):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_ff,
                                      dropout=0.1, activation="gelu")
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        """Pass input through transformer encoder layers and apply layer normalization."""
        out = src
        for l in self.layers:
            out = l(out)
        return self.norm(out)


if __name__ == "__main__":
    # Instantiate model
    transformer = SimpleTransformer(num_layers=4).npu()

    # Create 1D DeviceMesh (8 NPU cards)
    init_dist()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,))

    # Shard each sub-layer first
    for layer in transformer.layers:
        fully_shard(layer, mesh=mesh)

    # Then shard the root transformer
    fsdp_transformer = fully_shard(transformer, mesh=mesh)

    # Dummy data (batch_size, seq_len, d_model)
    batch_size, seq_len = 8, 16
    dummy_input = torch.randn(batch_size, seq_len, 256).npu()

    # Optimizer and loss
    optimizer = torch.optim.Adam(fsdp_transformer.parameters(), lr=1e-4)

    # Simple training loop
    with SkipDTensorDispatch():
        for step in range(2):
            optimizer.zero_grad()
            # Forward pass
            logits = fsdp_transformer(dummy_input)  # shape: (batch, seq_len, d_model)
            # For illustration, use dummy loss (mean of logits)
            loss = logits.mean()
            loss.backward()
            optimizer.step()
            print(f"[step {step}] loss = {loss.item():.4f}")
