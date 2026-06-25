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
"""FSDP demonstration with MindSpore MultiLayerNets.

This script shows how to use fully_shard for distributed training.
Run with: msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True fsdp_demo.py
"""
import os

from mindspore import nn
from mindspore import communication as dist
from mindspore._c_expression import NoFallbackGuard
from mindspore import mint

# hyper_parallel selects its backend from HYPER_PARALLEL_PLATFORM at import time,
# so the environment must be configured before importing it.
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from hyper_parallel import init_device_mesh  # pylint: disable=C0413
from hyper_parallel.core.fully_shard.api import fully_shard  # pylint: disable=C0413


class MultiLayerNets(nn.Cell):
    """Multi-layer network for FSDP demonstration."""
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 16, weight_init="normal", bias_init="zeros")
        )

    def construct(self, x):
        """Flatten input and pass through the dense ReLU sequential layers."""
        x = self.flatten(x)
        return self.dense_relu_sequential(x)


def get_forward_fn(net):
    """Create forward function for the given network."""
    def _forward_fn(data):
        logits = net(data)
        dummy_loss = logits.mean()
        return dummy_loss
    return _forward_fn


if __name__ == "__main__":
    # Create 1D DeviceMesh (8 NPU cards)
    dist.init()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(8,))

    # Instantiate model
    model = MultiLayerNets()

    # Shard each sub-layer first
    for layer in model.dense_relu_sequential:
        fully_shard(layer, mesh=mesh)

    # Then shard the root cell
    fsdp_model = fully_shard(model, mesh=mesh)

    # Dummy data (batch_size, seq_len, d_model)
    batch_size, seq_len = 1, 2
    dummy_data = mint.randn(batch_size, seq_len, 256)

    # Optimizer and loss
    forward_fn = get_forward_fn(fsdp_model)
    optimizer = nn.Adam(fsdp_model.trainable_params(), learning_rate=1e-5)

    # Simple training loop
    for step in range(10):
        fsdp_model.zero_grad()
        loss = forward_fn(dummy_data)
        loss.backward()
        grads = tuple(param.grad for param in fsdp_model.trainable_params())
        with NoFallbackGuard():
            optimizer(grads)
        print(f"[step {step}] loss = {loss.item():.4f}")
