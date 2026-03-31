"""FSDP demonstration with MindSpore MultiLayerNets.

This script shows how to use fully_shard for distributed training.
Run with: msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True fsdp_demo.py
"""
# pylint: disable=C0413
import os
os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

from mindspore import nn
from mindspore import communication as dist
from mindspore._c_expression import NoFallbackGuard
from mindspore import mint

from hyper_parallel import init_device_mesh
from hyper_parallel.core.fully_shard.api import fully_shard


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
