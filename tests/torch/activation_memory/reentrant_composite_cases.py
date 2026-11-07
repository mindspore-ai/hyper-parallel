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
"""Two-rank NPU FSDP + reentrant checkpoint exclusion acceptance cases."""

import gc
import weakref

import torch
import torch.distributed as dist

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.activation_memory import checkpoint, checkpoint_exclude_wrapper, checkpoint_wrapper
from hyper_parallel.core.fully_shard.api import fully_shard
from tests.torch.utils import init_dist


class _ExcludedMatmul(torch.nn.Module):
    """Own the parameter whose sharded/unsharded identity changes across replay."""

    def __init__(self) -> None:
        """Create a trainable projection and its execution counter."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(64, 64) * 0.02)
        self.calls = 0

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        """Apply the parameterized excluded operator."""
        self.calls += 1
        return activation @ self.weight


class _Block(torch.nn.Module):
    """Keep a recomputable activation before the excluded matmul."""

    def __init__(self) -> None:
        """Wrap the projection in a shared checkpoint exclusion."""
        super().__init__()
        self.excluded = checkpoint_exclude_wrapper(_ExcludedMatmul())
        self.activation_storage = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Expose only a weak storage reference from the original forward."""
        activation = torch.nn.functional.gelu(value * 1.125).repeat(32, 1)
        if not torch.is_grad_enabled():
            self.activation_storage = weakref.ref(activation.untyped_storage())
        return torch.nn.functional.silu(self.excluded(activation))


class _Model(torch.nn.Module):
    """Host the checkpoint inside an actual fully_shard root."""

    def __init__(self, mode: str) -> None:
        """Select the wrapper, explicit API, or eager reference path."""
        super().__init__()
        block = _Block()
        self.block = checkpoint_wrapper(block, use_reentrant=True) if mode == "wrapper" else block
        self.mode = mode

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Use the requested public checkpoint entry point."""
        if self.mode == "wrapper":
            return self.block(value)
        if self.mode == "api":
            return checkpoint(self.block, value, use_reentrant=True)
        return self.block(value)


def _local(value: torch.Tensor) -> torch.Tensor:
    """Extract a local shard without running DTensor dispatch on the comparison."""
    return value.to_local() if isinstance(value, DTensor) else value


def _close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Include the phase and rank in parity failures."""
    torch.testing.assert_close(actual.detach().float(), expected.detach().float(), rtol=3e-3, atol=1e-5,
                               msg=lambda detail: f"rank {dist.get_rank()} {label}: {detail}")


def _run(mode: str) -> None:
    """Compare two sharded steps to a replicated eager reference on both ranks."""
    rank, device_id = init_dist()
    try:
        _run_steps(mode, rank, device_id)
    finally:
        dist.destroy_process_group()


def _run_steps(mode: str, rank: int, device_id: int) -> None:
    """Exercise parameter swaps, exclusion and optimizer updates in one process group."""
    assert dist.get_world_size() == 2, "this FSDP acceptance case requires exactly two ranks"
    device = torch.device("npu", device_id)
    mesh = init_device_mesh("npu", (2,), mesh_dim_names=("fsdp",))
    torch.manual_seed(712)
    reference = _Model("eager").to(device)
    torch.manual_seed(712)
    model = _Model(mode).to(device)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)
    model.set_reduce_op_type("avg")
    reference_weight = reference.block.excluded.weight
    sharded_weight = model.block.excluded.weight
    assert isinstance(sharded_weight, DTensor), "excluded weight was not sharded by fully_shard"
    assert _local(sharded_weight).shape[0] == reference_weight.shape[0] // 2
    shard = slice(rank * 32, (rank + 1) * 32)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    sharded_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for step in range(2):
        reference_optimizer.zero_grad(set_to_none=True)
        sharded_optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(920 + step)
        data = torch.randn(8, 64, device=device) * 0.1
        reference_input = data.detach().clone().requires_grad_()
        sharded_input = data.detach().clone().requires_grad_()
        expected = reference(reference_input).square().sum()
        expected.backward()
        with SkipDTensorDispatch():
            actual = model(sharded_input).square().sum()
        gc.collect()
        storage = model.block.activation_storage
        assert storage is not None, f"rank {rank} step {step}: checkpoint forward did not record activation"
        assert storage() is None, f"rank {rank} step {step}: excluded region retained recomputable storage"
        with SkipDTensorDispatch():
            actual.backward()
        _close(actual, expected, f"step {step} loss")
        assert reference_input.grad.abs().max().item() > 1e-3, f"rank {rank} step {step}: trivial reference dx"
        assert reference_weight.grad.abs().max().item() > 1e-3, f"rank {rank} step {step}: trivial reference dw"
        _close(sharded_input.grad, reference_input.grad, f"step {step} dx")
        assert sharded_weight.grad is not None, f"rank {rank} step {step}: missing sharded dw"
        _close(_local(sharded_weight.grad), reference_weight.grad[shard], f"step {step} dw")
        assert model.block.excluded.calls == step + 1, f"rank {rank} step {step}: excluded matmul replayed"
        torch.npu.synchronize()
        del actual, sharded_input
        gc.collect()
        assert storage() is None, f"rank {rank} step {step}: completed backward retained activation"
        with SkipDTensorDispatch():
            sharded_optimizer.step()
            reference_optimizer.step()
        _close(_local(sharded_weight), reference_weight[shard], f"step {step} updated weight")


def test_fsdp_checkpoint_wrapper_exclude() -> None:
    """Verify FSDP gradients and storage with checkpoint_wrapper on two NPUs."""
    _run("wrapper")


def test_fsdp_checkpoint_api_exclude() -> None:
    """Verify the explicit checkpoint(use_reentrant=True) API under FSDP."""
    _run("api")
