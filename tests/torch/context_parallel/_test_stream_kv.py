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

"""Distributed stream KV tests against independent CPU attention gradients."""

import gc
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch_npu  # pylint: disable=unused-import

from hyper_parallel import init_device_mesh
from hyper_parallel.components.modules.gqa_attention import GatedGQAAttention
from hyper_parallel.core.activation_memory import checkpoint as hyper_checkpoint
from hyper_parallel.distributed.context_parallel import StreamKVConfig, StreamKVGQAAttention


def _setup():
    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("hccl")
    return dist.get_rank(), dist.get_world_size()


def _assert_close(actual, expected, name):
    """Require both absolute and relative accuracy; report the failing quantity."""
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    difference = actual - expected
    absolute = difference.abs().max().item()
    relative = (difference.norm() / expected.norm().clamp_min(1e-30)).item()
    assert bool(torch.isfinite(actual).all()), f"Expected finite {name} on rank {dist.get_rank()}"
    assert absolute <= .12, f"Expected {name} max-abs <= 0.12, got {absolute} on rank {dist.get_rank()}"
    assert relative <= .035, f"Expected {name} relative-L2 <= 0.035, got {relative} on rank {dist.get_rank()}"


def _dense_attention(query, key, value, incoming):
    operands = [tensor.double().requires_grad_(True) for tensor in (query, key, value)]
    q, k, v = operands
    mapping = torch.arange(q.shape[1]) // (q.shape[1] // k.shape[1])
    scores = (q @ k[:, mapping].transpose(-1, -2)) * q.shape[-1]**-0.5
    scores = scores.masked_fill(torch.ones(q.shape[2], k.shape[2], dtype=torch.bool).triu(1), -torch.inf)
    output = scores.softmax(-1) @ v[:, mapping]
    return output.detach(), *torch.autograd.grad(output, operands, incoming.double())


def test_stream_kv_attention():
    """Check supported Ulysses degrees, replica SUM and one-token tail panels."""
    rank, world = _setup()
    length, local = world * 16, 16
    for seed in (271, 913):
        generator = torch.Generator().manual_seed(seed)
        query = torch.randn(1, 32, length, 256, generator=generator).bfloat16()
        key, value = [torch.randn(1, 2, length, 256, generator=generator).bfloat16() for _ in range(2)]
        incoming = torch.randn(query.shape, generator=generator).bfloat16()
        reference = _dense_attention(query, key, value, incoming)
        selection = slice(rank * local, (rank + 1) * local)
        for pu in (1, 2, 4, 8):
            if world % pu:
                continue
            mesh = init_device_mesh("npu", (world // pu, pu), mesh_dim_names=("kv", "u"))
            for config in (StreamKVConfig(t, w, h, causal_load_balance=balanced)
                           for balanced in (False, True) for t, w, h in ((7, 5, 2), (11, 32, None), (3, 2, 3))):
                interface = StreamKVGQAAttention(mesh["u"], mesh["kv"], config)
                operands = [tensor[:, :, selection].to("npu").requires_grad_(True)
                            for tensor in (query, key, value)]
                with torch.autocast("npu", dtype=torch.bfloat16):
                    output = interface.attention(*operands)
                gradients = torch.autograd.grad(output, operands, incoming[:, :, selection].to("npu"))
                for name, actual, expected in zip(("output", "dq", "dk", "dv"), (output, *gradients), reference):
                    label = f"{name}, seed={seed}, Pu={pu}, config={config}"
                    _assert_close(actual, expected[:, :, selection], label)
                del operands, gradients, output
    dist.barrier()


class _Norm(nn.Module):
    """Qwen-style zero-centered RMS norm preserving activation dtype."""

    def __init__(self, dim: int) -> None:
        """Create one zero-centered, shared-per-dimension scale."""
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Accumulate normalization in FP32, or FP64 for the CPU oracle.

        Args:
            tensor: Query or key activation with head dimension last.

        Returns:
            Normalized activation in the original dtype.
        """
        dtype = tensor.dtype
        values = tensor if dtype == torch.float64 else tensor.float()
        return (values * (values.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
                * (1 + self.weight)).to(dtype)


def _component(interface):
    source = nn.Module()
    source.config = SimpleNamespace(hidden_size=256, num_attention_heads=8, num_key_value_heads=2,
                                    head_dim=256, sliding_window=None)
    source.q_proj = nn.Linear(256, 8 * 2 * 256, bias=False)
    source.k_proj, source.v_proj = (nn.Linear(256, 2 * 256, bias=False) for _ in range(2))
    source.o_proj = nn.Linear(8 * 256, 256, bias=False)
    source.q_norm, source.k_norm = _Norm(256), _Norm(256)
    model = GatedGQAAttention(module=source, attention_interface=interface)
    generator = torch.Generator().manual_seed(619)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * .03)
            else:
                parameter.copy_((torch.randn(parameter.shape, generator=generator)
                                 * parameter.shape[1]**-.5).bfloat16().float())
    return model


def _cpu_interface(module, query, key, value, attention_mask=None, **kwargs):
    del module, attention_mask
    mapping = torch.arange(query.shape[1]) // (query.shape[1] // key.shape[1])
    scores = (query @ key[:, mapping].transpose(-1, -2)) * kwargs["scaling"]
    scores = scores.masked_fill(torch.ones(query.shape[2], key.shape[2], dtype=torch.bool).triu(1), -torch.inf)
    return (scores.softmax(-1) @ value[:, mapping]).transpose(1, 2), None


@pytest.mark.parametrize("balanced", (False, True))
def test_stream_kv_component(balanced):
    """Use actual GatedGQAAttention, FP32 parameters/AMP and three optimizer steps.

    CPU FP64 references consume the same BF16 execution values of the linears.
    Parameter SUM is performed once here, outside the attention interface. This
    is replicated-parameter CP qualification, not an FSDP/full-model claim.

    Args:
        balanced: Enable mirrored causal Q/KV layout when True.
    """
    rank, world = _setup()
    pu = min(4, world)
    mesh = init_device_mesh("npu", (world // pu, pu), mesh_dim_names=("kv", "u"))
    interface = StreamKVGQAAttention(mesh["u"], mesh["kv"], StreamKVConfig(7, 5, 2, causal_load_balance=balanced))
    model = _component(interface).to("npu")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    live_bytes = []
    local, length = 16, 16 * world
    selection = slice(rank * local, (rank + 1) * local)
    generator = torch.Generator().manual_seed(823)
    x, dy = [torch.randn(1, length, 256, generator=generator).bfloat16() for _ in range(2)]
    frequencies = 10000000.0**(-torch.arange(0, 64, 2).float() / 64)
    angles = torch.arange(length).float()[:, None] * frequencies[None]
    angles = torch.cat((angles, angles), -1)[None]
    cosine, sine = angles.cos().bfloat16(), angles.sin().bfloat16()
    for step in range(3):
        reference = _component(_cpu_interface).double()
        for (name, parameter), (_, actual) in zip(reference.named_parameters(), model.named_parameters()):
            values = actual.detach().cpu()
            parameter.data.copy_(values.double() if "norm" in name else values.bfloat16().double())
        rx = x.double().requires_grad_(True)
        expected = reference(rx, position_embeddings=(cosine.double(), sine.double()))[0]
        expected_gradients = torch.autograd.grad(expected, (rx, *reference.parameters()), dy.double())
        local_x = x[:, selection].to("npu").requires_grad_(True)
        positions = tuple(tensor[:, selection].to("npu") for tensor in (cosine, sine))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("npu", dtype=torch.bfloat16):
            if step == 2:
                output = hyper_checkpoint(lambda hidden: model(hidden, position_embeddings=positions)[0], local_x)
            elif step == 1:
                output = checkpoint(lambda hidden: model(hidden, position_embeddings=positions)[0],
                                    local_x, use_reentrant=False)
            else:
                output = model(local_x, position_embeddings=positions)[0]
        output.backward(dy[:, selection].to("npu"))
        for parameter in model.parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        actuals = (output, local_x.grad, *(parameter.grad for parameter in model.parameters()))
        expecteds = (expected[:, selection], expected_gradients[0][:, selection], *expected_gradients[1:])
        names = ("output", "dx", *(name for name, _ in model.named_parameters()))
        for name, actual, expected_value in zip(names, actuals, expecteds):
            _assert_close(actual, expected_value, f"{name}, step={step}, balanced={balanced}")
        del actuals, actual, local_x, output
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del reference, rx, expected, expected_gradients
        gc.collect()
        torch.npu.synchronize()
        live_bytes.append(torch.npu.memory_allocated())
    # Allow fixed checkpoint bookkeeping, but reject retained activation storage between steps.
    assert live_bytes[-1] <= live_bytes[0] + 4 * 2**20, (
        f"Expected allocated memory to remain within 4 MiB of step 0, got {live_bytes} on rank {rank}"
    )
    dist.barrier()
