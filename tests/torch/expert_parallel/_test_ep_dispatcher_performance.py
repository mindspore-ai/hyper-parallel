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
"""Torchrun worker for ExpertParallel token dispatcher performance comparison."""
import os
import time

import torch
import torch.distributed as dist

from hyper_parallel import init_device_mesh
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel
from hyper_parallel.platform.torch.common import MoE
from tests.torch.utils import _DEVICE_TYPE, init_backend


def _sync_device() -> None:
    """Synchronize the current backend device before/after timing windows."""
    if _DEVICE_TYPE == "npu":
        torch.npu.synchronize()


def _build_moe(
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
    seed: int,
) -> MoE:
    """Build a deterministic MoE module on the target device."""
    torch.manual_seed(seed)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(seed)
    return MoE(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device)


def _run_step(moe: MoE, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one forward+backward step and return detached output and input grad."""
    moe.zero_grad(set_to_none=True)
    x_in = x.detach().clone().requires_grad_(True)
    out = moe(x_in)
    out.sum().backward()
    return out.detach(), x_in.grad.detach()


def _assert_close(result: torch.Tensor, reference: torch.Tensor, label: str) -> None:
    """Assert two dispatcher outputs are numerically close."""
    result_cpu = result.detach().cpu()
    reference_cpu = reference.detach().cpu()
    max_diff = (result_cpu - reference_cpu).abs().max().item()
    assert torch.allclose(result_cpu, reference_cpu, rtol=1e-3, atol=1e-3), (
        f"{label} mismatch: max_diff={max_diff:.6f}, "
        f"result_shape={tuple(result.shape)}, reference_shape={tuple(reference.shape)}"
    )


def _measure_step_time(moe: MoE, x: torch.Tensor, warmup_steps: int, perf_steps: int) -> float:
    """Measure max-rank average step time in milliseconds."""
    for _ in range(warmup_steps):
        _run_step(moe, x)

    dist.barrier()
    _sync_device()
    start = time.perf_counter()
    for _ in range(perf_steps):
        _run_step(moe, x)
    _sync_device()
    elapsed = time.perf_counter() - start

    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float32, device=x.device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    return elapsed_tensor.item() * 1000.0 / perf_steps


def test_ep_dispatcher_performance_compare_npu():
    """
    Feature: ExpertParallel token dispatcher performance comparison.
    Description:
        1. Build two identical EP MoE modules with the same weights and input.
        2. Apply ``ExpertParallel(token_dispatcher="all_to_all")`` on a 1-D EP mesh.
        3. Apply ``ExpertParallel(token_dispatcher="deredundency")`` on a 2-D
           ``[oep=2, iep=2]`` EP mesh.
        4. Check forward output and input gradient consistency.
        5. Measure average forward+backward step time for both dispatchers.
    Expectation: Run success and print timing comparison on rank0.
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError(f"This performance case expects 4 ranks, got world_size={world_size}.")

    device = torch.device(_DEVICE_TYPE)
    dim, hidden_dim = 256, 512
    batch_size, seq_len = 4, 128
    num_experts = world_size * 2
    top_k = 2
    model_seed = 2026
    input_seed = 2027
    warmup_steps = int(os.environ.get("HP_EP_DISPATCHER_PERF_WARMUP", "3"))
    perf_steps = int(os.environ.get("HP_EP_DISPATCHER_PERF_STEPS", "10"))

    all_to_all_moe = _build_moe(dim, hidden_dim, num_experts, top_k, device, model_seed)
    deredundency_moe = _build_moe(dim, hidden_dim, num_experts, top_k, device, model_seed)

    all_to_all_mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(world_size,),
        mesh_dim_names=("ep",),
    )
    deredundency_mesh = init_device_mesh(
        device_type=_DEVICE_TYPE,
        mesh_shape=(2, 2),
        mesh_dim_names=("oep", "iep"),
    )

    ExpertParallel(token_dispatcher="all_to_all").apply(all_to_all_moe.experts, all_to_all_mesh)
    ExpertParallel(token_dispatcher="deredundency").apply(deredundency_moe.experts, deredundency_mesh)

    torch.manual_seed(input_seed)
    if _DEVICE_TYPE == "npu":
        torch.npu.manual_seed(input_seed)
    x = torch.randn(batch_size, seq_len, dim, device=device)

    all_out, all_grad = _run_step(all_to_all_moe, x)
    der_out, der_grad = _run_step(deredundency_moe, x)
    _assert_close(der_out, all_out, f"rank{rank} dispatcher forward output")
    _assert_close(der_grad, all_grad, f"rank{rank} dispatcher input gradient")

    all_to_all_avg_ms = _measure_step_time(all_to_all_moe, x, warmup_steps, perf_steps)
    deredundency_avg_ms = _measure_step_time(deredundency_moe, x, warmup_steps, perf_steps)

    assert all_to_all_avg_ms > 0.0
    assert deredundency_avg_ms > 0.0
    ratio = deredundency_avg_ms / all_to_all_avg_ms
    if rank == 0:
        print(
            "[EP_DISPATCHER_PERF] "
            f"all_to_all_avg_ms={all_to_all_avg_ms:.3f}, "
            f"deredundency_avg_ms={deredundency_avg_ms:.3f}, "
            f"ratio={ratio:.3f}, "
            f"warmup_steps={warmup_steps}, perf_steps={perf_steps}, "
            f"tokens={batch_size * seq_len}, routed_tokens={batch_size * seq_len * top_k}, "
            f"num_experts={num_experts}"
        )
