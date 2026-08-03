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
"""Distributed accuracy tests for Qwen3.5 linear-attention CP."""
import torch
import torch.distributed as dist

import hyper_parallel as hp
from hyper_parallel.core.context_parallel.linear_attention_context_parallel import (
    LinearAttentionContextParallel,
    _differentiable_all_to_all_shard,
)
from hyper_parallel.models.qwen3_5.model import Qwen3_5GatedDeltaNet
from tests.torch.utils import init_dist


_MODES = ("ulysses", "p2p", "all_gather")
_OUTPUT_MAX_ABS_TOL = 5e-2
_INPUT_GRAD_MAX_ABS_TOL = 5e-1
_MAX_REL_SCALE_TOL = 1e-1
_REL_L2_TOL = 5e-2
_PARAM_GRAD_REL_L2_TOL = 1e-1
_GRAD_NORM_REL_TOL = 1e-2


def _global_max(value):
    value = value.detach().float()
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def _global_relative_l2(actual, expected):
    diff_norm_sq = (actual.detach().float() - expected.detach().float()).square().sum()
    expected_norm_sq = expected.detach().float().square().sum()
    dist.all_reduce(diff_norm_sq, op=dist.ReduceOp.SUM)
    dist.all_reduce(expected_norm_sq, op=dist.ReduceOp.SUM)
    return diff_norm_sq.sqrt() / expected_norm_sq.sqrt().clamp_min(1e-12)


def _build_module(device):
    return Qwen3_5GatedDeltaNet(
        hidden_size=128,
        num_v_heads=8,
        num_k_heads=4,
        head_k_dim=16,
        head_v_dim=16,
        conv_kernel_size=4,
    ).to(device=device, dtype=torch.bfloat16)


def _check_single_token_a2a_round_trip(mesh, rank, device):
    """Cover batch=1/local-sequence=1 in both Ulysses redistribution directions."""
    local = (
        torch.arange(rank * 4, (rank + 1) * 4, device=device, dtype=torch.float32)
        .reshape(1, 1, 4)
        .requires_grad_(True)
    )
    head_shard = _differentiable_all_to_all_shard(
        local,
        mesh,
        split_dim=2,
        concat_dim=1,
    )
    restored = _differentiable_all_to_all_shard(
        head_shard,
        mesh,
        split_dim=1,
        concat_dim=2,
    )

    torch.testing.assert_close(restored, local)
    restored.sum().backward()
    torch.testing.assert_close(local.grad, torch.ones_like(local))


def _run_mode(mode, mesh, rank, world, device):
    """Compare one linear-attention CP mode with a full-sequence reference."""
    torch.manual_seed(20260726)
    reference = _build_module(device)
    candidate = _build_module(device)
    candidate.load_state_dict(reference.state_dict())
    LinearAttentionContextParallel(mode=mode).apply(candidate, mesh)

    full_seq = 128
    local_seq = full_seq // world
    local_slice = slice(rank * local_seq, (rank + 1) * local_seq)
    torch.manual_seed(20260727)
    full_input = torch.randn(
        1,
        full_seq,
        128,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    local_input = (
        full_input[:, local_slice].detach().clone().contiguous().requires_grad_(True)
    )

    expected_full = reference(full_input)
    actual = candidate(local_input)
    expected = expected_full[:, local_slice]
    output_max_abs = _global_max((actual - expected).abs().max())
    output_expected_max_abs = _global_max(expected.abs().max())
    output_max_rel_scale = output_max_abs / output_expected_max_abs.clamp_min(1e-12)
    output_rel_l2 = _global_relative_l2(actual, expected)

    torch.manual_seed(20260800 + rank)
    local_grad_output = torch.randn_like(actual)
    gathered_grad_output = [torch.empty_like(local_grad_output) for _ in range(world)]
    dist.all_gather(gathered_grad_output, local_grad_output)
    expected_full.backward(torch.cat(gathered_grad_output, dim=1))
    actual.backward(local_grad_output)

    input_grad_max_abs = _global_max(
        (local_input.grad - full_input.grad[:, local_slice]).abs().max()
    )
    input_grad_expected_max_abs = _global_max(
        full_input.grad[:, local_slice].abs().max()
    )
    input_grad_max_rel_scale = (
        input_grad_max_abs / input_grad_expected_max_abs.clamp_min(1e-12)
    )
    input_grad_rel_l2 = _global_relative_l2(
        local_input.grad,
        full_input.grad[:, local_slice],
    )
    param_grad_max_abs = torch.zeros((), device=device, dtype=torch.float32)
    param_grad_max_rel_scale = torch.zeros_like(param_grad_max_abs)
    param_grad_max_rel_l2 = torch.zeros_like(param_grad_max_abs)
    max_rel_scale_param = ""
    max_rel_l2_param = ""
    expected_grad_norm_sq = torch.zeros_like(param_grad_max_abs)
    actual_grad_norm_sq = torch.zeros_like(param_grad_max_abs)
    reference_params = dict(reference.named_parameters())
    for name, parameter in candidate.named_parameters():
        assert parameter.grad is not None, f"{name}.grad is missing"
        assert torch.isfinite(parameter.grad).all(), f"{name}.grad is not finite"
        actual_grad = parameter.grad.detach().float().clone()
        dist.all_reduce(actual_grad, op=dist.ReduceOp.SUM)
        assert torch.isfinite(actual_grad).all(), f"reduced {name}.grad is not finite"
        assert reference_params[name].grad is not None, f"reference {name}.grad is missing"
        expected_grad = reference_params[name].grad.detach().float()
        param_grad_max_abs = torch.maximum(
            param_grad_max_abs,
            (actual_grad - expected_grad).abs().max(),
        )
        grad_diff = actual_grad - expected_grad
        expected_max_abs = expected_grad.abs().max()
        expected_l2 = expected_grad.square().sum().sqrt()
        current_max_rel_scale = (
            grad_diff.abs().max() / expected_max_abs.clamp_min(1e-12)
        )
        current_rel_l2 = (
            grad_diff.square().sum().sqrt()
            / expected_l2.clamp_min(1e-12)
        )
        if current_max_rel_scale > param_grad_max_rel_scale:
            param_grad_max_rel_scale = current_max_rel_scale
            max_rel_scale_param = name
        if current_rel_l2 > param_grad_max_rel_l2:
            param_grad_max_rel_l2 = current_rel_l2
            max_rel_l2_param = name
        expected_grad_norm_sq += expected_grad.square().sum()
        actual_grad_norm_sq += actual_grad.square().sum()

    param_grad_max_abs = _global_max(param_grad_max_abs)
    param_grad_max_rel_scale = _global_max(param_grad_max_rel_scale)
    param_grad_max_rel_l2 = _global_max(param_grad_max_rel_l2)
    grad_norm_rel = (
        (actual_grad_norm_sq.sqrt() - expected_grad_norm_sq.sqrt()).abs()
        / expected_grad_norm_sq.sqrt().clamp_min(1e-12)
    )
    grad_norm_rel = _global_max(grad_norm_rel)
    if rank == 0:
        print(
            f"mode={mode} "
            f"output_max_abs={output_max_abs.item():.6e} "
            f"output_max_rel_scale={output_max_rel_scale.item():.6e} "
            f"output_rel_l2={output_rel_l2.item():.6e} "
            f"input_grad_max_abs={input_grad_max_abs.item():.6e} "
            f"input_grad_max_rel_scale={input_grad_max_rel_scale.item():.6e} "
            f"input_grad_rel_l2={input_grad_rel_l2.item():.6e} "
            f"param_grad_max_abs={param_grad_max_abs.item():.6e} "
            f"param_grad_max_rel_scale={param_grad_max_rel_scale.item():.6e} "
            f"param_grad_max_rel_l2={param_grad_max_rel_l2.item():.6e} "
            f"grad_norm_rel={grad_norm_rel.item():.6e}",
            flush=True,
        )
        print(
            f"mode={mode} max_rel_scale_param={max_rel_scale_param} "
            f"max_rel_l2_param={max_rel_l2_param}",
            flush=True,
        )

    # BF16 kernels may change reduction order across eager and fused backends.
    # Loose max-abs guards catch outliers; relative L2 guards broad numerical drift.
    assert output_max_abs.item() <= _OUTPUT_MAX_ABS_TOL
    assert input_grad_max_abs.item() <= _INPUT_GRAD_MAX_ABS_TOL
    assert output_max_rel_scale.item() <= _MAX_REL_SCALE_TOL
    assert input_grad_max_rel_scale.item() <= _MAX_REL_SCALE_TOL
    assert param_grad_max_rel_scale.item() <= _MAX_REL_SCALE_TOL
    assert output_rel_l2.item() <= _REL_L2_TOL
    assert input_grad_rel_l2.item() <= _REL_L2_TOL
    assert param_grad_max_rel_l2.item() <= _PARAM_GRAD_REL_L2_TOL
    assert grad_norm_rel.item() <= _GRAD_NORM_REL_TOL


def test_linear_attention_cp_forward_backward_accuracy():
    """All three CP modes match a full-sequence Gated DeltaNet reference."""
    rank, device_id = init_dist()
    world = dist.get_world_size()
    assert world == 2
    device = torch.device(f"npu:{device_id}")
    mesh = hp.init_device_mesh("npu", (world,), mesh_dim_names=("cp",))

    _check_single_token_a2a_round_trip(mesh, rank, device)
    for mode in _MODES:
        _run_mode(mode, mesh, rank, world, device)
        dist.barrier()
