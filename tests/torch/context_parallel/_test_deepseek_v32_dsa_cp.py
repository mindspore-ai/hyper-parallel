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
"""Two-card DeepSeek-V3.2 CANN DSA CP precision worker."""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Attention

from hyper_parallel.components.modules.dsa_attention import DeepseekV32DSAAttention
from hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel import (
    deepseek_v32_dsa_attention_cp_wrapper,
)


class _CPMesh:
    """Minimal CP mesh backed by the initialized world process group."""

    @staticmethod
    def size() -> int:
        """Return the CP world size."""
        return dist.get_world_size()

    @staticmethod
    def get_local_rank() -> int:
        """Return this process's CP rank."""
        return dist.get_rank()

    @staticmethod
    def get_group() -> Any:
        """Return the process group used by CP collectives."""
        return dist.group.WORLD


def _build_attention(loss_coeff: float) -> DeepseekV32DSAAttention:
    """Build a kernel-compatible cropped DeepSeek-V3.2 attention module."""
    config = DeepseekV32Config(  # pylint: disable=unexpected-keyword-arg
        hidden_size=512,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=64,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=2048,
        attention_dropout=0.0,
    )
    config.dsa_loss_coeff = loss_coeff
    source = DeepseekV32Attention(config, layer_idx=0)
    return DeepseekV32DSAAttention(module=source)


def _position_embeddings(seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic Transformers-format rotary embeddings."""
    positions = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
    frequencies = torch.arange(32, dtype=torch.float32).view(1, 1, 32) / 32
    angles = positions * frequencies
    return (
        torch.cat((angles.cos(), angles.cos()), dim=-1).to(device=device, dtype=torch.bfloat16),
        torch.cat((angles.sin(), angles.sin()), dim=-1).to(device=device, dtype=torch.bfloat16),
    )


def _install_cp_forward(module: DeepseekV32DSAAttention):
    """Return the wrapper forward after installing its companion attributes."""
    request = deepseek_v32_dsa_attention_cp_wrapper(module, None, None, _CPMesh(), None)
    for name, value in request.companion_attrs.items():
        setattr(module, name, value)
    return request.forward


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[float, float]:
    """Compare tensors and include auditable error metrics on failure."""
    actual_fp32 = actual.detach().float().cpu()
    expected_fp32 = expected.detach().float().cpu()
    error = actual_fp32 - expected_fp32
    relative_l2 = error.norm() / expected_fp32.norm().clamp_min(1.0e-12)
    max_abs = error.abs().max()
    assert torch.isfinite(actual_fp32).all(), f"{name} contains non-finite values"
    assert relative_l2 <= rtol, f"{name} relative L2={relative_l2}, allowed={rtol}; max_abs={max_abs}"
    assert max_abs <= atol, f"{name} max_abs={max_abs}, allowed={atol}; relative L2={relative_l2}"
    return float(relative_l2), float(max_abs)


def _average_cp_parameter_gradients(module: torch.nn.Module) -> None:
    """Apply the CP replica averaging performed by the training sharding path."""
    for parameter in module.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(dist.get_world_size())


def _run_case(loss_coeff: float, actual_seq_len: list[int] | None, label: str) -> None:
    """Compare one auxiliary-loss setting at CP1 and CP2."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(197)
    reference = _build_attention(loss_coeff)
    candidate = deepcopy(reference)
    device = torch.device("npu", rank)
    reference = reference.to(device=device, dtype=torch.bfloat16).train()
    candidate = candidate.to(device=device, dtype=torch.bfloat16).train()
    cp_forward = _install_cp_forward(candidate)

    batch_size = 1
    global_seq_len = 32
    local_seq_len = global_seq_len // world_size
    generator = torch.Generator().manual_seed(293)
    source = torch.randn(batch_size, global_seq_len, 512, generator=generator, dtype=torch.float32)
    reference_input = source.to(device=device, dtype=torch.bfloat16).requires_grad_()
    local_input = source[:, rank * local_seq_len : (rank + 1) * local_seq_len]
    local_input = local_input.to(device=device, dtype=torch.bfloat16).requires_grad_()
    cos, sin = _position_embeddings(global_seq_len, device)
    local_cos = cos[:, rank * local_seq_len : (rank + 1) * local_seq_len]
    local_sin = sin[:, rank * local_seq_len : (rank + 1) * local_seq_len]

    reference_output = reference(
        reference_input,
        position_embeddings=(cos, sin),
        position_ids=torch.arange(global_seq_len, device=device).unsqueeze(0),
        actual_seq_len=actual_seq_len,
    )[0]
    local_output = cp_forward(
        local_input,
        position_embeddings=(local_cos, local_sin),
        position_ids=torch.arange(
            rank * local_seq_len,
            (rank + 1) * local_seq_len,
            device=device,
        ).unsqueeze(0),
        actual_seq_len=actual_seq_len,
    )[0]
    output_metrics = _assert_close(
        f"output[rank={rank},aux={loss_coeff}]",
        local_output,
        reference_output[:, rank * local_seq_len : (rank + 1) * local_seq_len],
        rtol=0.025,
        atol=0.03,
    )

    reference_output.float().square().mean().backward()
    local_output.float().square().mean().backward()
    _average_cp_parameter_gradients(candidate)
    reference_slice = reference_input.grad[:, rank * local_seq_len : (rank + 1) * local_seq_len]
    input_metrics = _assert_close(
        f"input_gradient[rank={rank},aux={loss_coeff}]",
        local_input.grad / world_size,
        reference_slice,
        rtol=0.04,
        atol=0.003,
    )
    reference_gradients = dict(reference.named_parameters())
    parameter_metrics = (0.0, 0.0)
    for name, parameter in candidate.named_parameters():
        expected = reference_gradients[name].grad
        if parameter.grad is None or expected is None:
            assert parameter.grad is expected, f"gradient presence differs for {name}"
            continue
        metrics = _assert_close(
            f"parameter_gradient[{name},aux={loss_coeff}]",
            parameter.grad,
            expected,
            rtol=0.04,
            atol=0.003,
        )
        parameter_metrics = tuple(max(current, measured) for current, measured in zip(parameter_metrics, metrics))
    indexer_gradients = [
        parameter.grad
        for name, parameter in candidate.named_parameters()
        if name.startswith("indexer.")
    ]
    if loss_coeff:
        assert indexer_gradients and all(gradient is not None for gradient in indexer_gradients)
        assert any(bool(torch.count_nonzero(gradient)) for gradient in indexer_gradients)
    else:
        assert all(gradient is None for gradient in indexer_gradients)
    summary = torch.tensor(
        (*output_metrics, *input_metrics, *parameter_metrics),
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(summary, op=dist.ReduceOp.MAX)
    if rank == 0:
        values = summary.cpu().tolist()
        print(
            f"{label}: output(rel_l2={values[0]:.6g}, max_abs={values[1]:.6g}), "
            f"input_grad(rel_l2={values[2]:.6g}, max_abs={values[3]:.6g}), "
            f"parameter_grad(rel_l2={values[4]:.6g}, max_abs={values[5]:.6g})"
        )


def test_deepseek_v32_dsa_cp2_precision():
    """Match CP1 output, input gradients and averaged parameter gradients."""
    if not torch_npu.npu.is_available():
        raise RuntimeError("DeepSeek-V3.2 DSA CP precision requires Ascend NPU")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    try:
        assert dist.get_world_size() == 2
        dist.all_reduce(torch.ones(1, device=torch.device("npu", local_rank)))
        _run_case(loss_coeff=0.0, actual_seq_len=None, label="dense, auxiliary KL disabled")
        _run_case(loss_coeff=0.1, actual_seq_len=None, label="dense, auxiliary KL enabled")
        _run_case(loss_coeff=0.0, actual_seq_len=[7, 21, 32], label="packed, auxiliary KL disabled")
    finally:
        dist.destroy_process_group()
