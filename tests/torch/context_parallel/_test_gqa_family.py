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
"""Small GQA family regression over the real planner, HCCL and NPU FA."""

from copy import deepcopy
from datetime import timedelta
import os

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from hyper_parallel import init_device_mesh
from hyper_parallel.components.modules import (
    GroupedQueryAttention, GatedGroupedQueryAttention,
)
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan


class _AttentionModel(nn.Module):
    """Expose one family boundary to the production planner."""

    def __init__(self, attention):
        """Retain source configuration and the ordinary module call boundary."""
        super().__init__()
        self.self_attn = attention
        self.config = attention.config

    def forward(self, hidden_states, **kwargs):
        """Return the attention output for gradient comparison."""
        return self.self_attn(hidden_states, **kwargs)[0]


def _build(factory, source, *, cp=1, tp=1, degree=None, cp_comm_overlap=False, fuse_qkv=True):
    """Construct the family and apply its declared source-weight conversion."""
    arguments = {} if factory is GatedGroupedQueryAttention else {"fuse_qkv": fuse_qkv}
    attention = factory(module=source, **arguments)
    for transform in attention.make_transforms():
        converted = transform.operations[0].convert(
            dict(source.named_parameters()), transform.source_patterns, transform.target_patterns)
        with torch.no_grad():
            for name, value in converted.items():
                attention.get_parameter(name).copy_(value)
    if cp > 1:
        # The unsharded reference retains source conversion. The distributed side
        # starts from config, so the same cases exercise native CP/TP construction.
        config = deepcopy(source.config)
        config.qk_norm = True
        config.dtype = next(source.parameters()).dtype
        cp_degree = degree or cp
        with torch.device(next(source.parameters()).device):
            native = factory(config=config, context={"cp_size": cp, "tp_size": tp},
                             ulysses_cp_size=cp_degree, kv_cp_size=cp // cp_degree,
                             cp_comm_overlap=cp_comm_overlap, **arguments)
        state = attention.state_dict()
        if factory is GatedGroupedQueryAttention:
            # This test source is Qwen3.5: its Norm stores gamma - 1, whereas native
            # config uses direct gamma. This is an explicit test conversion, not native loading.
            state = {name: value + 1 if name in ("q_norm.weight", "k_norm.weight") else value
                     for name, value in state.items()}
        native.load_state_dict(state)
        attention = native
    return _AttentionModel(attention)


def _close(actual, expected):
    """Use existing MLA BF16 bounds plus a relative-L2 guard for accumulated gradients."""
    actual, expected = actual.detach().float(), expected.detach().float()
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=0.06)
    if expected.norm() > 1e-6:
        relative = (actual - expected).norm() / expected.norm()
        assert relative < 0.025, relative.item()


def _check_parameter_gradients(parallel, reference, mesh, tp, tp_rank):
    """Compare accumulated gradients after the existing CP/TP SUM reductions."""
    for name, parameter in parallel.named_parameters():
        assert parameter.grad is not None, name
        dist.all_reduce(parameter.grad, group=mesh["cp"].get_group())
        rows = any(part in name for part in (".linear_qkv.", ".q_proj.", ".k_proj.", ".v_proj."))
        columns = name.endswith("o_proj.weight")
        expected_grad = reference.get_parameter(name).grad
        if tp > 1:
            if rows or columns:
                expected_grad = expected_grad.chunk(tp, 1 if columns else 0)[tp_rank]
            else:
                dist.all_reduce(parameter.grad, group=mesh["tp"].get_group())
        _close(parameter.grad, expected_grad)


def _case(mesh, factory, degree, packed, device, *, tp=1, sp=False, padding=False, fuse_qkv=True):
    """Compare overlap off/on with the same representation and unpartitioned reference."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3Config  # pylint: disable=C0415
    from transformers.models.qwen3_5.modeling_qwen3_5 import (  # pylint: disable=C0415
        Qwen3_5Attention, Qwen3_5TextConfig,
    )

    cp = mesh["cp"].size()
    cp_rank, tp_rank = mesh["cp"].get_local_rank(), mesh["tp"].get_local_rank()
    gated = factory is GatedGroupedQueryAttention
    config_type = Qwen3_5TextConfig if gated else Qwen3Config
    attention_type = Qwen3_5Attention if gated else Qwen3Attention
    config = config_type(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                         num_attention_heads=8, num_key_value_heads=4, head_dim=16, attention_bias=True)
    for overlap in (False, True):
        torch.manual_seed(724)
        source = attention_type(config, 0).to(device=device, dtype=torch.bfloat16)
        reference = _build(factory, deepcopy(source), fuse_qkv=fuse_qkv)
        parallel = _build(factory, deepcopy(source), cp=cp, tp=tp, degree=degree,
                          cp_comm_overlap=overlap, fuse_qkv=fuse_qkv)
        plan = ShardingPlanner().plan(parallel, mesh, tp_size=tp, cp_size=cp, sequence_parallel=sp)
        parallel, _ = apply_sharding_plan(parallel, plan, mesh)
        options = {"actual_seq_len": (7, 19, 41, 70, 97, 128)} if packed else {}
        if padding:
            mask = torch.ones(2, 64, device=device, dtype=torch.bool)
            mask[0, :9], mask[1, 55:] = False, False
            options["attention_mask"] = mask
        for microbatch in range(2):
            torch.manual_seed(881 + microbatch)
            inputs = torch.randn(2, 64, 128, dtype=torch.bfloat16).to(device).requires_grad_()
            target = torch.randn_like(inputs)
            frequency = torch.randn(2, 64, 4 if gated else 8, dtype=torch.bfloat16).repeat(1, 1, 2).to(device)
            positions = frequency.cos(), frequency.sin()
            local = inputs.detach().chunk(cp, 1)[cp_rank]
            if sp:
                local = local.chunk(tp, 1)[tp_rank]
            local = local.contiguous().requires_grad_()
            local_positions = tuple(part.chunk(cp, 1)[cp_rank] for part in positions)
            expected = reference(inputs, position_embeddings=positions, **options)
            actual = checkpoint(parallel, local, position_embeddings=local_positions,
                                use_reentrant=False, **options)
            expected_slice, local_target = (part.chunk(cp, 1)[cp_rank] for part in (expected, target))
            if sp:
                expected_slice, local_target = (part.chunk(tp, 1)[tp_rank]
                                                for part in (expected_slice, local_target))
            _close(actual, expected_slice)
            (expected.float() * target.float()).sum().div(inputs.numel() * 2).backward()
            # Replicated TP outputs repeat one logical loss. This is the existing
            # common TP SUM-adjoint contract; SP outputs own distinct tokens.
            (actual.float() * local_target.float()).sum().div(
                inputs.numel() * 2 * (1 if sp else tp)).backward()
            reference_dx = inputs.grad.chunk(cp, 1)[cp_rank]
            if sp:
                reference_dx = reference_dx.chunk(tp, 1)[tp_rank]
            elif tp > 1:
                dist.all_reduce(local.grad, group=mesh["tp"].get_group())
            _close(local.grad, reference_dx)
        _check_parameter_gradients(parallel, reference, mesh, tp, tp_rank)


def test_gqa_family_npu():
    """Feature: Family-owned GQA CP with packed boundaries and stable backward scheduling.
    Description: Cover grouped/split/gated, U/AG/hybrid, B2, TP/SP and checkpointed accumulation on four cards.
    Expectation: Outputs, dX and all parameter gradients agree with the same unsharded representation.
    """
    import torch_npu  # pylint: disable=C0415,unused-import

    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        mesh = init_device_mesh("npu", (4, 1), mesh_dim_names=("cp", "tp"))
        for factory, degree, packed, fused in (
            (GroupedQueryAttention, 4, False, True), (GroupedQueryAttention, 1, True, True),
            (GroupedQueryAttention, 2, True, False), (GatedGroupedQueryAttention, 4, True, True),
        ):
            _case(mesh, factory, degree, packed, device, fuse_qkv=fused)
        _case(mesh, GroupedQueryAttention, 1, False, device, padding=True)
        mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("cp", "tp"))
        _case(mesh, GatedGroupedQueryAttention, 2, False, device, tp=2, sp=True)
        _case(mesh, GroupedQueryAttention, 1, True, device, tp=2, sp=False, fuse_qkv=False)
    finally:
        dist.destroy_process_group()
