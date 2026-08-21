# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""perf_kernels: 用户自定义高性能 kernel 库（perf_replacement 示例的 YAML 载体）。

真实场景中这是用户自己的 kernel 包（flash-attn、融合算子等），只要在
PYTHONPATH 上可 import，就能被 YAML 的 ``_target_`` 引用。本文件演示两条
计算注入通道和一条模块替换通道：

- **local_compute_fn 工厂契约**（`flash_attention_factory` /
  `fused_swiglu_factory`）：apply 时框架以通用上下文（module/mesh/
  expert_mesh，按签名过滤）把工厂 build 一次，工厂返回区域 compute fn
  ``fn(module, *local_args)``——在 local-region 骨架内以本地张量执行，
  参数解包与 I/O 契约由骨架托管，kernel 只需关注计算本身；
- **inner_wrapper 契约**（`flash_attention_wrapper`）：
  ``@inner_wrapper fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh)``
  原地替换 target.forward。上述计算 kernel 都是纯标准算子（可 dispatch），
  示例声明 region_dispatch=True——validate 下 dispatch 穿透注入实现，
  out_src 真校验（不再是适配器 to_local + 声明式重包的黑盒路径）；
- **replace_module 契约**（`CheckpointMappedLinear` / `NpuGroupedMoe`）：
  替换完整 `nn.Module`，并由 replacement 的 `make_transforms()` 声明相对
  source module 到 replacement 的参数重命名或布局转换。

两个计数器（FLASH_CALLS/FUSED_CALLS）用于证明替换真正生效。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.core_model_loading import Transpose, WeightConverter, WeightRenaming

from hyper_parallel.auto_models.components.distributed.injection import (
    inner_wrapper,
    local_compute,
)
from hyper_parallel.auto_models.components.model_transform import module_replacement

FLASH_CALLS = {"n": 0}
FUSED_CALLS = {"n": 0}


@module_replacement
class CheckpointMappedLinear(nn.Module):
    """Linear implementation with a replacement-specific parameter schema."""

    def __init__(self, *, module, module_fqn, context):
        super().__init__()
        del module_fqn, context
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.register_parameter("packed_weight", module.weight)
        self.register_parameter("bias", module.bias)
        self.train(module.training)

    def forward(self, input):
        return F.linear(input, self.packed_weight, self.bias)

    def make_transforms(self):
        return [WeightRenaming("weight", "packed_weight")]


class _GroupedMatmul(torch.autograd.Function):
    """Autograd wrapper consuming checkpoint-transposed ``[E, K, N]`` weights."""

    @staticmethod
    def forward(ctx, inputs, weight, tokens_per_expert):
        import torch_npu  # pylint: disable=import-outside-toplevel

        ctx.save_for_backward(inputs, weight)
        ctx.tokens_per_expert = tokens_per_expert
        return torch_npu.npu_grouped_matmul(
            [inputs],
            [weight],
            bias=None,
            group_list=tokens_per_expert,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    @staticmethod
    def backward(ctx, grad_output):
        import torch_npu  # pylint: disable=import-outside-toplevel

        inputs, weight = ctx.saved_tensors
        tokens_per_expert = ctx.tokens_per_expert
        grad_inputs = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight.transpose(1, 2).contiguous()],
            bias=None,
            group_list=tokens_per_expert,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        grad_weight = torch_npu.npu_grouped_matmul(
            [inputs.transpose(0, 1)],
            [grad_output],
            bias=None,
            group_list=tokens_per_expert,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        return grad_inputs, grad_weight, None


@module_replacement
class NpuGroupedMoe(nn.Module):
    """Complete MoE replacement holding checkpoint-transposed expert parameters."""

    def __init__(self, *, module, module_fqn, context):
        super().__init__()
        del module_fqn, context
        self.gate = module.gate
        self.experts = nn.Module()
        self.experts.num_experts = module.experts.num_experts

        gate_up_proj = module.experts.gate_up_proj
        self.experts.gate_up_proj = nn.Parameter(
            gate_up_proj.detach().transpose(1, 2).contiguous(),
            requires_grad=gate_up_proj.requires_grad,
        )

        down_proj = module.experts.down_proj
        self.experts.down_proj = nn.Parameter(
            down_proj.detach().transpose(1, 2).contiguous(),
            requires_grad=down_proj.requires_grad,
        )
        self.train(module.training)

    def forward(self, hidden_states):
        import torch_npu  # pylint: disable=import-outside-toplevel

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states)
        permuted_states, row_ids_map = torch_npu.npu_moe_token_permute(
            hidden_states,
            selected_experts.to(torch.int32),
        )
        tokens_per_expert = torch.histc(
            selected_experts,
            bins=self.experts.num_experts,
            min=0,
            max=self.experts.num_experts,
        ).to(torch.int64)
        gate_up_output = _GroupedMatmul.apply(
            permuted_states,
            self.experts.gate_up_proj,
            tokens_per_expert,
        )
        activated_states = torch_npu.npu_swiglu(gate_up_output, dim=-1)
        expert_output = _GroupedMatmul.apply(
            activated_states,
            self.experts.down_proj,
            tokens_per_expert,
        )
        output = torch_npu.npu_moe_token_unpermute(
            expert_output,
            row_ids_map,
            probs=routing_weights,
        )
        return output.reshape(batch_size, sequence_length, hidden_dim)

    def make_transforms(self):
        return [
            WeightConverter(
                source_patterns="experts.gate_up_proj",
                target_patterns="experts.gate_up_proj",
                operations=[Transpose(dim0=1, dim1=2)],
            ),
            WeightConverter(
                source_patterns="experts.down_proj",
                target_patterns="experts.down_proj",
                operations=[Transpose(dim0=1, dim1=2)],
            ),
        ]


def _fast_attention(module, hidden_states):
    """共享的 fast attention 数学：一次 F.sdpa 替换 eager 多 kernel 路径。

    TP 语义不在 kernel 里：q/k/v_proj 列切（头被分到各 rank）、o_proj 行切
    （输出 partial 由边界按契约 all-reduce）都是参数分片的事，kernel 看到的
    就是"本地 shard 上的标准 attention"。
    """
    FLASH_CALLS["n"] += 1
    b, s, _ = hidden_states.shape
    q = module.q_proj(hidden_states).view(b, s, -1, module.head_dim)
    k = module.k_proj(hidden_states).view(b, s, -1, module.head_dim)
    v = module.v_proj(hidden_states).view(b, s, -1, module.head_dim)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return module.o_proj(o.transpose(1, 2).reshape(b, s, -1))


@local_compute
def flash_attention_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
    """local_compute_fn 通道：@local_compute 是强制纪律（声明上下文
    需求——mesh 家族四个参数为必选（框架填充，本例不使用）；工厂返回
    compute_fn。compute_fn 的入参必须与
    SlowCausalAttention.forward(hidden_states) 匹配（apply 时校验）。
    本 kernel 是纯标准算子（F.sdpa/线性层）——示例声明
    region_dispatch=True：validate 下 DTensor 直入本函数、策略传播穿透、
    out_src 真校验；production 恒 local 直通，两模式零分支差异。"""
    def compute_fn(module, hidden_states):
        return _fast_attention(module, hidden_states)
    return compute_fn


@inner_wrapper
def flash_attention_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """inner_wrapper 通道（registry 契约 + @inner_wrapper 强制纪律）：
    原地替换 target.forward。

    与 local_compute_fn 通道的差别：没有 local-region 骨架托管——但
    **双模转换同样不需要用户写**：本例 region_dispatch=True（纯算子
    替换），validate 下适配器 DTensor 直入、dispatch 穿透、按边界
    out_src 声明真校验；若注入物含通信/自定义 kernel 则声明 False，
    适配器转为黑盒托管（to_local / 参数临时解包 / 声明式重包）。
    本例与 CP 无关，框架填入的 cp_mesh 为 None。
    """

    def fast_forward(hidden_states, *args, **kwargs):
        return _fast_attention(target_module, hidden_states)

    target_module.forward = fast_forward


@local_compute
def fused_swiglu_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
    """高性能 MLP：融合的 F.silu 替换逐步分解的朴素 silu（一串小 kernel）。"""
    def compute_fn(module, hidden_states):
        FUSED_CALLS["n"] += 1
        return module.down_proj(
            F.silu(module.gate_proj(hidden_states))
            * module.up_proj(hidden_states))
    return compute_fn
