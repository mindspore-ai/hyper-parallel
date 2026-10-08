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
"""JT MLA Ulysses preserving the model's rotary and QK observation contracts."""

# This model adapter uses the same native Torch runtime as JT.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from functools import wraps
from types import MethodType, SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.utils.communication import differentiable_all_reduce
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.mla_context_parallel import MLAContextParallel, MLACPStrategy
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import JTDeepseekV3MLAAttention, JTDeepseekV3MoE


def _head_projection_request(projection: nn.Linear, head_dim: int) -> _ForwardRewriteRequest:
    """Slice rows inside Linear.forward so parameter materialization hooks still run."""
    original = projection.forward

    @wraps(original)
    def project(input: torch.Tensor, *, mla_head_range: tuple[int, int] | None = None) -> torch.Tensor:
        """Project selected heads using the original managed parameters."""
        if mla_head_range is None:
            return original(input)
        begin, end = mla_head_range
        if not 0 <= begin < end <= projection.weight.shape[0] // head_dim:
            raise ValueError("MLA head range is outside the TP-local parameter")
        rows = slice(begin * head_dim, end * head_dim)
        bias = None if projection.bias is None else projection.bias[rows]
        # Pylint cannot infer this Torch C-extension callable.
        return F.linear(input, projection.weight[rows], bias)  # pylint: disable=not-callable

    return _ForwardRewriteRequest(projection, project)


def _latent_inputs(module: JTDeepseekV3MLAAttention, runtime: MLAContextParallel,
                   hidden: torch.Tensor, positions: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retain JT's Norm, shared-key gather and native-dtype rotary operations."""
    query_latent, kv_latent, key_rope = module.project_latent_inputs(hidden)
    batch, sequence = query_latent.shape[:2]
    query = module.q_b_proj(query_latent).reshape(batch, sequence, module.num_heads, module.qk_head_dim)
    query_pass, query_rope = query.split((module.qk_nope_head_dim, module.qk_rope_head_dim), dim=-1)
    cos, sin = positions
    query = torch.cat((query_pass.transpose(1, 2),
                       module.explicit_rotary(query_rope.transpose(1, 2), cos, sin)), dim=-1)
    key_rope = module.explicit_rotary(key_rope.unsqueeze(1), cos, sin).squeeze(1)
    query, kv_latent, key_rope = runtime.latent(query, kv_latent, key_rope)
    kv_states = module.kv_b_proj(kv_latent.unsqueeze(1), mla_head_range=runtime.head_range)
    kv_states = kv_states.reshape(batch, sequence * runtime.degree, runtime.compute_heads,
                                  module.qk_nope_head_dim + module.v_head_dim).transpose(1, 2)
    key_pass, value = kv_states.split((module.qk_nope_head_dim, module.v_head_dim), dim=-1)
    key = torch.cat((key_pass, key_rope.unsqueeze(1).expand(-1, runtime.compute_heads, -1, -1)), dim=-1)
    return query, key, value


def _observed_attention(module: JTDeepseekV3MLAAttention, runtime: MLAContextParallel,
                        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        sequence_ends: tuple[int, ...]) -> torch.Tensor:
    """Keep statistics in persistent TP-head coordinates for the existing clip reducer."""
    observer = SimpleNamespace(max_logits_val=None, is_causal=True)
    output, _ = module.attention_interface(
        observer, query.contiguous(), key.contiguous(), value.contiguous(), None,
        dropout=0.0, scaling=module.scaling, actual_seq_len=sequence_ends,
    )
    with torch.no_grad():
        maximum = observer.max_logits_val
        if maximum is None or maximum.numel() != runtime.compute_heads:
            raise RuntimeError("JT MLA attention must return one QK maximum per compute head")
        if module.max_logits_val is None:
            module.max_logits_val = maximum.new_zeros(runtime.local_heads)
        begin, end = runtime.head_range
        module.max_logits_val[begin:end].copy_(torch.maximum(module.max_logits_val[begin:end], maximum))
    return output


@inner_wrapper
def mla_cp_wrapper(target_module: nn.Module, mesh: Any, tp_mesh: Any, cp_mesh: Any, ep_mesh: Any,
                   *, strategy: MLACPStrategy = "expanded_ulysses") -> list[_ForwardRewriteRequest]:
    """Install JT's causal MLA CP boundary after TP and before FSDP.

    Args:
        target_module: JT MLA replacement with TP-local projection parameters.
        mesh: Framework mesh supplied by the rewriter.
        tp_mesh: Tensor-parallel axis.
        cp_mesh: Context-parallel axis with contiguous sequence shards.
        ep_mesh: Expert axis, unused by attention.
        strategy: Expanded QKV or latent KV communication.
    """
    del mesh, ep_mesh
    if not isinstance(target_module, JTDeepseekV3MLAAttention):
        raise ValueError("JT MLA CP requires JTDeepseekV3MLAAttention replacement")
    if getattr(target_module, "mla_cp_runtime", None) is not None:
        raise ValueError("MLA CP is already installed")
    tp_degree = 1 if tp_mesh is None else tp_mesh.size()
    heads = target_module.config.num_attention_heads // tp_degree
    runtime = MLAContextParallel(cp_mesh, heads, strategy)
    if target_module.num_heads != heads:
        raise ValueError("Install MLA CP after TP head sharding")
    if target_module.attention_dropout != 0 or target_module.sliding_window is not None:
        raise ValueError("JT MLA CP requires dropout=0 and full causal attention")
    for projection, width in ((target_module.q_b_proj, target_module.qk_head_dim),
                              (target_module.kv_b_proj, target_module.qk_nope_head_dim + target_module.v_head_dim)):
        weight = projection.weight.to_local() if isinstance(projection.weight, DTensor) else projection.weight
        # A subclass may change Linear.forward semantics, so require the exact class.
        if type(projection) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("MLA CP requires plain Linear projections")
        if weight.shape[0] != heads * width:
            raise ValueError("MLA CP requires plain Linear projections with TP-local head rows")
    if runtime.degree == 1:
        return []
    original = target_module.forward

    @wraps(original)
    def forward(hidden_states: torch.Tensor, position_embeddings: Any = None, attention_mask: Any = None,
                past_key_values: Any = None, actual_seq_len: Any = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        """Compute contiguous causal shards using the model's original projection children."""
        if attention_mask is not None or past_key_values is not None or position_embeddings is None:
            raise ValueError("JT MLA CP requires explicit RoPE positions, no mask and no KV cache")
        if kwargs.pop("use_cache", False) or kwargs.pop("output_attentions", False):
            raise ValueError("JT MLA CP does not support cached decoding or attention weights")
        kwargs.pop("position_ids", None)
        if kwargs:
            raise ValueError(f"Unsupported JT MLA CP options: {sorted(kwargs)}")
        local_sequence = position_embeddings[0].shape[-2]
        global_sequence = local_sequence * runtime.degree
        sequence_ends = (global_sequence,) if actual_seq_len is None else tuple(actual_seq_len)
        if sequence_ends == (local_sequence,):
            sequence_ends = (global_sequence,)
        if (not sequence_ends or sequence_ends[0] <= 0 or sequence_ends[-1] != global_sequence
                or any(left >= right for left, right in zip(sequence_ends, sequence_ends[1:]))):
            raise ValueError("JT MLA CP requires global increasing cumulative document ends")
        if hidden_states.shape[0] != 1:
            raise ValueError("JT MLA CP currently requires batch_size=1")
        if strategy == "expanded_ulysses":
            query, key, value = target_module._project_attention_inputs(
                hidden_states, position_embeddings, None)
            query, key, value = runtime.expanded(query, key, value)
        else:
            query, key, value = _latent_inputs(target_module, runtime, hidden_states, position_embeddings)
        if query.shape[2] != global_sequence:
            raise ValueError("JT MLA SP projection and RoPE sequence lengths disagree")
        output = runtime.restore(_observed_attention(target_module, runtime, query, key, value, sequence_ends))
        return target_module.o_proj(output.reshape(1, local_sequence, -1).contiguous()), None

    requests = [_ForwardRewriteRequest(target_module, forward, companion_attrs={"mla_cp_runtime": runtime})]
    if strategy == "latent_kv_head":
        requests.append(_head_projection_request(target_module.kv_b_proj,
                                                 target_module.qk_nope_head_dim + target_module.v_head_dim))
    return requests


class _JTTrainingContext:
    """Own token shifts and objective weighting only on explicitly parallelized models."""

    def __init__(self, model: nn.Module, mesh: Any) -> None:
        """Bind only the model and process groups used by JT training boundaries."""
        self.model = model
        self.cp_mesh = mesh.cp_mesh
        self.cp_group = self.cp_mesh.get_group()
        self.degree = self.cp_mesh.size()
        self.rank = self.cp_mesh.get_local_rank()
        self.statistics_group = mesh.device_mesh[("cp", "tp")]._flatten().get_group()
        self.tp_group = mesh.device_mesh["tp"].get_group()
        self.tp_size = mesh.tp_size
        self.loss_parallel = mesh.loss_parallel

    def prepare(self, module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Establish per-forward supervision counts and global RoPE positions."""
        del module
        tokens = kwargs.get("input_ids", args[0] if args else None)
        labels = kwargs.get("shift_labels", args[1] if len(args) > 1 else None)
        if tokens is None or labels is None or tokens.shape != labels.shape:
            raise ValueError("JT CP requires aligned input_ids and pre-shifted labels")
        count = labels.ge(0).sum()
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=self.cp_group)
        if count.item() == 0:
            raise ValueError("JT CP requires supervision somewhere in the global sequence")
        kwargs = dict(kwargs)
        kwargs["sequence_start"] = self.rank * tokens.shape[1]
        global_sequence = self.degree * tokens.shape[1]
        if kwargs.get("actual_seq_len") is None:
            kwargs["actual_seq_len"] = (global_sequence,)
        elif not tuple(kwargs["actual_seq_len"]) or tuple(kwargs["actual_seq_len"])[-1] != global_sequence:
            raise ValueError("JT CP document ends must cover the global input sequence")
        return args, kwargs

    def reduce_losses(self, module: nn.Module, args: tuple, output: Any) -> Any:
        """Replicate global objectives so zero-supervision shards retain their contributions."""
        del module, args
        names = tuple(output.loss)
        losses = differentiable_all_reduce(torch.stack([output.loss[name] for name in names]),
                                            "sum", self.cp_group)
        # Trainer weights replicas by foundation counts; autograd redistributes those weights
        # before FSDP averages parameter gradients, including ranks with zero local labels.
        output.loss = dict(zip(names, losses.unbind()))
        return output

    def shift_inputs(self, input_ids: torch.Tensor, labels: torch.Tensor,
                     loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather only the next shard's first token/target/mask, without wraparound."""
        values = (input_ids, labels, loss_mask)
        first = torch.stack([value[:, 0].to(torch.int64) for value in values], dim=-1).contiguous()
        gathered = [torch.empty_like(first) for _ in range(self.degree)]
        dist.all_gather(gathered, first, group=self.cp_group)
        following = gathered[self.rank + 1] if self.rank + 1 < self.degree else torch.zeros_like(first)
        return tuple(torch.cat((value[:, 1:], following[:, index:index + 1].to(value.dtype)), dim=1)
                     for index, value in enumerate(values))

    def token_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the local CE sum divided by the global count for this objective."""
        valid = (mask != 0) & (labels >= 0)
        count = valid.sum().float()
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=self.cp_group)
        numerator = self.model.loss_function(
            logits=logits, labels=None, vocab_size=self.model.config.vocab_size,
            shift_labels=labels.masked_fill(~valid, -100), num_items_in_batch=1,
        ).reshape(())
        return numerator / count.clamp_min(1)

    def routing_statistics(self, module: JTDeepseekV3MoE, indices: torch.Tensor,
                           scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate non-gradient routing counts over the sequence's CP/TP domain."""
        counts = torch.bincount(indices.flatten(), minlength=module.config.n_routed_experts).float()
        dist.all_reduce(counts, group=self.statistics_group)
        frequency = counts / counts.sum().clamp_min(1)
        normalized = scores / (scores.sum(-1, keepdim=True) + 1e-20)
        auxiliary = (normalized.mean(0) * frequency).sum() * scores.shape[-1] * module.config.moe_aux_loss_coeff
        if self.tp_size > 1:
            mean = auxiliary.detach().clone()
            dist.all_reduce(mean, group=self.tp_group)
            mean = mean / self.tp_size
            # Vocab-parallel CE removes the FSDP TP divisor; the replicated auxiliary objective still needs it.
            gradient = auxiliary / self.tp_size if self.loss_parallel else auxiliary
            auxiliary = gradient + mean - gradient.detach()
        # The model-level differentiable sum combines all CP contributions before Trainer weighting.
        auxiliary = auxiliary / self.degree
        return frequency.detach(), auxiliary


def configure_context_parallel(model: nn.Module, mesh: Any) -> None:
    """Bind JT supervision boundaries after attention, TP and FSDP installation.

    Args:
        model: Final JT model with explicit MLA CP wrappers selected in its plan.
        mesh: Framework mesh context, including CP and TP domains.
    """
    if mesh.cp_size <= 1:
        return
    if getattr(model, "jt_cp_context", None) is not None:
        raise ValueError("JT model context parallelism is already installed")
    attentions = [module for module in model.modules() if isinstance(module, JTDeepseekV3MLAAttention)]
    if not attentions or any(getattr(module, "mla_cp_runtime", None) is None for module in attentions):
        raise ValueError("JT CP requires explicit MLA wrappers for every trunk and MTP attention")
    context = _JTTrainingContext(model, mesh)
    model.jt_cp_context = context
    model.register_forward_pre_hook(context.prepare, with_kwargs=True)
    model.register_forward_hook(context.reduce_losses)
    model._token_loss = context.token_loss
    model.mtp.execution.shift_inputs = context.shift_inputs
    for module in model.modules():
        if isinstance(module, JTDeepseekV3MoE):
            module.routing_statistics = MethodType(context.routing_statistics, module)
    model.qk_clip_group = mesh.dp_cp_mesh.get_group()
