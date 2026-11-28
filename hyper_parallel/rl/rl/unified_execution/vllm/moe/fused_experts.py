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
"""Ascend BF16 Qwen3 MoE stages with slot-ordered output on each source rank."""

from dataclasses import replace
from math import gcd
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch.nn import functional as F


def expert_source_ranks(owners: list[int], domain: list[int], tp_size: int, sp: bool,
                      rank: int) -> list[int]:
    """Find the EP ranks holding distinct token rows for this TP lane.

    HP flattens DP × PCP × TP into consecutive expert groups. Only ranks in
    the same TP plane may replicate rows. SP makes every TP lane a source.
    Each row of this grid has identical tokens and each column covers every
    source. Non-contiguous placements can always treat every rank as a source.
    """
    if rank not in owners or not set(owners).issubset(domain):
        raise ValueError("Published expert shard is outside the native worker domain")
    width = 1 if sp else gcd(len(owners), tp_size)
    start = domain.index(owners[0])
    if start % width or owners != domain[start:start + len(owners)]:
        width = 1
    index = owners.index(rank)
    return owners[index % width::width]


def _row_group(ranks, available):
    if len(ranks) == 1:
        return None
    for group in available:
        if ranks == dist.get_process_group_ranks(group):
            return group
    # Different source lanes form disjoint groups in the published EP domain.
    return dist.new_group(ranks, use_local_synchronization=True)


class _ExpertWeights(NamedTuple):
    """Native BF16 kernel interface, backed by HP's published local tensors."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor


class AscendFusedExpertsBackend(torch.nn.Module):
    """Share EP communication resources and Ascend expert kernels across the worker's layers.

    The Qwen3 router is shared with training. Each call reads the active HP
    layer's [E,H,2I] and [E,I,H] tensors.
    """

    def __init__(self, config: Any, group: Any, topology: Any) -> None:
        """Initialize the optional vLLM backend inside its inference worker."""
        # pylint: disable=import-outside-toplevel
        from vllm.distributed import get_dp_group, get_ep_group, get_pcp_group, get_tp_group
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, get_routing_method_type
        from vllm.model_executor.layers.fused_moe.layer import make_parallel_config
        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

        super().__init__()
        hf = config.model_config.hf_text_config
        self.group = group
        size, rank = topology.ep_size, topology.ep_rank
        self.local_count = hf.num_experts // size
        self.first = rank * self.local_count
        if self.local_count * size != hf.num_experts or self.group.rank() != rank:
            raise ValueError("Native expert range differs from the published HP shard")

        owners = dist.get_process_group_ranks(self.group)
        sources = expert_source_ranks(owners, get_ep_group().ranks, topology.tp_size,
                                      topology.sequence_parallel, owners[rank])
        tp = get_tp_group()
        available = [self.group, tp.device_group]
        if topology.dp_size > 1:
            available.append(get_dp_group().device_group)
        if topology.pcp_size > 1:
            available.append(get_pcp_group().device_group)
        self.source_group = _row_group(sources, available)
        self.source_index = sources.index(owners[rank])
        self.sharded_rows = len(sources) == size
        self.source_count = len(sources)
        self.sp_size = topology.tp_size if topology.sequence_parallel else 1
        self.common_source_rows = set(sources).issubset(get_tp_group().ranks)
        self.dp_size = topology.dp_size

        # Only the FFN consumes this config. Each rank owns complete expert
        # matrices; routing uses HP's EP rank (which can be a native subgroup).
        parallel = make_parallel_config(None, None, None, False, config.parallel_config)
        parallel = replace(parallel, tp_size=1, tp_rank=0, ep_size=size, ep_rank=rank, use_ep=True)
        self.config = FusedMoEConfig(
            num_experts=hf.num_experts, experts_per_token=hf.num_experts_per_tok,
            hidden_dim=hf.hidden_size, intermediate_size=hf.moe_intermediate_size,
            num_local_experts=self.local_count, num_logical_experts=hf.num_experts,
            activation=MoEActivation.SILU, device=config.device_config.device,
            routing_method=get_routing_method_type("softmax", hf.num_experts_per_tok, hf.norm_topk_prob, None, False),
            moe_parallel_config=parallel, in_dtype=config.model_config.dtype,
            moe_backend=config.kernel_config.moe_backend, max_num_tokens=config.scheduler_config.max_num_batched_tokens,
        )
        self.method = AscendUnquantizedFusedMoEMethod(self.config)

    def _gather_source_rows(self, states, ids, context):
        """Collect distinct sources at the common physical row count used by vLLM."""
        local_rows = states.shape[0]
        bucket = max(1, local_rows)
        if not self.common_source_rows:
            # DP schedules independently. PCP within one replica shares a
            # padded physical lane extent. The scheduler reports pre-SP rows.
            if context.dp_metadata is not None:
                counts = context.dp_metadata.num_tokens_across_dp_cpu
                if counts.device.type != "cpu":
                    raise ValueError("Expected host scheduling metadata for MoE source rows")
                scheduled = int(counts.max())
            elif self.dp_size > 1:
                raise ValueError("MoE requires synchronized DP schedule metadata")
            elif context.batch_descriptor is not None:
                scheduled = context.batch_descriptor.num_tokens
            else:
                scheduled = local_rows * self.sp_size
            bucket = max(1, (scheduled + self.sp_size - 1) // self.sp_size)
        if bucket < local_rows:
            raise ValueError(f"MoE source extent {bucket} is smaller than {local_rows} local rows")

        if bucket > local_rows:
            states = F.pad(states, (0, 0, 0, bucket - local_rows))
            ids = F.pad(ids, (0, 0, 0, bucket - local_rows), value=self.config.num_experts)

        # Send the raw int32 IDs in the same HCCL message as BF16 activations.
        # Reinterpreting preserves every ID bit, including the padding sentinel.
        packet = torch.cat((states, ids.contiguous().view(states.dtype)), dim=1)
        received = packet.new_empty((self.source_count * bucket, packet.shape[1]))
        dist.all_gather_into_tensor(received, packet, group=self.source_group)
        hidden = states.shape[1]
        values, routes = received.split((hidden, packet.shape[1] - hidden), dim=1)
        return values.contiguous(), routes.contiguous().view(torch.int32), bucket

    def token_dispatch(self, states, indices):
        """Gather distinct source rows and pack them by local expert; router weights stay at the source."""
        # Load worker-only dependencies when the native backend is actually used.
        # pylint: disable=import-outside-toplevel
        from vllm.forward_context import get_forward_context
        from vllm_ascend.device.device_op import DeviceOperator

        ids = indices.int().contiguous()
        bucket = states.shape[0]
        if self.source_group is not None:
            states, ids, bucket = self._gather_source_rows(states, ids, get_forward_context())

        packed, inverse, counts, _ = DeviceOperator.npu_moe_init_routing(
            states, ids, active_num=ids.numel(), expert_num=self.config.num_experts + 1,
            active_expert_range=[self.first, self.first + self.local_count],
        )
        return packed, counts.long(), ids, inverse, bucket

    def compute_experts(self, packed, counts, gate_up, down):
        """Run the native grouped gate/up projection, SwiGLU and down projection on unweighted rows."""
        # pylint: disable=import-outside-toplevel
        from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
        from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
        from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
        from vllm_ascend.ops.fused_moe.moe_mlp import apply_moe_mlp

        inputs = MoEMlpComputeInput(
            hidden_states=packed, group_list=counts, group_list_type=1, activation=self.config.activation,
            weights=MoEWeights(w1=gate_up, w2=down), layer=_ExpertWeights(gate_up, down),
            quant=MoEQuantParams(), dynamic_scale=None, topk_scales=None, fusion=False,
        )
        values, _ = apply_moe_mlp(inputs, self.method)
        return values

    def token_combine(self, values, ids, inverse, local_rows, bucket):
        """Restore [T,K,H] unweighted slots and return them to the ranks owning their token rows."""
        # Triton is optional until the accelerator backend executes.
        from .restore_slots import restore_slots  # pylint: disable=import-outside-toplevel

        slots = restore_slots(values, ids, inverse, self.first, self.local_count)

        # Every slot has one contributing owner. Disjoint rows go straight
        # to their source. TP replicas all need the same complete slots.
        # Each slot has exactly one owner. Sum its BF16 bit pattern as an equal-width integer
        # with zeros from the other ranks, preserving even negative zero.
        payload = slots.view(torch.int16)
        if self.sharded_rows:
            local = slots.new_empty((bucket, ids.shape[1], values.shape[-1]))
            dist.reduce_scatter_tensor(local.view(torch.int16), payload, group=self.group)
            slots = local[:local_rows]
        else:
            dist.all_reduce(payload, group=self.group)
            if self.source_group is not None:
                slots = slots.narrow(0, self.source_index * bucket, local_rows)
        return slots
