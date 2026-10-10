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
"""CPU/Gloo worker checking token-partitioned V4.1 router gradients."""

from copy import deepcopy
from datetime import timedelta
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn import functional

from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.utils.moe_utils import sync_and_update_expert_bias
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41TopKRouter


def test_partitioned_router_gradients() -> None:
    """Real SUM collectives and rank-average gradients match a full-token oracle."""
    dist.init_process_group(
        "gloo", timeout=timedelta(seconds=60),
        init_method=os.environ.get("HP_AUX_TEST_INIT_METHOD", "env://"),
        rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]),
    )
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        config = SimpleNamespace(
            hidden_size=8, num_local_experts=4, num_experts_per_tok=2,
            scoring_func="sqrtsoftplus", routed_scaling_factor=1.25, router_aux_loss_coef=0.1,
            v41_vision_enabled=True, router_bias_update_rate=0.002,
        )
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("balance",))
        for num_tokens in (5, 1):
            torch.manual_seed(2026)
            router = DeepseekV41TopKRouter(config)
            torch.nn.init.normal_(router.weight)
            reference = deepcopy(router)
            reference.aux_loss_coeff = 0
            inputs = torch.randn(num_tokens, config.hidden_size)
            sequence_ids = torch.arange(num_tokens) % 2
            image_mask = torch.arange(num_tokens) % 3 == 0
            with aux_loss_scale_context():
                _, weights, _ = router(
                    inputs[rank::world_size], image_mask=image_mask[rank::world_size],
                    sequence_ids=sequence_ids[rank::world_size], num_sequences=2,
                    sequence_partition_groups=(dist.group.WORLD,),
                )
                local_loss = weights.sum() * 0
                bind_aux_loss_scale(local_loss)
                (local_loss * 0.25).backward()
            dist.all_reduce(router.weight.grad)
            router.weight.grad.div_(world_size)

            logits, _, indices = reference(inputs, image_mask=image_mask)
            scores = functional.softplus(logits).sqrt()  # pylint: disable=not-callable
            probabilities = functional.normalize(scores, p=1, dim=-1)
            routing_map = functional.one_hot(  # pylint: disable=not-callable
                indices, config.num_local_experts,
            ).sum(1).float()
            sample_losses = []
            for sample in range(2):
                selected = sequence_ids == sample
                if selected.any():
                    counts = routing_map[selected].sum(0)
                    sample_losses.append(config.num_local_experts * torch.dot(
                        counts / counts.sum(), probabilities[selected].mean(0),
                    ))
            aux = torch.stack(sample_losses).mean()
            (aux * config.router_aux_loss_coef * 0.25).backward()
            torch.testing.assert_close(router.last_aux_loss, aux.detach())
            torch.testing.assert_close(router.weight.grad, reference.weight.grad, atol=1.0e-7, rtol=1.0e-5)

            expected_biases = []
            for mask, bias in ((~image_mask, router.bias), (image_mask, router.bias_vl)):
                loads = routing_map[mask].sum(0)
                expected_biases.append(bias.detach() + config.router_bias_update_rate * (loads.mean() - loads).sign())
            # Match the parameter layout seen after FSDP has resharded the gate.
            router.bias = torch.nn.Parameter(distribute_tensor(router.bias, mesh, [Shard(0)]), False)
            router.bias_vl = torch.nn.Parameter(distribute_tensor(router.bias_vl, mesh, [Shard(0)]), False)
            router.expert_bias_update_groups = (dist.group.WORLD,)
            sync_and_update_expert_bias(router)
            for bias, expected in zip((router.bias, router.bias_vl), expected_biases):
                torch.testing.assert_close(bias.redistribute(mesh, [Replicate()]).to_local(), expected)
            torch.testing.assert_close(router.tokens_per_expert, torch.zeros_like(router.tokens_per_expert))
    finally:
        dist.destroy_process_group()
