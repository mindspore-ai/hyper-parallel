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
"""AutoModels Qwen3-MoE static-graph EP and FSDP+EP correctness example."""
# The environment selection and optional-dependency guard must precede the
# backend imports in this executable example.
# pylint: disable=wrong-import-position,forbidden-backend-import,unused-import

import os

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from transformers.utils import import_utils as transformers_import_utils

# The text-only model does not require torchvision.  This also permits an
# isolated Torch overlay to coexist with a torchvision built for the host's
# default Torch.
transformers_import_utils._torchvision_available = False

from transformers import Qwen3MoeConfig

from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel import (
    qwen3moe_ep_compute_fn,
)
from hyper_parallel.distributed.mesh import (
    DistributedSetup,
    MeshContext,
)
from hyper_parallel.trainer.config import PlanOverride, Target, entries_to_plan_overrides
from hyper_parallel.compile import GraphTrainer, PassConfig


def _build_distributed_setup(world_size: int, mode: str) -> DistributedSetup:
    """Build EP-only or FSDP+EP topology and reuse the dynamic EP factory."""
    ep_size = world_size if mode == "ep" else 2
    fsdp_shard_size = 1 if mode == "ep" else world_size
    fsdp_replicate_size = world_size // fsdp_shard_size
    edp_shard_size = 1 if mode == "ep" else world_size // ep_size
    mesh = MeshContext(
        dp_size=world_size,
        dp_replicate_size=fsdp_replicate_size,
        dp_shard_size=fsdp_shard_size,
        edp_shard_size=edp_shard_size,
        tp_size=1,
        cp_size=1,
        pp_size=1,
        ep_size=ep_size,
        sequence_parallel=False,
        loss_parallel=False,
    )
    mesh.build_meshs("npu", world_size)
    factory_path = (
        "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel."
        "qwen3moe_ep_compute_fn"
    )
    return DistributedSetup(
        mesh_context=mesh,
        strategy_config=None,
        pipeline_config=None,
        plan_overrides=entries_to_plan_overrides([
            PlanOverride(
                match="*.mlp",
                when="ep",
                region_dispatch=False,
                local_compute_fn=Target(
                    qwen3moe_ep_compute_fn,
                    target_path=factory_path,
                ),
            )
        ], ep_size=ep_size),
    )


def _build_model(setup: DistributedSetup):
    """Build a tiny random Qwen3-MoE through HyperAutoModel."""
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        router_aux_loss_coef=0.0,
        output_router_logits=False,
        use_cache=False,
    )
    return HyperAutoModelForCausalLM.from_config(
        config,
        distributed_setup=setup,
        torch_dtype="bfloat16",
        attn_implementation="eager",
        validate_placement=False,
        activation_checkpoint="off",
    )


def _loss_fn(model, input_ids, labels):
    return model(input_ids=input_ids, labels=labels, use_cache=False).loss


def _reference_grads(
    model: torch.nn.Module,
    mesh: MeshContext,
    fsdp_enabled: bool,
) -> tuple[list[str], list[torch.Tensor]]:
    """Build the expected local gradients after graph-mode reduce-scatter."""
    expert_param_names = GraphTrainer._find_expert_param_names(model)  # pylint: disable=protected-access
    dense_sub = mesh.fsdp_non_moe_mesh["fsdp_shard"]
    expert_sub = mesh.fsdp_moe_mesh["edp_shard"] if mesh.fsdp_moe_mesh else None
    names = []
    grads = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        names.append(name)
        grad = (
            parameter.grad.detach().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter)
        )
        if not fsdp_enabled:
            grads.append(grad)
            continue
        sub = expert_sub if name in expert_param_names else dense_sub
        degree = sub.size()
        if degree > 1 and grad.ndim > 0 and grad.shape[0] % degree == 0:
            reduced_shape = (grad.shape[0] // degree, *grad.shape[1:])
            reduced_grad = torch.empty(
                reduced_shape,
                dtype=grad.dtype,
                device=grad.device,
            )
            dist.reduce_scatter_tensor(
                reduced_grad,
                grad,
                group=sub.get_group(),
            )
            grad = reduced_grad
        grads.append(grad)
    return names, grads


def main() -> None:
    """Run eager dynamic EP, capture it, and verify graph loss/gradients."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mode = os.environ.get("STATIC_PARALLEL_MODE", "ep")
    expected_world_size = 2 if mode == "ep" else 4
    if mode not in ("ep", "fsdp_ep"):
        raise ValueError(f"STATIC_PARALLEL_MODE must be ep or fsdp_ep, got {mode}")
    if world_size != expected_world_size:
        raise ValueError(
            f"Mode {mode} expects {expected_world_size} ranks, got {world_size}"
        )

    torch.manual_seed(20260829)
    torch.npu.manual_seed_all(20260829)
    setup = _build_distributed_setup(world_size, mode)
    model = _build_model(setup)
    model.train()
    input_ids = torch.tensor([[1, 7, 3, 9, 5, 2, 11, 4]], device="npu")
    input_ids = ((input_ids + rank * 17 - 1) % (model.config.vocab_size - 1)) + 1
    labels = input_ids.clone()

    eager_loss = _loss_fn(model, input_ids, labels)
    eager_loss.backward()
    torch.npu.synchronize()
    dist.barrier()
    fsdp_enabled = mode == "fsdp_ep"
    trainable_names, eager_grads = _reference_grads(
        model,
        setup.mesh_context,
        fsdp_enabled,
    )
    model.zero_grad(set_to_none=True)

    trainer = GraphTrainer(
        model,
        _loss_fn,
        PassConfig(
            ep_degree=setup.mesh_context.ep_size,
            fsdp_enabled=fsdp_enabled,
            fsdp_degree=None,
            tp_size=1,
            enable_overlap=False,
        ),
        optimizer_config={"lr": 1e-4},
        device=torch.device("npu", local_rank),
        mesh_context=setup.mesh_context,
    )
    graph_loss = trainer.train_step(input_ids, labels)
    torch.npu.synchronize()
    dist.barrier()
    graph_grads = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    loss_diff = (graph_loss.detach() - eager_loss.detach()).abs().float()
    grad_diffs = [
        (actual - expected).abs().max().float()
        for actual, expected in zip(graph_grads, eager_grads)
    ]
    grad_diff_tensor = torch.stack(grad_diffs)
    local_max_index = int(grad_diff_tensor.argmax().cpu())
    grad_diff = grad_diff_tensor[local_max_index]
    loss_tolerance = 1e-5
    grad_tolerance = 2e-3 if fsdp_enabled else 1e-5
    if float(grad_diff.cpu()) > grad_tolerance:
        actual = graph_grads[local_max_index]
        expected = eager_grads[local_max_index]
        print(
            f"rank={rank} GRAD_MISMATCH name={trainable_names[local_max_index]} "
            f"diff={float(grad_diff.cpu())} "
            f"eager_max={float(expected.abs().max().cpu())} "
            f"graph_max={float(actual.abs().max().cpu())}"
        )
    collective_count = trainer._joint_graph.graph_module.ep_collective_count
    fsdp_collective_count = sum(
        "fsdp_" in str(node.meta.get("comm_type", ""))
        for node in trainer._joint_graph.graph_module.graph.nodes
    )
    global_diffs = torch.stack((loss_diff, grad_diff))
    dist.all_reduce(global_diffs, op=dist.ReduceOp.MAX)
    global_loss_diff, global_grad_diff = global_diffs
    if global_loss_diff > loss_tolerance or global_grad_diff > grad_tolerance:
        raise AssertionError(
            "Static parallel graph mismatch across ranks: "
            f"loss={float(global_loss_diff.cpu())}, "
            f"grad={float(global_grad_diff.cpu())}, "
            f"loss_tolerance={loss_tolerance}, grad_tolerance={grad_tolerance}"
        )
    trainer.optimizer_step()
    dist.barrier()
    if rank == 0:
        print(
            "STATIC_PARALLEL_OK "
            f"mode={mode} ep_collectives={collective_count} "
            f"fsdp_collectives={fsdp_collective_count} "
            f"loss_max_abs_diff={float(global_loss_diff.cpu())} "
            f"grad_max_abs_diff={float(global_grad_diff.cpu())} "
            f"loss_tolerance={loss_tolerance} grad_tolerance={grad_tolerance}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
