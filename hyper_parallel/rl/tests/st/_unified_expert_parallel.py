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
"""Device acceptance of production EP transports, graph replay and gradients."""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.functional.moe_token_unpermute import moe_token_unpermute
from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward
from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsConfig
from hyper_parallel.distributed.expert_parallel.ep_context import EPContext
from hyper_parallel.distributed.module_binding import ParallelBinding
from rl.unified_execution.training.grouped_experts import bind_native_unpermute
from rl.unified_execution.vllm.moe.fused_experts import AscendFusedExpertsBackend
from rl.unified_execution.vllm.moe.grouped_experts import rollout_experts


def _experts(count: int, offset: int) -> torch.nn.Module:
    """Build deterministic expert weights for one contiguous logical expert range."""
    source = torch.nn.Module()
    source.num_experts, source.hidden_size, source.intermediate_size = count, 2048, 768
    source.is_transposed = False
    source.config = SimpleNamespace(hidden_act="silu", initializer_range=.02)
    gate, down = [], []
    for expert in range(offset, offset + count):
        torch.manual_seed(12000 + expert)
        gate.append(torch.randn(1536, 2048, device="npu", dtype=torch.bfloat16) * .02)
        down.append(torch.randn(2048, 768, device="npu", dtype=torch.bfloat16) * .02)
    source.gate_up_proj = torch.nn.Parameter(torch.stack(gate))
    source.down_proj = torch.nn.Parameter(torch.stack(down))
    module = torch.nn.Module()
    module.experts = GroupedExperts(module=source)
    # The fixture already owns a shard; HP's binding reads the local tensor shape.
    bind_local_expert_forward(module, 1, use_grouped_gemm=True)
    return module


def _reference_forward(full: torch.nn.Module, states: torch.Tensor, ids: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
    """Evaluate each local token against a complete expert bank without EP routing."""
    slots = ids.shape[1]
    routes = full.experts(states.repeat_interleave(slots, 0), ids.flatten())
    if not states.shape[0]:
        return torch.empty_like(states)
    inverse = torch.arange(ids.numel(), dtype=torch.int32, device=ids.device)
    return moe_token_unpermute(routes.contiguous(), inverse, weights.contiguous())


def _update_weights(module: torch.nn.Module, full: torch.nn.Module) -> None:
    """Apply the same pointwise update to the local shard and independent full bank."""
    for model in (module, full):
        for parameter in model.parameters():
            parameter.add_(torch.full_like(parameter, .0002))


def _equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(f"EP output {actual.shape}/{actual.dtype} differs from {expected.shape}/{expected.dtype}")
    if not torch.isfinite(actual).all() or not torch.equal(actual.contiguous().view(torch.uint8),
                                                         expected.contiguous().view(torch.uint8)):
        raise AssertionError(f"EP bitwise gate failed: max_abs={(actual.float() - expected.float()).abs().max()}")


def _training_experts(module, options):
    """Exercise the HP training stages with the same logical weights as rollout."""
    mesh = SimpleNamespace(ndim=1, size=lambda: options["world_size"], get_group=lambda: options["group"],
                           get_local_rank=lambda: options["rank"])
    config = GroupedExpertsConfig(128, 2048, 768, routing_weight_mode="output", num_layers=1)
    with torch.device("meta"):
        experts = EPGroupedExperts(config=config)
    for name in ("gate_up_proj", "down_proj"):
        local = getattr(module.experts, name).detach().transpose(-1, -2).contiguous()
        setattr(experts, name, torch.nn.Parameter(local))
    experts.bind_parallel(ParallelBinding(mesh_binding=None, ep_context=EPContext(mesh, num_experts=128)))
    bind_native_unpermute(experts)
    return experts


def _gradients(module: torch.nn.Module, states: torch.Tensor, ids: torch.Tensor,
               weights: torch.Tensor, options: dict) -> None:
    """Compare input, routing-weight and expert gradients with a full expert bank."""
    size, rank, local = options["world_size"], options["rank"], options["local_experts"]

    def gather(value: torch.Tensor) -> torch.Tensor:
        output = torch.empty((size * value.shape[0], *value.shape[1:]), device=value.device, dtype=value.dtype)
        dist.all_gather_into_tensor(output, value.detach().contiguous(), group=options["group"])
        return output

    x = states.clone().requires_grad_()
    probs = weights.clone().requires_grad_()
    upstream = torch.randn_like(states)
    experts = _training_experts(module, options)
    experts(x, ids, probs).float().mul(upstream).sum().backward()
    actual = [x.grad, probs.grad, experts.gate_up_proj.grad.transpose(-1, -2),
              experts.down_proj.grad.transpose(-1, -2)]
    full = _experts(128, 0)
    with torch.no_grad():
        full.experts.gate_up_proj.copy_(gather(module.experts.gate_up_proj))
        full.experts.down_proj.copy_(gather(module.experts.down_proj))
    all_x = gather(states).requires_grad_()
    all_probs = gather(weights).requires_grad_()
    all_ids, all_upstream = gather(ids), gather(upstream)
    routes = full.experts(all_x.repeat_interleave(8, 0), all_ids.flatten()).view(size * states.shape[0], 8, 2048)
    # Independent FP32 weighted sum supplies the autograd reference.
    expected_output = (routes.float() * all_probs.float().unsqueeze(-1)).sum(dim=1).to(routes.dtype)
    expected_output.float().mul(all_upstream).sum().backward()
    rows = states.shape[0]
    expected = [all_x.grad[rank * rows:(rank + 1) * rows], all_probs.grad[rank * rows:(rank + 1) * rows],
                full.experts.gate_up_proj.grad[rank * local:(rank + 1) * local],
                full.experts.down_proj.grad[rank * local:(rank + 1) * local]]
    for gradient, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(gradient, reference, rtol=.02, atol=2e-4)


def main() -> None:
    """Test real ownership, changing routes, empty ranks, graph updates and all gradients."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    torch.npu.set_deterministic_level(3)
    domain = list(range(dist.get_world_size()))
    group = None
    for first in range(0, len(domain), args.ep):
        ranks = domain[first:first + args.ep]
        created_group = dist.new_group(ranks, backend="hccl")
        if dist.get_rank() in ranks:
            group = created_group
    rank, local = group.rank(), 128 // args.ep
    # Worker-only imports follow explicit NPU and distributed initialization.
    # pylint: disable=import-outside-toplevel
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import override_forward_context
    from vllm_ascend.ascend_config import init_ascend_config
    # pylint: enable=import-outside-toplevel

    config = EngineArgs(model=args.model, dtype="bfloat16", max_model_len=512,
                        enforce_eager=True).create_engine_config()
    with set_current_vllm_config(config):
        init_ascend_config(config)
        init_distributed_environment(dist.get_world_size(), dist.get_rank(),
                                     local_rank=int(os.environ["LOCAL_RANK"]), backend="hccl")
        initialize_model_parallel(tensor_model_parallel_size=1, prefill_context_model_parallel_size=len(domain),
                                  backend="hccl")
        options = {"group": group, "rank": rank, "world_size": args.ep, "local_experts": local}
        topology = SimpleNamespace(tp_size=1, dp_size=1, pcp_size=len(domain), sequence_parallel=False,
                                   ep_size=args.ep, ep_rank=rank)
        kernels = AscendFusedExpertsBackend(config, group, topology)
    context = SimpleNamespace(no_compile_layers={"experts": kernels}, dp_metadata=None,
                              batch_descriptor=SimpleNamespace(num_tokens=args.rows))
    # One owner contributes each slot. A BF16 sum with the other ranks'
    # positive zero would silently flip the sign before native unpermute.
    slot_ids = (torch.arange(8, device="npu", dtype=torch.int32) % args.ep * local).repeat(args.ep, 1)
    inverse = torch.arange(slot_ids.numel(), device="npu", dtype=torch.int32)
    negative_zero = torch.full((slot_ids.numel(), 64), -0., device="npu", dtype=torch.bfloat16)
    returned = kernels.token_combine(negative_zero, slot_ids, inverse, local_rows=1, bucket=1)
    _equal(returned, negative_zero[:8].view(1, 8, 64))

    module = _experts(local, rank * local)
    full = _experts(128, 0)
    training = _training_experts(module, options)
    # Different subgroups must carry different data to detect cross-group mixing.
    torch.manual_seed(21000 + dist.get_rank())
    # Unequal sources (including an empty rank) must issue the same collective.
    for rows in (args.rows, 0 if rank == 0 else args.rows // 2):
        x = torch.randn(rows, 2048, device="npu", dtype=torch.bfloat16)
        ids = torch.zeros(rows, 8, device="npu", dtype=torch.int64)
        weights = torch.zeros(rows, 8, device="npu", dtype=torch.float32)

        # Bind each row case now; graph replay must mutate these exact tensors.
        def prepare(step: int, states=x, choices=ids, probabilities=weights) -> None:
            logits = torch.randn(states.shape[0], 128, device="npu", dtype=torch.float32)
            if step % 2:
                logits[:, :8] += 20
            probs, selected = logits.softmax(-1).topk(8, dim=-1)
            probs /= probs.sum(-1, keepdim=True)
            choices.copy_(selected)
            probabilities.copy_(probs)
            states.normal_()

        def candidate(states=x, choices=ids, probabilities=weights) -> torch.Tensor:
            with override_forward_context(context):
                return rollout_experts(
                    states, choices, probabilities, module.experts.gate_up_proj, module.experts.down_proj, "experts",
                )

        with torch.no_grad():
            prepare(0)
            for _ in range(3):
                candidate()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = candidate()
            for step in range(2):
                prepare(step)
                expected = _reference_forward(full, x, ids, weights)
                actual = candidate()
                dynamic = training(x, ids, weights)
                graph.replay()
                torch.npu.synchronize()
                _equal(actual, expected)
                _equal(dynamic, expected)
                _equal(captured, expected)
            _update_weights(module, full)
            training = _training_experts(module, options)
            graph.replay()
            torch.npu.synchronize()
            _equal(captured, candidate())
        del graph, captured
    states = torch.randn(4, 2048, device="npu", dtype=torch.bfloat16)
    probs, ids = torch.randn(4, 128, device="npu").softmax(-1).topk(8, dim=-1)
    _gradients(module, states, ids, probs / probs.sum(-1, keepdim=True), options)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
