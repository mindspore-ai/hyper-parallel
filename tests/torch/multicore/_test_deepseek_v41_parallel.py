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

"""NPU parity of DSV4.1 MegaMoe with production TP/SP and CP boundaries."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from functools import wraps
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import
from torch import nn
from torch.nn import functional
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4SparseMoeBlock,
)

from hyper_parallel.distributed import ShardingPlanner, apply_sharding_plan
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models.deepseek_v41.adapter.distributed.megamoe import (
    DeepseekV41MegaMoeExperts,
)
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_megamoe_compute_fn,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41TopKRouter,
    _v41_sparse_moe_forward,
    bind_shared_expert_forward,
)
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence
from tests.torch.multicore._test_mega_moe_dynamic_tokens import _assert_close


def _text_signature(self: nn.Module, hidden_states: torch.Tensor,
                    input_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Retain the text block signature until the declared compute injection."""
    raise AssertionError(f"Missing MegaMoe injection for {type(self).__name__}: {hidden_states.shape}, {input_ids}")


@wraps(_v41_sparse_moe_forward)
def _vision_signature(self: nn.Module, *_args: object, **_kwargs: object) -> torch.Tensor:
    """Follow the production VLM signature while requiring compute injection."""
    raise AssertionError(f"Missing MegaMoe injection for {type(self).__name__}")


def _model(tp: int, cp: int, vision: bool, mode: str) -> nn.Module:
    """Build identical full expert weights before the production EP partition."""
    torch.manual_seed(83)
    config = DeepseekV4Config.from_dict({
        'hidden_size': 512, 'moe_intermediate_size': 128, 'num_hidden_layers': 1,
        'num_local_experts': 48, 'num_experts_per_tok': 6, 'mlp_layer_types': ['moe'],
        'swiglu_limit': 10., 'scoring_func': 'sigmoid', 'routed_scaling_factor': 1.,
    })
    config.v41_vision_enabled = vision
    block = DeepseekV4SparseMoeBlock(config, 0)
    block.gate = DeepseekV41TopKRouter(config)
    bind_shared_expert_forward(block.shared_experts)
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.04)
    with torch.no_grad():
        block.gate.bias.zero_()
        block.gate.bias[:6].fill_(2.)
        if vision:
            block.gate.bias_vl.zero_()
            block.gate.bias_vl[-6:].fill_(2.)
    block.experts = DeepseekV41MegaMoeExperts(
        module=block.experts, dispatch_mode=mode,
        context={'tp': tp > 1, 'cp': cp > 1, 'ep': True, 'sequence_parallel': tp > 1},
    )
    block.forward = MethodType(_vision_signature if vision else _text_signature, block)
    model = nn.Module()
    model.config = config
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = block
    model.model.layers = nn.ModuleList([layer])
    return model.npu().bfloat16()


def _reference(block: nn.Module, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Use independent per-expert matmuls with the source FP32 clipped activation."""
    _, probabilities, indices = block.gate(hidden, image_mask=mask)
    inputs = hidden.reshape(-1, hidden.shape[-1])
    first, second = block.experts.gate_up_proj, block.experts.down_proj
    output = inputs.float() * 0 + (first.float().sum() + second.float().sum()) * 0
    for expert in range(first.shape[0]):
        rows, slots = torch.where(indices == expert)
        gate, up = (inputs[rows] @ first[expert].to(hidden.dtype)).float().chunk(2, dim=-1)
        activation = functional.silu(gate.clamp(max=10.)) * up.clamp(-10., 10.)
        values = activation.to(hidden.dtype) @ second[expert].to(hidden.dtype)
        output = output.index_add(0, rows, values.float() * probabilities[rows, slots, None])
    return output.to(hidden.dtype).reshape_as(hidden)


def _shared_reference(shared: nn.Module, hidden: torch.Tensor, tp: int) -> torch.Tensor:
    """Independently compute column/row TP, including BF16 partial-output rounding."""
    width = shared.gate_proj.weight.shape[0] // tp
    partials = []
    for index in range(tp):
        columns = slice(index * width, (index + 1) * width)
        gate = (hidden @ shared.gate_proj.weight[columns].to(hidden.dtype).T).float()
        up = (hidden @ shared.up_proj.weight[columns].to(hidden.dtype).T).float()
        activation = functional.silu(gate.clamp(max=10.)) * up.clamp(-10., 10.)
        partials.append(activation.to(hidden.dtype) @ shared.down_proj.weight[:, columns].to(hidden.dtype).T)
    return torch.stack(partials).float().sum(dim=0).to(hidden.dtype)


def _collective_results(comparisons: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """Finish every collective before reporting rank-local comparison failures."""
    errors, failures = {}, {}
    for name, (actual, expected) in comparisons.items():
        try:
            errors[name] = _assert_close(name, actual, expected)
        except AssertionError as error:
            failures[name] = str(error)
    peer_failures = [None] * dist.get_world_size()
    dist.all_gather_object(peer_failures, failures)
    if any(peer_failures):
        raise AssertionError(f"Parallel MoE parity failures by rank: {peer_failures}")
    return errors


def _source_inputs(rank: int, tp: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay one source's CP-local sequence before its TP/SP slice."""
    generator = torch.Generator().manual_seed(701 + rank // tp + count)
    hidden = torch.randn(1, count * tp, 512, generator=generator).bfloat16()
    upstream = torch.randn(hidden.shape, generator=generator).bfloat16() * 0.1
    return hidden, upstream


def _compare(block: nn.Module, reference: nn.Module, mesh: MeshContext,
             count: int, vision: bool) -> dict:
    """Check local results, EP-owner gradients, and TP-sharded shared-expert gradients."""
    rank, world = dist.get_rank(), dist.get_world_size()
    tp = mesh.tp_size
    tp_rank = mesh.device_mesh.get_local_rank('tp')
    full, upstream = _source_inputs(rank, tp, count)
    interval = slice(tp_rank * count, (tp_rank + 1) * count)
    hidden = full[:, interval].npu().requires_grad_()
    original = hidden.detach().clone().requires_grad_()
    mask = (torch.arange(count * tp).reshape(1, -1) % 3 == 0).npu() if vision else None
    kwargs = {'image_mask': mask} if vision else {}
    block.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    routed_reference = _reference(reference, original, None if mask is None else mask[:, interval])
    shared_input = full.npu()
    shared_input = torch.cat((shared_input[:, :interval.start], original, shared_input[:, interval.stop:]), dim=1)
    shared_reference = _shared_reference(reference.shared_experts, shared_input, tp)[:, interval]
    expected = routed_reference + shared_reference
    branches = {}
    handles = [
        block.experts.register_forward_hook(
            lambda _module, _args, output: branches.__setitem__('routed', output.detach().cpu())),
        block.shared_experts.register_forward_hook(
            lambda _module, _args, output: branches.__setitem__('shared', output.detach().cpu())),
    ]
    try:
        actual = block(hidden, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    # Independent BF16 rounding in two branches can amplify relative error after
    # cancellation. Check each branch at the original tolerance and their exact sum.
    comparisons = {
        'routed_output': (branches['routed'], routed_reference.detach().cpu()),
        'shared_output': (branches['shared'], shared_reference.detach().cpu()),
        'output': (actual.detach().cpu(), branches['routed'] + branches['shared']),
    }
    derivative = upstream[:, interval].npu()
    (actual.float() * derivative.float()).sum().backward()
    (expected.float() * derivative.float()).sum().backward()
    comparisons['input_gradient'] = (hidden.grad.cpu(), original.grad.cpu())
    comparisons['router_gradient'] = (block.gate.weight.grad.cpu(), reference.gate.weight.grad.cpu())
    # Native expert GMMs differentiate the combined EP token stream. Do the same
    # instead of rounding each source's BF16 dW before summing its contributions.
    states, derivatives, masks = [], [], []
    for peer in range(world):
        peer_hidden, peer_derivative = _source_inputs(peer, tp, count)
        peer_interval = slice((peer % tp) * count, (peer % tp + 1) * count)
        states.append(peer_hidden[:, peer_interval])
        derivatives.append(peer_derivative[:, peer_interval])
        if vision:
            masks.append(mask[:, peer_interval])
    reference.zero_grad(set_to_none=True)
    combined = _reference(reference, torch.cat(states, dim=1).npu(),
                          torch.cat(masks, dim=1) if vision else None)
    (combined.float() * torch.cat(derivatives, dim=1).npu().float()).sum().backward()
    for name in ('gate_up_proj', 'down_proj'):
        expected_grad = getattr(reference.experts, name).grad
        comparisons[name] = (getattr(block.experts, name).grad.cpu(), expected_grad.chunk(world)[rank].cpu())
    # Shared TP sees every TP/SP row before its matmuls. Differentiate those rows
    # together too, avoiding an extra BF16 rounding of per-source parameter gradients.
    reference.shared_experts.zero_grad(set_to_none=True)
    shared_output = _shared_reference(reference.shared_experts, full.npu(), tp)
    (shared_output.float() * upstream.npu().float()).sum().backward()
    for name, parameter in block.shared_experts.named_parameters():
        expected_grad = dict(reference.shared_experts.named_parameters())[name].grad
        if tp > 1:
            dimension = 1 if name == 'down_proj.weight' else 0
            expected_grad = expected_grad.chunk(tp, dim=dimension)[tp_rank]
        comparisons[name] = (parameter.grad.cpu(), expected_grad.cpu())
    return {'local_rows': count, 'max_abs_errors': _collective_results(comparisons)}


def test_deepseek_v41_parallel() -> None:
    """Exercise the real planner, shared-expert collectives, and native MegaMoe kernel."""
    tp, cp = int(os.environ['HP_DSV41_TP']), int(os.environ['HP_DSV41_CP'])
    vision = os.environ['HP_DSV41_VISION'] == '1'
    mode = os.environ['HP_DYNAMIC_DISPATCH']
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8 or (tp, cp) not in ((2, 1), (1, 2), (2, 2)):
        raise ValueError('This parity test requires EP8 with TP2, CP2, or TP2+CP2')
    start_shmem_lifetime()
    mesh = MeshContext(dp_size=world // tp // cp, dp_shard_size=world // tp,
                       tp_size=tp, cp_size=cp, ep_size=world, sequence_parallel=tp > 1)
    mesh.build_meshs('npu', world)
    model = _model(tp, cp, vision, mode)
    block = model.model.layers[0].mlp
    # Keep reference gradients in FP32: EP/TP aggregation must not round each source twice.
    reference = deepcopy(block).float()
    plan = ShardingPlanner().plan(model, mesh.device_mesh, tp_size=tp, cp_size=cp,
                                  ep_size=world, sequence_parallel=tp > 1)
    spec = plan.modules['model.layers.0.mlp']
    spec.region_dispatch = False
    spec.local_compute_fn = deepseek_v41_megamoe_compute_fn
    apply_sharding_plan(model, plan, mesh)
    cases = [_compare(block, reference, mesh, count, vision) for count in (17, 129)]
    report = {'rank': rank, 'tp': tp, 'cp': cp, 'ep': world, 'vision': vision,
              'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'dispatch_mode': mode, 'passed': True, 'cases': cases}
    if os.getenv('HP_MEGA_MOE_LEVEL1_RESULT'):
        write_evidence(report)
    elif os.getenv('HP_DSV41_PARALLEL_RESULT_DIR'):
        output = (Path(os.environ['HP_DSV41_PARALLEL_RESULT_DIR'])
                  / f'{mode}-tp{tp}-cp{cp}-vision{int(vision)}-rank{rank}.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    block.experts.close_runtime()
    dist.destroy_process_group()
