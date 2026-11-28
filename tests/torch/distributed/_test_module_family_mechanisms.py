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
"""Distributed worker acceptance for actual communication, state and FSDP2."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.dtensor.random import manual_seed
from hyper_parallel.components.modules import (
    DeepseekSparseAttention, GatedDeltaNet, GatedGroupedQueryAttention, GatedMultiLatentAttention,
    GroupedQueryAttention, KimiDeltaAttention, MultiLatentAttention, OffsetRMSNorm,
)
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.components.checkpoint.dcp_checkpointer import DistributedCheckpointer
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.context_parallel.cp_context import CPContext
from hyper_parallel.distributed.expert_parallel.ep_context import EPContext
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights, _move_model_to_device
from hyper_parallel.models._transformers import model_builder
from hyper_parallel.models.module_build import (
    attach_build_context, build_module_context, prepare_model_components, resolve_effective_model_spec,
)
from hyper_parallel.models.replacement import ModuleReplacementRequest
from tests.common.module_family_fixtures import LinearConfig, LinearFamily, register_linear_family
from tests.common.parallel_module_fixtures import (
    AttentionConfig, LocalAttention, MODEL_SPEC, create_attention_model,
    ExpertConfig, LocalScaleExperts, create_expert_model,
)

from tests.common.training_state import (
    capture_full_training_state, load_full_training_state, restore_full_training_state, save_full_training_state,
)


def _initialize(device):
    if device == "npu":
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl" if device == "npu" else "gloo", timeout=timedelta(seconds=180))
    mesh = MeshContext(dp_size=2, cp_size=2, dp_shard_size=4)
    mesh.build_meshs(device, dist.get_world_size())
    return mesh


def _check_sequence_transport(mesh, device):
    """Compare differentiable CP gathers and sequence/head exchanges with references."""
    context = CPContext(mesh.cp_mesh)
    tensor = (torch.arange(16, device=device, dtype=torch.float32).reshape(1, 2, 4, 2)
              + dist.get_rank() * 20).requires_grad_()
    pending = context.launch_gather_sequence(tensor)
    gathered, receipt = pending.wait()
    peers = [torch.empty_like(tensor) for _ in range(2)]
    dist.all_gather(peers, tensor.detach(), group=mesh.cp_mesh.get_group())
    torch.testing.assert_close(gathered, torch.cat(peers, dim=1))
    assert receipt.offset == context.rank * 2
    gathered.sum().backward()
    torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 2))
    exchanged = context.sequence_to_head(tensor)
    restored = context.head_to_sequence(exchanged)
    torch.testing.assert_close(restored, tensor)
    tensor.grad = None
    restored.sum().backward()
    torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))


def _linear_sharding(*, module, module_fqn, member, context):
    del module_fqn, member, context
    return ModuleShardingSpec(params={name: {"tp": Shard(0)} for name, _ in module.named_parameters()},
                              in_src={}, in_dst={}, is_boundary=False)


def _check_family_aliases():
    """Verify shared family parameters retain their identity and gradient ownership."""
    register_linear_family(sharding=_linear_sharding)
    model = nn.Module()
    model.component = LinearFamily(config=LinearConfig(output_size=4))
    model.alias = model.component
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    plan = ShardingPlanner(derive=False).plan(model, mesh, tp_size=2)
    model, source = apply_sharding_plan(model, plan, mesh)
    assert model.component.weight is model.alias.weight
    assert model.component.weight.shape == (2, 4)
    assert set(source) == {"component.weight", "component.bias"}
    assert set(plan.parameter_aliases) == {("component.weight", "alias.weight"), ("component.bias", "alias.bias")}
    tokens = torch.ones(2, 4)
    (model.component(tokens).sum() + model.alias(tokens).sum()).backward()
    torch.testing.assert_close(model.component.weight.grad, torch.full((2, 4), 4.))


def _source_slots(rank, iteration, device):
    count = 0 if rank % 2 == iteration % 2 else 2
    tokens = (torch.arange(count * 3, device=device, dtype=torch.float32).reshape(count, 3) / 10
              + rank * 0.2).requires_grad_()
    indices = (torch.arange(count * 2, device=device).reshape(count, 2) + rank) % 4
    probabilities = torch.full((count, 2), 0.3 + rank * 0.05, device=device, requires_grad=True)
    return tokens, indices, probabilities


def _reference_experts(tokens, indices, probabilities, weight):
    return (torch.tanh(tokens[:, None, :] * weight[indices]) * probabilities[:, :, None]).sum(dim=1)


def _permute_routed_slots(hidden_states, expert_indices, routing_weights):
    """Prepare the worker's selected routes in expert-major order."""
    top_k = expert_indices.shape[1]
    ids, order = expert_indices.reshape(-1).sort(stable=True)
    rows = torch.arange(order.numel(), device=order.device)
    source_inverse = torch.empty_like(order).scatter_(0, order, rows)
    hidden = hidden_states.index_select(0, order // top_k)
    weights = routing_weights.reshape(-1).index_select(0, order)
    return hidden, ids, weights, source_inverse


def _sum_routed_slots(values, source_inverse, top_k):
    """Restore the worker's source routes and sum top-k outputs."""
    values = values.index_select(0, source_inverse)
    return values.reshape(-1, top_k, values.shape[-1]).sum(dim=1)


def _check_expert_transport(device):
    """Compare routed EP transport and probability gradients with dense experts."""
    mesh = init_device_mesh(device, (2, 2), mesh_dim_names=("edp", "ep"))
    context = EPContext(mesh["ep"], num_experts=4)
    global_weight = torch.arange(12, device=device, dtype=torch.float32).reshape(4, 3) / 10 + 0.1
    for iteration in range(2):
        tokens, indices, probabilities = _source_slots(dist.get_rank(), iteration, device)
        local_weight = global_weight.chunk(2)[context.rank].clone().requires_grad_()
        slots, expert_ids, weights, source_inverse = _permute_routed_slots(tokens, indices, probabilities)
        hidden, counts, weights, *transport = context.dispatch(slots, expert_ids, weights)
        local_ids = torch.arange(counts.numel(), device=counts.device).repeat_interleave(
            counts, output_size=hidden.shape[0],
        )
        values = torch.tanh(hidden * local_weight[local_ids]) * weights[:, None]
        actual = _sum_routed_slots(context.combine(values, *transport), source_inverse, indices.shape[1])
        expected = _reference_experts(tokens, indices, probabilities, global_weight)
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.sum(), (tokens, probabilities, local_weight))
        expected_grads = torch.autograd.grad(expected.sum(), (tokens, probabilities))
        for value, reference in zip(actual_grads[:2], expected_grads):
            torch.testing.assert_close(value, reference)
        reference_weight = global_weight.clone().requires_grad_()
        group_base = (dist.get_rank() // 2) * 2
        reference_loss = sum(_reference_experts(*_source_slots(rank, iteration, device), reference_weight).sum()
                             for rank in range(group_base, group_base + 2))
        reference_grad, = torch.autograd.grad(reference_loss, (reference_weight,))
        torch.testing.assert_close(actual_grads[2], reference_grad.chunk(2)[context.rank])


def _checkpoint_path():
    paths = [tempfile.mkdtemp(prefix="hp-family-state-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(paths)
    return Path(paths[0])


def _zero_tensors(value):
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    if isinstance(value, dict):
        return {key: _zero_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_zero_tensors(item) for item in value]
    return deepcopy(value)


def _update(model, optimizer, tokens):
    with SkipDTensorDispatch():
        output = model(*tokens) if isinstance(tokens, tuple) else model(tokens)
        output.square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if hasattr(model, "reset_iter_state"):
        model.reset_iter_state()
    return output.detach()


def _roundtrip_state(model, optimizer, tokens, make_target):
    """Save complete training state and compare the resumed update with uninterrupted training."""
    state = capture_full_training_state(model, optimizer, global_step=1,
                                       data_progress={"rank": dist.get_rank(), "sample": 12})
    directory = _checkpoint_path()
    checkpointer = DistributedCheckpointer(extra_state_per_rank=True)
    save_full_training_state(checkpointer, str(directory / "global_step_1"), state)
    loaded = load_full_training_state(checkpointer, str(directory / "global_step_1"), _zero_tensors(state))
    target = make_target()
    target_opt = torch.optim.AdamW(target.parameters(), lr=0.01)
    identities = [id(parameter) for parameter in target.parameters()]
    extra = restore_full_training_state(target, target_opt, loaded)
    assert identities == [id(parameter) for parameter in target.parameters()]
    assert extra["global_step"] == 1 and extra["data_progress"] == {"rank": dist.get_rank(), "sample": 12}
    actual = _update(target, target_opt, tokens)
    expected = _update(model, optimizer, tokens)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual_state = capture_full_training_state(target, target_opt, global_step=2, data_progress={})
    expected_state = capture_full_training_state(model, optimizer, global_step=2, data_progress={})
    for name, value in expected_state["model"].items():
        torch.testing.assert_close(actual_state["model"][name], value, rtol=2e-5, atol=2e-6)
    for name, values in expected_state["optimizer"]["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(actual_state["optimizer"]["state"][name][key], value, rtol=2e-5, atol=2e-6)
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(directory)


def _build_cp_model(mesh, device, fsdp):
    """Build a CP family with optional main-repository FSDP2 wrapping."""
    context = build_module_context(DistributedSetup(mesh_context=mesh), device=device, dtype=torch.float32,
                                   model_spec=MODEL_SPEC)
    model = create_attention_model(AttentionConfig(width=4), context={})
    original_weight = model.attention.weight
    model, _ = prepare_model_components(model, context)
    assert model.attention.weight is original_weight
    model = model.to(device)
    plan = ShardingPlanner(derive=False).plan(model, mesh.device_mesh, cp_size=2)
    implementation = type(model.attention).forward
    model, source = apply_sharding_plan(model, plan, mesh)
    assert type(model.attention).forward is implementation and "forward" not in vars(model.attention)
    assert set(source) == {"attention.weight"}
    if fsdp:
        model = FSDP2Manager(FSDP2Config(dp_shard_size=4), mesh).parallelize(model, source)
    return model


def _check_direct_cp_model(mesh, device, fsdp):
    """Compare CP model updates and checkpoint continuation with a dense reference."""
    model = _build_cp_model(mesh, device, fsdp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    all_tokens = torch.arange(32, device=device, dtype=torch.float32).reshape(2, 4, 4) / 10
    dp_rank, cp_rank = mesh.device_mesh["dp"].get_local_rank(), mesh.cp_mesh.get_local_rank()
    tokens = all_tokens[dp_rank:dp_rank + 1, cp_rank * 2:(cp_rank + 1) * 2].contiguous()
    reference = LocalAttention(config=AttentionConfig(width=4)).to(device)
    reference_opt = torch.optim.AdamW(reference.parameters(), lr=0.01)
    actual = _update(model, optimizer, tokens)
    expected = reference(all_tokens)
    torch.testing.assert_close(actual, expected[dp_rank:dp_rank + 1, cp_rank * 2:(cp_rank + 1) * 2],
                               rtol=2e-5, atol=2e-6)
    if fsdp:
        expected.square().mean().backward()
        reference_opt.step()
        reference_opt.zero_grad(set_to_none=True)
        state = capture_full_training_state(model, optimizer, global_step=1, data_progress={})
        torch.testing.assert_close(state["model"]["attention.weight"].cpu(), reference.weight.cpu(),
                                   rtol=2e-5, atol=2e-6)
        for key, value in reference_opt.state[reference.weight].items():
            torch.testing.assert_close(state["optimizer"]["state"]["attention.weight"][key].cpu(), value.cpu(),
                                       rtol=2e-5, atol=2e-6)
        _roundtrip_state(model, optimizer, tokens, lambda: _build_cp_model(mesh, device, True))


def _build_ep_model(mesh):
    """Build an EP family and apply its owned expert layout."""
    context = build_module_context(DistributedSetup(mesh_context=mesh), device="npu", dtype=torch.float32)
    model = create_expert_model(ExpertConfig(), context={})
    original_weight = model.routed_bank.weight
    model, _ = prepare_model_components(model, context)
    assert model.routed_bank.weight is original_weight
    model = model.to("npu")
    plan = ShardingPlanner(derive=False).plan(model, mesh.device_mesh, ep_size=2)
    model, source = apply_sharding_plan(model, plan, mesh)
    assert set(source) == {"routed_bank.weight"}
    return FSDP2Manager(FSDP2Config(dp_shard_size=4, edp_shard_size=2), mesh).parallelize(model, source)


def _check_ep_fsdp_model():
    """Verify expert ownership, optimizer moments and DCP continuation under FSDP2."""
    mesh = MeshContext(dp_size=4, dp_shard_size=4, ep_size=2, edp_shard_size=2)
    mesh.build_meshs("npu", dist.get_world_size())
    model = _build_ep_model(mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    hidden = torch.arange(32, device="npu", dtype=torch.float32).reshape(4, 2, 4) / 20
    ids = torch.stack([(torch.arange(4, device="npu").reshape(2, 2) + rank) % 4 for rank in range(4)])
    weights = torch.stack([torch.full((2, 2), 0.2 + rank * 0.05, device="npu") for rank in range(4)])
    rank = dist.get_rank()
    tokens = (hidden[rank], ids[rank], weights[rank])
    reference = LocalScaleExperts(config=ExpertConfig()).to("npu")
    reference_opt = torch.optim.AdamW(reference.parameters(), lr=0.01)
    actual = _update(model, optimizer, tokens)
    expected = reference(hidden.reshape(-1, 4), ids.reshape(-1, 2), weights.reshape(-1, 2))
    torch.testing.assert_close(actual.cpu(), expected.reshape(4, 2, 4)[rank].cpu(), rtol=2e-5, atol=2e-6)
    expected.square().mean().backward()
    reference_opt.step()
    reference_opt.zero_grad(set_to_none=True)
    state = capture_full_training_state(model, optimizer, global_step=1, data_progress={})
    torch.testing.assert_close(state["model"]["routed_bank.weight"].cpu(), reference.weight.cpu(),
                               rtol=2e-5, atol=2e-6)
    for key, value in reference_opt.state[reference.weight].items():
        torch.testing.assert_close(state["optimizer"]["state"]["routed_bank.weight"][key].cpu(), value.cpu(),
                                   rtol=2e-5, atol=2e-6)
    _roundtrip_state(model, optimizer, tokens, lambda: _build_ep_model(mesh))


def test_parallel_contexts_and_dcp_cpu() -> None:
    """Run actual CP/EP collectives and a complete optimizer-state DCP restart."""
    mesh = _initialize("cpu")
    try:
        _check_sequence_transport(mesh, "cpu")
        _check_family_aliases()
        _check_expert_transport("cpu")
        _check_direct_cp_model(mesh, "cpu", False)
        torch.manual_seed(3)
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        tokens = torch.ones(2, 2)
        _update(model, optimizer, tokens)
        _roundtrip_state(model, optimizer, tokens, lambda: nn.Linear(2, 2))
    finally:
        dist.destroy_process_group()


def test_family_cp_fsdp_dcp_npu() -> None:
    """Compare real main-FSDP2 gradients/updates and continuation on DP2 x CP2."""
    mesh = _initialize("npu")
    try:
        _check_sequence_transport(mesh, "npu")
        _check_expert_transport("npu")
        _check_direct_cp_model(mesh, "npu", True)
        _check_ep_fsdp_model()
    finally:
        dist.destroy_process_group()


class _AttentionInitializationModel(nn.Module):
    """Retain a native projection policy and inspect its final local parameter shards."""

    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.attention = attention
        self.native_values = {}

    @torch.no_grad()
    def initialize_weights(self) -> None:
        """Use the model's distribution and capture final parameter values for comparison."""
        for child in self.modules():
            if isinstance(child, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(child.weight, std=.017)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
        self.native_values = {name: value.to_local().detach().clone() for name, value in self.named_parameters()}


def test_attention_meta_initialization_cp_fsdp_npu() -> None:
    """Feature: Attention defaults and native initialization on actual CP/FSDP shards.
    Description: Promote local meta members to CP2, bind layouts, shard and initialize storage on two NPUs.
    Expectation: All shards are finite and exactly preserve the native policy, including recurrence states.
    """
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", torch.npu.current_device())
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        mesh = MeshContext(dp_size=1, cp_size=2, dp_shard_size=2, sequence_parallel=False)
        mesh.build_meshs("npu", dist.get_world_size())
        config = SimpleNamespace(
            hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=8,
            attention_bias=True, qk_norm=True, rms_norm_eps=1e-6, index_head_dim=8,
            index_n_heads=2, index_topk=1024, num_heads=4, head_k_dim=4, head_v_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=4, linear_value_head_dim=8,
        )
        cases = ((GroupedQueryAttention, {}), (GroupedQueryAttention, {"fuse_qkv": False}),
                 (GatedGroupedQueryAttention, {}), (MultiLatentAttention, {}), (GatedMultiLatentAttention, {}),
                 (GatedDeltaNet, {"ulysses_degree": None}), (KimiDeltaAttention, {"ulysses_degree": None}),
                 (DeepseekSparseAttention, {"kv_cp_size": 2}))
        context = build_module_context(DistributedSetup(mesh_context=mesh), device=device, dtype=torch.float32)
        for factory, options in cases:
            with torch.device("meta"):
                model = _AttentionInitializationModel(factory(config=config, **options))
            model, _ = prepare_model_components(model, context)
            plan = ShardingPlanner(derive=False).plan(model, mesh.device_mesh, cp_size=2, sequence_parallel=False)
            model, source = apply_sharding_plan(model, plan, mesh)
            fsdp = FSDP2Config(dp_shard_size=2, wrapper_class=[type(model.attention).__name__])
            model = FSDP2Manager(fsdp, mesh).parallelize(model, source)
            model = _move_model_to_device(model, True, device)
            manual_seed(421, mesh.device_mesh)
            _initialize_model_weights(model)
            local_numel, global_numel = 0, 0
            for name, parameter in model.named_parameters():
                value = parameter.to_local()
                local_numel += value.numel()
                global_numel += parameter.numel()
                assert torch.isfinite(value).all(), name
                torch.testing.assert_close(value, model.native_values[name], rtol=0, atol=0, msg=name)
            assert local_numel < global_numel
    finally:
        dist.destroy_process_group()


def _poison_materialized_parameters(model, is_meta_device, device):
    """Make any missed random initialization fail deterministically on its local shard."""
    model = _move_model_to_device(model, is_meta_device, device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.to_local().fill_(float("nan"))
    return model


def test_qwen35_native_initialization_cp_fsdp_npu() -> None:
    """Feature: Real Qwen dense/MoE model initialization after CP2 and FSDP2.
    Description: Poison every materialized shard and run native HF initialization after source replacement.
    Expectation: Norms, router and all parameters initialize; distributed A_log matches native HF exactly.
    """
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", torch.npu.current_device())
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        mesh = MeshContext(dp_size=1, cp_size=2, dp_shard_size=2, sequence_parallel=False)
        mesh.build_meshs("npu", dist.get_world_size())
        for model_type, config_type in ((Qwen3_5ForCausalLM, Qwen3_5TextConfig),
                                        (Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig)):
            config = config_type.from_dict({
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "layer_types": ["linear_attention"], "initializer_range": 0.017,
                "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
                "num_experts": 4, "num_experts_per_tok": 2,
            })
            reference = model_type(config).to(device)
            with torch.device("meta"):
                model = model_type(config)
            context = dict(build_module_context(DistributedSetup(mesh_context=mesh), device=device,
                           model_spec=resolve_effective_model_spec(config), config=config))
            context["module_replacement_requests"] = (
                ModuleReplacementRequest(match=("model.layers.*.linear_attn",), target=GatedDeltaNet,
                                         kwargs={"ulysses_degree": None}),
                ModuleReplacementRequest(match=("model.layers.*.input_layernorm",
                                                "model.layers.*.post_attention_layernorm", "model.norm"),
                                         target=OffsetRMSNorm),
            )
            attach_build_context(model, context)
            fsdp = FSDP2Config(dp_shard_size=2,
                              wrapper_class=[model_type.__name__.replace("ForCausalLM", "DecoderLayer")])
            manual_seed(421, mesh.device_mesh)
            with patch.object(model_builder, "_move_model_to_device", _poison_materialized_parameters):
                model = model_builder.apply_model_infrastructure(
                    model, mesh=mesh, sharding_planner=ShardingPlanner(derive=False),
                    fsdp2_manager=FSDP2Manager(fsdp, mesh), is_meta_device=True, is_hf_model=True, device=device)
            local_numel = 0
            for name, parameter in model.named_parameters():
                value = parameter.to_local()
                local_numel += value.numel()
                assert torch.isfinite(value).all(), name
                if name.endswith("dt_bias"):
                    torch.testing.assert_close(value, torch.ones_like(value), rtol=0, atol=0)
                elif "layernorm" in name or name == "model.norm.weight":
                    torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
            assert local_numel < sum(parameter.numel() for parameter in model.parameters())
            assert not model.is_custom_code()
            expected = reference.model.layers[0].linear_attn
            actual = model.model.layers[0].linear_attn
            torch.npu.manual_seed(35)
            reference.model._init_weights(expected)
            torch.npu.manual_seed(35)
            model.model._init_weights(actual)
            torch.testing.assert_close(actual.A_log.full_tensor(), expected.A_log, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
