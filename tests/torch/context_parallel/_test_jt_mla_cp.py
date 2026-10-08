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
"""NPU worker for JT MLA, MTP halos and FSDP/TP optimizer ownership."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.components.optim.builders import AdamW
from hyper_parallel.components.optim.mixed_precision_optimizer import Float16OptimizerWithFloat16Params
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.core.tensor_parallel.loss_parallel import loss_parallel
from hyper_parallel.models._transformers.loss_parallel import causal_lm_loss_parallel
from hyper_parallel.models.jt_deepseek_v3.adapter.jt_builder import _with_model_ep_overrides
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models.build_options import FSDP2Config, FSDP2MixedPrecisionConfig
from hyper_parallel.models.jt_deepseek_v3.configuration_jt_deepseek_v3 import JTDeepseekV3Config
from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3ForCausalLM, observed_fusion_attention,
)
from hyper_parallel.models.jt_deepseek_v3.adapter.distributed.context_parallel import (
    mla_cp_wrapper, configure_context_parallel,
)
from hyper_parallel.models.jt_deepseek_v3.adapter.runtime.jt_optimizer import _after_update
from hyper_parallel.models.replacement import compile_module_replacements, apply_module_replacements
from hyper_parallel.trainer.config import Target, entries_to_module_replacements, entries_to_plan_overrides
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.runtime.metrics import mean_global_loss


def _optimizer(model):
    inner = AdamW({"adamw_lr": 2.4e-5, "adamw_betas": (0.9, 0.95),
                   "adamw_eps": 1e-8, "adamw_weight_decay": 0.1},
                  model=model, no_decay_params=["bias", "norm"]).get_optimizer()
    return Float16OptimizerWithFloat16Params(inner, model)


def _build_pair(tp, sp, strategy):
    world = dist.get_world_size()
    cp = world // tp
    # The JT builder uses EP for fused expert tensors, which cannot be plain TP row shards.
    ep = world if tp > 1 else 1
    mesh = MeshContext(dp_size=1, dp_shard_size=cp, tp_size=tp, cp_size=cp, ep_size=ep,
                       sequence_parallel=sp, loss_parallel=tp > 1)
    mesh.build_meshs("npu", world)
    # Keep TP/EP/SP identical so this comparison measures only the effect of CP.
    reference_mesh = MeshContext(dp_size=cp, dp_shard_size=cp, tp_size=tp, ep_size=ep,
                                 sequence_parallel=sp, loss_parallel=tp > 1)
    reference_mesh.build_meshs("npu", world)
    recipe = parse_training_args([str(Path(__file__).resolve().parents[3] /
                                     "examples/training_demo/jt_deepseek_v3/jt_deepseek_v3.yaml")])
    torch.manual_seed(112)
    base = JTDeepseekV3ForCausalLM(JTDeepseekV3Config(**dict(recipe.model.config))).float()
    replacements = entries_to_module_replacements(recipe.plan_overrides)
    apply_module_replacements(base, compile_module_replacements(base, replacements), weights_mapping=[])
    models = []
    for candidate in (False, True):
        model, info = deepcopy(base).npu(), None
        context = mesh if candidate else reference_mesh
        if candidate or tp > 1:
            overrides = entries_to_plan_overrides(recipe.plan_overrides, cp_size=context.cp_size, ep_size=ep,
                                                 sequence_parallel=sp)
            overrides["*.self_attn"] = replace(
                overrides["*.self_attn"], inner_target="self", inner_wrapper=Target(
                    mla_cp_wrapper,
                    target_path=("hyper_parallel.models.jt_deepseek_v3.adapter.distributed."
                                 "context_parallel.mla_cp_wrapper"),
                    strategy=strategy))
            if ep > 1:
                overrides = _with_model_ep_overrides(
                    DistributedSetup(mesh_context=context, plan_overrides=overrides), model.config).plan_overrides
            plan = ShardingPlanner(plan_overrides=overrides).plan(
                model, context.device_mesh, tp_size=tp, cp_size=context.cp_size, ep_size=ep,
                sequence_parallel=sp, loss_parallel=mesh.loss_parallel)
            model, info = apply_sharding_plan(model, plan, context)
        manager = FSDP2Manager(FSDP2Config(
            dp_shard_size=context.dp_shard_size, reshard_after_forward=True,
            mix_precision=FSDP2MixedPrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32",
                                                   cast_forward_inputs=False)), context, fp32_main_params=True)
        model = manager.parallelize(model, info)
        model.qk_clip_group = context.dp_cp_mesh.get_group()
        if candidate:
            configure_context_parallel(model, mesh)
        if context.loss_parallel:
            model.loss_function = causal_lm_loss_parallel
        models.append((model, _optimizer(model), context))
    return models, mesh


def _backward(model, optimizer, mesh, inputs, labels, sequence_ends=None):
    optimizer.zero_grad()
    loss_context = loss_parallel(mesh=mesh.device_mesh["tp"]) if mesh.loss_parallel else nullcontext()
    with SkipDTensorDispatch(), loss_context:
        output = model(input_ids=inputs, shift_labels=labels, actual_seq_len=sequence_ends)
        assert all(loss.dtype == torch.float32 for loss in output.loss.values())
        count = labels.ge(0).sum()
        if mesh.sequence_parallel:
            count = count / mesh.tp_size
        losses = mean_global_loss(output.loss, {"foundation_tokens": count}, {"foundation_tokens": count}, mesh)
        sum(losses.values()).backward()
    return losses


def _compare_gradients(reference, candidate):
    errors, norms, details = [], [], []
    for name, expected in reference.named_parameters():
        actual = candidate.get_parameter(name)
        reference_grad = getattr(expected, "main_grad", None)
        candidate_grad = getattr(actual, "main_grad", None)
        assert reference_grad is not None and candidate_grad is not None, name
        full_reference, full_candidate = reference_grad.full_tensor().float(), candidate_grad.full_tensor().float()
        assert torch.isfinite(full_candidate).all(), name
        errors.append((full_candidate - full_reference).square().sum())
        norms.append(full_reference.square().sum())
        details.append((name, errors[-1].item(), norms[-1].item(), full_candidate.square().sum().item()))
    relative = (torch.stack(errors).sum() / torch.stack(norms).sum()).sqrt()
    worst = sorted(details, key=lambda item: item[1], reverse=True)[:5]
    if dist.get_rank() == 0:
        print({"gradient_relative_l2": relative.item(), "largest_squared_errors": worst}, flush=True)
    assert relative < 0.035, f"Global parameter-gradient relative L2: {relative.item()}, largest errors: {worst}"


def _compare_parameters(reference, candidate):
    for name, expected in reference.named_parameters():
        actual = candidate.get_parameter(name)
        assert expected.main_param.dtype == actual.main_param.dtype == torch.float32
        expected_value = expected.main_param.full_tensor().float()
        actual_value = actual.main_param.full_tensor().float()
        relative = (actual_value - expected_value).norm() / expected_value.norm().clamp_min(1e-8)
        assert relative < 0.025, f"{name} main parameter relative L2: {relative.item()}"


def _training_case(tp, sp, strategy):
    models, mesh = _build_pair(tp, sp, strategy)
    rank = mesh.cp_mesh.get_local_rank()
    torch.manual_seed(913)
    for _step in range(2):
        tokens = torch.randint(0, 512, (1, 257), device="cpu").npu()
        # A one-row offset slice is already contiguous and still aliases input IDs.
        inputs, labels = tokens[:, :-1].contiguous(), tokens[:, 1:].clone()
        labels[:, 17::29] = -100
        sequence_ends = None
        if _step == 1:
            # Boundaries cross shards and coincide with a shard edge; supervision exists only in the last shard.
            sequence_ends = (37, 128, 177, 256)
            labels[:, :192] = -100
            labels[:, [end - 1 for end in sequence_ends]] = -100
        assert inputs.min() >= 0
        expected = _backward(*models[0], inputs, labels, sequence_ends)
        actual = _backward(*models[1], inputs.chunk(mesh.cp_size, 1)[rank].contiguous(),
                           labels.chunk(mesh.cp_size, 1)[rank].contiguous(), sequence_ends)
        if dist.get_rank() == 0:
            print({"strategy": strategy, "tp": tp, "cp": mesh.cp_size, "sp": sp, "step": _step + 1,
                   "reference": {key: value.item() for key, value in expected.items()},
                   "candidate": {key: value.item() for key, value in actual.items()}}, flush=True)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, rtol=0.01,
                                       atol=1e-6 if name.endswith("/aux") else 0.002)
        _compare_gradients(models[0][0], models[1][0])
        for model, optimizer, _context in models:
            with SkipDTensorDispatch(no_skip={torch.zeros_like}):
                optimizer.step()
            # Force clipping to exercise both FP32 main weights and FSDP row ownership.
            _after_update(model, 0.5, optimizer, (), {})
            model.reset_iter_state()
        _compare_parameters(models[0][0], models[1][0])


def _check_fa_statistics():
    """Compare packed FA outputs, gradients and QK statistics with a literal block-causal oracle."""
    torch.manual_seed(102)
    for ends in ((64,), (13, 41, 64)):
        query = torch.randn(1, 4, 64, 192, dtype=torch.bfloat16).npu().requires_grad_()
        key = torch.randn_like(query).requires_grad_()
        value = torch.randn(1, 4, 64, 128, dtype=torch.bfloat16).npu().requires_grad_()
        module = torch.nn.Module()
        module.max_logits_val, module.is_causal = None, True
        output, _ = observed_fusion_attention(
            module, query, key, value, None, scaling=192 ** -0.5, actual_seq_len=ends)
        reference = [tensor.detach().cpu().float().requires_grad_() for tensor in (query, key, value)]
        scores = reference[0] @ reference[1].transpose(-1, -2) * (192 ** -0.5)
        allowed = torch.zeros(64, 64, dtype=torch.bool)
        for begin, end in zip((0, *ends[:-1]), ends):
            allowed[begin:end, begin:end] = torch.ones(end - begin, end - begin, dtype=torch.bool).tril()
        scores = scores.masked_fill(~allowed, -torch.inf)
        expected = (scores.softmax(-1) @ reference[2]).transpose(1, 2)
        torch.testing.assert_close(module.max_logits_val.cpu(), scores.detach().amax((0, 2, 3)),
                                   rtol=2e-5, atol=2e-5)
        upstream = torch.randn_like(output)
        output.backward(upstream)
        expected.backward(upstream.cpu().float())
        pairs = [(output.detach().cpu().float(), expected.detach())]
        pairs.extend((actual.grad.cpu().float(), target.grad) for actual, target in zip((query, key, value), reference))
        for actual, target in pairs:
            relative = (actual - target).norm() / target.norm().clamp_min(1e-8)
            assert relative < 0.025, f"Packed FA relative L2 for ends {ends}: {relative.item()}"


def test_jt_mla_cp_training():
    """Check both CP paths, TP/SP and active QK clipping on the real NPU backend."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch_npu.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group("hccl", timeout=timedelta(seconds=240))
    try:
        _check_fa_statistics()
        for strategy in ("expanded_ulysses", "latent_kv_head"):
            for tp, sp in ((1, False), (2, False), (2, True)):
                _training_case(tp, sp, strategy)
    finally:
        dist.destroy_process_group()
