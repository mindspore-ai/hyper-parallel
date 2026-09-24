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
"""Reduced-depth pretraining CP8 comparison with real attention dimensions."""

from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from importlib import import_module
import json
import os
from pathlib import Path
import tempfile
from types import MethodType
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.nn import functional

from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.components.functional.compressed_cann import _operators
from hyper_parallel.components.modules import shared_compressed_dsa_attention as shared
from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM
from hyper_parallel.models.deepseek_v41.adapter.distributed.shared_attention_context_parallel import (
    _build_shared_attention_cp_context,
)
from tests.torch.context_parallel._test_sequence_halo import WorldMesh, init_group
from tests.ut.auto_models.models.deepseek_v41.test_deepseek_v41_crop import _tiny_config, _write_engram_assets


def _model(directory, length):
    """Six pretraining layers with two source banks and Full/Reindex/Reuse."""
    assets = _write_engram_assets(directory, num_hidden_layers=6)
    config = _tiny_config(assets, num_hidden_layers=6)
    config.hidden_size = 512
    config.moe_intermediate_size = 512
    config.n_routed_experts = 8
    config.num_attention_heads = 8
    config.head_dim = 512
    config.q_lora_rank = 256
    config.o_lora_rank = 128
    config.qk_rope_head_dim = 64
    config.partial_rotary_factor = .125
    for rope in config.rope_parameters.values():
        if isinstance(rope, dict):
            rope["partial_rotary_factor"] = .125
    config.index_head_dim = 128
    config.index_n_heads = 32
    config.index_topk = 512
    config.sliding_window = 128
    config.max_position_embeddings = length
    config.v41_compress_ratios = [0, 2, 2, 1, 1, 1]
    config.v41_kv_source_layer_ids = [1, 3]
    config.v41_index_source_layer_ids = [1, 3, 4]
    config.v41_candidate_source_layer_id = -1
    config.v41_candidate_topk_blocks = 4
    config.v41_candidate_block_size = 128
    config.v41_indexer_loss_coeff = .001
    config.tie_word_embeddings = False
    model = DeepseekV41ForCausalLM(config)
    for layer in model.model.layers:
        layer.self_attn = shared.SharedCompressedDSAAttention(layer.self_attn)
        if getattr(layer.self_attn, "indexer", None) is not None:
            assert not layer.self_attn.indexer.is_candidate_source
            assert not layer.self_attn.indexer.uses_candidates
        if hasattr(layer, "engram"):
            layer.engram = EngramModule(module=layer.engram)
    return model


def _attach(model, halo, ep_group):
    """Install production communication, with AG retained only as the test oracle."""
    for layer in model.model.layers:
        context = _build_shared_attention_cp_context(WorldMesh())
        if not halo:
            context = replace(context, launch_raw_halo=None)
        layer.self_attn.forward = partial(
            MethodType(shared.SharedCompressedDSAAttention.forward, layer.self_attn),
            shared_attention_cp_context=context,
        )
        if hasattr(layer, "engram"):
            layer.engram.forward = partial(
                layer.engram.parallel_forward, ep_group=ep_group, ep_rank=0, ep_size=1,
                cp_group=dist.group.WORLD, cp_rank=dist.get_rank(), cp_size=dist.get_world_size(),
            )


def _gradient_metrics(model, expected):
    """Report actual deviations rather than introducing a global percentage gate."""
    accumulators = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            assert name not in expected, name
            continue
        assert name in expected and bool(parameter.grad.isfinite().all()), name
        group = "indexer" if "indexer" in name else "sink" if "sink" in name else "body"
        target = expected[name].to(parameter.device).float()
        torch.testing.assert_close(parameter.grad, expected[name].to(parameter.device), msg=name)
        difference = parameter.grad.float() - target
        stats = accumulators.setdefault(group, torch.zeros(4, device=parameter.device))
        stats[0] += difference.square().sum()
        stats[1] += target.square().sum()
        stats[2] = torch.maximum(stats[2], difference.abs().max())
        stats[3] = torch.maximum(stats[3], target.abs().max())
    result = {}
    for name, stats in accumulators.items():
        error, norm, maximum, scale = stats.cpu().tolist()
        result[name] = {"relative_l2": (error/max(norm, 1e-30))**.5,
                        "max_abs": maximum, "reference_max": scale}
    return result


def _final_weight_record(model, reference, initial, mode, device, li_calls, kl_calls):
    """Verify final parameters and summarize actual native calls and updates."""
    calls = torch.tensor([li_calls, kl_calls], device=device)
    dist.all_reduce(calls)
    if "fused" in mode:
        assert bool((calls > 0).all()), calls
    if mode == "ag_reference":
        reference["weights"] = {
            name: value.detach().cpu().clone() for name, value in model.named_parameters()
        }
    weight_stats = torch.zeros(4, device=device, dtype=torch.float64)
    for name, value in model.named_parameters():
        target_weight = reference["weights"][name].to(device).double()
        torch.testing.assert_close(value.detach().cpu(), reference["weights"][name], msg=name)
        difference = value.detach().double() - target_weight
        weight_stats[0] += difference.square().sum()
        weight_stats[1] += target_weight.square().sum()
        weight_stats[2] = torch.maximum(weight_stats[2], difference.abs().max())
        weight_stats[3] += (value.detach().cpu() != initial[name]).sum().to(device)
    weight_error, weight_norm, weight_max, changed = weight_stats.cpu().tolist()
    return {"mode": mode, "native_calls_all_ranks": calls.cpu().tolist(),
            "parameters": sum(value.numel() for value in model.parameters()),
            "candidate_source": -1, "topk": 512,
            "final_weight_relative_l2": (weight_error/max(weight_norm, 1e-30))**.5,
            "final_weight_max_abs": weight_max, "updated_elements": int(changed)}


def _sum_parameter_gradients(model):
    """Combine local-query contributions before the identical optimizer update."""
    for parameter in model.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def test_v41_training_step_cp8():
    """Run three optimizer steps with mutated packed boundaries on the final combined path."""
    device = init_group("hccl")
    rank, size, length = dist.get_rank(), dist.get_world_size(), 4096
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260923)
    # All arms share the available reference mHC post, independently of CSA.
    mhc = import_module("hyper_parallel.components.functional.mhc_post")
    ep_groups = [dist.new_group([owner]) for owner in range(size)]
    reference = {}
    records = []
    with tempfile.TemporaryDirectory() as directory:
        model = _model(directory, length).to(device=device, dtype=torch.bfloat16)
    for layer in model.model.layers:
        layer.attn_hc.float()
        layer.ffn_hc.float()
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    local = length // size
    start = rank * local
    tokens = torch.randint(3, 64, (1, length+1))
    input_ids = tokens[:, start:start+local].to(device)
    positions = torch.arange(start, start+local, device=device).unsqueeze(0)
    boundaries = torch.tensor([0, 126, 2050, length], device=device)
    packed = shared.SharedCompressedPackedSequence(boundaries, start, local, length)
    set_aux_loss_scale(torch.tensor(1./size, device=device))
    ops = _operators()
    try:
        for mode in ("ag_reference", "ag_reference_repeat", "halo_reference", "halo_fused", "halo_fused_repeat"):
            model.load_state_dict(initial)
            _attach(model, not mode.startswith("ag_"), ep_groups[rank])
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            with ExitStack() as stack:
                stack.enter_context(patch.object(mhc, "omni_training_custom_ops", None))
                if "reference" in mode:
                    stack.enter_context(patch.object(shared, "cann_indexer_available", return_value=False))
                    stack.enter_context(patch.object(shared, "cann_kl_available", return_value=False))
                li = stack.enter_context(patch.object(ops, "lightning_indexer", wraps=ops.lightning_indexer))
                kl = stack.enter_context(patch.object(ops, "sparse_lightning_indexer_kl_loss_grad",
                                                     wraps=ops.sparse_lightning_indexer_kl_loss_grad))
                for step in range(3):
                    splits = [0, 126, 2050, length] if step % 2 == 0 else [0, 130, 2046, length]
                    boundaries.copy_(torch.tensor(splits, device=device))
                    target = tokens[:, 1:].clone()
                    target[:, torch.tensor(splits[1:])-1] = -100
                    target = target[:, start:start+local].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(input_ids=input_ids, position_ids=positions, packed_seq_params=packed,
                                   use_cache=False)
                    loss = functional.cross_entropy(output.logits.float().flatten(0, 1), target.flatten(),
                                                    reduction="sum") / (length-3)
                    loss.backward()
                    _sum_parameter_gradients(model)
                    global_loss = loss.detach().clone()
                    dist.all_reduce(global_loss)
                    assert bool(global_loss.isfinite())
                    record = {"mode": mode, "step": step, "loss": float(global_loss)}
                    if mode == "ag_reference":
                        reference[step] = {
                            "logits": output.logits.detach().cpu(), "loss": global_loss.cpu(),
                            "grads": {name: value.grad.detach().cpu().clone()
                                      for name, value in model.named_parameters() if value.grad is not None},
                        }
                    else:
                        record["gradients"] = _gradient_metrics(model, reference[step]["grads"])
                        torch.testing.assert_close(output.logits.detach().cpu(), reference[step]["logits"])
                        torch.testing.assert_close(global_loss.cpu(), reference[step]["loss"])
                        record["logits_max_abs"] = float(
                            (output.logits.detach().cpu()-reference[step]["logits"]).abs().max()
                        )
                        record["loss_abs"] = float((global_loss.cpu()-reference[step]["loss"]).abs())
                    optimizer.step()
                    records.append(record)
                    if rank == 0:
                        print("TRAINING " + json.dumps(record), flush=True)
                records.append(_final_weight_record(
                    model, reference, initial, mode, device, li.call_count, kl.call_count,
                ))
            optimizer.zero_grad(set_to_none=True)
        if os.environ.get("V41_TEST_REPORT_DIR"):
            path = Path(os.environ["V41_TEST_REPORT_DIR"]) / "model"
            path.mkdir(parents=True, exist_ok=True)
            (path / f"rank{rank}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    finally:
        set_aux_loss_scale(torch.tensor(1.))
        dist.destroy_process_group()
