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
"""Six-layer ordinary-pretraining CP8 integration, using upstream AG collectives."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import timedelta
import json
import os
from time import perf_counter
from unittest.mock import patch

import torch
import torch.distributed as dist

from hyper_parallel.components.functional.compressed_indexer_ops import _operators
from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.components.modules import shared_compressed_dsa_attention as shared
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Attention
from hyper_parallel.models.deepseek_v41.adapter.distributed.shared_attention_context_parallel import (
    _build_shared_attention_cp_context,
)
from tests.torch.context_parallel._test_deepseek_v41_dsa_cp import _config, _CPMesh
from tests.torch.context_parallel._test_v41_native_ops import _device


def _chain(native=False):
    """Cover SWA, r2 Full/Reuse, r1 Full/Reindex/Reuse and two source banks."""
    config = _config()
    config.num_hidden_layers = 6
    config.layer_types = ["sliding_attention"] * 6
    config.mlp_layer_types = ["moe"] * 6
    config.v41_compress_ratios = [0, 2, 2, 1, 1, 1]
    config.v41_kv_source_layer_ids = [1, 3]
    config.v41_index_source_layer_ids = [1, 3, 4]
    config.v41_candidate_source_layer_id = -1
    config.v41_indexer_loss_coeff = .01
    config.index_topk = 512
    if native:
        config.hidden_size = 64
        config.head_dim = 512
        config.num_attention_heads = 8
        config.index_n_heads = 8
        config.index_head_dim = 128
        config.sliding_window = 128
        config.qk_rope_head_dim = 64
    torch.manual_seed(975)
    modules = torch.nn.ModuleList([DeepseekV41Attention(config, layer) for layer in range(6)])
    with torch.no_grad():
        for name, value in modules.named_parameters():
            if name.endswith("sinks"):
                value.copy_(torch.linspace(-2, 2, value.numel()))
            elif value.ndim == 1:
                value.fill_(1)
            else:
                value.normal_(0, .03)
    return modules


def _forward(modules, hidden, start, boundaries, context, use_fused=False):
    """Recreate forward-scoped sharing while preserving learned parameter identity."""
    length = hidden.shape[1]
    positions = torch.arange(start, start+length, device=hidden.device).unsqueeze(0)
    rotary = modules[0].rope_head_dim
    freq = 10000 ** (-torch.arange(0, rotary, 2, device=hidden.device).float()/rotary)
    angles = positions.float().unsqueeze(-1)*freq
    embeddings = {kind: (angles.cos(), angles.sin()) for kind in ("main", "compress")}
    packed = shared.SharedCompressedPackedSequence(boundaries, start, length, int(boundaries[-1]))
    packed = packed.prepare(hidden.device, (0, 1, 2))
    state = shared.SharedCompressedAttentionState()
    for module in modules:
        module.use_optimized_sparse_attention = use_fused
        if module.is_index_source:
            module.indexer.use_fused = use_fused
        parameters = {name: value if name == "sinks" or hidden.dtype == torch.float32 else value.bfloat16()
                      for name, value in module.named_parameters()}
        update = torch.func.functional_call(
            module, parameters, (hidden, embeddings, positions, None),
            {"shared_attention_state": state, "shared_attention_cp_context": context, "packed_seq_params": packed},
        )[0]
        hidden = hidden + update
    return hidden


def _sum_gradients(modules):
    """Combine replicated parameter gradients using the same original SUM semantics."""
    for value in modules.parameters():
        if value.grad is not None:
            dist.all_reduce(value.grad, op=dist.ReduceOp.SUM)


def test_native_geometry_cp8_gloo():
    """Compare full and CP8 outputs, every gradient and ten SGD updates in FP32."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    try:
        rank, size, length = dist.get_rank(), dist.get_world_size(), 128
        assert size == 8
        local = length//size
        reference = _chain()
        candidate = torch.nn.ModuleList([
            shared.SharedCompressedDSAAttention(module, use_fused_kernels=False) for module in deepcopy(reference)
        ])
        context = _build_shared_attention_cp_context(_CPMesh())
        optimizers = [torch.optim.SGD(model.parameters(), lr=.01) for model in (reference, candidate)]
        for step in range(10):
            torch.manual_seed(100+step)
            boundaries = torch.tensor([0, 30 if step % 2 else 34, 78, length])
            source = torch.randn(1, length, 32)
            full = source.clone().requires_grad_()
            shard = source[:, rank*local:(rank+1)*local].clone().requires_grad_()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            set_aux_loss_scale(torch.tensor(1.))
            expected = _forward(reference, full, 0, boundaries, None)
            expected.square().mean().backward()
            set_aux_loss_scale(torch.tensor(1./size))
            actual = _forward(candidate, shard, rank*local, boundaries, context)
            (actual.square().sum()/expected.numel()).backward()
            _sum_gradients(candidate)
            torch.testing.assert_close(actual, expected[:, rank*local:(rank+1)*local])
            torch.testing.assert_close(shard.grad, full.grad[:, rank*local:(rank+1)*local])
            for (name, value), (other_name, target) in zip(candidate.named_parameters(), reference.named_parameters()):
                assert name == other_name
                if value.grad is None or target.grad is None:
                    assert value.grad is target.grad, name
                else:
                    torch.testing.assert_close(value.grad, target.grad, msg=name)
            for optimizer in optimizers:
                optimizer.step()
            for value, target in zip(candidate.parameters(), reference.parameters()):
                torch.testing.assert_close(value, target)
        print(json.dumps({"case": "native_geometry_cp8_gloo", "rank": rank, "steps": 10,
                              "ratios": [0, 2, 2, 1, 1, 1], "all_parameter_gradients": True}), flush=True)
    finally:
        set_aux_loss_scale(torch.tensor(1.))
        dist.destroy_process_group()


def _deviation(actual, expected):
    """Measure every element; default_close is diagnostic, never silently relaxed."""
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    delta = actual-expected
    try:
        torch.testing.assert_close(actual, expected)
        close = True
    except AssertionError:
        close = False
    return {"relative_l2": float(delta.norm()/expected.norm().clamp_min(1e-30)),
            "max_abs": float(delta.abs().max()), "finite": bool(actual.isfinite().all()), "default_fp32_close": close}


def _collect_step_gradients(model, output, loss):
    """Check finite BF16 execution and keep FP32 master gradients for comparison."""
    current = {name: value.grad.detach().cpu().clone() for name, value in model.named_parameters()
               if value.grad is not None}
    assert current and all(bool(value.isfinite().all()) for value in current.values())
    assert all(value.grad is None or value.grad.dtype == torch.float32 for value in model.parameters())
    assert bool(output.isfinite().all()) and bool(loss.isfinite())
    return current


def _report_npu_step(rank, step, output, hidden, current, expected, milliseconds):
    """Report numerical differences without redefining model acceptance thresholds."""
    expected_output, expected_hidden, expected_gradients = expected
    assert current.keys() == expected_gradients.keys()
    gradients = {}
    for group in ("indexer", "sinks", "body"):
        names = [name for name in current if (
            "indexer" if "indexer" in name else "sinks" if "sinks" in name else "body") == group]
        gradients[group] = _deviation(torch.cat([current[name].flatten() for name in names]),
                                     torch.cat([expected_gradients[name].flatten() for name in names]))
    print(json.dumps({"case": "native_fused_cp8_smoke", "rank": rank, "step": step,
                      "output": _deviation(output, expected_output),
                      "input_gradient": _deviation(hidden.grad, expected_hidden),
                      "gradients": gradients, "forward_backward_ms": milliseconds,
                      "peak_allocated_mib": torch.npu.max_memory_allocated()/2**20}), flush=True)


def _count_native_dispatch(stack):
    """Track calls without a Mock retaining accelerator activation tensors."""
    calls = {}
    for name in ("lightning_indexer", "sparse_flash_mla", "sparse_lightning_indexer_kl_loss_grad"):
        calls[name] = 0

        def _counted(*args, _name=name, _native=getattr(_operators(), name), **kwargs):
            """Count dispatch without keeping tensor arguments alive."""
            calls[_name] += 1
            return _native(*args, **kwargs)

        stack.enter_context(patch.object(_operators(), name, new=_counted))
    return calls


def test_native_cp8_training_smoke():
    """Run ten BF16 steps with FP32 master weights and unchanged CP8 collectives.

    This is a finite/update/dispatch smoke with an explicit numerical report;
    it does not certify the full Trainer, FSDP, EP, or convergence.
    """
    torch.set_num_threads(1)
    device = _device()
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        length = int(os.environ.get("V41_CP_LENGTH", "256"))
        steps = int(os.environ.get("V41_CP_STEPS", "10"))
        assert size == 8
        assert length >= 256 and length % 16 == 0 and steps >= 2
        # Initialize communication before optional native metadata services.
        ready = torch.ones(1, device=device)
        dist.all_reduce(ready)
        torch.npu.synchronize()
        torch.use_deterministic_algorithms(True)
        initial = _chain(native=True)
        weights = deepcopy(initial.state_dict())
        context = _build_shared_attention_cp_context(_CPMesh())
        local = length//size
        reference = []
        baseline_weights = None
        set_aux_loss_scale(torch.tensor(1./size, device=device))
        for fused in (False, True):
            model = deepcopy(initial).to(device)
            model.load_state_dict(weights)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            with ExitStack() as stack:
                calls = _count_native_dispatch(stack) if fused else {}
                for step in range(steps):
                    torch.manual_seed(200+step)
                    source = torch.randn(1, length, 64).bfloat16()
                    hidden = source[:, rank*local:(rank+1)*local].to(device).requires_grad_()
                    boundaries = torch.tensor([0, 62 if step % 2 else 66, length//2+30, length])
                    optimizer.zero_grad(set_to_none=True)
                    torch.npu.synchronize()
                    torch.npu.reset_peak_memory_stats()
                    begin = perf_counter()
                    output = _forward(model, hidden, rank*local, boundaries, context, use_fused=fused)
                    loss = output.float().square().sum()/(length*64)
                    loss.backward()
                    _sum_gradients(model)
                    torch.npu.synchronize()
                    milliseconds = (perf_counter()-begin)*1000
                    current = _collect_step_gradients(model, output, loss)
                    if not fused:
                        reference.append((output.detach().cpu(), hidden.grad.detach().cpu(), current))
                    else:
                        _report_npu_step(rank, step, output, hidden, current, reference[step], milliseconds)
                    optimizer.step()
                if fused:
                    counts = torch.tensor(list(calls.values()), device=device)
                    dist.all_reduce(counts)
                    assert bool((counts > 0).all()), counts
                    actual_weights = torch.cat([value.detach().cpu().flatten() for value in model.parameters()])
                    deviation = _deviation(actual_weights, baseline_weights)
                    assert deviation["finite"]
                    assert any(not torch.equal(value.detach().cpu(), weights[name])
                               for name, value in model.named_parameters())
                    print(json.dumps({"case": "native_fused_cp8_final", "rank": rank, "steps": steps, "length": length,
                                          "native_calls": counts.cpu().tolist(), "weights": deviation}), flush=True)
                else:
                    baseline_weights = torch.cat([value.detach().cpu().flatten() for value in model.parameters()])
            del model, optimizer, output, loss, hidden
            torch.npu.empty_cache()
    finally:
        set_aux_loss_scale(torch.tensor(1.))
        dist.destroy_process_group()
