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
from importlib import import_module
from unittest.mock import patch
from typing import Any

import torch
import torch.distributed as dist
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.utils import is_torchvision_available

from hyper_parallel.components.functional import compressed_indexer_ops, compressed_smla
from hyper_parallel.components.functional.aux_loss import set_aux_loss_scale
from hyper_parallel.components.modules import shared_compressed_dsa_attention as shared
from hyper_parallel.models.deepseek_v41.adapter.distributed.shared_attention_context_parallel import (
    _build_shared_attention_cp_context,
)
from tests.torch.context_parallel._test_v41_native_ops import _device


class _CPMesh:
    """Minimal mesh interface backed by the initialized world group."""

    @staticmethod
    def size() -> int:
        """Return the CP world size."""
        return dist.get_world_size()

    @staticmethod
    def get_local_rank() -> int:
        """Return this process's CP rank."""
        return dist.get_rank()

    @staticmethod
    def get_group() -> Any:
        """Return the raw process group expected by the collective API."""
        return dist.group.WORLD


def _config() -> DeepseekV4Config:
    """Create a small V4.1 attention configuration."""
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=16,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        max_position_embeddings=128,
        layer_types=["sliding_attention"] * 4,
        mlp_layer_types=["moe"] * 4,
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 2},
        compress_rope_theta=10000.0,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1.0e-6,
        use_cache=False,
        partial_rotary_factor=0.5,
    )
    config.v41_compress_ratios = [0, 0, 2, 2]
    config.v41_kv_source_layer_ids = [2]
    config.v41_index_source_layer_ids = [2]
    return config


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
    attention_type = import_module("hyper_parallel.models.deepseek_v41.modeling_deepseek_v41").DeepseekV41Attention
    modules = torch.nn.ModuleList([attention_type(config, layer) for layer in range(6)])
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
        module.use_fused_ops = use_fused
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


# Transformers imports optional NPU/vision extensions with the text model.
# This CPU-only test needs neither extension; gate images can contain wheels
# without their native libraries. Keep the real model and CP math enabled.
@patch.dict("sys.modules", {"torch_npu": None, "torchvision": None})
def test_native_geometry_cp8_gloo():
    """Compare full and CP8 outputs, every gradient and two SGD updates in FP32."""
    # Transformers caches this probe before the scoped module isolation above.
    is_torchvision_available.cache_clear()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    try:
        rank, size, length = dist.get_rank(), dist.get_world_size(), 128
        assert size == 8
        local = length//size
        reference = _chain()
        candidate = torch.nn.ModuleList([
            shared.SharedCompressedDSAAttention(module, use_fused_ops=False) for module in deepcopy(reference)
        ])
        context = _build_shared_attention_cp_context(_CPMesh())
        # Keep this CPU oracle independent of accelerator optimizer dispatch hooks.
        optimizers = [torch.optim.SGD(model.parameters(), lr=.01, foreach=False)
                      for model in (reference, candidate)]
        for step in range(2):
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
    finally:
        set_aux_loss_scale(torch.tensor(1.))
        dist.destroy_process_group()


def _count_native_dispatch(stack):
    """Track calls without a Mock retaining accelerator activation tensors."""
    calls = {}
    entries = ((compressed_indexer_ops, "_load_indexer_op"),
               (compressed_smla, "_load_attention_ops"), (compressed_indexer_ops, "_load_kl_ops"))
    for module, name in entries:
        loaded = getattr(module, name)()
        native = loaded[0] if isinstance(loaded, tuple) else loaded
        calls[name] = 0

        def _counted(*args, _name=name, _native=native, **kwargs):
            """Count dispatch without keeping tensor arguments alive."""
            calls[_name] += 1
            return _native(*args, **kwargs)

        replacement = (_counted, *loaded[1:]) if isinstance(loaded, tuple) else _counted
        stack.enter_context(patch.object(module, name, return_value=replacement))
    return calls


def test_native_cp8_training_smoke():
    """Exercise all fused entries and finite parameter updates across two packed CP8 steps.

    Single-device tests check numerical precision independently; this integration
    smoke checks dispatch and training with the unchanged shared CP collectives.
    """
    torch.set_num_threads(1)
    device = _device()
    dist.init_process_group("hccl", timeout=timedelta(minutes=5))
    try:
        rank, size, length = dist.get_rank(), dist.get_world_size(), 256
        assert size == 8
        # Initialize communication before optional native metadata services.
        dist.all_reduce(torch.ones(1, device=device))
        modules = _chain(native=True)
        initial = deepcopy(modules.state_dict())
        model = torch.nn.ModuleList([
            shared.SharedCompressedDSAAttention(module, use_fused_ops=True) for module in modules
        ]).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        context = _build_shared_attention_cp_context(_CPMesh())
        local = length//size
        set_aux_loss_scale(torch.tensor(1./size, device=device))
        with ExitStack() as stack:
            calls = _count_native_dispatch(stack)
            for step in range(2):
                torch.manual_seed(200+step)
                source = torch.randn(1, length, 64).bfloat16()
                hidden = source[:, rank*local:(rank+1)*local].to(device).requires_grad_()
                boundaries = torch.tensor([0, 62 if step % 2 else 66, length//2+30, length])
                optimizer.zero_grad(set_to_none=True)
                output = _forward(model, hidden, rank*local, boundaries, context, use_fused=True)
                loss = output.float().square().sum()/(length*64)
                loss.backward()
                _sum_gradients(model)
                assert bool(output.isfinite().all()) and bool(loss.isfinite())
                assert bool(hidden.grad.isfinite().all())
                gradients = [value.grad for value in model.parameters() if value.grad is not None]
                assert gradients and all(grad.dtype == torch.float32 and bool(grad.isfinite().all())
                                         for grad in gradients)
                optimizer.step()
            counts = torch.tensor(list(calls.values()), device=device)
            dist.all_reduce(counts)
            assert bool((counts > 0).all()), counts
        assert all(bool(value.isfinite().all()) for value in model.parameters())
        assert any(not torch.equal(value.detach().cpu(), initial[name]) for name, value in model.named_parameters())
    finally:
        set_aux_loss_scale(torch.tensor(1.))
        dist.destroy_process_group()
