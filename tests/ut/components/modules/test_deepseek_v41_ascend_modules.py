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
"""CPU contract tests for DeepSeek-V4.1 optional Ascend replacements."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

import hyper_parallel.components.modules.shared_compressed_dsa_attention as dsa_module
import hyper_parallel.components.modules.tile_kernels as tile_kernels_module
from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.components.modules.mhc import PipelinedMhcModule
from hyper_parallel.components.modules.tile_kernels import (
    TileKernelsEngramModule,
    TileKernelsV41MhcModule,
    TileKernelsV41TopKRouter,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41Model,
    DeepseekV41TopKRouter,
)
from hyper_parallel.models._transformers.model_builder import _initialize_model_weights
from hyper_parallel.models.deepseek_v41.adapter.distributed import (
    shared_attention_context_parallel as attention_parallel_module,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)


class _IdentityHash(nn.Module):
    max_ngram_size = 2

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return each token ID as one n-gram hash index."""
        del segment_starts, token_mask
        return input_ids.unsqueeze(-1)


@pytest.mark.parametrize("vision_enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_router_meta_replacement_native_initialization(vision_enabled, dtype, monkeypatch):
    """Initialize replaced meta parameters through the native model lifecycle."""
    config = DeepseekV4Config(
        hidden_size=64,
        num_local_experts=16,
        num_experts_per_tok=2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.25,
        v41_vision_enabled=vision_enabled,
    )
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    model.config = config
    model.vision = None
    with torch.device("meta"):
        model.gate = DeepseekV41TopKRouter(config).to(dtype=dtype)
    plan = compile_module_replacements(
        model,
        [ModuleReplacementSpec(
            match=("gate",),
            factory=TileKernelsV41TopKRouter,
            module_type=DeepseekV41TopKRouter,
        )],
    )
    apply_module_replacements(model, plan)
    model.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    with torch.random.fork_rng():
        torch.manual_seed(42)
        _initialize_model_weights(model)

    reference = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(reference)
    reference.config = config
    reference.vision = None
    reference.gate = DeepseekV41TopKRouter(config).to(dtype=dtype)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        _initialize_model_weights(reference)
    for name, parameter in model.gate.named_parameters():
        torch.testing.assert_close(parameter, reference.gate.get_parameter(name), rtol=0, atol=0)

    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), f"Native initialization missed {name}, dtype={dtype}"
    torch.testing.assert_close(model.gate.bias, torch.zeros_like(model.gate.bias))
    if vision_enabled:
        torch.testing.assert_close(model.gate.bias_vl, torch.zeros_like(model.gate.bias_vl))
    assert model.gate.weight.std().float() > 0, f"Router weight was not randomly initialized, dtype={dtype}"
    monkeypatch.setattr(
        tile_kernels_module, "deepseek_topk_gate",
        lambda scores, top_k: torch.topk(scores, top_k, dim=-1, sorted=False).indices,
    )
    hidden = torch.randn(2, 4, config.hidden_size, dtype=dtype, requires_grad=True)
    logits, weights, _ = model.gate(hidden)
    assert torch.isfinite(logits).all(), f"Non-finite initialized router logits, dtype={dtype}"
    assert torch.isfinite(weights).all(), f"Non-finite initialized routing weights, dtype={dtype}"
    weights.square().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all(), (
        f"Non-finite initialized router input gradient, dtype={dtype}"
    )


@pytest.mark.parametrize("attention_type", [
    dsa_module.SharedCompressedDSAAttention,
    dsa_module.DeepSelectSharedCompressedDSAAttention,
])
def test_shared_attention_meta_sinks_native_initialization(attention_type):
    """Initialize sinks for every training backend after meta materialization."""
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    model.config = DeepseekV4Config()
    model.vision = None
    model.attention = attention_type.__new__(attention_type)
    nn.Module.__init__(model.attention)
    model.attention.sinks = nn.Parameter(torch.empty(4, device="meta"))
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.attention.sinks.fill_(float("nan"))
    _initialize_model_weights(model)
    torch.testing.assert_close(model.attention.sinks, torch.zeros(4))


def _engram_source() -> nn.Module:
    module = nn.Module()
    module.hash_mapping = _IdentityHash()
    module.embed = nn.Embedding(32, 6)
    module.wkv = nn.Linear(6, 12, bias=False)
    module.q_weight = nn.Parameter(torch.randn(2, 4))
    module.k_weight = nn.Parameter(torch.randn(2, 4))
    module.layer_id = 0
    module.hidden_size = 4
    module.hc_mult = 2
    module.eps = 1.0e-6
    module.clamp_value = 1.0e-3
    module.logical_num_embeddings = 32
    module.padded_num_embeddings = 32
    return module


def test_pipelined_mhc_preserves_parameters_and_extended_contract():
    source = nn.Module()
    source.input_norm = nn.RMSNorm(12)
    source.fn = nn.Parameter(torch.randn(15, 12) * 0.1)
    source.base = nn.Parameter(torch.randn(15) * 0.1)
    source.scale = nn.Parameter(torch.ones(3))
    source.hc_mult = 3
    source.hc_sinkhorn_iters = 4
    source.hc_eps = 1.0e-6
    replacement = PipelinedMhcModule(module=source)

    assert replacement.fn is source.fn
    assert replacement.base is source.base
    assert replacement.scale is source.scale
    assert replacement.state_dict().keys() == source.state_dict().keys()

    hidden = torch.randn(2, 5, 3, 4, requires_grad=True)
    incoming_mix = torch.softmax(torch.randn(2, 5, 3), dim=-1).requires_grad_(True)
    collapsed, pre, post, combine = replacement(hidden, incoming_mix)
    expected = (incoming_mix.unsqueeze(-1) * hidden).sum(dim=2)
    torch.testing.assert_close(collapsed, expected)
    assert pre.shape == post.shape == (2, 5, 3)
    assert combine.shape == (2, 5, 3, 3)

    sublayer = torch.randn_like(collapsed, requires_grad=True)
    output = replacement.post(sublayer, hidden, post, combine)
    reference = post.unsqueeze(-1) * sublayer.unsqueeze(-2)
    reference = reference + (combine.unsqueeze(-1) * hidden.unsqueeze(-2)).sum(dim=-3)
    torch.testing.assert_close(output, reference)
    (output.sum() + collapsed.sum()).backward()
    assert hidden.grad is not None
    assert incoming_mix.grad is not None
    assert sublayer.grad is not None


def test_tilekernels_mhc_preserves_parameters_and_rejects_cpu_execution():
    source = nn.Module()
    source.input_norm = nn.RMSNorm(16)
    source.fn = nn.Parameter(torch.randn(24, 16) * 0.1)
    source.base = nn.Parameter(torch.randn(24) * 0.1)
    source.scale = nn.Parameter(torch.ones(3))
    source.hc_mult = 4
    source.hc_sinkhorn_iters = 4
    source.hc_eps = 1.0e-6
    replacement = TileKernelsV41MhcModule(module=source)

    assert replacement.fn is source.fn
    assert replacement.base is source.base
    assert replacement.scale is source.scale
    assert replacement.state_dict().keys() == source.state_dict().keys()

    hidden = torch.randn(1, 2, 4, 4, dtype=torch.bfloat16)
    pre_mix = torch.randn(1, 2, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="requires NPU hidden streams"):
        replacement(hidden, pre_mix)


def test_tilekernels_engram_matches_reference_with_mask(monkeypatch):
    source = _engram_source()
    reference = EngramModule(module=source)
    replacement = TileKernelsEngramModule(module=source)
    assert replacement.state_dict().keys() == reference.state_dict().keys()

    def reference_gate(
        hidden: torch.Tensor,
        key_value: torch.Tensor,
        hidden_weight: torch.Tensor,
        embedding_weight: torch.Tensor,
        clamp: float,
        eps: float,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate the unfused Engram gate formula."""
        key, value = key_value.split([8, 4], dim=-1)
        key = key.float().unflatten(-1, (2, 4))
        hidden_fp32 = hidden.float()
        reciprocal_std = torch.rsqrt(hidden_fp32.square().mean(-1) + eps)
        reciprocal_std = reciprocal_std * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (
            hidden_fp32 * hidden_weight.float() * embedding_weight.float() * key
        ).sum(-1)
        dot *= reciprocal_std * 0.5
        signed_root = torch.where(
            dot < 0,
            -dot.abs().clamp_min(clamp).sqrt(),
            dot.abs().clamp_min(clamp).sqrt(),
        )
        output = hidden_fp32 + signed_root.sigmoid().unsqueeze(
            -1
        ) * value.float().unsqueeze(-2)
        output = output.to(hidden.dtype)
        if mask is not None:
            output = torch.where(mask.reshape(*hidden.shape[:-2], 1, 1), hidden, output)
        return output

    monkeypatch.setattr(tile_kernels_module, "deepseek_engram_gate", reference_gate)
    hidden = torch.randn(2, 5, 2, 4, requires_grad=True)
    input_ids = torch.randint(0, 32, (2, 5))
    token_mask = torch.tensor(
        [[True, False, True, True, False], [True, True, False, True, True]]
    )
    expected = reference(hidden, input_ids, token_mask=token_mask)
    actual = replacement(hidden, input_ids, token_mask=token_mask)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert hidden.grad is not None
    assert source.wkv.weight.grad is not None


def test_tilekernels_router_preserves_reference_semantics(monkeypatch):
    source = nn.Module()
    source.hidden_size = 8
    source.num_experts = 6
    source.top_k = 2
    source.scoring_func = "sqrtsoftplus"
    source.routed_scaling_factor = 1.25
    source.weight = nn.Parameter(torch.randn(6, 8))
    source.bias = nn.Parameter(torch.randn(6))
    source.bias_vl = nn.Parameter(torch.randn(6))
    replacement = TileKernelsV41TopKRouter(module=source)
    assert replacement.state_dict().keys() == source.state_dict().keys()

    monkeypatch.setattr(
        tile_kernels_module,
        "deepseek_topk_gate",
        lambda scores, top_k: torch.topk(scores, top_k, dim=-1, sorted=False).indices,
    )
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    image_mask = torch.tensor([[False, True, False, True], [True, False, False, True]])
    logits, weights, indices = replacement(hidden, image_mask)
    scores = torch.nn.functional.softplus(logits).sqrt()  # pylint: disable=not-callable
    correction = torch.where(
        image_mask.reshape(-1, 1), source.bias_vl.unsqueeze(0), source.bias.unsqueeze(0)
    )
    expected_indices = torch.topk(scores + correction, 2, dim=-1, sorted=False).indices
    expected_weights = scores.gather(1, expected_indices)
    expected_weights = expected_weights / (
        expected_weights.sum(-1, keepdim=True) + 1.0e-20
    )
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights * 1.25)
    weights.sum().backward()
    assert hidden.grad is not None


@pytest.mark.parametrize("vision_enabled", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_tilekernels_router_preserves_registered_parameters_in_replacement(vision_enabled, device):
    """Preserve optional parameter registration through the real replacement executor."""
    config = SimpleNamespace(
        hidden_size=8,
        num_local_experts=6,
        num_experts_per_tok=2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.25,
        v41_vision_enabled=vision_enabled,
    )
    with torch.device(device):
        source = DeepseekV41TopKRouter(config).eval()
    model = nn.Module()
    model.gate = source
    plan = compile_module_replacements(
        model,
        [ModuleReplacementSpec(
            match=("gate",),
            factory=TileKernelsV41TopKRouter,
            module_type=DeepseekV41TopKRouter,
        )],
    )

    apply_module_replacements(model, plan)

    replacement = model.gate
    assert tuple(replacement._parameters) == tuple(source._parameters), (
        f"Parameter registry changed: source={tuple(source._parameters)}, "
        f"replacement={tuple(replacement._parameters)}"
    )
    assert replacement.bias_vl is source.bias_vl, (
        f"bias_vl identity changed: source={id(source.bias_vl)}, replacement={id(replacement.bias_vl)}"
    )
    assert replacement.weight is source.weight and replacement.bias is source.bias, (
        f"Router parameter identity changed for device={device}, vision_enabled={vision_enabled}"
    )
    assert replacement.state_dict().keys() == source.state_dict().keys(), (
        f"Checkpoint keys changed: source={tuple(source.state_dict())}, "
        f"replacement={tuple(replacement.state_dict())}"
    )
    assert replacement.training == source.training, (
        f"Training state changed: source={source.training}, replacement={replacement.training}"
    )


@pytest.mark.parametrize("selection_mode", ["full", "source", "candidate"])
def test_deepselect_selection_exclusion_preserves_backend(monkeypatch, selection_mode):
    """Keep the backend choice inside the upstream recompute exclusion region."""
    source = nn.Module()
    source.q_b_proj = nn.Linear(8, 16)
    source.weights_proj = nn.Linear(8, 2)
    for name, value in {
        "compress_ratio": 2, "num_heads": 2, "head_dim": 8, "index_topk": 2,
        "owns_key": False, "is_candidate_source": selection_mode == "source",
        "uses_candidates": selection_mode == "candidate", "candidate_topk_blocks": 1,
        "candidate_block_size": 2, "loss_coeff": 0.0,
    }.items():
        setattr(source, name, value)
    indexer = dsa_module.SharedCompressedDSAIndexer(source, use_deepselect=True)
    indices = torch.zeros(1, 4, 2, dtype=torch.int32)
    blocks = torch.zeros(1, 4, 1, dtype=torch.int32)
    calls = []

    def select(*args, **kwargs):
        """Record the backend flag at the actual selection dispatch boundary."""
        del args
        calls.append(kwargs["use_deepselect"])
        return (indices, blocks) if selection_mode == "source" else indices

    operator = {
        "full": "compressed_causal_topk",
        "source": "compressed_causal_topk_and_candidates",
        "candidate": "compressed_candidate_topk",
    }[selection_mode]
    monkeypatch.setattr(dsa_module, operator, select)
    query, key, weights = torch.randn(1, 4, 2, 8), torch.randn(1, 2, 8), torch.randn(1, 4, 2)
    kwargs = dict(candidate_blocks=blocks, query_offset=0, minimum_key_indices=None, reduce_sum=None)
    direct = indexer._run_selection(query, key, weights, **kwargs)
    keys_before = tuple(indexer.state_dict())
    indexer.enable_selection_exclude()
    excluded = indexer._run_selection(query, key, weights, **kwargs)
    assert calls == [True, True], f"DeepSelect dispatch changed after exclusion: {calls}"
    assert tuple(indexer.state_dict()) == keys_before, "Exclusion changed checkpoint names"
    torch.testing.assert_close(excluded[0], direct[0])


def test_deepselect_input_alignment(monkeypatch):
    monkeypatch.setattr(
        dsa_module,
        "deepseek_select_stride_requirement",
        lambda: (32, 32),
    )
    scores = torch.arange(42, dtype=torch.float32).reshape(2, 3, 7)
    padded, end = dsa_module._prepare_deepselect_input(scores)
    assert padded.dtype == torch.bfloat16
    assert padded.shape == (6, 16)
    assert padded.stride() == (16, 1)
    torch.testing.assert_close(padded[:, :7].float(), scores.reshape(6, 7))
    assert torch.isneginf(padded[:, 7:]).all()
    torch.testing.assert_close(end, torch.full((6,), 7, dtype=torch.int32))

    aligned, aligned_end = dsa_module._prepare_deepselect_input(torch.randn(2, 3, 16))
    assert aligned.shape == (6, 16)
    assert aligned_end is None


def test_deepselect_attention_enables_training_backend(monkeypatch):
    captured = {}

    def fake_init(self: nn.Module, module: Any, **kwargs: Any) -> None:
        """Capture replacement constructor flags without building attention."""
        captured.update(kwargs)
        nn.Module.__init__(self)
        self.source = module

    monkeypatch.setattr(
        dsa_module.SharedCompressedDSAAttentionBase, "__init__", fake_init
    )
    source = SimpleNamespace()
    replacement = dsa_module.DeepSelectSharedCompressedDSAAttention(source)
    assert replacement.source is source
    assert captured == {
        "replace_indexer": True,
        "use_optimized_sparse_attention": True,
        "use_deepselect_indexer": True,
    }


@pytest.mark.parametrize("use_deepselect", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_shared_attention_wrapper_preserves_backend_and_contexts(monkeypatch, use_deepselect, parallel):
    """Allow both training backends without changing wrapper-owned parallel contexts."""
    captured = {}

    def fake_init(self: nn.Module, module: Any, **kwargs: Any) -> None:
        """Initialize module bookkeeping without model weights or hardware."""
        del module, kwargs
        nn.Module.__init__(self)

    def forward(self: nn.Module, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Capture the wrapper's arguments and preserve the original tensor."""
        del self
        captured.update(kwargs)
        return hidden_states

    monkeypatch.setattr(dsa_module.SharedCompressedDSAAttentionBase, "__init__", fake_init)
    attention_type = (
        dsa_module.DeepSelectSharedCompressedDSAAttention
        if use_deepselect else dsa_module.SharedCompressedDSAAttention
    )
    monkeypatch.setattr(attention_type, "forward", forward)
    target = attention_type(nn.Module())
    cp_context, tp_context = object(), object()
    monkeypatch.setattr(attention_parallel_module, "_build_shared_attention_cp_context", lambda mesh: cp_context)
    monkeypatch.setattr(attention_parallel_module, "_build_shared_attention_tp_context", lambda mesh: tp_context)
    mesh = SimpleNamespace(size=lambda: 2) if parallel else None

    request = attention_parallel_module.deepseek_v41_shared_attention_parallel_wrapper(
        target, None, mesh, mesh, None,
    )
    hidden = torch.ones(1, 2, 4)
    result = request.forward(hidden)

    assert request.target is target and result is hidden, (
        f"Wrapper changed target or output identity: backend={attention_type.__name__}, parallel={parallel}"
    )
    expected = {
        "shared_attention_cp_context": cp_context,
        "shared_attention_tp_context": tp_context,
    } if parallel else {}
    assert captured == expected, f"Parallel contexts differ: expected={expected}, got={captured}"
    with pytest.raises(ValueError, match="contexts are owned by the wrapper"):
        request.forward(hidden, shared_attention_cp_context=cp_context)


def test_shared_attention_wrapper_rejects_unrelated_module():
    """Keep the type guard for modules without the shared-attention contract."""
    with pytest.raises(TypeError, match="requires SharedCompressedDSAAttention"):
        attention_parallel_module.deepseek_v41_shared_attention_parallel_wrapper(
            nn.Linear(4, 4), None, None, None, None,
        )


def test_flashmla_attention_does_not_require_deepselect(monkeypatch):
    captured = {}

    def fake_init(self: nn.Module, module: Any, **kwargs: Any) -> None:
        """Capture FlashMLA replacement flags without building attention."""
        captured.update(kwargs)
        nn.Module.__init__(self)
        self.source = module
        self.scaling = 0.125
        self.head_dim = 64

    sentinel = nn.Identity()
    monkeypatch.setattr(
        dsa_module.SharedCompressedDSAAttentionBase,
        "__init__",
        fake_init,
    )
    monkeypatch.setattr(
        dsa_module,
        "DeepseekV41SparsePrefillAttention",
        lambda *args, **kwargs: sentinel,
    )
    replacement = dsa_module.FlashMLASharedCompressedDSAAttention(SimpleNamespace())
    assert replacement.flash_mla is sentinel
    assert captured == {
        "replace_indexer": True,
        "use_optimized_sparse_attention": False,
        "use_deepselect_indexer": False,
    }

    replacement.sinks = nn.Parameter(torch.zeros(2))
    query = torch.randn(1, 2, 3, 64)
    key_value = torch.randn(1, 1, 3, 64)
    sparse_indices = torch.tensor([[[0, -1], [0, 1], [1, 2]]])

    with pytest.raises(RuntimeError, match="does not provide an Ascend training backward"):
        replacement._run_sparse_attention(
            query,
            key_value,
            sparse_indices,
            query.device,
        )

    replacement.eval()
    with torch.no_grad():
        output = replacement._run_sparse_attention(
            query,
            key_value,
            sparse_indices,
            query.device,
        )
        expected = dsa_module._reference_sparse_attention(
            query,
            key_value,
            sparse_indices,
            replacement.sinks,
            replacement.scaling,
        )
    torch.testing.assert_close(output, expected)
