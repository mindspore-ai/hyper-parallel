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
"""Compare cropped V4.1 Engram, MoE, and CSA2 modules with the native repo.

The released inference repository imports CUDA/TileLang kernels at module load.
This validator keeps the released Python module composition and replaces only
those kernel entry points with independent eager PyTorch references. The same
weights and inputs are then executed by the native module on CPU and by the
HyperParallel module on CPU or Ascend.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as functional  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts, DeepseekV4MLP

from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedAttentionState,
    SharedCompressedDSAAttention,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41AttentionPlaceholder,
    DeepseekV41EngramPlaceholder,
    DeepseekV41TopKRouter,
    _v41_sparse_moe_forward,
)


DEFAULT_NATIVE_REPO = Path("/home/ma-user/work/y00512198/DeepSeek-V4.1-Flash")


@dataclass(frozen=True)
class Tolerance:
    """Absolute and relative tolerances for one validation precision."""

    atol: float
    rtol: float


class HyperParallelMoE(nn.Module):
    """Minimal owner that invokes the adapter's real sparse-MoE forward."""

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.gate = DeepseekV41TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)
        self.forward = MethodType(_v41_sparse_moe_forward, self)


def _eager_sparse_attention(
        query: torch.Tensor,
        key_value: torch.Tensor,
        sinks: torch.Tensor,
        sparse_indices: torch.Tensor,
        scale: float,
) -> torch.Tensor:
    """Independent eager equivalent of the native selected-token kernel."""
    batch_size, sequence_length, _, _ = query.shape
    key_length = key_value.shape[1]
    valid = sparse_indices >= 0
    safe = sparse_indices.clamp(min=0, max=max(key_length - 1, 0)).long()
    expanded = key_value.unsqueeze(1).expand(-1, sequence_length, -1, -1)
    selected = expanded.gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, key_value.shape[-1]))
    logits = torch.einsum("bshd,bskd->bshk", query.float(), selected.float()) * scale
    logits.masked_fill_(~valid.unsqueeze(2), float("-inf"))
    sink_logits = sinks.float().view(1, 1, -1, 1).expand(batch_size, sequence_length, -1, -1)
    probabilities = torch.cat((logits, sink_logits), dim=-1).softmax(dim=-1)
    output = torch.einsum("bshk,bskd->bshd", probabilities[..., :-1], selected.float())
    return output.to(query.dtype)


def _identity_quant(value: torch.Tensor, *_args: Any, **_kwargs: Any) -> torch.Tensor:
    """Leave the native in-place activation quantization calls unchanged."""
    return value


def _unsupported_kernel(*_args: Any, **_kwargs: Any) -> torch.Tensor:
    """Reject quantized GEMMs that are outside the requested high-precision route."""
    raise NotImplementedError("the parity validator does not emulate native FP8/FP4 GEMMs")


def _load_native_modules(native_repo: Path) -> tuple[Any, Any]:
    """Load the released Python implementation with high-precision kernel stubs."""
    inference_dir = native_repo / "inference"
    model_path = inference_dir / "model.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"native model.py is absent: {model_path}")

    kernel = types.ModuleType("kernel")
    kernel.act_quant = _identity_quant
    kernel.fp4_act_quant = _identity_quant
    kernel.fp4_gemm = _unsupported_kernel
    kernel.fp8_gemm = _unsupported_kernel
    kernel.hc_split_sinkhorn = _unsupported_kernel
    kernel.sparse_attn = _eager_sparse_attention
    sys.modules["kernel"] = kernel
    sys.path.insert(0, str(inference_dir))
    try:
        model_spec = importlib.util.spec_from_file_location("deepseek_v41_native_model", model_path)
        if model_spec is None or model_spec.loader is None:
            raise ImportError(f"cannot create an import spec for {model_path}")
        native_model = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(native_model)
        native_engram = sys.modules["engram"]
    finally:
        sys.path.remove(str(inference_dir))
    native_model.world_size = 1
    native_model.rank = 0
    native_model.default_dtype = torch.float32
    return native_model, native_engram


def _device_from_args(name: str, index: int) -> torch.device:
    """Resolve CPU or initialize the requested Ascend device."""
    if name == "cpu":
        return torch.device("cpu")
    import torch_npu  # noqa: F401  # pylint: disable=C0415,unused-import

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    torch.npu.set_device(index)
    return torch.device(f"npu:{index}")


def _synchronize(device: torch.device) -> None:
    """Synchronize accelerator work before timing or reading results."""
    if device.type == "npu":
        torch.npu.synchronize(device)


def _cast_parameters(
        module: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        keep_fp32: tuple[str, ...] = (),
) -> nn.Module:
    """Move parameters while retaining explicitly FP32 control parameters."""
    module.to(device=device)
    for name, parameter in module.named_parameters():
        target_dtype = torch.float32 if any(name.endswith(suffix) for suffix in keep_fp32) else dtype
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(device=device, dtype=target_dtype)
    return module


def _fill_parameters(module: nn.Module, seed: int) -> None:
    """Initialize non-quantized parameters deterministically away from Top-K ties."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not parameter.is_floating_point():
                continue
            value = torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.08
            if name.endswith("norm.weight") or name.endswith("_weight"):
                value = value * 0.05 + 1.0
            parameter.copy_(value.to(parameter.dtype))


def _copy_parameter(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy one mapped parameter with a shape assertion."""
    if target.shape != source.shape:
        raise ValueError(f"mapped parameter shape mismatch: {tuple(target.shape)} versus {tuple(source.shape)}")
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    """Return stable scalar error metrics on CPU FP32 values."""
    reference_fp32 = reference.detach().cpu().float()
    actual_fp32 = actual.detach().cpu().float()
    difference = (reference_fp32 - actual_fp32).abs()
    denominator = reference_fp32.abs().clamp_min(1.0e-8)
    flat_reference = reference_fp32.flatten()
    flat_actual = actual_fp32.flatten()
    cosine = 1.0
    if flat_reference.numel() and flat_reference.norm() and flat_actual.norm():
        cosine = float(functional.cosine_similarity(flat_reference, flat_actual, dim=0))
    return {
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
        "max_rel": float((difference / denominator).max()) if difference.numel() else 0.0,
        "relative_l2": float(difference.norm() / flat_reference.norm().clamp_min(1.0e-8)),
        "cosine": cosine,
    }


def _compare(
        cases: list[dict[str, Any]],
        name: str,
        reference: torch.Tensor,
        actual: torch.Tensor,
        tolerance: Tolerance,
        elapsed_ms: float,
) -> None:
    """Record one exact or tolerance-based tensor comparison."""
    integer = not reference.is_floating_point()
    metric_values = None if integer else _metrics(reference, actual)
    passed = torch.equal(reference.detach().cpu(), actual.detach().cpu()) if integer else (
        torch.allclose(
            reference.detach().cpu().float(),
            actual.detach().cpu().float(),
            atol=tolerance.atol,
            rtol=tolerance.rtol,
        )
        and metric_values["relative_l2"] <= 1.0e-2
        and metric_values["cosine"] >= 0.9999
    )
    result: dict[str, Any] = {
        "name": name,
        "status": "pass" if passed else "fail",
        "shape": list(reference.shape),
        "dtype": str(actual.dtype),
        "elapsed_ms": elapsed_ms,
    }
    if not integer:
        result["metrics"] = metric_values
        result["tolerance"] = {"atol": tolerance.atol, "rtol": tolerance.rtol}
        result["shape_acceptance"] = {"max_relative_l2": 1.0e-2, "min_cosine": 0.9999}
    cases.append(result)


def _timed_forward(device: torch.device, function: Any) -> tuple[Any, float]:
    """Run one callable and return synchronized wall time."""
    _synchronize(device)
    start = time.perf_counter()
    output = function()
    _synchronize(device)
    return output, (time.perf_counter() - start) * 1000.0


def _engram_assets() -> dict[str, Any]:
    """Build a compact synchronized Engram hash/table layout."""
    return {
        "layer_ids": [0],
        "max_ngram_size": 3,
        "num_heads": 2,
        "head_dim": 4,
        "primes": [[[17, 19], [23, 29]]],
        "num_embeddings": [88],
        "multipliers": [[101, 103, 107]],
        "token_map": list(range(32)),
        "pad_token_id": 0,
    }


def _native_hash_state(native_engram: Any, assets: dict[str, Any]) -> nn.Module:
    """Instantiate the native state with prepared assets, bypassing a tokenizer download."""
    primes = tuple(tuple(tuple(row) for row in layer) for layer in assets["primes"])
    layout = native_engram.EngramLayout(
        max_ngram_size=assets["max_ngram_size"],
        layer_ids=tuple(assets["layer_ids"]),
        num_embeddings=tuple(assets["num_embeddings"]),
        primes=primes,
        n_heads=assets["num_heads"],
        head_dim=assets["head_dim"],
    )
    state = native_engram.NgramHashState.__new__(native_engram.NgramHashState)
    nn.Module.__init__(state)
    state.layout = layout
    state.pad_id = assets["token_map"][assets["pad_token_id"]]
    flattened = [value for row in assets["primes"][0] for value in row]
    offsets = np.cumsum([0, *flattened[:-1]]).reshape(1, -1)
    state.register_buffer("primes", torch.tensor(assets["primes"], dtype=torch.long), persistent=False)
    state.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
    state.register_buffer("multipliers", torch.tensor(assets["multipliers"], dtype=torch.long), persistent=False)
    state.register_buffer("token_map", torch.tensor(assets["token_map"], dtype=torch.long), persistent=False)
    state.register_buffer("cache", torch.empty(2, 16, dtype=torch.long), persistent=False)
    return state


def _run_engram(
        native_model: Any,
        native_engram: Any,
        target_device: torch.device,
        dtype: torch.dtype,
        tolerance: Tolerance,
) -> list[dict[str, Any]]:
    """Compare native and HyperParallel Engram hash, fusion, and input gradient."""
    cases: list[dict[str, Any]] = []
    assets = _engram_assets()
    hash_state = _native_hash_state(native_engram, assets)
    layout = hash_state.layout
    native_args = native_model.ModelArgs(
        dtype="bf16",
        expert_dtype=None,
        dim=16,
        n_layers=1,
        n_mtp_layers=0,
        hc_mult=2,
        norm_eps=1.0e-6,
    )
    native = native_model.Engram(native_args, 0, layout)
    native.embed = nn.Embedding(assets["num_embeddings"][0], assets["head_dim"])
    hp_config = SimpleNamespace(
        hidden_size=16,
        hc_mult=2,
        rms_norm_eps=1.0e-6,
        v41_engram_table_pad_multiple=1,
    )
    hp = EngramModule(module=DeepseekV41EngramPlaceholder(hp_config, 0, assets))
    _cast_parameters(native, torch.device("cpu"), dtype, ("q_weight", "k_weight"))
    _fill_parameters(native, 101)
    _copy_parameter(hp.embed.weight, native.embed.weight)
    _copy_parameter(hp.wkv.weight, native.wkv.weight)
    _copy_parameter(hp.q_weight, native.q_weight)
    _copy_parameter(hp.k_weight, native.k_weight)
    _cast_parameters(hp, target_device, dtype, ("q_weight", "k_weight"))

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    token_mask = torch.tensor(
        [[True, True, False, True, True, True], [True, True, True, True, False, True]]
    )
    # Native hash state intentionally runs under inference_mode; clone after it
    # returns so embedding backward receives an ordinary tensor.
    native_hash = hash_state(input_ids, 0, token_mask)[:, :, 0].clone()
    hp_hash, hash_ms = _timed_forward(
        target_device,
        lambda: hp.hash_mapping(input_ids.to(target_device), token_mask=token_mask.to(target_device)),
    )
    _compare(cases, "engram.hash_ids", native_hash, hp_hash, tolerance, hash_ms)

    generator = torch.Generator(device="cpu").manual_seed(102)
    source = torch.randn(2, 6, 2, 16, generator=generator, dtype=torch.float32) * 0.2
    native_input = source.to(dtype).requires_grad_(True)
    hp_input = source.to(device=target_device, dtype=dtype).requires_grad_(True)
    native_output = native(native_input, native_hash, token_mask)
    hp_output, output_ms = _timed_forward(
        target_device,
        lambda: hp(hp_input, input_ids.to(target_device), token_mask=token_mask.to(target_device)),
    )
    _compare(cases, "engram.output", native_output, hp_output, tolerance, output_ms)
    native_output.float().sum().backward()
    hp_output.float().sum().backward()
    _synchronize(target_device)
    _compare(cases, "engram.input_grad", native_input.grad, hp_input.grad, tolerance, 0.0)
    return cases


def _moe_config() -> DeepseekV4Config:
    """Build the small released-shape-invariant MoE configuration."""
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.25,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
        swiglu_limit=1.5,
        sliding_window=4,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        partial_rotary_factor=0.5,
    )
    config.v41_vision_enabled = True
    return config


def _copy_moe_weights(native: nn.Module, hp: HyperParallelMoE) -> None:
    """Map released w1/w3/w2 experts into the HF fused expert container."""
    _copy_parameter(hp.gate.weight, native.gate.weight)
    _copy_parameter(hp.gate.bias, native.gate.bias)
    _copy_parameter(hp.gate.bias_vl, native.gate.bias_vl)
    for expert_index, expert in enumerate(native.experts):
        intermediate = expert.w1.weight.shape[0]
        _copy_parameter(hp.experts.gate_up_proj[expert_index, :intermediate], expert.w1.weight)
        _copy_parameter(hp.experts.gate_up_proj[expert_index, intermediate:], expert.w3.weight)
        _copy_parameter(hp.experts.down_proj[expert_index], expert.w2.weight)
    _copy_parameter(hp.shared_experts.gate_proj.weight, native.shared_experts.w1.weight)
    _copy_parameter(hp.shared_experts.up_proj.weight, native.shared_experts.w3.weight)
    _copy_parameter(hp.shared_experts.down_proj.weight, native.shared_experts.w2.weight)


def _sort_routes(indices: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize Top-K order without changing expert/weight association."""
    # Ascend ArgSort sends integer inputs to AiCPU; expert ids are exactly
    # representable as FP32 and this is validation-only canonicalization.
    order = indices.float().argsort(dim=-1)
    return indices.gather(-1, order), weights.gather(-1, order)


def _run_moe(
        native_model: Any,
        target_device: torch.device,
        dtype: torch.dtype,
        tolerance: Tolerance,
) -> list[dict[str, Any]]:
    """Compare text/VL routing, expert output, and the input gradient."""
    cases: list[dict[str, Any]] = []
    args = native_model.ModelArgs(
        dtype="bf16",
        expert_dtype=None,
        dim=16,
        moe_inter_dim=24,
        n_layers=1,
        n_mtp_layers=0,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        score_func="sqrtsoftplus",
        norm_topk_prob=True,
        route_scale=1.25,
        swiglu_limit=1.5,
        vision_n_layers=1,
    )
    native = native_model.MoE(0, args)
    hp = HyperParallelMoE(_moe_config())
    _cast_parameters(native, torch.device("cpu"), dtype, ("bias", "bias_vl"))
    _fill_parameters(native, 201)
    _copy_moe_weights(native, hp)
    _cast_parameters(hp, target_device, dtype, ("bias", "bias_vl"))

    generator = torch.Generator(device="cpu").manual_seed(202)
    source = torch.randn(2, 5, 16, generator=generator, dtype=torch.float32) * 0.3
    image_mask = torch.tensor(
        [[False, False, True, True, False], [False, True, True, False, False]]
    )
    native_input = source.to(dtype).requires_grad_(True)
    hp_input = source.to(device=target_device, dtype=dtype).requires_grad_(True)
    native_weights, native_indices = native.gate(native_input.flatten(0, 1), image_mask.flatten())
    hp_logits, hp_weights, hp_indices = hp.gate(hp_input, image_mask=image_mask.to(target_device))
    native_logits = functional.linear(native_input.flatten(0, 1).float(), native.gate.weight.float())
    native_scores = functional.softplus(native_logits).sqrt()
    hp_scores = functional.softplus(hp_logits).sqrt()
    native_score_grad = torch.autograd.grad(native_scores.sum(), native_logits, retain_graph=True)[0]
    hp_score_grad = torch.autograd.grad(hp_scores.sum(), hp_logits, retain_graph=True)[0]
    if hp_scores.device != target_device or hp_score_grad.device != target_device:
        raise RuntimeError("sqrt(softplus) output or gradient unexpectedly left the target device")
    _compare(cases, "moe.router_logits", native_logits, hp_logits, tolerance, 0.0)
    _compare(cases, "moe.sqrt_softplus_scores", native_scores, hp_scores, tolerance, 0.0)
    _compare(cases, "moe.sqrt_softplus_logit_grad", native_score_grad, hp_score_grad, tolerance, 0.0)
    native_indices, native_weights = _sort_routes(native_indices, native_weights)
    hp_indices, hp_weights = _sort_routes(hp_indices, hp_weights)
    _compare(cases, "moe.router_indices", native_indices, hp_indices, tolerance, 0.0)
    _compare(cases, "moe.router_weights", native_weights, hp_weights, tolerance, 0.0)

    native_output = native(native_input, image_mask)
    hp_output, output_ms = _timed_forward(
        target_device,
        lambda: hp(hp_input, image_mask=image_mask.to(target_device)),
    )
    _compare(cases, "moe.output", native_output, hp_output, tolerance, output_ms)
    native_output.float().sum().backward()
    hp_output.float().sum().backward()
    _synchronize(target_device)
    _compare(cases, "moe.input_grad", native_input.grad, hp_input.grad, tolerance, 0.0)
    return cases


def _attention_config(
        ratios: tuple[int, ...],
        kv_sources: tuple[int, ...],
        index_sources: tuple[int, ...],
        candidate_source: int,
) -> DeepseekV4Config:
    """Build a shape-compatible V4.1 attention-only configuration."""
    layer_count = len(ratios)
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=layer_count,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=512,
        q_lora_rank=128,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        max_position_embeddings=16,
        rope_theta=10000.0,
        layer_types=["sliding_attention"] * layer_count,
        mlp_layer_types=["moe"] * layer_count,
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 2},
        compress_rope_theta=10000.0,
        sliding_window=4,
        o_groups=8,
        o_lora_rank=128,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=2,
        rms_norm_eps=1.0e-6,
        partial_rotary_factor=0.125,
        attention_dropout=0.0,
    )
    config.v41_compress_ratios = list(ratios)
    config.v41_kv_source_layer_ids = list(kv_sources)
    config.v41_index_source_layer_ids = list(index_sources)
    config.v41_candidate_source_layer_id = candidate_source
    config.v41_candidate_topk_blocks = 1 if candidate_source >= 0 else 0
    config.v41_candidate_block_size = 2
    config.v41_indexer_loss_coeff = 0.0
    return config


def _native_attention_args(
        native_model: Any,
        ratios: tuple[int, ...],
        kv_sources: tuple[int, ...],
        index_sources: tuple[int, ...],
        candidate_source: int,
) -> Any:
    """Build the equivalent native attention configuration."""
    return native_model.ModelArgs(
        max_batch_size=2,
        max_seq_len=16,
        dtype="bf16",
        expert_dtype=None,
        dim=512,
        n_layers=len(ratios),
        n_mtp_layers=0,
        n_heads=8,
        q_lora_rank=128,
        head_dim=512,
        rope_head_dim=64,
        norm_eps=1.0e-6,
        o_groups=8,
        o_lora_rank=128,
        window_size=4,
        compress_ratios=ratios,
        kv_source_layers=kv_sources,
        index_source_layers=index_sources,
        compress_rope_theta=10000.0,
        original_seq_len=0,
        rope_theta=10000.0,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=2,
        candidate_source_layer=candidate_source,
        candidate_topk_blocks=1 if candidate_source >= 0 else 0,
        candidate_block_size=2,
    )


def _copy_attention_weights(native: nn.Module, hp: SharedCompressedDSAAttention) -> None:
    """Apply the released checkpoint-name mapping to one attention layer."""
    pairs = (
        (hp.sinks, native.attn_sink),
        (hp.q_a_proj.weight, native.wq_a.weight),
        (hp.q_a_norm.weight, native.q_norm.weight),
        (hp.q_b_proj.weight, native.wq_b.weight),
        (hp.kv_proj.weight, native.wkv.weight),
        (hp.kv_norm.weight, native.kv_norm.weight),
        (hp.o_a_proj.weight, native.wo_a.weight),
        (hp.o_b_proj.weight, native.wo_b.weight),
    )
    for target, source in pairs:
        _copy_parameter(target, source)
    if native.compressor is not None:
        _copy_parameter(hp.compressor.wkv.weight, native.compressor.wkv.weight)
        _copy_parameter(hp.compressor.norm.weight, native.compressor.norm.weight)
        if hasattr(native.compressor, "wgate"):
            _copy_parameter(hp.compressor.wgate.weight, native.compressor.wgate.weight)
    if native.indexer is not None:
        _copy_parameter(hp.indexer.q_b_proj.weight, native.indexer.wq_b.weight)
        _copy_parameter(hp.indexer.weights_proj.weight, native.indexer.weights_proj.weight)
        if native.indexer.owns_k:
            _copy_parameter(hp.indexer.wk.weight, native.indexer.wk.weight)
            _copy_parameter(hp.indexer.k_norm.weight, native.indexer.k_norm.weight)


def _position_embeddings(native: nn.Module, device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Convert the native complex frequency buffer into the adapter's real contract."""
    frequencies = native.freqs_cis[:8]
    cos = frequencies.real.unsqueeze(0).to(device=device)
    sin = frequencies.imag.unsqueeze(0).to(device=device)
    return {"main": (cos, sin), "compress": (cos, sin)}


def _candidate_mask(blocks: torch.Tensor, block_size: int, width: int) -> torch.Tensor:
    """Expand HyperParallel compact candidate block ids to the native bool mask."""
    positions = blocks.long().unsqueeze(-1) * block_size
    positions = positions + torch.arange(block_size, device=blocks.device)
    positions = positions.flatten(-2)
    valid = (positions >= 0) & (positions < width)
    result = torch.zeros(*blocks.shape[:2], width, dtype=torch.bool, device=blocks.device)
    result.scatter_(2, positions.clamp(min=0, max=max(width - 1, 0)), valid)
    return result


def _reset_native_attention_state(native_model: Any) -> None:
    """Clear the released process-global cross-layer attention slots."""
    native_model.shared_attn.compress_kv = None
    native_model.shared_attn.index_k = None
    native_model.shared_attn.topk_idxs = None
    native_model.shared_attn.candidates = None


def _run_attention_topology(
        native_model: Any,
        target_device: torch.device,
        dtype: torch.dtype,
        tolerance: Tolerance,
        name: str,
        ratios: tuple[int, ...],
        kv_sources: tuple[int, ...],
        index_sources: tuple[int, ...],
        candidate_source: int,
) -> list[dict[str, Any]]:
    """Compare one Full/Reuse/Reindex or ratio-one attention topology."""
    cases: list[dict[str, Any]] = []
    native_args = _native_attention_args(
        native_model, ratios, kv_sources, index_sources, candidate_source
    )
    hp_config = _attention_config(ratios, kv_sources, index_sources, candidate_source)
    # generate.py sets the process default dtype before model construction, so
    # unannotated native KV/index caches inherit BF16 in the released path.
    with native_model.set_dtype(dtype):
        native_layers = [native_model.Attention(layer, native_args) for layer in range(len(ratios))]
    hp_layers = [
        SharedCompressedDSAAttention(DeepseekV41AttentionPlaceholder(hp_config, layer))
        for layer in range(len(ratios))
    ]
    for layer_index, (native, hp) in enumerate(zip(native_layers, hp_layers)):
        native_fp32 = ("attn_sink",)
        if ratios[layer_index] > 1 and native.compressor is not None:
            # The released high-precision path declares ratio>1 pooling
            # projections in FP32 and casts the compressor input accordingly.
            native_fp32 += ("compressor.wkv.weight", "compressor.wgate.weight")
        _cast_parameters(native, torch.device("cpu"), dtype, native_fp32)
        _fill_parameters(native, 301 + layer_index)
        _copy_attention_weights(native, hp)
        _cast_parameters(hp, target_device, dtype, ("sinks",))

    generator = torch.Generator(device="cpu").manual_seed(310 + len(ratios))
    sources = [
        torch.randn(2, 8, 512, generator=generator, dtype=torch.float32) * 0.2
        for _ in ratios
    ]
    native_inputs = [source.to(dtype).requires_grad_(True) for source in sources]
    hp_inputs = [source.to(device=target_device, dtype=dtype).requires_grad_(True) for source in sources]
    position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
    output_tolerance = tolerance
    gradient_tolerance = tolerance
    if dtype == torch.bfloat16:
        output_tolerance = Tolerance(8.0e-2, 8.0e-2)
        gradient_tolerance = Tolerance(1.25, 1.0e-1)
    _reset_native_attention_state(native_model)
    hp_state = SharedCompressedAttentionState()
    native_outputs = []
    hp_outputs = []
    total_ms = 0.0
    for layer_index, (native, hp) in enumerate(zip(native_layers, hp_layers)):
        native_outputs.append(native(native_inputs[layer_index], 0))
        hp_output, elapsed_ms = _timed_forward(
            target_device,
            lambda hp=hp, layer_index=layer_index: hp(
                hp_inputs[layer_index],
                _position_embeddings(native_layers[layer_index], target_device),
                position_ids.to(target_device),
                None,
                shared_attention_state=hp_state,
            )[0],
        )
        hp_outputs.append(hp_output)
        total_ms += elapsed_ms
        _compare(
            cases,
            f"attention.{name}.layer{layer_index}.output",
            native_outputs[-1],
            hp_output,
            output_tolerance,
            elapsed_ms,
        )

        if layer_index in kv_sources:
            compressed_length = 8 // ratios[layer_index]
            native_compressed = native_model.shared_attn.compress_kv[:2, :compressed_length]
            _compare(
                cases,
                f"attention.{name}.layer{layer_index}.compressed_kv",
                native_compressed,
                hp_state.compressed_kv_by_source[layer_index],
                tolerance,
                0.0,
            )
            native_key = native_model.shared_attn.index_k[:2, :compressed_length]
            _compare(
                cases,
                f"attention.{name}.layer{layer_index}.index_key",
                native_key,
                hp_state.index_key_by_source[layer_index],
                tolerance,
                0.0,
            )
        if layer_index in index_sources:
            native_indices = native_model.shared_attn.topk_idxs
            native_indices = torch.where(native_indices >= 0, native_indices - 8, native_indices)
            _compare(
                cases,
                f"attention.{name}.layer{layer_index}.topk_indices",
                native_indices,
                hp_state.topk_indices_by_source[layer_index],
                tolerance,
                0.0,
            )
        if layer_index == candidate_source:
            compact = hp_state.candidate_blocks_by_source[layer_index]
            candidate_mask = _candidate_mask(compact, hp.indexer.candidate_block_size, 8 // ratios[layer_index])
            _compare(
                cases,
                f"attention.{name}.layer{layer_index}.candidate_mask",
                native_model.shared_attn.candidates,
                candidate_mask,
                tolerance,
                0.0,
            )

    torch.stack([output.float().sum() for output in native_outputs]).sum().backward()
    torch.stack([output.float().sum() for output in hp_outputs]).sum().backward()
    _synchronize(target_device)
    for layer_index, (native_input, hp_input) in enumerate(zip(native_inputs, hp_inputs)):
        _compare(
            cases,
            f"attention.{name}.layer{layer_index}.input_grad",
            native_input.grad,
            hp_input.grad,
            gradient_tolerance,
            total_ms,
        )
    return cases


def _run_attention(
        native_model: Any,
        target_device: torch.device,
        dtype: torch.dtype,
        tolerance: Tolerance,
) -> list[dict[str, Any]]:
    """Cover ratio-2 Full/Reuse/Reindex and the ratio-1 no-wgate path."""
    cases = _run_attention_topology(
        native_model,
        target_device,
        dtype,
        tolerance,
        "shared_ratio2",
        (2, 2, 2),
        (0,),
        (0, 2),
        0,
    )
    cases.extend(
        _run_attention_topology(
            native_model,
            target_device,
            dtype,
            tolerance,
            "full_ratio1",
            (1,),
            (0,),
            (0,),
            -1,
        )
    )
    return cases


def _probe_native_kernel_import(native_repo: Path) -> dict[str, Any]:
    """Probe the unmodified released CUDA/TileLang import in a clean process."""
    command = [
        sys.executable,
        "-c",
        f"import sys; sys.path.insert(0, {str(native_repo / 'inference')!r}); import kernel",
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired as error:
        return {"name": "native.kernel_import", "status": "unsupported", "reason": str(error)}
    output = (process.stderr or process.stdout).strip()
    return {
        "name": "native.kernel_import",
        "status": "pass" if process.returncode == 0 else "unsupported",
        "returncode": process.returncode,
        "reason": output[-2000:],
    }


def _probe_npu_sparse_shapes(device: torch.device) -> list[dict[str, Any]]:
    """Record the current enhanced sparse-FA head-dimension contract."""
    if device.type != "npu":
        return []
    from hyper_parallel.components.modules.shared_compressed_dsa_attention import (  # pylint: disable=C0415
        npu_sparse_attention_with_scalar_sink,
    )

    results = []
    for head_dim in (128, 512):
        try:
            query = torch.zeros(
                1, 8, 8, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True
            )
            key_value = torch.zeros(
                1, 1, 12, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True
            )
            indices = torch.arange(6, device=device, dtype=torch.int32).view(1, 1, 6).expand(1, 8, 6)
            sinks = torch.zeros(8, device=device, dtype=torch.float32)
            output = npu_sparse_attention_with_scalar_sink(
                query,
                key_value,
                indices.contiguous(),
                sinks,
                64,
                head_dim**-0.5,
            )
            output.float().sum().backward()
            _synchronize(device)
        except Exception as error:  # pylint: disable=broad-except
            results.append(
                {
                    "name": f"hyper_parallel.npu_sparse_attention.head_dim_{head_dim}",
                    "status": "unsupported",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
        else:
            results.append(
                {
                    "name": f"hyper_parallel.npu_sparse_attention.head_dim_{head_dim}",
                    "status": "pass",
                    "forward": "pass",
                    "backward": "pass",
                }
            )
    return results


def _probe_native_rope(native_model: Any, device: torch.device) -> dict[str, Any] | None:
    """Check whether the released complex-number RoPE can execute on Ascend."""
    if device.type != "npu":
        return None
    try:
        value = torch.zeros(1, 8, 8, 64, device=device, dtype=torch.bfloat16)
        frequencies = native_model.precompute_freqs_cis(64, 8, 0, 10000.0, 1.0, 32, 1).to(device)
        native_model.apply_rotary_emb(value, frequencies)
        _synchronize(device)
    except Exception as error:  # pylint: disable=broad-except
        return {
            "name": "native.apply_rotary_emb.npu",
            "status": "unsupported",
            "reason": f"{type(error).__name__}: {error}",
        }
    return {"name": "native.apply_rotary_emb.npu", "status": "pass"}


def _run_group(
        report: dict[str, Any],
        name: str,
        function: Any,
) -> None:
    """Run one module group without hiding later compatibility results."""
    try:
        report["cases"].extend(function())
    except Exception as error:  # pylint: disable=broad-except
        report["cases"].append(
            {
                "name": name,
                "status": "error",
                "error_type": type(error).__name__,
                "reason": str(error),
                "traceback": traceback.format_exc(),
            }
        )


def _parse_args() -> argparse.Namespace:
    """Parse standalone validation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-repo", type=Path, default=DEFAULT_NATIVE_REPO)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Execute every module comparison and persist machine-readable evidence."""
    args = _parse_args()
    target_device = _device_from_args(args.device, args.device_index)
    dtype_name = args.dtype or ("bfloat16" if target_device.type == "npu" else "float32")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    tolerance = Tolerance(2.0e-5, 2.0e-4) if dtype == torch.float32 else Tolerance(5.0e-2, 5.0e-2)
    native_model, native_engram = _load_native_modules(args.native_repo.resolve())
    report: dict[str, Any] = {
        "native_repo": str(args.native_repo.resolve()),
        "native_commit": subprocess.run(
            ["git", "-C", str(args.native_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "hyper_parallel_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "device": str(target_device),
        "dtype": dtype_name,
        "torch_version": torch.__version__,
        "cases": [],
        "compatibility": [_probe_native_kernel_import(args.native_repo.resolve())],
    }
    report["compatibility"].extend(_probe_npu_sparse_shapes(target_device))
    native_rope = _probe_native_rope(native_model, target_device)
    if native_rope is not None:
        report["compatibility"].append(native_rope)
    _run_group(
        report,
        "engram",
        lambda: _run_engram(native_model, native_engram, target_device, dtype, tolerance),
    )
    _run_group(
        report,
        "moe",
        lambda: _run_moe(native_model, target_device, dtype, tolerance),
    )
    _run_group(
        report,
        "attention",
        lambda: _run_attention(native_model, target_device, dtype, tolerance),
    )
    report["summary"] = {
        status: sum(case["status"] == status for case in report["cases"])
        for status in ("pass", "fail", "error")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0 if report["summary"]["fail"] == 0 and report["summary"]["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
