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
"""Declare K3 LLM structure, persistent state, data and limited operator gates."""
# Native Torch model adapters follow the same exception as other HF model families.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from torch import nn

from hyper_parallel.models.kimi_k3.adapter.distributed.expert_parallel import KimiK3EPMoeBlock
from hyper_parallel.models.kimi_k3.adapter.mla import KimiMLAAttention as SharedKimiMLAAttention
from hyper_parallel.models.kimi_k3.adapter.validation.native_module_parity import get_module_cases
from hyper_parallel.models.kimi_k3.configuration_kimi_k3 import KimiK3Config, KimiLinearConfig
from hyper_parallel.models.kimi_k3.modeling_kimi_linear import (
    KimiDeltaAttention, KimiMLAAttention, KimiMLP, KimiMoEGate, KimiSparseMoeBlock,
)
from hyper_parallel.models.kimi_k3.quantile_balancing import QuantileBalancer
from hyper_parallel.models.kimi_k3.vision import MoonViT3dPretrainedModel, PatchMergerMLPV2
from hyper_parallel.models.validation_spec import (
    CheckpointValidationSpec, DataValidationSpec, ModelValidationSpec, ParameterProbeSpec,
    StateInvariantSpec, TopologyConstraint,
)


def _validate_module_tree(model: Any, context: Any) -> dict[str, Any] | None:
    """Check configured branch roles on the final architecture, including packed EP."""
    del context
    config = getattr(model, "config", None)
    if not isinstance(config, KimiLinearConfig):
        return {"expected_config": "KimiLinearConfig"}
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", ())
    failures = []
    if len(layers) != config.num_hidden_layers:
        failures.append("model.layers: configured layer count differs")
    for index, layer in enumerate(layers):
        failures.extend(_validate_decoder_layer(layer, index, config))
    if config.attn_res_block_size is not None:
        for name in ("output_attn_res_norm", "output_attn_res_proj"):
            if not isinstance(getattr(backbone, name, None), nn.Module):
                failures.append(f"model.{name}: missing final AttnRes module")
    return {"invalid_modules": failures} if failures else None


def _validate_decoder_layer(layer: nn.Module, index: int, config: KimiLinearConfig) -> list[str]:
    """Check one configured attention, feed-forward and residual branch."""
    prefix = f"model.layers.{index}"
    failures = []
    expected_attention = (KimiDeltaAttention,) if config.is_kda_layer(index) else (
        KimiMLAAttention, SharedKimiMLAAttention)
    if not isinstance(getattr(layer, "self_attn", None), expected_attention):
        failures.append(f"{prefix}.self_attn: expected {', '.join(cls.__name__ for cls in expected_attention)}")
    is_moe = config.is_moe and index >= config.first_k_dense_replace and index % config.moe_layer_freq == 0
    expected_mlp = (KimiSparseMoeBlock, KimiK3EPMoeBlock) if is_moe else (KimiMLP,)
    branch = "block_sparse_moe" if is_moe else "mlp"
    if not isinstance(getattr(layer, branch, None), expected_mlp):
        failures.append(f"{prefix}.{branch}: configured dense/MoE branch differs")
    if is_moe:
        gate = getattr(getattr(layer, "block_sparse_moe", None), "gate", None)
        if not _router_matches_config(gate, config.quantile_balance):
            failures.append(f"{prefix}.block_sparse_moe.gate: router/QB structure differs from configuration")
    if config.attn_res_block_size is not None:
        for branch in ("self_attention", "mlp"):
            for suffix in ("res_norm", "res_proj"):
                name = f"{branch}_{suffix}"
                if not isinstance(getattr(layer, name, None), nn.Module):
                    failures.append(f"{prefix}.{name}: missing block AttnRes module")
    return failures


def _router_matches_config(gate: Any, quantile_balance: bool) -> bool:
    """Require QB only when the model configuration enables its training state."""
    if not isinstance(gate, KimiMoEGate):
        return False
    qb = getattr(gate, "quantile_balancer", None)
    return isinstance(qb, QuantileBalancer) if quantile_balance else qb is None


def _local(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if callable(getattr(value, "to_local", None)) else value


def _router_state_fields(module: KimiMoEGate) -> dict[str, tuple[torch.Tensor, torch.dtype, tuple[int, ...]]]:
    """Retain the selection bias contract even when QB updates are disabled."""
    fields = {
        "e_score_correction_bias": (module.e_score_correction_bias, torch.float32, (module.num_experts,)),
    }
    qb = module.quantile_balancer
    if qb is not None:
        fields.update({
            "histogram": (qb.histogram, torch.int64, (qb.num_experts, qb.num_bins)),
            "token_count": (qb.token_count, torch.int64, ()),
            "update_steps": (qb.update_steps, torch.int64, ()),
            "lower_bound": (qb.lower_bound, torch.float32, ()),
            "bin_width": (qb.bin_width, torch.float32, ()),
        })
    return fields


def _validate_qb_state(model: Any, context: Any) -> dict[str, Any] | None:
    """Require exact statistics dtypes and valid persisted values, including after resume."""
    del context
    failures = []
    for fqn, module in model.named_modules():
        if not isinstance(module, KimiMoEGate):
            continue
        if not _router_matches_config(module, module.config.quantile_balance):
            failures.append(f"{fqn}: QB structure differs from configuration")
            continue
        fields = _router_state_fields(module)
        for name, (tensor, dtype, shape) in fields.items():
            value = _local(tensor).detach()
            if tensor.dtype != dtype or tuple(tensor.shape) != shape or value.is_meta:
                failures.append(f"{fqn}.{name}: invalid dtype/shape or unmaterialized state")
                continue
            if not bool(torch.isfinite(value).all()):
                failures.append(f"{fqn}.{name}: nonfinite state")
            if name in ("histogram", "token_count", "update_steps") and bool((value < 0).any()):
                failures.append(f"{fqn}.{name}: negative counter")
            if name == "bin_width" and not bool((value > 0).all()):
                failures.append(f"{fqn}.{name}: nonpositive bin width")
    return {"invalid_qb_state": failures} if failures else None


def _llm_topology(topology: dict[str, Any]) -> bool:
    return (all(int(topology.get(axis, 1)) == 1 for axis in ("tp", "pp"))
            and not topology.get("sequence_parallel", False) and not topology.get("loss_parallel", False))


def get_validation_spec() -> ModelValidationSpec:
    """Return LLM contracts, with module-parity coverage stated explicitly."""
    optional_probes = (
        "model.layers.*.self_attn.A_log", "model.layers.*.self_attn.dt_bias",
        "model.layers.*.self_attn.g_proj.weight", "model.layers.*.block_sparse_moe.gate.weight",
        "model.layers.*.block_sparse_moe.experts.0.w[123].weight",
        "model.layers.*.block_sparse_moe.experts.w[123]",
        "model.layers.*.block_sparse_moe.routed_expert_*_proj.weight",
        "model.layers.*.*res_proj.weight", "model.layers.*.*res_norm.weight",
        "model.output_attn_res_proj.weight", "model.output_attn_res_norm.weight",
    )
    return ModelValidationSpec(
        module_cases=get_module_cases(),
        parameter_probes=(ParameterProbeSpec("model.embed_tokens.weight"), ParameterProbeSpec("lm_head.weight"),
                          *(ParameterProbeSpec(pattern, required=False) for pattern in optional_probes)),
        state_invariants=(
            StateInvariantSpec("kimi_k3.configured_module_tree", _validate_module_tree, phase="structure"),
            StateInvariantSpec("kimi_k3.qb_persistent_state", _validate_qb_state, phase="materialization"),
        ),
        data=DataValidationSpec(
            required_forward_fields=("input_ids", "position_ids"), runtime_fields=("valid_token_mask",),
            labels_are_shifted=True, deterministic_replay_required=True,
        ),
        checkpoint=CheckpointValidationSpec(
            training_only_target_patterns=("model.layers.*.block_sparse_moe.gate.quantile_balancer.*",),
        ),
        topology_constraints=(TopologyConstraint(
            "kimi_k3.llm_supported_axes", _llm_topology,
            "Use TP=PP=1 and disable sequence/loss parallelism; EP requires the explicit packed recipe.",
        ),),
        metadata={
            "scope": "K3 LLM only; TextTrainer labels are shifted, public HF forward labels are unshifted",
            "target_fields": "labels, shift_labels or chunk_loss_targets; selected loss validates the target contract",
            "module_parity_coverage": ("SiTU", "block AttnRes"),
            "module_parity_remaining": ("KDA", "gated NoPE MLA", "latent MoE/router", "QB step lifecycle"),
            "distributed_acceptance": "Separate topology/recompute/checkpoint evidence is required",
            "packing": "Explicit packed runtime, B1, shared document boundaries for KDA/conv/MLA; separate acceptance",
            "expert_probes": "Native expert 0 per layer or packed weights; full expert parity is a separate gate",
        },
    )


def _validate_multimodal_module_tree(model: Any, context: Any) -> dict[str, Any] | None:
    """Check visual ownership and reuse the complete language branch invariant."""
    config = getattr(model, "config", None)
    if not isinstance(config, KimiK3Config):
        return {"expected_config": "KimiK3Config"}
    failures = []
    vision = getattr(model, "vision_tower", None)
    if not isinstance(vision, MoonViT3dPretrainedModel):
        failures.append("vision_tower: expected MoonViT3dPretrainedModel")
    elif len(vision.encoder.blocks) != config.vision_config.vt_num_hidden_layers:
        failures.append("vision_tower.encoder.blocks: configured layer count differs")
    if not isinstance(getattr(model, "mm_projector", None), PatchMergerMLPV2):
        failures.append("mm_projector: expected PatchMergerMLPV2")
    if config.vision_config.text_hidden_size != config.text_config.hidden_size:
        failures.append("mm_projector: output width must match language hidden size")
    language_failure = _validate_module_tree(getattr(model, "language_model", None), context)
    if language_failure:
        failures.append({"language_model": language_failure})
    return {"invalid_modules": failures} if failures else None


def get_multimodal_validation_spec() -> ModelValidationSpec:
    """Declare image/sampled-video contracts separately from numerical acceptance."""
    language_spec = get_validation_spec()
    image_fields = ("pixel_values", "grid_thws", "media_spans", "media_patch_offsets", "media_token_mask")
    return ModelValidationSpec(
        parameter_probes=(
            ParameterProbeSpec("vision_tower.patch_embed.proj.weight"),
            ParameterProbeSpec("vision_tower.encoder.blocks.*.wqkv.weight"),
            ParameterProbeSpec("mm_projector.proj.0.weight"),
            ParameterProbeSpec("mm_projector.proj.2.weight"),
            *(replace(probe, match=f"language_model.{probe.match}") for probe in language_spec.parameter_probes),
        ),
        state_invariants=(
            StateInvariantSpec("kimi_k3.image_module_tree", _validate_multimodal_module_tree, phase="structure"),
            StateInvariantSpec("kimi_k3.qb_persistent_state", _validate_qb_state, phase="materialization"),
        ),
        data=DataValidationSpec(
            required_forward_fields=("input_ids", "attention_mask", "position_ids", "media_token_mask"),
            runtime_fields=("packed_sequence_info", "cu_seq_lens_q", "cu_seq_lens_k"),
            cp_replicated_forward_fields=image_fields + ("labels_are_shifted", "media_metadata"),
            modality_fields=image_fields + ("encoder_parallel_inputs",),
            modality_parameter_patterns=("vision_tower.*", "mm_projector.*"),
            labels_are_shifted=None, deterministic_replay_required=True, processor_revision_required=True,
        ),
        checkpoint=CheckpointValidationSpec(training_only_target_patterns=(
            "language_model.model.layers.*.block_sparse_moe.gate.quantile_balancer.*",
        )),
        topology_constraints=(TopologyConstraint(
            "kimi_k3.image_supported_axes", _llm_topology,
            "Use TP=PP=1 and disable sequence/loss parallelism; CP and EP require their explicit recipes. "
            "Encoder CP must satisfy the model bridge and shared encoder runtime constraints.",
        ),),
        metadata={
            "scope": "Independent or packed image/text and pre-sampled video; optional decoder and fixed encoder CP",
            "label_contract": {
                "raw_cp1": "Raw labels are shifted once by the model or K3 chunked loss adapter",
                "cp_or_packed": "The collator shifts complete documents before decoder CP slices token fields",
                "explicit_shift_labels": "Aligned targets at model/loss boundary; Omni datasets still provide labels",
                "ownership": "K3 collator/runtime/loss validate the selected mode; no single static shift flag applies",
            },
            "target_validation": "validate_multimodal_targets observes raw, pre-shifted and chunked target modes",
            "media_metadata": "Optional CPU provenance follows global media order; record its canonical JSON identity",
            "module_parity_coverage": (),
            "module_parity_remaining": ("MoonViT", "projector/scatter", "complete image/text model"),
            "distributed_acceptance": "Separate FSDP/CP/EP/encoder-CP/recompute/checkpoint evidence is required",
            "packing": "Explicit B1 packing with shared document boundaries for KDA, convolution and MLA",
            "media": "Images and pre-sampled videos with patch/grid/span metadata; audio is unsupported",
            "encoder_parallel": "Fixed encoder CP uses eager vision attention; replay must restore its encoder context",
            "qualification": "Functional topology support does not establish cross-topology numerical acceptance",
        },
    )


def _multimodal_target_source(inputs: Mapping[str, Any]) -> tuple[str, torch.Tensor, bool]:
    """Resolve the one target source accepted by the production model."""
    names = tuple(name for name in ("chunk_loss_targets", "shift_labels", "labels")
                  if inputs.get(name) is not None)
    if not names:
        raise ValueError("K3 training requires labels, shift_labels or chunk_loss_targets")
    if len(names) != 1:
        raise ValueError("K3 training requires exactly one target source")
    targets = inputs[names[0]]
    if not isinstance(targets, torch.Tensor) or targets.dtype != torch.int64:
        raise ValueError("K3 training targets must be an int64 Tensor")
    aligned = names[0] != "labels" or bool(inputs.get("labels_are_shifted", False))
    return names[0], targets, aligned


def _effective_multimodal_targets(inputs: Mapping[str, Any], targets: torch.Tensor,
                                  aligned: bool) -> torch.Tensor:
    """Observe target masking and the one raw-label shift without changing inputs."""
    mask = inputs.get("chunk_loss_mask")
    if mask is not None:
        if mask.shape != targets.shape:
            raise ValueError("K3 chunk_loss_mask must match targets")
        targets = targets.masked_fill(~mask.bool(), -100)
    if aligned:
        return targets
    targets = targets.masked_fill(~inputs["attention_mask"].bool() | inputs["media_token_mask"].bool(), -100)
    return torch.nn.functional.pad(targets[:, 1:], (0, 1), value=-100)


def _check_multimodal_target_boundaries(inputs: Mapping[str, Any], targets: torch.Tensor,
                                       cp_rank: int, cp_size: int) -> None:
    """Reject supervision of padding, visual target positions and document transitions."""
    local_length = targets.shape[1]
    global_media = inputs["media_token_mask"]
    if tuple(global_media.shape) != (targets.shape[0], local_length * cp_size):
        raise ValueError("K3 media_token_mask must retain the global decoder sequence")
    target_media = torch.nn.functional.pad(global_media[:, 1:].bool(), (0, 1), value=False)
    start = cp_rank * local_length
    target_media = target_media[:, start:start + local_length]
    padding = ~inputs["attention_mask"].bool()
    target_padding = torch.nn.functional.pad(padding[:, 1:], (0, 1), value=cp_rank == cp_size - 1)
    forbidden = padding | target_padding | target_media
    if bool((targets.ne(-100) & forbidden).any()):
        raise ValueError("K3 aligned targets must ignore padding and visual next-token positions")
    info = inputs.get("packed_sequence_info")
    if info is not None:
        ends = [end - 1 - start for end in info.global_offsets[1:] if start < end <= start + local_length]
        if ends and bool(targets[:, ends].ne(-100).any()):
            raise ValueError("K3 packed targets must ignore every document's final prediction")


def validate_multimodal_targets(model_inputs: Mapping[str, Any], *, cp_rank: int = 0,
                                cp_size: int = 1) -> dict[str, Any]:
    """Check actual training targets at the final model/loss boundary.

    Static data declarations accept either raw or globally shifted labels. This
    diagnostic checks the selected mode after optional chunk-loss preparation;
    it does not modify tensors, install hooks or replace runtime/loss validation.

    Args:
        model_inputs: Actual forward kwargs, including local token fields,
            global media masks and one supported training-target path.
        cp_rank: Rank in the chronological decoder context group.
        cp_size: Size of that decoder context group.

    Returns:
        Target source, alignment mode and effective local supervised-token count.

    Raises:
        ValueError: Target shape, alignment, padding or document/media boundaries
            disagree with the production data contract.
    """
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("K3 target validation requires a valid decoder CP coordinate")
    name, targets, aligned = _multimodal_target_source(model_inputs)
    tokens = model_inputs["input_ids"]
    if tokens.ndim != 2 or targets.shape != tokens.shape:
        raise ValueError("K3 training targets must match local input_ids [batch, sequence]")
    if model_inputs["attention_mask"].shape != tokens.shape:
        raise ValueError("K3 attention_mask must match local training targets")
    if (cp_size > 1 or model_inputs.get("packed_sequence_info") is not None) and not aligned:
        raise ValueError("K3 CP and packed targets must be globally shifted before partitioning")
    effective = _effective_multimodal_targets(model_inputs, targets, aligned)
    _check_multimodal_target_boundaries(model_inputs, effective, cp_rank, cp_size)
    return {"target_field": name, "labels_are_shifted": aligned,
            "local_supervised_tokens": int(effective.ne(-100).sum()), "cp_rank": cp_rank, "cp_size": cp_size}


__all__ = ["get_validation_spec", "get_multimodal_validation_spec", "validate_multimodal_targets"]
