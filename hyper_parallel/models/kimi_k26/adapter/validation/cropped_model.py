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
"""Cropped Kimi-K2.5/K2.6 multimodal model used by the VLM training demo."""

from __future__ import annotations

import logging
from typing import Any

from transformers import AutoConfig, PreTrainedModel

from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models._transformers import HyperAutoModelForImageTextToText
from hyper_parallel.models._transformers.checkpoint_loader import load_pretrained_weights
from hyper_parallel.models.build_options import CompileConfig
from hyper_parallel.models.kimi_k26.adapter import KIMI_MODEL_TYPES

logger = logging.getLogger(__name__)


def build_cropped_kimi_vlm(
        config_path: str | None = None,
        pretrained_model_name_or_path: str | None = None,
        num_hidden_layers: int = 2,
        n_routed_experts: int = 8,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        validate_placement: bool = False,
        distributed_setup: DistributedSetup | None = None,
        peft_config: Any | None = None,
        compile_config: CompileConfig | dict[str, Any] | None = None,
        activation_checkpoint: str | None = None,
        activation_swap: str = "none",
        swap_inputs: bool = False,
        freeze_patterns: list[str] | None = None,
        load_pretrained: bool = False,
) -> PreTrainedModel:
    """Create a Kimi-K2.5/K2.6 VLM with a cropped text tower.

    Weights are randomly initialized by default (the throughput arms only need the
    geometry); pass ``load_pretrained=True`` together with a checkpoint path to
    load the real weights instead.

    Reads the top-level ``Kimi_K25Config`` natively (never the repo's
    remote-code classes), crops only the nested text tower, keeps the 27-layer
    vision tower and the multimodal projector untouched, and parallelizes the
    whole ``KimiK25ForConditionalGeneration`` through the HyperParallel model
    stack. The family's TP layout (MLA down-projections and the vision path
    replicated, up-projections colwise) is provided by
    ``hyper_parallel/models/kimi_k26/adapter/registration.py`` and resolved
    through the shared adapter registry, so this builder carries no layout
    knowledge of its own.

    Args:
        config_path: Local Hugging Face Kimi-K2.5/K2.6 model directory (config
            only). Preferred over ``pretrained_model_name_or_path`` when both
            are given.
        pretrained_model_name_or_path: Alternative local model directory. The
            VLM trainer reads this attribute to derive the processor path, so
            loader-based configs should set it explicitly.
        num_hidden_layers: Text decoder layers retained in the cropped model.
        n_routed_experts: Routed experts per text MoE layer.
        trust_remote_code: Mirrored to the VLM trainer so the processor is
            loaded natively (``False``, the ``image_grid_thw`` protocol) instead
            of through the repo's remote-code processor (``grid_thws``). Not
            used for model loading, which is always native.
        local_files_only: Disable implicit Hub downloads when true.
        torch_dtype: Model parameter dtype accepted by HyperAutoModel.
        attn_implementation: Hugging Face attention implementation name.
        validate_placement: Enable HyperParallel placement validation.
        distributed_setup: Trainer-provided distributed topology.
        peft_config: Optional Trainer-provided PEFT configuration.
        compile_config: Optional Trainer-provided compile configuration.
        activation_checkpoint: Activation checkpoint mode.
        activation_swap: Activation swap mode.
        swap_inputs: Offload each checkpointed block's boundary activation to
            host memory during the forward pass. Independent of
            ``activation_swap``; requires ``activation_checkpoint`` to be
            ``"full"`` or ``"selective"``.
        load_pretrained: Load the Hugging Face weights from ``model_dir`` instead
            of keeping the random initialization. ``config_path`` still selects
            the configuration, so a cropped geometry with full weights would be a
            mismatch: use this with ``num_hidden_layers`` / ``n_routed_experts``
            matching the checkpoint. Loading is strict -- a partially matching
            checkpoint raises rather than silently training a half-random model.
        freeze_patterns: Module-name globs whose parameters are frozen
            (``requires_grad = False``) before plan/FSDP derivation, applied with
            ``fnmatch`` against each module's fully qualified name. Freezing the
            media tower, e.g. ``["model.vision_tower*"]``, also removes its
            backward pass and saved activations, which is what makes a
            like-for-like comparison against a pipeline that freezes it.

    Returns:
        A parallelized ``KimiK25ForConditionalGeneration``, loaded from
        ``model_dir`` when ``load_pretrained`` is set and randomly initialized
        otherwise.

    Note:
        The demo drives the model with an offline content-list dataset (see
        ``prepare_kimi_vlm_data.py``). When running with ``tp_size > 1``, the
        template-less vision leaves additionally need explicit parameter specs
        (``plan_overrides`` YAML entries) to satisfy FSDP owner coverage.
    """
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if n_routed_experts <= 0:
        raise ValueError("n_routed_experts must be positive")
    del trust_remote_code  # processor-level flag consumed by the VLM trainer

    model_dir = config_path or pretrained_model_name_or_path
    if model_dir is None:
        raise ValueError("config_path or pretrained_model_name_or_path is required")

    config = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if getattr(config, "model_type", None) not in KIMI_MODEL_TYPES:
        raise ValueError(
            "config_path must contain a Kimi-K2.5/K2.6 configuration "
            f"(config.model_type in {KIMI_MODEL_TYPES}); got model_type="
            f"{getattr(config, 'model_type', None)!r}"
        )

    text_cfg = config.text_config
    text_cfg.architectures = ["DeepseekV3ForCausalLM"]
    text_cfg.num_hidden_layers = num_hidden_layers
    text_cfg.n_routed_experts = n_routed_experts
    text_cfg.use_cache = False
    text_cfg.quantization_config = None
    # nn.Embedding uses config.pad_token_id as padding_idx; the HF generic
    # _init_weights then runs `weight[padding_idx]`, an integer index on a
    # DTensor row-sharded embedding that is not supported. None disables that
    # branch; harmless for a random-init demo because the pad row is unused.
    text_cfg.pad_token_id = None
    try:
        text_cfg._attn_implementation = attn_implementation
    except Exception:  # pylint: disable=broad-except
        pass  # optional hint; native sub-tower dispatch falls back gracefully

    from transformers.models.kimi_k25 import (  # pylint: disable=C0415
        modeling_kimi_k25 as _kimi_mod,
    )

    _orig_init_weights = _kimi_mod.Kimi_K25PreTrainedModel._init_weights

    def _patched_init_weights(self, module):  # noqa: ANN001
        # The vision position embeddings use trunc_normal_, whose erfinv_ has no
        # HyperParallel DTensor layout-inference registration; swap in the
        # equivalent normal_ initialisation so the random init stays supported.
        if isinstance(module, _kimi_mod.Kimi_K25VisionPositionEmbeddings):
            module.position_embeddings.data.normal_(mean=0.0, std=0.02)
            return None
        _orig_init_weights(self, module)
        return None

    _kimi_mod.Kimi_K25PreTrainedModel._init_weights = _patched_init_weights
    try:
        model = HyperAutoModelForImageTextToText.from_config(
            config,
            distributed_setup=distributed_setup,
            peft_config=peft_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            validate_placement=validate_placement,
            compile_config=compile_config,
            activation_checkpoint=activation_checkpoint,
            activation_swap=activation_swap,
            swap_inputs=swap_inputs,
            freeze_config=freeze_patterns,
        )
    finally:
        _kimi_mod.Kimi_K25PreTrainedModel._init_weights = _orig_init_weights

    if load_pretrained:
        # ``from_config`` never loads weights by design, so a real training run has
        # to say so explicitly.  This is the same load the ``from_pretrained`` entry
        # point performs internally, applied to the already-sharded model.
        load_pretrained_weights(model, model_dir, strict=True)
        logger.info("Loaded pretrained weights for the Kimi VLM from %s", model_dir)

    mesh_context = getattr(distributed_setup, "mesh_context", None)
    if mesh_context is not None and int(getattr(mesh_context, "cp_size", 1)) > 1:
        from hyper_parallel.models.kimi_k26.adapter.distributed.context_parallel import (  # pylint: disable=C0415
            bind_context_parallel,
        )

        bind_context_parallel(model, mesh_context)
    return model
