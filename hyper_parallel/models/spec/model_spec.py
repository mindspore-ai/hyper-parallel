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
"""ModelSpec dataclass — per-model bundle for trainer registration.

Each model registers a ModelSpec that tells the trainer how to construct and
(51 lines) with adaptations for hyper's DTensor-first architecture.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Type

@dataclass
class ModelSpec:
    """Per-model registration bundle.

    Args:
        name: Unique identifier used in YAML ``model.name``.
        build_model_fn: Factory ``(cfg) -> nn.Module``. Can return any
            ``nn.Module`` — self-written or ``AutoModel.from_pretrained``.
        parallelize_fn: Optional custom parallelize ``(model, mesh, cfg) -> model``.
            When ``None``, ``BaseTrainer._build_parallelized_model`` applies the
            default TP → CP → AC → FSDP pipeline by reading ``model.layers``,
            ``model._tp_plan``, and ``model._cp_modules``.
        clip_grad_fn: Optional custom gradient clipping callable.
            When ``None``, hyper's DTensor-aware ``clip_grad_norm_`` is used.
        pipelining_fn: Optional PP setup ``(model, mesh, cfg) -> (schedule, stages)``.
        state_dict_adapter: Optional class for HF ↔ hyper weight key translation.

    Example:
        Standard transformer (uses defaults, ~15 lines to register)::

            register_spec("qwen3_5_7b", ModelSpec(
                name="qwen3_5_7b",
                build_model_fn=lambda cfg: Qwen35ForCausalLM(cfg.model),
            ))

        MoE model (custom parallelize for EP, ~170 lines total)::

            register_spec("deepseek_v3", ModelSpec(
                name="deepseek_v3",
                build_model_fn=lambda cfg: DeepSeekV3ForCausalLM(cfg.model),
                parallelize_fn=parallelize_deepseek_v3,
            ))

    Model-specific parallel validation (e.g. "TP unsupported for linear-attn
    layers") is **inlined at the top of the model's** ``parallelize_<name>()``
    function — convention. There is no separate
    ``validate_parallel_fn`` field; that would be one indirection too many.
    """

    name: str
    build_model_fn: Callable
    parallelize_fn: Optional[Callable] = None
    clip_grad_fn: Optional[Callable] = None
    pipelining_fn: Optional[Callable] = None
    state_dict_adapter: Optional[Type] = None
