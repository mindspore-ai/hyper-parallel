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
"""Wan provider for the generic ``HyperAutoModelForDiT`` facade."""

from __future__ import annotations

import logging
from typing import Any

from diffusers import WanTransformer3DModel

from hyper_parallel.models._diffusers import DiffusersDiTModelProvider
from hyper_parallel.models.wan.condition import build_wan_condition_model
from hyper_parallel.models.wan.configuration import load_wan_transformer_config

logger = logging.getLogger(__name__)


def _validate_wan_parallel_axes(distributed_setup: Any) -> None:
    """Reject parallel modes that the current Wan training path does not support."""
    mesh = getattr(distributed_setup, "mesh_context", None)
    if mesh is None:
        return
    active = {
        name: value
        for name, value in {
            "tp_size": getattr(mesh, "tp_size", 1),
            "cp_size": getattr(mesh, "cp_size", 1),
            "pp_size": getattr(mesh, "pp_size", 1),
            "ep_size": getattr(mesh, "ep_size", 1),
        }.items()
        if int(value) != 1
    }
    for flag_name in ("sequence_parallel", "loss_parallel"):
        if bool(getattr(mesh, flag_name, False)):
            active[flag_name] = True
    if active:
        raise NotImplementedError(
            "Wan AutoModels migration currently supports DP/FSDP2 full finetuning only; "
            f"unsupported active axes/options: {active}."
        )


def _disable_fsdp_forward_input_cast_for_wan(distributed_setup: Any) -> None:
    """Keep official Wan RoPE tensors in fp32 across FSDP block calls."""
    fsdp_config = getattr(distributed_setup, "strategy_config", None)
    mix_precision = getattr(fsdp_config, "mix_precision", None)
    if mix_precision is None or not getattr(mix_precision, "cast_forward_inputs", False):
        return
    mix_precision.cast_forward_inputs = False
    logger.warning(
        "Disabled fsdp_config.mix_precision.cast_forward_inputs for Wan because "
        "FSDP block input casting would downcast rotary_emb from fp32 to bf16."
    )


class WanDiTModelProvider(DiffusersDiTModelProvider):
    """Wan-specific config, path and model-class adapter."""

    model_class = WanTransformer3DModel

    @staticmethod
    def prepare(distributed_setup: Any, *, peft_config: Any = None, **kwargs: Any) -> None:
        del kwargs
        if peft_config is not None:
            raise ValueError("Wan full-finetune config must not set peft/lora options.")
        _validate_wan_parallel_axes(distributed_setup)
        _disable_fsdp_forward_input_cast_for_wan(distributed_setup)

    @staticmethod
    def validate_pretrained_options(*, task: str | None, **kwargs: Any) -> None:
        del kwargs
        if task not in {"t2v", "i2v"}:
            raise ValueError("Wan task must be either 't2v' or 'i2v'")

    @staticmethod
    def load_config(config_source: str, **overrides: Any) -> dict[str, Any]:
        return load_wan_transformer_config(config_source, **overrides)

    @staticmethod
    def resolve_condition_model_path(model_target: Any) -> str | None:
        condition_path = getattr(model_target, "condition_model_name_or_path", None)
        model_path = getattr(model_target, "pretrained_model_name_or_path", None)
        if condition_path and str(condition_path).lower() != "auto":
            return str(condition_path)
        if not model_path:
            return None
        normalized = str(model_path).replace("\\", "/")
        return str(model_path)[: -len("/transformer")] if normalized.endswith("/transformer") else str(model_path)

    @staticmethod
    def build_condition_model(*, model_target: Any, device: Any, dp_rank: int, seed: int) -> Any:
        condition_path = WanDiTModelProvider.resolve_condition_model_path(model_target)
        if condition_path is None:
            raise ValueError("Wan DiT training requires model.condition_model_name_or_path")
        task = getattr(model_target, "task", "t2v")
        condition_kwargs = dict(getattr(model_target, "condition_model", {}) or {})
        condition_kwargs.setdefault("transformer_dtype", getattr(model_target, "torch_dtype", "bfloat16"))
        condition_kwargs.setdefault("local_files_only", bool(getattr(model_target, "local_files_only", False)))
        return build_wan_condition_model(
            base_model_path=condition_path,
            task=task,
            device=device,
            dp_rank=dp_rank,
            seed=seed,
            **condition_kwargs,
        )


__all__ = ["WanDiTModelProvider"]
