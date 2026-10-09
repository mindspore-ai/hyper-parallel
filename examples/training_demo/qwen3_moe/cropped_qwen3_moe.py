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
"""Build a layer-cropped Qwen3-MoE model from configuration only."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

from transformers import AutoConfig, PreTrainedModel

from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import CompileConfig


@dataclass(frozen=True)
class CroppedQwen3MoeOptions:
    """Model construction options for the cropped Qwen3-MoE example.

    Args:
        num_hidden_layers: Decoder layers retained in the cropped model.
        local_files_only: Disable implicit Hub downloads when true.
        torch_dtype: Model parameter dtype accepted by HyperAutoModel.
        attn_implementation: Hugging Face attention implementation name.
        validate_placement: Enable HyperParallel placement validation.
        distributed_setup: Trainer-provided distributed topology.
        peft_config: Optional Trainer-provided PEFT configuration.
        compile_config: Optional Trainer-provided compile configuration.
        activation_checkpoint: Activation checkpoint mode.
        activation_swap: Activation swap mode.
    """

    num_hidden_layers: int = 4
    local_files_only: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    validate_placement: bool = False
    distributed_setup: DistributedSetup | None = None
    peft_config: Any | None = None
    compile_config: CompileConfig | dict[str, Any] | None = None
    activation_checkpoint: str | None = None
    activation_swap: str = "none"


def _resolve_build_options(
        legacy_args: tuple[Any, ...],
        legacy_kwargs: dict[str, Any],
        options: CroppedQwen3MoeOptions | None,
) -> CroppedQwen3MoeOptions:
    """Bind the original build arguments while supporting grouped options."""
    option_names = tuple(field.name for field in fields(CroppedQwen3MoeOptions))
    if len(legacy_args) > len(option_names):
        raise TypeError(
            "build_cropped_qwen3_moe() takes from 1 to "
            f"{len(option_names) + 1} positional arguments "
            f"but {len(legacy_args) + 1} were given"
        )

    positional_names = option_names[:len(legacy_args)]
    duplicated_name = next(
        (name for name in positional_names if name in legacy_kwargs),
        None,
    )
    if duplicated_name is not None:
        raise TypeError(
            "build_cropped_qwen3_moe() got multiple values for argument "
            f"'{duplicated_name}'"
        )

    unexpected_name = next(
        (name for name in legacy_kwargs if name not in option_names),
        None,
    )
    if unexpected_name is not None:
        raise TypeError(
            "build_cropped_qwen3_moe() got an unexpected keyword argument "
            f"'{unexpected_name}'"
        )

    option_updates = dict(zip(positional_names, legacy_args))
    option_updates.update(legacy_kwargs)
    return replace(options or CroppedQwen3MoeOptions(), **option_updates)


def build_cropped_qwen3_moe(
        config_path: str,
        *legacy_args: Any,
        options: CroppedQwen3MoeOptions | None = None,
        **legacy_kwargs: Any,
) -> PreTrainedModel:
    """Create a Qwen3-MoE model with fewer decoder layers and random weights.

    The function intentionally calls ``from_config`` instead of
    ``from_pretrained``. It reads only the Hugging Face configuration and
    tokenizer assets from ``config_path``; no checkpoint tensor is loaded.

    Args:
        config_path: Local Hugging Face Qwen3-30B-A3B model directory.
        *legacy_args: Existing positional build options in their original order.
        options: Grouped model construction options.
        **legacy_kwargs: Existing keyword build options, which override ``options``.

    Returns:
        A parallelized, randomly initialized cropped Qwen3-MoE model.
    """
    options = _resolve_build_options(legacy_args, legacy_kwargs, options)
    if options.num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")

    config = AutoConfig.from_pretrained(
        config_path,
        local_files_only=options.local_files_only,
        trust_remote_code=False,
    )
    if getattr(config, "model_type", None) != "qwen3_moe":
        raise ValueError(
            "config_path must contain a Qwen3-MoE configuration; "
            f"got model_type={getattr(config, 'model_type', None)!r}"
        )

    config.num_hidden_layers = options.num_hidden_layers
    config.use_cache = False
    return HyperAutoModelForCausalLM.from_config(
        config,
        distributed_setup=options.distributed_setup,
        peft_config=options.peft_config,
        torch_dtype=options.torch_dtype,
        attn_implementation=options.attn_implementation,
        validate_placement=options.validate_placement,
        compile_config=options.compile_config,
        activation_checkpoint=options.activation_checkpoint,
        activation_swap=options.activation_swap,
    )
