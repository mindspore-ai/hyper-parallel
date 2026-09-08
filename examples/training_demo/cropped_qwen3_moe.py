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

from typing import Any

from transformers import AutoConfig, PreTrainedModel

from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import CompileConfig


def build_cropped_qwen3_moe(
        config_path: str,
        num_hidden_layers: int = 4,
        local_files_only: bool = True,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        validate_placement: bool = False,
        distributed_setup: DistributedSetup | None = None,
        peft_config: Any | None = None,
        compile_config: CompileConfig | dict[str, Any] | None = None,
        activation_checkpoint: str | None = None,
        activation_swap: str = "none",
) -> PreTrainedModel:
    """Create a Qwen3-MoE model with fewer decoder layers and random weights.

    The function intentionally calls ``from_config`` instead of
    ``from_pretrained``. It reads only the Hugging Face configuration and
    tokenizer assets from ``config_path``; no checkpoint tensor is loaded.

    Args:
        config_path: Local Hugging Face Qwen3-30B-A3B model directory.
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

    Returns:
        A parallelized, randomly initialized cropped Qwen3-MoE model.
    """
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")

    config = AutoConfig.from_pretrained(
        config_path,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    if getattr(config, "model_type", None) != "qwen3_moe":
        raise ValueError(
            "config_path must contain a Qwen3-MoE configuration; "
            f"got model_type={getattr(config, 'model_type', None)!r}"
        )

    config.num_hidden_layers = num_hidden_layers
    config.use_cache = False
    return HyperAutoModelForCausalLM.from_config(
        config,
        distributed_setup=distributed_setup,
        peft_config=peft_config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        validate_placement=validate_placement,
        compile_config=compile_config,
        activation_checkpoint=activation_checkpoint,
        activation_swap=activation_swap,
    )
