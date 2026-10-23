# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Build a small, layer-cropped dense Qwen3 model for PP smoke tests."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PreTrainedModel, Qwen3Config

from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import CompileConfig


def build_cropped_qwen3(
    config_path: str | None = None,
    num_hidden_layers: int = 4,
    vocab_size: int = 256,
    hidden_size: int = 128,
    intermediate_size: int = 256,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 512,
    local_files_only: bool = True,
    torch_dtype: str = "float32",
    attn_implementation: str = "eager",
    validate_placement: bool = False,
    distributed_setup: DistributedSetup | None = None,
    peft_config: Any | None = None,
    compile_config: CompileConfig | dict[str, Any] | None = None,
    activation_checkpoint: str | None = None,
    activation_checkpoint_selection: Any | None = None,
    swap_inputs: bool = False,
    allow_cache_entry_mutation: bool = False,
    model_init_dtype: str | None = None,
    activation_swap: str = "none",
) -> PreTrainedModel:
    """Create a randomly initialized Qwen3 with a reduced layer count.

    When ``config_path`` is omitted, a self-contained tiny config is used,
    which makes this function suitable for a local PP smoke test without any
    checkpoint or tokenizer download.
    """
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if config_path:
        config = AutoConfig.from_pretrained(
            config_path, local_files_only=local_files_only, trust_remote_code=False
        )
        if getattr(config, "model_type", None) != "qwen3":
            raise ValueError(
                "config_path must contain a dense Qwen3 config; "
                f"got model_type={getattr(config, 'model_type', None)!r}"
            )
        config.num_hidden_layers = num_hidden_layers
    else:
        config = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            use_cache=False,
        )
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
        activation_checkpoint_selection=activation_checkpoint_selection,
        swap_inputs=swap_inputs,
        allow_cache_entry_mutation=allow_cache_entry_mutation,
        model_init_dtype=model_init_dtype,
        activation_swap=activation_swap,
    )


__all__ = ["build_cropped_qwen3"]
