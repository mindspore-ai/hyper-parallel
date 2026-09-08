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
# ============================================================================
"""Build the large Qwen3-MoE compile benchmark without loading a checkpoint."""

from __future__ import annotations

from typing import Any, Optional

from transformers import Qwen3MoeConfig, PreTrainedModel

from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import CompileConfig


def build_large_qwen3_moe_config(
    num_hidden_layers: int = 6,
) -> Qwen3MoeConfig:
    """Return the parameter-sized Qwen3-MoE configuration used by the demo.

    The dimensions keep the large hidden/expert projection sizes while fitting
    an eight-card 64-GiB NPU benchmark: 6 decoder layers, 64 routed experts,
    and 8 experts selected per token.  No checkpoint is read; Trainer
    materializes and initializes the sharded model directly on the target NPU
    mesh.
    """
    return Qwen3MoeConfig(
        vocab_size=65536,
        hidden_size=8192,
        intermediate_size=24576,
        moe_intermediate_size=3840,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=64,
        num_key_value_heads=8,
        max_position_embeddings=32768,
        num_experts=64,
        num_experts_per_tok=8,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        output_router_logits=False,
        use_cache=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        # The benchmark uses pretokenized mock data and does not need a fixed
        # padding row.  Keeping this unset also avoids integer-indexing a
        # TP+FSDP-sharded embedding during HF random initialization.
        pad_token_id=None,
        architectures=["Qwen3MoeForCausalLM"],
        attn_implementation="sdpa",
    )


def build_large_qwen3_moe(
    *,
    distributed_setup: Optional[DistributedSetup] = None,
    torch_dtype: Any = "bfloat16",
    attn_implementation: str = "sdpa",
    num_hidden_layers: int = 6,
    compile_config: Optional[CompileConfig] = None,
    activation_checkpoint: Optional[str] = None,
    activation_swap: str = "none",
    validate_placement: bool = False,
) -> PreTrainedModel:
    """Build and parallelize the large model through the Trainer contract."""
    config = build_large_qwen3_moe_config(num_hidden_layers=num_hidden_layers)
    config.attn_implementation = attn_implementation
    return HyperAutoModelForCausalLM.from_config(
        config,
        distributed_setup=distributed_setup,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        compile_config=compile_config,
        activation_checkpoint=activation_checkpoint,
        activation_swap=activation_swap,
        validate_placement=validate_placement,
    )


__all__ = ["build_large_qwen3_moe", "build_large_qwen3_moe_config"]
