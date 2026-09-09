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
"""CPU unit tests for model-owned canonical weight mappings."""
# pylint: disable=forbidden-backend-import,missing-public-docstring

import torch

from rl.roles.model import ModelRegistration, resolve_vllm_model
from rl.roles.weight_sync.model_adapter import (
    ModelWeightAdapter,
    PackedExpertWeightAdapter,
    aggregate_direct_content_identity,
    build_model_weight_adapter,
    direct_fragment_record,
)


def _model(family: str):
    """Return one registered dense or MoE rollout model."""
    identities = {
        "qwen3": ("Qwen3ForCausalLM", "qwen3"),
        "qwen3_moe": ("Qwen3MoeForCausalLM", "qwen3_moe"),
        "deepseek_v3": ("DeepseekV3ForCausalLM", "deepseek_v3"),
    }
    architecture, model_type = identities[family]
    registration = ModelRegistration(
        name=family,
        hyper_model_name=family,
        weights_path="/model",
        tokenizer_path="/tokenizer",
        hf_architecture=architecture,
        model_type=model_type,
        text_model_type=model_type,
        tie_word_embeddings=True,
    )
    return resolve_vllm_model(registration, "hyper")


def test_model_adapter_maps_dense_and_tied_actor_names() -> None:
    """Dense mapping preserves canonical tensors and transfers tied storage once."""
    adapter = ModelWeightAdapter(_model("qwen3"))
    embedding = torch.arange(4, dtype=torch.float32)
    mapped = adapter.map_local_state_dict(
        {
            "model.embed_tokens.weight": embedding,
            "lm_head.weight": embedding,
            "model.norm.weight": torch.ones(2),
        }
    )

    assert list(mapped) == ["model.embed_tokens.weight", "model.norm.weight"]
    assert mapped["model.embed_tokens.weight"] is embedding
    descriptions = adapter.direct_source_descriptions(mapped, source_rank=3)
    assert [description["name"] for description in descriptions] == [
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]
    assert {description["source_rank"] for description in descriptions} == {3}


def test_packed_expert_adapter_splits_gate_up_into_canonical_regions() -> None:
    """Packed expert storage becomes separate canonical gate and up projections."""
    adapter = PackedExpertWeightAdapter(_model("qwen3_moe"))
    packed = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4)

    descriptions = adapter.direct_source_descriptions(
        {"model.layers.0.mlp.experts.gate_up_proj": packed},
        source_rank=1,
    )

    assert [description["name"] for description in descriptions] == [
        "model.layers.0.mlp.experts.gate_proj.weight",
        "model.layers.0.mlp.experts.up_proj.weight",
    ]
    assert [description["global_shape"] for description in descriptions] == [
        [2, 3, 4],
        [2, 3, 4],
    ]
    assert [description["source_starts"] for description in descriptions] == [
        [0, 0, 0],
        [0, 3, 0],
    ]
    assert all(
        description["source_name"] == "model.layers.0.mlp.experts.gate_up_proj"
        for description in descriptions
    )


def test_direct_content_identity_is_stable_for_canonical_fragments() -> None:
    """Fragment records aggregate deterministically regardless of insertion order."""
    first_key, first = direct_fragment_record(
        "weight",
        (0,),
        (2,),
        "float32",
        torch.tensor([1.0, 2.0]).numpy().tobytes(),
    )
    second_key, second = direct_fragment_record(
        "weight",
        (2,),
        (2,),
        "float32",
        torch.tensor([3.0, 4.0]).numpy().tobytes(),
    )

    forward = aggregate_direct_content_identity(
        {first_key: first, second_key: second}
    )
    reverse = aggregate_direct_content_identity(
        {second_key: second, first_key: first}
    )

    assert forward == reverse, (
        f"Content identity depends on insertion order: forward={forward}, reverse={reverse}"
    )
    assert forward["fragment_count"] == 2
    assert forward["total_bytes"] == 16


def test_model_adapter_factory_selects_registered_family() -> None:
    """Dense and both supported MoE families select their shared adapters."""
    dense = build_model_weight_adapter(_model("qwen3"))
    qwen_moe = build_model_weight_adapter(_model("qwen3_moe"))
    deepseek = build_model_weight_adapter(_model("deepseek_v3"))

    assert dense.__class__ is ModelWeightAdapter
    assert qwen_moe.__class__ is PackedExpertWeightAdapter
    assert deepseek.__class__ is PackedExpertWeightAdapter
