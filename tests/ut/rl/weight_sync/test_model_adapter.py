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

import pytest
import torch

from rl.roles.model_setup import ModelRegistration, resolve_vllm_model
from rl.roles.weight_sync.model_adapter import (
    ModelWeightAdapter,
    build_model_weight_adapter,
)


def _model(family: str, *, tied: bool = True):
    """Return the registered Qwen3 dense rollout model."""
    identities = {
        "qwen3": ("Qwen3ForCausalLM", "qwen3"),
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
        tie_word_embeddings=tied,
    )
    return resolve_vllm_model(registration, "hyper")


@pytest.mark.parametrize("tied", [False, True])
def test_model_adapter_maps_dense_and_tied_actor_names(tied: bool) -> None:
    """Dense mapping preserves canonical tensors and transfers tied storage once."""
    adapter = ModelWeightAdapter(_model("qwen3", tied=tied))
    embedding = torch.arange(4, dtype=torch.float32)
    mapped = adapter.map_local_state_dict(
        {
            "model.embed_tokens.weight": embedding,
            "lm_head.weight": embedding,
            "model.norm.weight": torch.ones(2),
        }
    )

    expected_names = ["model.embed_tokens.weight", "model.norm.weight"]
    if not tied:
        expected_names.insert(1, "lm_head.weight")
    assert list(mapped) == expected_names
    assert mapped["model.embed_tokens.weight"] is embedding
    descriptions = adapter.direct_source_descriptions(mapped, source_rank=3)
    assert [description["name"] for description in descriptions] == sorted(expected_names)
    assert {description["source_rank"] for description in descriptions} == {3}


def test_model_adapter_factory_selects_registered_family() -> None:
    """The supported dense family selects its canonical adapter."""
    dense = build_model_weight_adapter(_model("qwen3"))

    assert dense.__class__ is ModelWeightAdapter
