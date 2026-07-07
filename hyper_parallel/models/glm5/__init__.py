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
"""GLM5 model registration."""
from hyper_parallel.models.glm5.model import (
    GLM5Config,
    GLM5Decoder,
    GLM5ForCausalLM,
    prepare_glm5_batch,
)
from hyper_parallel.models.glm5.parallelize import parallelize_glm5
from hyper_parallel.models.glm5.state_dict import GLM5StateDictAdapter
from hyper_parallel.models.spec import ModelSpec, register_spec

_UNIVERSAL_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
)


def _resolve_overrides(model_cfg) -> dict:
    """Collect GLM5 config overrides from model args."""
    overrides = {}
    if model_cfg is None:
        return overrides
    for field in _UNIVERSAL_FIELDS:
        value = getattr(model_cfg, field, None)
        if value is not None:
            overrides[field] = value
    extra = getattr(model_cfg, "config_overrides", None)
    if isinstance(extra, dict):
        overrides.update(extra)
    return overrides


def _build(cfg) -> GLM5ForCausalLM:
    overrides = _resolve_overrides(getattr(cfg, "model", None))
    config = GLM5Config(**overrides) if overrides else GLM5Config()
    return GLM5ForCausalLM(config)


register_spec(
    "glm5",
    ModelSpec(
        name="glm5",
        build_model_fn=_build,
        parallelize_fn=parallelize_glm5,
        state_dict_adapter=GLM5StateDictAdapter,
        prepare_batch_fn=prepare_glm5_batch,
    ),
)

__all__ = [
    "GLM5Config",
    "GLM5Decoder",
    "GLM5ForCausalLM",
    "prepare_glm5_batch",
]
