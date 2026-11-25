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
"""Lazy model identities; implementations remain owned by HyperParallel/HF."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from .qwen3 import QWEN3_MODELS


def resolve_runtime_symbol(target: str) -> object:
    """Load a declared class or factory only when its runtime needs it."""
    module, name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


@dataclass(frozen=True)
class ModelFamily:
    """Describe existing model classes without implementing another forward."""

    model_type: str
    vllm_architecture: str
    execution_targets: dict[str, str]
    builder_target: str
    gqa_factory: str
    row_projections: tuple[str, ...]
    attention_attribute: str = "self_attn"
    attention_output: str = "o_proj"

    @property
    def moe(self) -> bool:
        """A routed-expert model must declare its authoritative HP router."""
        return "moe_block" in self.execution_targets

    def execution_types(self) -> dict[str, type]:
        """Load authoritative forward owners only in a model execution process."""
        return {role: resolve_runtime_symbol(target) for role, target in self.execution_targets.items()}


_FAMILIES = {options["model_type"]: ModelFamily(**options) for options in QWEN3_MODELS}


def model_families() -> tuple[ModelFamily, ...]:
    """Expose the same supported identities to plan checks and worker registration."""
    return tuple(_FAMILIES.values())


def get_model_family(model_type: str) -> ModelFamily:
    """Select a supported family without importing Torch or allocating a model."""
    if model_type not in _FAMILIES:
        raise ValueError(f"Unsupported unified execution model family: {model_type!r}")
    return _FAMILIES[model_type]
