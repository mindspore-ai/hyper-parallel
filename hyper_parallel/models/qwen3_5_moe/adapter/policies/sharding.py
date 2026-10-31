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
"""FSDP execution policy for Qwen3.5-MoE and Qwen3.6 VLMs."""

from typing import Any


def get_fsdp_execution_order(model: Any, module_fqns: tuple[str, ...]) -> tuple[str, ...]:
    """Return child units in Qwen VLM embedding, vision, and decoder order.

    Args:
        model: Model being built or inspected.
        module_fqns: Selected FSDP child units in module traversal order.

    Returns:
        Every selected child unit once, with text embeddings before vision.
    """
    del model
    embedding_fqn = "model.language_model.embed_tokens"
    if embedding_fqn not in module_fqns:
        return module_fqns
    return (embedding_fqn,) + tuple(fqn for fqn in module_fqns if fqn != embedding_fqn)


__all__ = ["get_fsdp_execution_order"]
