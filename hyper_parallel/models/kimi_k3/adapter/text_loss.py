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
"""Native K3 LLM binding for globally aligned text chunk targets."""
# pylint: disable=forbidden-backend-import

from torch import nn


def bind_chunk_loss(model: nn.Module) -> None:
    """Validate the native chunked-objective entry without changing forward.

    Args:
        model: Registered Kimi Linear causal LLM exposing its native objective contract.

    Raises:
        ValueError: The model family or required native entry is missing.
    """
    if getattr(model.config, "model_type", None) != "kimi_linear" or not hasattr(model, "lm_head"):
        raise ValueError("K3 text chunk loss requires the registered KimiLinearForCausalLM")
