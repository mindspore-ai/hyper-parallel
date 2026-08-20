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
"""Build the tokenizer used by private Omni data stages."""

from typing import Any

from hyper_models.components.datasets.llm.build_tokenizer import AutoTokenizer


def build_tokenizer(pretrained_model_name_or_path: str, **kwargs: Any) -> Any:
    """Build an Omni tokenizer before constructing modality processors.

    Args:
        pretrained_model_name_or_path: Model identifier or local tokenizer path.
        **kwargs: Options forwarded to the shared tokenizer loader.

    Returns:
        The constructed tokenizer.
    """
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)


__all__ = ["build_tokenizer"]
