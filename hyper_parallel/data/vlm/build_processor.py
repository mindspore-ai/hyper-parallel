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
"""Build the VLM processor."""

from typing import Any

from transformers import AutoProcessor


def build_processor(pretrained_model_name_or_path: str, **kwargs: Any) -> Any:
    """Build the Qwen3-VL processor and surface the tokenizer chat template.

    Base checkpoints (e.g. Qwen3.5-0.8B-Base) carry the chat template on the
    tokenizer but not on the processor; fall back when the processor lacks one.

    Args:
        pretrained_model_name_or_path: Model identifier or local processor path.
        **kwargs: Options forwarded to ``AutoProcessor.from_pretrained``.

    Returns:
        The constructed processor.
    """
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, **kwargs
    )
    if getattr(processor, "chat_template", None) is None:
        tokenizer_chat_template = getattr(
            getattr(processor, "tokenizer", None), "chat_template", None
        )
        if tokenizer_chat_template is not None:
            processor.chat_template = tokenizer_chat_template
    return processor


__all__ = ["build_processor"]
