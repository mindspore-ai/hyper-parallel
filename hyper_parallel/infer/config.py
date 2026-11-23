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
"""Serializable generation parameters, independent of task configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any


def _validate_token_ids(name: str, value: Any) -> None:
    """Validate scalar special tokens and the optional EOS list."""
    if value is None:
        return
    values = value if name == "eos_token_id" and isinstance(value, list) else [value]
    invalid = any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in values)
    if not values or invalid:
        raise ValueError(f"{name} must contain non-negative token ids")


@dataclass
class GenerationOptions:
    """Sampling parameters; null special tokens resolve from model/tokenizer metadata."""

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = False
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    repetition_penalty: float = 1.0
    use_cache: bool = True

    def __post_init__(self) -> None:
        """Reject invalid or non-serializable parameters at the API boundary."""
        for name in ("max_new_tokens", "top_k"):
            value = getattr(self, name)
            if (not isinstance(value, int) or isinstance(value, bool)) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("temperature", "top_p", "repetition_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.top_p > 1:
            raise ValueError("top_p must be in (0, 1]")
        for name in ("do_sample", "use_cache"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        for name in ("eos_token_id", "pad_token_id"):
            _validate_token_ids(name, getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        """Return only the serializable sampling fields."""
        return asdict(self)


def resolve_generation_options(options: GenerationOptions, model: Any, tokenizer: Any = None) -> GenerationOptions:
    """Resolve special tokens: explicit options, model generation metadata, tokenizer.

    Returns:
        A new options object; the caller's configuration is never mutated.
    """
    metadata = getattr(model, "generation_config", None)
    resolved = {}
    for name in ("eos_token_id", "pad_token_id"):
        value = getattr(options, name)
        for source in (metadata, tokenizer):
            if value is None:
                value = getattr(source, name, None)
        resolved[name] = value
    if resolved["pad_token_id"] is None:
        eos = resolved["eos_token_id"]
        resolved["pad_token_id"] = eos[0] if isinstance(eos, list) and eos else eos
    if resolved["pad_token_id"] is None:
        raise ValueError("pad_token_id is unresolved; configure it or provide tokenizer/model metadata")
    return replace(options, **resolved)
