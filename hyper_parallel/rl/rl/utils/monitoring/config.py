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
"""Safe configuration conversion for external monitoring backends."""

import re
from typing import Any, Mapping

_SECRET_KEY_PARTS = frozenset(("api_key", "apikey", "password", "secret", "token"))


def sanitize_config(value: Any, key: str = "") -> Any:
    """Recursively redact secret-like configuration fields."""
    separated_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    key_parts = frozenset(re.findall(r"[a-z0-9]+", separated_key.lower()))
    compound_parts = key_parts | frozenset(("_".join(sorted(key_parts)),))
    if key_parts & _SECRET_KEY_PARTS or compound_parts & _SECRET_KEY_PARTS:
        return "***"
    if isinstance(value, Mapping):
        return {
            str(child_key): sanitize_config(child_value, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_config(item) for item in value]
    return value
