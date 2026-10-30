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
"""Small, training-independent helpers referenced by emitted source."""

from typing import Any


def to_local_if_dtensor(output: Any) -> Any:
    """Return the local tensor for DTensor-like values, otherwise the value."""
    to_local = getattr(output, "to_local", None)
    if callable(to_local):
        return to_local()
    return output


__all__ = ["to_local_if_dtensor"]
