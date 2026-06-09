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
"""
Global tiling registry: maps tiling_position -> pre-serialized tiling bytes.

Each operator registers its tiling data here; gen_runtime_data writes them
to *.bin files that are loaded by the C++ kernel at runtime.
"""

_registry: dict = {}


def register_tiling(tiling_position: int, data: bytes) -> None:
    """Register pre-serialized tiling bytes at the given position."""
    _registry[tiling_position] = data


def get_tiling(tiling_position: int) -> bytes:
    """Retrieve pre-serialized tiling bytes for the given position."""
    if tiling_position not in _registry:
        raise KeyError(f"No tiling registered at position {tiling_position}")
    return _registry[tiling_position]


def clear() -> None:
    """Clear all registered tiling data."""
    _registry.clear()
