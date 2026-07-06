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
"""Test-only parallel configuration templates for MoE combination tests."""

from typing import Dict, Tuple

_TEMPLATES: Dict[str, Tuple[int, int, int, int]] = {
    "ep-only": (1, 2, 1, 1),
    "tp-only": (1, 1, 2, 1),
    "dp-ep": (2, 2, 1, 1),
    "ep-tp": (1, 2, 2, 1),
    "dp-ep-tp": (2, 2, 2, 1),
    "dp-ep-cp": (2, 2, 1, 2),
}


def get_template(name: str, **overrides: int) -> Dict[str, int]:
    """Get a template configuration with optional dimension overrides."""
    if name not in _TEMPLATES:
        raise ValueError(
            f"Unknown template: {name}. Available: {list(_TEMPLATES.keys())}"
        )
    dp, ep, tp, cp = _TEMPLATES[name]
    config = {"dp": dp, "ep": ep, "tp": tp, "cp": cp}
    config.update(overrides)
    return config
