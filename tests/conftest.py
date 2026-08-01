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
"""Test-only source bootstrap for separately packaged local components."""

from __future__ import annotations

import sys
from pathlib import Path


_OBSERVER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "hyper_models"
    / "components"
    / "training"
    / "low_precision"
    / "precision_observer"
    / "src"
)

if str(_OBSERVER_SOURCE) not in sys.path:
    sys.path.insert(0, str(_OBSERVER_SOURCE))
