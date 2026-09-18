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
"""Shared, side-effect-free configuration for the Hyper-RL CPU UT suite."""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path
import sys

import pytest


# ``rl`` is an independent source root rather than an installed subpackage.
# CI deploys the wheel separately from the test archive. Resolve the active package.
_RL_SOURCE_ROOT = Path(find_spec("hyper_parallel").origin).parent / "rl"
if str(_RL_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RL_SOURCE_ROOT))  # pylint: disable=sys-path-mutation


os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")


def pytest_configure(config: pytest.Config) -> None:
    """Include the migrated Agentic suite despite its standalone script name."""
    config.addinivalue_line("python_files", "agentic_ut.py")
