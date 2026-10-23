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
"""Framework-free size contract; retain readable layout, comments and API documentation."""
import ast
from pathlib import Path


def test_async_core_code_budget():
    """Count every physical line, including licenses, comments, docstrings and blank lines."""
    root = Path(__file__).resolve().parents[2] / "rl/async_training"
    counts = {}
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        counts[str(path.relative_to(root))] = len(source.splitlines())
    assert sum(counts.values()) <= 1100, f"Async core exceeds its physical-line budget: {counts}"
