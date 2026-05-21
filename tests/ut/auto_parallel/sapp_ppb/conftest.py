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
"""Pytest fixtures that put ``sapp_ppb`` on ``sys.path`` for the pipeline balance UT.

``hyper_parallel/auto_parallel/`` is intentionally not a regular Python package
(no ``__init__.py``), so importing ``sapp_ppb`` requires its parent directory on
``sys.path``. We resolve that directory once at collection time via either the
source tree (when tests run in-repo) or an installed ``hyper_parallel`` wheel.
"""
import importlib.util
import os
import sys

import pytest


def _from_source_tree() -> str:
    """Walk up from this file looking for ``hyper_parallel/auto_parallel/sapp_ppb``."""
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cur = os.path.dirname(cur)
        if not cur or cur == os.path.dirname(cur):
            break
        cand = os.path.join(cur, "hyper_parallel", "auto_parallel")
        if os.path.isdir(os.path.join(cand, "sapp_ppb")):
            return cand
    return ""


def _from_installed_package() -> str:
    """Find ``auto_parallel`` via ``find_spec`` without triggering heavy imports."""
    try:
        spec = importlib.util.find_spec("hyper_parallel")
    except (ImportError, ValueError):
        return ""
    if spec is None or not spec.submodule_search_locations:
        return ""
    base = list(spec.submodule_search_locations)[0]
    cand = os.path.join(base, "auto_parallel")
    if os.path.isdir(os.path.join(cand, "sapp_ppb")):
        return cand
    return ""


_SAPP_PARENT = _from_source_tree() or _from_installed_package()
if _SAPP_PARENT and _SAPP_PARENT not in sys.path:
    sys.path.insert(0, _SAPP_PARENT)


def pytest_collection_modifyitems(config, items):
    """Skip every collected item if ``sapp_ppb`` sources are unavailable."""
    del config
    if _SAPP_PARENT:
        return
    skip_marker = pytest.mark.skip(reason="sapp_ppb sources not available")
    for item in items:
        item.add_marker(skip_marker)
