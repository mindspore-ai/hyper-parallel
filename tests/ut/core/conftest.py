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
"""Pytest hooks for ``tests/ut/core``.

``hyper_parallel.core`` reads the local rank directly from ``torch.distributed``
(``dist.get_rank``) instead of the platform facade, but ``tests/ut`` never
initializes a real process group.  This module-level autouse fixture stubs the
rank to ``0`` (and the world size to ``1``) so mesh construction, layout
resolution and redistribution run single-process without a backend.

``torch.distributed`` is a module singleton shared by every ``core`` submodule,
so patching it once here covers ``device_mesh``, ``layout``, ``random``,
``tensor_redistribution``, ``stage``, ``context_parallel`` and the shard ops.
"""

from __future__ import annotations

import torch.distributed as dist  # noqa: E402  pylint: disable=wrong-import-position
import pytest  # noqa: E402  pylint: disable=wrong-import-position


@pytest.fixture(autouse=True)
def _stub_core_dist_rank(monkeypatch):
    """Stub ``torch.distributed`` rank/world-size for single-process ``core`` UT."""
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    # ``get_world_size`` is read by DeviceMesh construction and sub-layout mapping.
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1)
