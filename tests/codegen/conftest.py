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
"""Shared fixtures for ``tests/codegen``.

The inline generator renders the real ``GQAAttention`` component and inlines
``run_qwen3_moe_flash_attention`` at build time; importing those resolves a
``torch_npu`` stub so NPU-backed modules import on CPU-only checkouts. Heavy
framework subsystems (torch, transformers) are already imported by the time a
test body runs, so a recording stub suffices — matching how
``tests/ut/auto_models/conftest.py`` injects ``torch_npu``.
"""

import sys
import types
import unittest.mock
import pytest


def _missing(_name):
    """Return an inert callable for any torch_npu API."""
    return lambda *args, **kwargs: None


@pytest.fixture(autouse=True)
def _torch_npu_stub():
    """Inject an inert ``torch_npu`` module for the duration of each test."""
    stub = types.ModuleType("torch_npu")
    stub.__getattr__ = _missing
    patcher = unittest.mock.patch.dict(sys.modules, {"torch_npu": stub})
    patcher.start()
    yield stub
    patcher.stop()