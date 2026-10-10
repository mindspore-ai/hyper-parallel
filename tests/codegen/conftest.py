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

import importlib.util
import sys
import types
import unittest.mock
import pytest


def _missing(name):
    """Return an inert callable for any torch_npu API.

    Dunder lookups (``__file__``, ``__loader__``, ...) raise
    ``AttributeError`` so the stub behaves like a module without those
    attributes: ``inspect.getmodule`` walks ``sys.modules`` reading
    ``module.__file__`` when torch registers custom ops, and a callable
    ``__file__`` crashes it.
    """
    if name.startswith("__"):
        raise AttributeError(name)
    return lambda *args, **kwargs: None


@pytest.fixture(autouse=True)
def _torch_npu_stub():
    """Inject an inert ``torch_npu`` module for the duration of each test.

    The stub carries a ``__spec__``: ``importlib.util.find_spec`` (used by
    third-party NPU probes such as accelerate's) raises ``ValueError`` for
    a ``sys.modules`` entry without one, and the lazy
    ``hyper_parallel.distributed`` imports inside codegen tests reach those
    probes through the ``build_options`` module chain.
    """
    stub = types.ModuleType("torch_npu")
    stub.__getattr__ = _missing
    stub.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
    patcher = unittest.mock.patch.dict(sys.modules, {"torch_npu": stub})
    patcher.start()
    yield stub
    patcher.stop()
