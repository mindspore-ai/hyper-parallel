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
"""Import-order coverage for the SwiGLU callable used by replacement modules."""

import os
from pathlib import Path
import subprocess
import sys
import unittest

from tests.common.mark_utils import arg_mark


_CHILD = """
import importlib
import importlib.machinery
from pathlib import Path
import sys
import types

# This stub checks wrapper argument forwarding, without testing NPU arithmetic.
stub = types.ModuleType('torch_npu')
stub.__spec__ = importlib.machinery.ModuleSpec('torch_npu', loader=None)
sentinel, output = object(), object()
calls = []
def npu_swiglu(value, dim=-1):
    calls.append((value, dim))
    return output
stub.npu_swiglu = npu_swiglu
sys.modules['torch_npu'] = stub

prefix = 'hyper_parallel.components.modules.'
names = ('swiglu_mlp', 'shared_expert', 'grouped_experts')
if sys.argv[2] == 'consumer_first':
    importlib.import_module(prefix + names[0])
definition = importlib.import_module('hyper_parallel.components.functional.swiglu')
root = Path(sys.argv[1]).resolve()
for name in names:
    module = importlib.import_module(prefix + name)
    assert Path(module.__file__).resolve().is_relative_to(root), module.__file__
    bound = module.swiglu
    assert bound is definition.swiglu, f'{name}: expected defining function, got {type(bound).__name__}'
    calls.clear()
    assert bound(sentinel, dim=0) is output, f'{name}: wrapper return identity changed'
    assert calls == [(sentinel, 0)], f'{name}: wrapper arguments changed'
"""


class TestSwiGLUImportOrder(unittest.TestCase):
    """Package/submodule imports must not change replacement callable bindings."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_consumer_callables_are_independent_of_import_order(self) -> None:
        """
        Feature: SwiGLU replacement imports.
        Description: Import the functional submodule before or after a consumer in fresh processes.
        Expectation: All three consumers call the defining wrapper with the original arguments.
        """
        root = Path(__file__).resolve().parents[4]
        env = dict(os.environ, PYTHONPATH=str(root), TORCH_DEVICE_BACKEND_AUTOLOAD="0",
                   PYTHONDONTWRITEBYTECODE="1")
        for order in ("submodule_first", "consumer_first"):
            with self.subTest(order=order):
                result = subprocess.run([sys.executable, "-B", "-c", _CHILD, str(root), order],
                                        env=env, capture_output=True, text=True, check=False, timeout=60)
                self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
