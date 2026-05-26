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
"""Pytest launchers for ``Module.parallelize`` NPU distributed tests.

Port allocation:
  10660–10661  2-card functional (toy linear TP + idempotent guard)
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run

_WORKER = str(Path(__file__).resolve().parent / "_test_module_parallelize.py")


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="unessential",
)
def test_module_parallelize_tp2():
    """
    Feature: Module.parallelize 2-rank TP
    Description: ToyLinear TP sharding via dmodule Module
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_WORKER, "test_parallelize_toy_linear_tp2", 10660, 2),
        TorchCase(_WORKER, "test_parallelize_idempotent_guard", 10661, 2),
    ])
