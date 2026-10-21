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
"""Thin launchers for packed KDA/GDN context parallelism."""
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_linear_tnd_cpu():
    """Feature: TND CP. Description: Four-rank independent reference. Expectation: Isolated samples and gradients."""
    torchrun_case(str(Path(__file__).with_name("_test_linear_tnd.py")), "test_linear_tnd_gloo", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("workers", [4, 8])
def test_linear_tnd_npu(workers, monkeypatch):
    """Feature: TND CP. Description: Kernels, protocols and replay. Expectation: Finite, isolated, identical replay."""
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    torchrun_case(str(Path(__file__).with_name("_test_linear_tnd.py")), "test_linear_tnd_npu", num_proc=workers)
