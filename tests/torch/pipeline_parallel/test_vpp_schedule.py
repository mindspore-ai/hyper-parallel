# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test vpp schedule"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

# Absolute path so torchrun+pytest finds the worker regardless of cwd.
_VPP_SCHEDULE = str(Path(__file__).resolve().parent / "vpp_schedule.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_vpp_schedule_group1():
    """
    Feature: parallel run case in pipeline_parallel
    Description:
        Runs ``test_vpp`` / ``run_parallel``. Before the VPP schedule, each **PP domain**
        (two ranks per domain when world size is 4) builds its own ``DeviceMesh``,
        calls ``manual_seed(parallel_seed, domain_mesh)`` with **different seeds per domain**,
        then runs ``torch.randn_like`` on a **sharded** ``DTensor`` on that mesh.

        ``simple_mlp.MLP`` still includes ``Dropout(p=0)`` for a no-op random-style module in the
        main network without affecting ``allclose`` vs standalone.

    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_VPP_SCHEDULE, "test_vpp", 12346, 4)
    ])
