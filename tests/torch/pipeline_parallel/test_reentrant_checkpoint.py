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
"""Distributed reentrant checkpoint and split pipeline backward."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run


_WORKER = str(Path(__file__).resolve().parent / "reentrant_checkpoint_worker.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_reentrant_checkpoint_split_small_meshes() -> None:
    """Exercise PP2 and PP4 on disjoint devices for an independent Level1 run."""
    parallel_run([
        TorchCase(_WORKER, "test_reentrant_pp2", num_proc=2),
        TorchCase(_WORKER, "test_reentrant_pp4", num_proc=4),
    ], global_num_proc=6)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards",
          essential_mark="unessential")
def test_reentrant_checkpoint_split_pp8():
    """Exercise eight-rank split backward beyond the PP2/PP4 Level0 cases."""
    parallel_run([TorchCase(_WORKER, "test_reentrant_pp8", num_proc=8)])
