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
"""Launch the four-rank checkpoint/HSDP finalization regression matrix."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_checkpoint_finalize_order_and_gradients(monkeypatch):
    """Both checkpoint modes and wrapped/unwrapped roots preserve finalization and gradients."""
    monkeypatch.chdir(Path(__file__).parent)
    msrun_case(3, "_test_fully_shard_checkpoint_finalize.py", "test_checkpoint_finalize_order_and_gradients",
               18538, worker_num=4, local_worker_num=4)
