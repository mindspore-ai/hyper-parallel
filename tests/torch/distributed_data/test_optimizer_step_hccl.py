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
"""Thin launcher for window prefetch with concurrent model communication."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import TorchCase, parallel_run_ascend_batches

_WORKER = str(Path(__file__).with_name("_test_optimizer_step_hccl.py").resolve())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_optimizer_step_loading_hccl() -> None:
    """Check DP2/TP2 input staging, concurrent collectives, and checkpoint replay."""
    # Online and metadata each start on a fresh WORLD. Share only the training ranks
    # across windows; each loader still creates and tears down its own data producer.
    parallel_run_ascend_batches([
        TorchCase(_WORKER, f"test_optimizer_step_loading_hccl[{source}]", num_proc=4)
        for source in ("online", "metadata")
    ])
