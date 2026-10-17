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
"""Framework-free launcher for two-process router auxiliary-loss checks."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_partitioned_router_aux_loss() -> None:
    """Launch real Gloo checks, including an empty local token shard.

    Feature: Distributed sequence and modality balancing.
    Description: Split logical sequences across two CPU/Gloo processes.
    Expectation: Loss, rank-average gradients and sharded bias updates match the full-token oracle.
    """
    torchrun_case(
        file_name=str(Path(__file__).with_name("_test_router_aux_loss.py")),
        case_name="test_partitioned_router_gradients",
        num_proc=2,
    )
