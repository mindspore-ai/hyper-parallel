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
"""Launch distributed Qwen vision CPU/Gloo workers without importing Torch."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_qwen_encoder_parallel() -> None:
    """Compare Encoder-DP and Encoder-CP features/gradients against native HF."""
    worker = str(Path(__file__).with_name("_qwen_encoder_impl.py"))
    for case in ("test_encoder_dp", "test_encoder_cp"):
        torchrun_case(worker, case, num_proc=2)
