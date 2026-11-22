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
"""Launch swap optimizer checkpoint resharding workers."""

from pathlib import Path

from pytest import MonkeyPatch

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).resolve().with_name("swap_optimizer_checkpoint.py"))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_swap_optimizer_checkpoint_resharding_gloo(monkeypatch: MonkeyPatch) -> None:
    """Run the native DTensor checkpoint regression on two CPU/Gloo ranks."""
    monkeypatch.setenv("HYPER_PARALLEL_TEST_DEVICE_TYPE", "cpu")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    torchrun_case(_WORKER, "test_swap_optimizer_checkpoint_reshards_native_dtensor", num_proc=2, timeout=420)
