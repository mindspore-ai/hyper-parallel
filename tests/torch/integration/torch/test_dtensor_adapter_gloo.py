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
"""Framework-free launcher for the two-rank CPU/Gloo adapter probe."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dtensor_adapter_gloo():
    """Run real two-rank layout and local optimizer checks without accelerators."""
    worker = Path(__file__).with_name("_test_dtensor_adapter_gloo.py")
    torchrun_case(str(worker), "test_adapter_gloo", num_proc=2)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_checkpoint_compatibility_gloo():
    """Verify four-rank sharded DCP storage sharing and restore identity."""
    worker = Path(__file__).with_name("_test_dtensor_adapter_gloo.py")
    torchrun_case(str(worker), "test_checkpoint_compatibility_gloo", num_proc=4)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_uneven_optimizer_checkpoint_gloo():
    """Verify original-width and empty FSDP optimizer shards through real DCP."""
    worker = Path(__file__).with_name("_test_dtensor_adapter_gloo.py")
    torchrun_case(str(worker), "test_uneven_optimizer_checkpoint_gloo", num_proc=4)
