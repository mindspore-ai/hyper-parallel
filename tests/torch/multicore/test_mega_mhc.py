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
"""Framework-free launchers for mHC numerical and multi-stream NPU regressions."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.torch.multicore._case_launcher import run_multicore_case
from tests.torch.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_mega_mhc_forward_reference_parity() -> None:
    """Validate all five native forward outputs before launching any backward work."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("mHC reference ST requires a payload built with --multicore on")
    worker = str(Path(__file__).resolve().with_name("_test_mega_mhc.py"))
    with without_inherited_rank_environment():
        run_multicore_case(worker, "test_reference_forward", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_mega_mhc_reference_parity() -> None:
    """Validate native forward/backward numerics against independent tensor operations."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("mHC reference ST requires a payload built with --multicore on")
    worker = str(Path(__file__).resolve().with_name("_test_mega_mhc.py"))
    with without_inherited_rank_environment():
        run_multicore_case(worker, "test_reference_forward_backward", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="unessential")
def test_mega_mhc_same_shape_multistream() -> None:
    """Run same-shape forward/backward calls on two independent NPU streams."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("mHC multi-stream ST requires a payload built with --multicore on")
    worker = str(Path(__file__).resolve().with_name("_test_mega_mhc.py"))
    with without_inherited_rank_environment():
        run_multicore_case(worker, "test_same_shape_multistream_forward_backward", num_proc=1)
