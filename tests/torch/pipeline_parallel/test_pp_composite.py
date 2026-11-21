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
"""Level0 launcher for the pipeline cases that need eight ranks.

The same eight workers cover dual-PP4 cold start, PP8 reentrant recomputation, HSDP gradient drain on
``(pp=2, dp=2, fsdp=2)``, and Llama3 loss parity on ``(pp=2, fsdp=2, tp=2)``.
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.distributed_launcher import torchrun_cases

_WORKER_DIR = Path(__file__).resolve().parent
_HSDP = _WORKER_DIR / "_test_pp_composite.py"
_FSDP_TP = _WORKER_DIR / "_test_pp_fsdp_tp_composite.py"
_VPP = _WORKER_DIR / "vpp_schedule.py"
_REENTRANT = _WORKER_DIR / "reentrant_checkpoint_worker.py"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_composite_level0() -> None:
    """Run the cold P2P checks before the HSDP and FSDP/TP composites."""
    torchrun_cases([
        f"{_VPP}::test_vpp_dynamic_batch_p2p_cold_start",
        f"{_VPP}::test_vpp_dynamic_multi_stream_p2p_cold_start",
        f"{_REENTRANT}::test_reentrant_pp8",
        f"{_HSDP}::test_pp_hsdp_vpp_interleaved_1f1b_matches_reference",
        f"{_HSDP}::test_pp_hsdp_vpp_dxdw_split_matches_reference",
        f"{_FSDP_TP}::test_pp_fsdp_tp_1f1b_composite_matches_reference",
    ], num_proc=8)
