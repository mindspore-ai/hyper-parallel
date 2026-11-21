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
"""One lightweight gate for checkpoint exclusion and split pipeline integration."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run_ascend_batches, TorchCase


_MEMORY = Path(__file__).resolve().parent
_TORCH = _MEMORY.parent
_EXCLUDE = str(_MEMORY / "checkpoint_exclude.py")
_REENTRANT = str(_MEMORY / "reentrant_checkpoint_cases.py")
_FSDP = str(_MEMORY / "reentrant_composite_cases.py")
_PP = str(_TORCH / "pipeline_parallel" / "reentrant_checkpoint_worker.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
def test_reentrant_exclude_suite() -> None:
    """Run checkpoint exclusion, recompute, and swap on up to four ranks per worker."""
    parallel_run_ascend_batches([
        TorchCase(_PP, "test_reentrant_pp4", num_proc=4),
        TorchCase(_EXCLUDE, "test_reentrant_checkpoint_exclude_swap", num_proc=1),
        TorchCase(_EXCLUDE, "test_checkpoint_exclude_save_chain_swap_level0", num_proc=1),
        TorchCase(_EXCLUDE, "test_rmsnorm_matmul_checkpoint_exclude_level0", num_proc=1),
        TorchCase(_EXCLUDE, "test_adjacent_save_regions_execute_once_and_replay_cached_outputs", num_proc=1),
        TorchCase(_REENTRANT, "test_memory_lifecycle", num_proc=1),
        TorchCase(_REENTRANT, ("test_operator_numerics", "test_rng_autocast"), num_proc=1),
        TorchCase(_PP, "test_reentrant_swap_stage_split_npu", num_proc=2),
        TorchCase(_PP, "test_reentrant_pp2", num_proc=2),
        TorchCase(_FSDP, "test_fsdp_checkpoint_exclude_level0", num_proc=2),
    ])
