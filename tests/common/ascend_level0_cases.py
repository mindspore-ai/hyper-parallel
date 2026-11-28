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
"""Ascend Level0 checkpoint and pipeline launches shared with the CPU scheduler tests."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_cases
from tests.common.parallel_case import TorchCase, parallel_run_ascend_batches


def run_reentrant_exclude_suite(memory_dir: Path, num_proc: int) -> None:
    """Schedule the checkpoint workers that require this rank count."""
    exclude = str(memory_dir / "checkpoint_exclude.py")
    reentrant = str(memory_dir / "reentrant_checkpoint_cases.py")
    fsdp = str(memory_dir / "reentrant_composite_cases.py")
    pipeline = str(memory_dir.parent / "pipeline_parallel" / "reentrant_checkpoint_worker.py")
    cases = (
        TorchCase(pipeline, "test_reentrant_pp4", num_proc=4),
        TorchCase(exclude, "test_reentrant_checkpoint_exclude_swap", num_proc=1),
        TorchCase(exclude, "test_checkpoint_exclude_save_chain_swap_level0", num_proc=1),
        TorchCase(exclude, "test_rmsnorm_matmul_checkpoint_exclude_level0", num_proc=1),
        TorchCase(exclude, "test_adjacent_save_regions_execute_once_and_replay_cached_outputs", num_proc=1),
        TorchCase(reentrant, "test_memory_lifecycle", num_proc=1),
        TorchCase(reentrant, ("test_operator_numerics", "test_rng_autocast"), num_proc=1),
        TorchCase(pipeline, "test_reentrant_swap_stage_split_npu", num_proc=2),
        TorchCase(pipeline, "test_reentrant_pp2", num_proc=2),
        TorchCase(fsdp, "test_fsdp_checkpoint_exclude_level0", num_proc=2),
    )
    parallel_run_ascend_batches([case for case in cases if case.num_proc == num_proc])


def run_pp_batch_cold_start_and_hsdp(worker_dir: Path) -> None:
    """Start batch P2P before any other collective in this eight-rank WORLD."""
    vpp = worker_dir / "vpp_schedule.py"
    hsdp = worker_dir / "_test_pp_composite.py"
    torchrun_cases([
        f"{vpp}::test_vpp_dynamic_batch_p2p_cold_start",
        f"{hsdp}::test_pp_hsdp_vpp_interleaved_1f1b_matches_reference",
        f"{hsdp}::test_pp_hsdp_vpp_dxdw_split_matches_reference",
    ], num_proc=8, timeout=420)


def run_pp_multi_stream_cold_start_and_composites(worker_dir: Path) -> None:
    """Start multi-stream P2P in its own WORLD before the two composite workers."""
    vpp = worker_dir / "vpp_schedule.py"
    reentrant = worker_dir / "reentrant_checkpoint_worker.py"
    fsdp_tp = worker_dir / "_test_pp_fsdp_tp_composite.py"
    torchrun_cases([
        f"{vpp}::test_vpp_dynamic_multi_stream_p2p_cold_start",
        f"{reentrant}::test_reentrant_pp8",
        f"{fsdp_tp}::test_pp_fsdp_tp_1f1b_composite_matches_reference",
    ], num_proc=8, timeout=420)
