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
"""UT: DCP layout round-trip for in-memory full weights.

Validates :func:`full_state_dict_to_dcp_format` and :func:`dcp_to_full_state_dict` only.
Hugging Face I/O, ``convert_full_checkpoint_to_dcp``, and file-based checkpoints are
covered in ST (see ``tests/torch/distributed_checkpoint/test_offline_convert_st_torch.py``).

Uses plain PyTorch tensors on CPU for this gate (``pytest tests/ut``).
"""
# pylint: disable=wrong-import-position
import os
import shutil

import pytest
import torch

from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemWriter
from hyper_parallel.core.distributed_checkpoint.offline_transform import (
    dcp_to_full_state_dict,
    full_state_dict_to_dcp_format,
)
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner

_WORKSPACE = os.path.join(os.path.dirname(__file__), "_dcp_roundtrip_ut_workspace")


def _cleanup_workspace() -> None:
    if os.path.isdir(_WORKSPACE):
        shutil.rmtree(_WORKSPACE, ignore_errors=True)


def _ensure_clean_dcp_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def _assert_state_dicts_close(original: dict, loaded: dict) -> None:
    """Recursively compare two (possibly nested) state dicts of plain tensors."""
    assert set(original.keys()) == set(loaded.keys())
    for key in original:
        a, b = original[key], loaded[key]
        if isinstance(a, dict):
            assert isinstance(b, dict), (key, type(b))
            _assert_state_dicts_close(a, b)
        else:
            assert torch.is_tensor(a) and torch.is_tensor(b), (key, type(a), type(b))
            assert a.shape == b.shape
            assert torch.allclose(a.cpu().float(), b.cpu().float())


def test_dcp_full_state_dict_roundtrip() -> None:
    """Write merged weights to DCP on disk and merge back; values match."""
    _cleanup_workspace()
    os.makedirs(_WORKSPACE, exist_ok=True)
    dcp_dir = os.path.join(_WORKSPACE, "dcp")
    try:
        _ensure_clean_dcp_dir(dcp_dir)
        state_dict = {"w": torch.randn(3, 5), "b": torch.randn(5)}
        full_state_dict_to_dcp_format(state_dict, dcp_dir)
        assert os.path.isfile(os.path.join(dcp_dir, ".metadata"))
        merged = dcp_to_full_state_dict(dcp_dir)
        _assert_state_dicts_close(state_dict, merged)
    finally:
        _cleanup_workspace()


def _save_nested_state_dict_as_dcp(state_dict: dict, dcp_dir: str) -> None:
    """Write a nested payload with the planner/writer pair the trainer's save uses.

    ``full_state_dict_to_dcp_format`` rejects nested payloads by contract, and the
    ``api.save`` path needs a process group, which UTs must not initialize. The
    orchestration below is the single-rank slice of ``api.save`` (same
    ``StandardSavePlanner`` + ``FileSystemWriter``), so the on-disk layout and the
    recorded planner-data nesting match a real trainer checkpoint.
    """
    storage_writer = FileSystemWriter(dcp_dir)
    planner = StandardSavePlanner()
    planner.configure_planner(state_dict=state_dict, is_coordinator=True)
    storage_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=True)
    local_plan = planner.build_local_plan()
    local_data = storage_writer.optimize_local_plan(local_plan)
    central_plan, global_metadata = planner.build_global_plan([local_data])
    central_plan = storage_writer.optimize_global_plan(central_plan)
    final_local_plan = planner.finalize_plan(central_plan)
    all_writes = storage_writer.execute_write(final_local_plan, planner)
    storage_writer.finalize_checkpoint(metadata=global_metadata, results=[all_writes])


def test_dcp_nested_state_dict_roundtrip() -> None:
    """A trainer-style nested payload merges back with its save-time nesting.

    ``_DcpMergeLoadPlanner`` used to write both the flat key and the planner-data
    nested position, so the re-flatten rejected the duplicate FQN and a checkpoint
    saved from ``{"model": ..., "optimizer": ...}`` could not be merged at all.
    """
    _cleanup_workspace()
    os.makedirs(_WORKSPACE, exist_ok=True)
    dcp_dir = os.path.join(_WORKSPACE, "dcp_nested")
    try:
        _ensure_clean_dcp_dir(dcp_dir)
        state_dict = {
            "model": {
                "scale_shift_table": torch.randn(6, 12),
                "blocks.0.attn1.qkv.weight": torch.randn(4, 4),
            },
            "optimizer": {"exp_avg": torch.randn(4, 4)},
        }
        _save_nested_state_dict_as_dcp(state_dict, dcp_dir)
        assert os.path.isfile(os.path.join(dcp_dir, ".metadata"))
        merged = dcp_to_full_state_dict(dcp_dir)
        assert isinstance(merged.get("model"), dict), f"model subtree lost: {sorted(merged)}"
        _assert_state_dicts_close(state_dict, merged)
    finally:
        _cleanup_workspace()
