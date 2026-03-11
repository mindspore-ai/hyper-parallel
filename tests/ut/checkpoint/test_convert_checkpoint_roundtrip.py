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
"""UT: DCP layout round-trip for in-memory full weights (platform-agnostic APIs only).

Validates :func:`full_state_dict_to_dcp_format` and :func:`dcp_to_full_state_dict` only.
Hugging Face I/O, ``convert_full_checkpoint_to_dcp``, and file-based checkpoints are
covered in ST (see ``tests/torch/checkpoint/test_offline_convert_st_torch.py`` and
``tests/mindspore/st/checkpoint/test_offline_convert_st_mindspore.py``).

Uses PyTorch tensors + ``HYPER_PARALLEL_PLATFORM=torch`` as the minimal tensor backend for
this gate (``pytest tests/ut``); MindSpore-specific paths are in ST.
"""
# pylint: disable=wrong-import-position
import os
import shutil
from typing import Iterator

import pytest
import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import hyper_parallel.platform.platform as _platform_mod
from hyper_parallel.core.distributed_checkpoint.offline_transform import (
    dcp_to_full_state_dict,
    full_state_dict_to_dcp_format,
)

_WORKSPACE = os.path.join(os.path.dirname(__file__), "_dcp_roundtrip_ut_workspace")


def _cleanup_workspace() -> None:
    if os.path.isdir(_WORKSPACE):
        shutil.rmtree(_WORKSPACE, ignore_errors=True)


def _ensure_clean_dcp_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def _assert_flat_state_dicts_close_torch(original: dict, loaded: dict) -> None:
    assert set(original.keys()) == set(loaded.keys())
    for key in original:
        a, b = original[key], loaded[key]
        assert torch.is_tensor(a) and torch.is_tensor(b), (key, type(a), type(b))
        assert a.shape == b.shape
        assert torch.allclose(a.cpu().float(), b.cpu().float())


@pytest.fixture(autouse=True)
def _reset_cached_platform_singleton() -> Iterator[None]:
    # Other test modules may set HYPER_PARALLEL_PLATFORM during collection; force torch for this UT.
    os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
    _platform_mod.platform = None
    yield
    _platform_mod.platform = None


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
        _assert_flat_state_dicts_close_torch(state_dict, merged)
    finally:
        _cleanup_workspace()
