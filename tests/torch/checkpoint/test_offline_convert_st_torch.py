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
"""ST (PyTorch): offline checkpoint — HF safetensors, ``convert_full_checkpoint_to_dcp``, .pt → DCP.

DCP full-weights ↔ disk round-trip without HF/file loaders is covered in ``tests/ut/core/distributed_checkpoint/``.

Module basename is distinct from the MindSpore ST file so pytest can collect both in one session.
"""
# pylint: disable=wrong-import-position
import os
import shutil

import pytest
import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import hyper_parallel.platform.platform as _platform_mod
from hyper_parallel.core.distributed_checkpoint.offline_transform import (
    convert_full_checkpoint_to_dcp,
    dcp_to_full_state_dict,
    full_state_dict_to_dcp_format,
    parse_checkpoint_from_huggingface,
    save_state_dict_as_huggingface_format,
)

_platform_mod.platform = None
_WORKSPACE = os.path.join(os.path.dirname(__file__), "_torch_offline_convert_checkpoint_workspace")


def _cleanup_workspace() -> None:
    if os.path.isdir(_WORKSPACE):
        shutil.rmtree(_WORKSPACE, ignore_errors=True)


def _subdir(name: str) -> str:
    return os.path.join(_WORKSPACE, name)


def _ensure_clean_subdir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def _count_safetensors_files(directory: str) -> int:
    return sum(1 for n in os.listdir(directory) if n.endswith(".safetensors"))


def _assert_flat_state_dicts_close_torch(original: dict, loaded: dict) -> None:
    assert set(original.keys()) == set(loaded.keys())
    for key in original:
        a, b = original[key], loaded[key]
        assert torch.is_tensor(a) and torch.is_tensor(b), (key, type(a), type(b))
        assert a.shape == b.shape
        assert torch.allclose(a.cpu().float(), b.cpu().float())


@pytest.fixture(autouse=True)
def _force_torch_platform():
    os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
    _platform_mod.platform = None
    yield


def test_offline_convert_checkpoint_roundtrip_suite_torch():
    """HF single+sharded (artifact checks), DCP round-trip, HF→DCP, .pt→DCP."""
    _cleanup_workspace()
    os.makedirs(_WORKSPACE, exist_ok=True)
    try:
        hf_single = _subdir("hf_single")
        _ensure_clean_subdir(hf_single)
        sd_single = {"layer.weight": torch.randn(4, 8), "bias": torch.randn(8)}
        save_state_dict_as_huggingface_format(hf_single, sd_single, max_shard_size="1GB")
        assert _count_safetensors_files(hf_single) == 1
        assert os.path.isfile(os.path.join(hf_single, "model.safetensors"))
        assert not os.path.isfile(os.path.join(hf_single, "model.safetensors.index.json"))
        out_single = parse_checkpoint_from_huggingface(hf_single)
        _assert_flat_state_dicts_close_torch(sd_single, out_single)

        hf_shard = _subdir("hf_shard")
        _ensure_clean_subdir(hf_shard)
        a = torch.randn(8, 8)
        b = torch.randn(8, 8)
        sd_shard = {"a": a, "b": b}
        cap = max(a.numel() * a.element_size(), b.numel() * b.element_size()) - 1
        save_state_dict_as_huggingface_format(hf_shard, sd_shard, max_shard_size=cap)
        n_st = _count_safetensors_files(hf_shard)
        assert n_st >= 2
        assert os.path.isfile(os.path.join(hf_shard, "model.safetensors.index.json"))
        out_shard = parse_checkpoint_from_huggingface(hf_shard)
        _assert_flat_state_dicts_close_torch(sd_shard, out_shard)

        dcp_direct = _subdir("dcp_direct")
        _ensure_clean_subdir(dcp_direct)
        sd_dcp = {"w": torch.randn(3, 5), "b": torch.randn(5)}
        full_state_dict_to_dcp_format(sd_dcp, dcp_direct)
        assert os.path.isfile(os.path.join(dcp_direct, ".metadata"))
        merged_dcp = dcp_to_full_state_dict(dcp_direct)
        _assert_flat_state_dicts_close_torch(sd_dcp, merged_dcp)

        hf_for_dcp = _subdir("hf_for_dcp")
        dcp_from_hf = _subdir("dcp_from_hf")
        _ensure_clean_subdir(hf_for_dcp)
        _ensure_clean_subdir(dcp_from_hf)
        sd_hf = {"p": torch.randn(6, 6)}
        save_state_dict_as_huggingface_format(hf_for_dcp, sd_hf, max_shard_size="1GB")
        assert _count_safetensors_files(hf_for_dcp) == 1
        expected = parse_checkpoint_from_huggingface(hf_for_dcp)
        convert_full_checkpoint_to_dcp(hf_for_dcp, dcp_from_hf, src_platform="huggingface")
        assert os.path.isfile(os.path.join(dcp_from_hf, ".metadata"))
        merged_hf = dcp_to_full_state_dict(dcp_from_hf)
        _assert_flat_state_dicts_close_torch(expected, merged_hf)

        pt_path = _subdir("full_checkpoint.pt")
        dcp_from_pt = _subdir("dcp_from_pt")
        sd_pt = {"x": torch.randn(2, 2)}
        torch.save(sd_pt, pt_path)
        _ensure_clean_subdir(dcp_from_pt)
        convert_full_checkpoint_to_dcp(pt_path, dcp_from_pt, src_platform="torch")
        assert os.path.isfile(os.path.join(dcp_from_pt, ".metadata"))
        merged_pt = dcp_to_full_state_dict(dcp_from_pt)
        _assert_flat_state_dicts_close_torch(sd_pt, merged_pt)
    finally:
        _cleanup_workspace()
