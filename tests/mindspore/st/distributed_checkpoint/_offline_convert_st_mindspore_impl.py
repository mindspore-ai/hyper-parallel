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
"""Impl for MindSpore offline checkpoint ST (lazy-loaded by test_offline_convert_st_mindspore).

ST (MindSpore): offline checkpoint — HF safetensors and ``convert_full_checkpoint_to_dcp``.

DCP full-weights ↔ disk round-trip without HF/file loaders is covered in ``tests/ut/core/distributed_checkpoint/``.

Module basename is distinct from the PyTorch ST file so pytest can collect both in one session.
"""
# pylint: disable=wrong-import-position
import os
import shutil

import numpy as np

import mindspore as ms
from mindspore import Tensor

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
import hyper_parallel.platform.platform as _platform_mod

_platform_mod.platform = None

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

from hyper_parallel.core.distributed_checkpoint.offline_transform import (  # noqa: E402
    convert_full_checkpoint_to_dcp,
    dcp_to_full_state_dict,
    full_state_dict_to_dcp_format,
    parse_checkpoint_from_huggingface,
    save_state_dict_as_huggingface_format,
)
from hyper_parallel.platform import get_platform  # noqa: E402

_WORKSPACE = os.path.join(os.path.dirname(__file__), "_ms_offline_convert_checkpoint_workspace")


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


def _assert_flat_state_dicts_close(original: dict, loaded: dict) -> None:
    platform = get_platform()
    assert set(original.keys()) == set(loaded.keys())
    for key in original:
        a, b = original[key], loaded[key]
        assert platform.is_tensor(a) and platform.is_tensor(b), (key, type(a), type(b))
        assert tuple(a.shape) == tuple(b.shape)
        np.testing.assert_allclose(a.asnumpy(), b.asnumpy(), rtol=1e-5, atol=1e-5)


def _random_state_dict_float32(keys_shapes: list[tuple[str, tuple[int, ...]]]) -> dict:
    rng = np.random.RandomState(42)  # pylint: disable=no-member
    out: dict = {}
    for name, shape in keys_shapes:
        out[name] = Tensor(rng.randn(*shape).astype(np.float32))
    return out


def test_offline_convert_checkpoint_roundtrip_suite_mindspore():
    """HF single+sharded (artifact checks), DCP merge, HF->DCP."""
    os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
    _platform_mod.platform = None
    _cleanup_workspace()
    os.makedirs(_WORKSPACE, exist_ok=True)
    try:
        hf_single = _subdir("hf_single")
        _ensure_clean_subdir(hf_single)
        sd_single = _random_state_dict_float32([("layer.weight", (4, 8)), ("bias", (8,))])
        save_state_dict_as_huggingface_format(hf_single, sd_single, max_shard_size="1GB")
        assert _count_safetensors_files(hf_single) == 1
        assert os.path.isfile(os.path.join(hf_single, "model.safetensors"))
        assert not os.path.isfile(os.path.join(hf_single, "model.safetensors.index.json"))
        out_single = parse_checkpoint_from_huggingface(hf_single)
        _assert_flat_state_dicts_close(sd_single, out_single)

        hf_shard = _subdir("hf_shard")
        _ensure_clean_subdir(hf_shard)
        sd_shard = _random_state_dict_float32([("a", (8, 8)), ("b", (8, 8))])
        a, b = sd_shard["a"], sd_shard["b"]
        cap = max(a.numel() * a.itemsize, b.numel() * b.itemsize) - 1
        save_state_dict_as_huggingface_format(hf_shard, sd_shard, max_shard_size=cap)
        assert _count_safetensors_files(hf_shard) >= 2
        assert os.path.isfile(os.path.join(hf_shard, "model.safetensors.index.json"))
        out_shard = parse_checkpoint_from_huggingface(hf_shard)
        _assert_flat_state_dicts_close(sd_shard, out_shard)

        dcp_direct = _subdir("dcp_direct")
        _ensure_clean_subdir(dcp_direct)
        sd_dcp = _random_state_dict_float32([("w", (3, 5)), ("b", (5,))])
        full_state_dict_to_dcp_format(sd_dcp, dcp_direct)
        assert os.path.isfile(os.path.join(dcp_direct, ".metadata"))
        merged_dcp = dcp_to_full_state_dict(dcp_direct)
        _assert_flat_state_dicts_close(sd_dcp, merged_dcp)

        hf_for_dcp = _subdir("hf_for_dcp")
        dcp_from_hf = _subdir("dcp_from_hf")
        _ensure_clean_subdir(hf_for_dcp)
        _ensure_clean_subdir(dcp_from_hf)
        sd_hf = _random_state_dict_float32([("p", (6, 6))])
        save_state_dict_as_huggingface_format(hf_for_dcp, sd_hf, max_shard_size="1GB")
        assert _count_safetensors_files(hf_for_dcp) == 1
        expected = parse_checkpoint_from_huggingface(hf_for_dcp)
        convert_full_checkpoint_to_dcp(hf_for_dcp, dcp_from_hf, src_platform="huggingface")
        assert os.path.isfile(os.path.join(dcp_from_hf, ".metadata"))
        merged_hf = dcp_to_full_state_dict(dcp_from_hf)
        _assert_flat_state_dicts_close(expected, merged_hf)
    finally:
        _cleanup_workspace()
