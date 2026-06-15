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
"""UT for :func:`_ensure_contiguous` in both MindSpore and Torch platform modules."""
import pytest

pytest.importorskip("mindspore")
pytest.importorskip("torch")

import mindspore as ms
import torch

from hyper_parallel.platform.mindspore.platform import _ensure_contiguous as _ms_ensure_contiguous
from hyper_parallel.platform.torch.platform import _ensure_contiguous as _torch_ensure_contiguous


class TestEnsureContiguousMindSpore:
    """Unit tests for the MindSpore ``_ensure_contiguous`` helper."""

    def test_contiguous_is_noop(self):
        """A contiguous tensor is returned as-is (identity)."""
        t = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ms.float32)
        out = _ms_ensure_contiguous(t)
        assert out is t
        assert out.is_contiguous()

    def test_non_contiguous_transpose(self):
        """A transposed tensor is made contiguous."""
        t = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ms.float32)
        t_t = t.T
        assert not t_t.is_contiguous()
        out = _ms_ensure_contiguous(t_t)
        assert out is not t_t
        assert out.is_contiguous()
        assert ms.ops.equal(out, t.T).all()

    def test_storage_offset_slice(self):
        """A sliced tensor with storage offset is made contiguous."""
        t = ms.Tensor([1.0, 2.0, 3.0, 4.0], dtype=ms.float32)
        t_s = t[1:]  # storage_offset != 0
        out = _ms_ensure_contiguous(t_s)
        assert out.is_contiguous()
        assert ms.ops.equal(out, ms.Tensor([2.0, 3.0, 4.0], dtype=ms.float32)).all()

    def test_scalar_is_noop(self):
        """A 0-d tensor is always contiguous."""
        t = ms.Tensor(42.0, dtype=ms.float32)
        out = _ms_ensure_contiguous(t)
        assert out is t


class TestEnsureContiguousTorch:
    """Unit tests for the Torch ``_ensure_contiguous`` helper."""

    def test_contiguous_is_noop(self):
        """A contiguous tensor is returned as-is (identity)."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = _torch_ensure_contiguous(t)
        assert out is t
        assert out.is_contiguous()

    def test_non_contiguous_transpose(self):
        """A transposed tensor is made contiguous."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        t_t = t.T
        assert not t_t.is_contiguous()
        out = _torch_ensure_contiguous(t_t)
        assert out is not t_t
        assert out.is_contiguous()
        assert torch.equal(out, t.T)

    def test_storage_offset_slice(self):
        """A sliced tensor with storage offset is made contiguous."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        t_s = t[1:]  # storage_offset != 0
        out = _torch_ensure_contiguous(t_s)
        assert out.is_contiguous()
        assert torch.equal(out, torch.tensor([2.0, 3.0, 4.0]))

    def test_scalar_is_noop(self):
        """A 0-d tensor is always contiguous."""
        t = torch.tensor(42.0)
        out = _torch_ensure_contiguous(t)
        assert out is t
