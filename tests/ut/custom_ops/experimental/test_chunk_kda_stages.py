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
"""Contracts for caller-managed KDA stages; no NPU execution required."""
# pytest intentionally injects fixtures by their function names.
# pylint: disable=redefined-outer-name
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hyper_parallel.custom_ops.experimental import chunk_kda_stages as public
from hyper_parallel.platform.torch.custom_ops import TorchCustomOps

_ROOT = Path(__file__).resolve().parents[4]
_BACKEND_PATH = _ROOT / "hyper_parallel/platform/mindspore/custom_ops/chunk_kda_stages.py"
_IMPL = "hyper_parallel.platform.mindspore.custom_ops.custom_op_impl"


class LocalTensor:
    """Minimal local Tensor contract, including observable contiguous copies."""

    def __init__(self, shape: tuple = (2, 3), data: object = None, contiguous: bool = True) -> None:
        """Set a fake shape, host payload and contiguity flag."""
        self.shape = shape
        self.data = data
        self.contiguous_flag = contiguous

    def asnumpy(self) -> np.ndarray:
        """Expose host metadata without initializing a device."""
        return np.asarray(self.data)

    def is_contiguous(self) -> bool:
        """Report whether the simulated buffer is contiguous."""
        return self.contiguous_flag

    def contiguous(self) -> "LocalTensor":
        """Return a copy only for a simulated noncontiguous input."""
        return self if self.contiguous_flag else LocalTensor(self.shape)


@pytest.fixture
def backend() -> Iterator[Tuple[object, SimpleNamespace]]:
    """Load only the adapter, avoiding extension builds and NPU initialization."""
    spec = importlib.util.spec_from_file_location("_kda_stage_adapter", _BACKEND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ms = SimpleNamespace(Tensor=LocalTensor)
    impl = SimpleNamespace(_custom_ops=SimpleNamespace())

    def indices(cu: Optional[Sequence[int]], size: int) -> Optional[list]:
        """Build the same host chunk plan as the real extension adapter."""
        if cu is None:
            return None
        return [x for seq, (begin, end) in enumerate(zip(cu, cu[1:]))
                for chunk in range((end - begin + size - 1) // size) for x in (seq, chunk)]

    impl._canonical_chunk_indices = indices
    package = SimpleNamespace(custom_op_impl=impl)
    with patch.dict("sys.modules", {_IMPL: impl,
                                   "hyper_parallel.platform.mindspore.custom_ops": package}):
        yield module, impl._custom_ops


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "NTD", "TND"])
def test_prepare_preserves_layout_and_all_flags(layout):
    """The public stage forwards layout and flags, including state-independent modes."""
    platform = MagicMock()
    with patch.object(public, "_platform", platform):
        result = public.npu_chunk_kda_fwd_prepare(
            "q", "k", "v", "g", "beta", .25, 128, layout=layout,
            a_log="a", dt_bias="bias", cu_seqlens=[0, 65], chunk_indices=[0, 0, 0, 1],
            epsilon=1e-5, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True, safe_gate=True,
            lower_bound=-4., use_exp2=False, backward_mode="recompute",
        )
    platform.custom_ops.npu_chunk_kda_fwd_prepare.assert_called_once_with(
        "q", "k", "v", "g", "beta", .25, 128, layout, "a", "bias",
        [0, 65], [0, 0, 0, 1], 1e-5, True, True, True, True, True, -4., False, "recompute",
    )
    assert result is platform.custom_ops.npu_chunk_kda_fwd_prepare.return_value


@pytest.mark.parametrize("name,args,kwargs,expected", [
    ("fwd_h", ("k", "w", "u"), {"gk": "g", "initial_state": "h0", "output_final_state": True,
                                    "state_v_first": True},
     ("k", "w", "u", "g", "h0", True, 64, None, None, True, True)),
    ("bwd_dhu", ("q", "k", "w", "do", "dv", .5), {"gk": "g", "h0": "h0", "dht": "dht"},
     ("q", "k", "w", "do", "dv", .5, 64, "g", "h0", "dht", None, None, True, False)),
    ("fwd_finalize", ("qs", "a", "vn", "h"), {"output_layout": "NTD"},
     ("qs", "a", "vn", "h", "NTD", False, None, None)),
    ("pre_process_fwd", ("k", "w", "u"), {"gk": "g", "cu_seqlens": [64, 129]},
     ("k", "w", "u", "g", [64, 129], 64)),
    ("pre_process_bwd", ("q", "k", "w", "do", "dv", .5), {"gk": "g"},
     ("q", "k", "w", "do", "dv", .5, 64, "g", None)),
    ("bwd_prepare", ("a", "vn", "do", "h"), {"scale": .5, "layout": "NTD"},
     ("a", "vn", "do", "h", .5, "NTD", 64, False, None, None)),
    ("merge_fwd_bwd", ("h", "summary", 2, 3), {"forward": False, "state_v_first": True},
     ("h", "summary", 2, 3, False, True)),
])
def test_stage_routes(name, args, kwargs, expected):
    """State boundaries, summaries and layout options reach the correct bridge."""
    platform = MagicMock()
    with patch.object(public, "_platform", platform):
        getattr(public, "npu_chunk_kda_" + name)(*args, **kwargs)
    getattr(platform.custom_ops, "npu_chunk_kda_" + name).assert_called_once_with(*expected)


def test_bwd_finalize_preserves_norm_and_beta_saves():
    """Saved normalized inputs and effective beta must reach the finalizer unchanged."""
    platform = MagicMock()
    tensors = tuple(f"tensor{i}" for i in range(15))
    with patch.object(public, "_platform", platform):
        public.npu_chunk_kda_bwd_finalize(
            *tensors, scale=.5, q_rstd="qr", k_rstd="kr", cu_seqlens=[0, 65],
            chunk_indices=[0, 0, 0, 1], use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True, output_layout="TND",
        )
    call = platform.custom_ops.npu_chunk_kda_bwd_finalize.call_args.args
    assert call[:15] == tensors
    assert call[22:] == ("qr", "kr", [0, 65], [0, 0, 0, 1], True, True, "TND")


def test_backward_metadata_does_not_overwrite_beta_flags(backend):
    """Finalize metadata precedes sigmoid flags in the native ABI."""
    adapter, extension = backend
    extension.npu_chunk_kda_bwd_finalize = MagicMock(return_value=[LocalTensor()] * 7)
    args = [LocalTensor()] * 15 + [.5, -5., 64, False, True, True, False,
                                 None, None, LocalTensor(data=[0, 65]), None, True, True, "BNSD"]
    adapter.run_chunk_kda_stage("npu_chunk_kda_bwd_finalize", *args)
    call = extension.npu_chunk_kda_bwd_finalize.call_args.args
    assert call[24:] == ([0, 65], [0, 0, 0, 1], True, True, "BNSD")


def test_packed_empty_sequence_and_tail_chunk_plan(backend):
    """Build canonical chunks per sequence, not ceil(total_tokens/chunk_size)."""
    adapter, extension = backend
    extension.npu_chunk_kda_fwd_h = MagicMock(return_value=[LocalTensor(), LocalTensor(), LocalTensor((0,))])
    result = adapter.run_chunk_kda_stage(
        "npu_chunk_kda_fwd_h", LocalTensor(), LocalTensor(), LocalTensor(), LocalTensor(),
        None, False, 64, [0, 1, 1, 66], None, True, False,
    )
    args = extension.npu_chunk_kda_fwd_h.call_args.args
    assert args[8] == [0, 0, 2, 0, 2, 1]
    assert len(result) == 3 and result[2] is None


def test_dhu_keeps_dh0_in_second_slot(backend):
    adapter, extension = backend
    dh, dh0, dv = LocalTensor(), LocalTensor(), LocalTensor()
    extension.npu_chunk_kda_bwd_dhu = MagicMock(return_value=[dh, dh0, dv])
    result = adapter.run_chunk_kda_stage(
        "npu_chunk_kda_bwd_dhu", *([LocalTensor()] * 5), .5, 64,
        LocalTensor(), LocalTensor(), LocalTensor(), None, None, True, True,
    )
    assert result == (dh, dh0, dv)


def test_missing_stage_does_not_rebuild_extension(backend):
    adapter, _ = backend
    with pytest.raises(RuntimeError, match="Rebuild.*extension"):
        adapter.run_chunk_kda_stage("npu_chunk_kda_fwd_prepare")


def test_merge_refuses_copy_of_mutable_state(backend):
    adapter, extension = backend
    extension.npu_chunk_kda_merge_fwd_bwd = MagicMock()
    with pytest.raises(ValueError, match="in place"):
        adapter.run_chunk_kda_stage("npu_chunk_kda_merge_fwd_bwd",
                                    LocalTensor(contiguous=False), LocalTensor(), 1, 1, True, False)
    extension.npu_chunk_kda_merge_fwd_bwd.assert_not_called()


def test_torch_has_explicit_stage_errors():
    for name in public.__dict__:
        if name.startswith("npu_chunk_kda_"):
            with pytest.raises(NotImplementedError, match="PyTorch"):
                getattr(TorchCustomOps, name)()
