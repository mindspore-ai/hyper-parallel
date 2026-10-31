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
"""Shape-preserving mHC stubs for CPU tests of other Xing40 behavior."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch


def _pre(module: Any, streams: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Select one stream and return unused coefficient placeholders."""
    count = module.num_stream
    output = streams.unflatten(-1, (count, -1))[..., 0, :]
    post = streams.new_zeros((*streams.shape[:-1], count))
    mix = streams.new_zeros((*streams.shape[:-1], count, count))
    return output, post, mix, streams


def _post(module: Any, output: torch.Tensor, residual: torch.Tensor,
          _post_scale: torch.Tensor, _mix: torch.Tensor) -> torch.Tensor:
    """Keep a differentiable path through each stream without computing mHC."""
    streams = residual.unflatten(-1, (module.num_stream, -1))
    return (streams + output.unsqueeze(-2)).flatten(-2)


def _head(_module: Any, streams: torch.Tensor) -> torch.Tensor:
    """Use a stream directly; no mHC head computation is performed."""
    return streams[..., 0, :]


def is_mhc_parameter(name: str) -> bool:
    """Identify parameters deliberately disconnected by the test stubs."""
    return any(part in {"attn_hc_pre", "ffn_hc_pre", "hc_head"} for part in name.split('.'))


@contextmanager
def stub_mhc_for_cpu() -> Iterator[None]:
    """Isolate mHC while exercising unrelated Xing40 CPU functionality."""
    model = "hyper_parallel.models.xing40.modeling_xing40"
    with (patch(f"{model}.Xing40MhcPre.forward", new=_pre),
          patch(f"{model}.Xing40MhcPost.forward", new=_post),
          patch(f"{model}.DeepseekV4HyperHead.forward", new=_head)):
        yield
