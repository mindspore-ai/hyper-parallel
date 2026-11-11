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
"""Generated fixed-order Ascend Triton tree, one launch for all levels."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile

import torch
import triton


_TEMPLATE = '''import triton
import triton.language as tl

@triton.jit(do_not_specialize=["size", "stride"])
def kernel({arguments}, output, size, stride, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < size
{body}
    tl.store(output + offsets, value0, mask=mask)
'''

def _emit(low: int, high: int, depth: int, packed: bool, lines: list[str]) -> None:
    if high - low == 1:
        pointer = f"partials + {low} * stride.to(tl.int64)" if packed else f"input{low}"
        lines.append(f"    value{depth} = tl.load({pointer} + offsets, mask=mask, other=0.0).to(tl.float32)")
        return
    middle = (low + high) // 2
    _emit(low, middle, depth, packed, lines)
    _emit(middle, high, depth + 1, packed, lines)
    lines.append(f"    value{depth} = value{depth} + value{depth + 1}")


@lru_cache(maxsize=None)
def _kernel(count: int, packed: bool):
    """Reduce each output tile with the fixed global binary tree."""
    if count <= 0 or count & (count - 1):
        raise ValueError("the Triton tree requires a power-of-two leaf count")
    lines: list[str] = []
    _emit(0, count, 0, packed, lines)
    arguments = "partials" if packed else ", ".join(f"input{index}" for index in range(count))
    source = _TEMPLATE.format(arguments=arguments, body="\n".join(lines))
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    name = f"pik_tree_{count}_{'packed' if packed else 'pointers'}_{digest}"
    directory = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache")) / "pik_tree_source"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    # Triton inspects a real source file; atomic installation also permits many
    # ranks to compile the same immutable source concurrently.
    if not path.exists() or path.read_text(encoding="utf-8") != source:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(source)
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel


def reduce_tree(partials: torch.Tensor) -> torch.Tensor:
    """Accumulate contiguous [leaves, rows, columns] into an FP32 root."""
    output = torch.empty(partials.shape[1:], dtype=torch.float32, device=partials.device)
    block = 1024
    grid = (triton.cdiv(output.numel(), block), 1, 1)
    _kernel(partials.shape[0], True)[grid](
        partials, output, output.numel(), partials.stride(0), BLOCK=block, enable_fp_fusion=False,
    )
    return output
