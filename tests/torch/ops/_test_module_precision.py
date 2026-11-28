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
"""Single-NPU module native/backend acceptance, including real VJPs and updates."""

import pytest
import torch

from tests.common.op_module_precision import (
    ModulePrecisionCase, check_module_case, module_precision_cases, position_precision_cases,
)
from tests.common.op_precision import PrecisionCase, check_case
from tests.torch.ops._test_backend_precision import _device, _require_implementation

_BACKENDS = {
    "rms_norm": ("torch_npu",), "swiglu": ("torch_npu",), "grouped_matmul": ("torch_npu",),
    "moe_token_permute": ("torch_npu",), "moe_token_unpermute": ("torch_npu",),
    "mhc_post": ("cann", "omni"), "sinkhorn": ("omni",),
    "moe_router": ("megagate",),
}


def _matrix():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for case in module_precision_cases(dtype, hardware=True):
            if case.name == "mhc_post" and dtype == torch.float32:
                continue
            if case.name == "mega_gate" and dtype != torch.bfloat16:
                continue
            if "empty_True" in case.name:
                continue
            for family in dict(case.variants):
                for backend in _BACKENDS[family]:
                    for recompute in (False, True):
                        yield pytest.param(case, family, backend, recompute,
                                           id=f"{case.name}/{dtype}/{family}/{backend}/recompute_{recompute}")


@pytest.mark.parametrize("case,family,backend,recompute", list(_matrix()))
def test_module_backend_precision(
    case: ModulePrecisionCase, family: str, backend: str, recompute: bool,
) -> None:
    """Switch one operator family after the complete native backward and assert actual use."""
    device = _device("npu")
    for owner, variant in case.variants:
        if owner == family:
            _require_implementation(family, variant, backend, "npu", case.args[0].to(device))
    selections = {owner: "native" for owner, _ in case.variants}
    selections[family] = backend
    check_module_case(case, selections=selections, device=device, recompute=recompute)


@pytest.mark.parametrize("case", [pytest.param(case, id=f"{case.name}/{dtype}")
                                 for dtype in (torch.float32, torch.float16, torch.bfloat16)
                                 for case in position_precision_cases(dtype)])
def test_module_position_precision(case: PrecisionCase) -> None:
    """Require the actual GQA/MLA adapters to execute fused RoPE and match native VJPs."""
    device = _device("npu")
    _require_implementation("rope", case.variant, "torch_npu", "npu", case.args[0].to(device))
    check_case(case, backend="torch_npu", device=device)
