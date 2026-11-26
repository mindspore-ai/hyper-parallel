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
"""Real-kernel single-device precision worker; absent hardware/packages report skips.

Run directly with ``pytest -rs tests/torch/ops/_test_backend_precision.py`` to
see per-backend availability. Missing operator symbols in an installed package,
kernel runtime errors and numerical failures remain failures.
"""

from __future__ import annotations

import importlib
import importlib.util
import os

import pytest
import torch

from hyper_parallel.components.ops.catalog import builtin_registry
from hyper_parallel.components.ops._dependency import require_package
from hyper_parallel.components.ops.registry import ImplementationUnavailable
from tests.common.op_precision import PrecisionCase, check_case
from tests.common.op_precision_suite import (
    FAMILY_BUILDERS, backend_precision_dtypes, precision_cases, registered_backends,
)

_BACKENDS = [entry for entry in registered_backends() if entry[2] != "native" and entry[3] in ("cuda", "npu")]
_OPTIONAL_PACKAGES = {
    "tile_kernels": ("tile_kernels",),
    "omni": ("omni_training_custom_ops",),
    "megagate": ("hyper_multicore",),
    "triton_ascend": ("triton.backends.ascend", "fla"),
    "ascendc": ("fla_npu",),
}


@pytest.fixture(autouse=True)
def _strict_npu_precision(request):
    """Disable reduced-mantissa FP32 convolution for strict native/oracle checks."""
    if (request.node.callspec.params.get("device_type", "npu") != "npu"
            or importlib.util.find_spec("torch_npu") is None):
        yield
        return
    importlib.import_module("torch_npu")
    previous = torch.npu.conv.allow_hf32
    torch.npu.conv.allow_hf32 = False
    try:
        yield
    finally:
        torch.npu.conv.allow_hf32 = previous


def _device(device_type: str) -> str:
    if device_type == "npu" and importlib.util.find_spec("torch_npu") is not None:
        importlib.import_module("torch_npu")
    api = getattr(torch, device_type, None)
    if api is None or not api.is_available():
        pytest.skip(f"{device_type} hardware is unavailable")
    index = int(os.environ.get("LOCAL_RANK", "0"))
    api.set_device(index)
    return f"{device_type}:{index}"


def _require_implementation(family, variant, backend, device_type, sample):
    # The inventory audit intentionally uses registry metadata; no public
    # introspection or dependency-probe API is needed by application code.
    """Check hardware compatibility and load the explicitly requested backend."""
    spec = builtin_registry._families[family][variant]  # pylint: disable=protected-access
    implementation = builtin_registry._implementations[(spec, backend, device_type)]  # pylint: disable=protected-access
    if implementation.check_compatibility is not None:
        reason = implementation.check_compatibility(sample)
        if reason is not None:
            pytest.skip(f"{family}.{variant}/{backend}: {reason}")
    try:
        for package in _OPTIONAL_PACKAGES.get(backend, ()):
            require_package(package)
        implementation.load()
    except ImplementationUnavailable as error:
        if str(error).startswith("missing optional package "):
            pytest.skip(f"{family}.{variant}/{backend}: {error}")
        raise


def _precision_matrix(backend_filter: str | None = None):
    # Keep each dtype and boundary independent: one skip/failure must not hide
    # another kernel invocation. DSA kernels currently accept BF16 fixtures.
    """Enumerate independent backend, dtype and boundary precision cases."""
    for family, variant, backend, device_type in _BACKENDS:
        if backend_filter is not None and backend != backend_filter:
            continue
        for dtype in backend_precision_dtypes(family, backend, variant):
            cases = [case for case in precision_cases(family, dtype, hardware=True, backend=backend)
                     if case.variant == variant]
            if not cases:
                raise AssertionError(f"No real-backend fixture for {family}.{variant}/{backend}/{dtype}")
            for case in cases:
                name = f"{family}/{variant}/{backend}/{device_type}/{dtype}/{case.name}"
                yield pytest.param(case, backend, device_type, id=name)


@pytest.mark.parametrize("case,backend,device_type", list(_precision_matrix()))
def test_backend_precision(case: PrecisionCase, backend: str, device_type: str) -> None:
    """Require actual selection and compare every output and argument/parameter VJP."""
    device = _device(device_type)
    first_tensor = next(arg for arg in case.args if isinstance(arg, torch.Tensor))
    _require_implementation(case.family, case.variant, backend, device_type, first_tensor.to(device))
    check_case(case, backend=backend, device=device)


@pytest.mark.parametrize("case,backend,device_type", list(_precision_matrix("tile_kernels")))
def test_tile_kernels_precision(case: PrecisionCase, backend: str, device_type: str) -> None:
    """Require real TileKernels forward/backward; missing prerequisites are failures."""
    importlib.import_module("torch_npu")
    api = getattr(torch, device_type)
    if not api.is_available():
        pytest.fail("TileKernels acceptance requires an available NPU")
    index = int(os.environ.get("LOCAL_RANK", "0"))
    api.set_device(index)
    device = f"{device_type}:{index}"
    check_case(case, device=device)
    check_case(case, backend=backend, device=device)
    api.synchronize()


def _native_matrix():
    for family in FAMILY_BUILDERS:
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases(family, dtype, hardware=True):
                yield pytest.param(case, id=f"{family}/{case.variant}/{dtype}/{case.name}")


@pytest.mark.parametrize("case", list(_native_matrix()))
def test_native_precision(case: PrecisionCase) -> None:
    """Validate all native contracts against independent oracles on the NPU."""
    check_case(case, device=_device("npu"))
