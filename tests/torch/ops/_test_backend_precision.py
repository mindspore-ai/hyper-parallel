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

import importlib
import importlib.util
import os

import pytest
import torch

from hyper_parallel.components.ops.catalog import builtin_registry
from hyper_parallel.components.ops._dependency import require_package
from hyper_parallel.components.ops.registry import ImplementationUnavailable
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases, registered_backends

_BACKENDS = [entry for entry in registered_backends() if entry[2] != "native" and entry[3] in ("cuda", "npu")]
_OPTIONAL_PACKAGES = {
    "omni": ("omni_training_custom_ops",),
    "cann": ("cann_ops_transformer",),
    "megagate": ("hyper_multicore",),
    "triton_ascend": ("triton.backends.ascend", "fla"),
    "ascendc": ("fla_npu",),
}


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


@pytest.mark.parametrize("family,variant,backend,device_type", _BACKENDS,
                         ids=["/".join(entry) for entry in _BACKENDS])
def test_backend_precision(family: str, variant: str, backend: str, device_type: str) -> None:
    """Require actual backend selection, compare every output and all argument/parameter VJPs."""
    device = _device(device_type)
    dtypes = (torch.bfloat16,)
    if family in ("rms_norm", "swiglu", "rope", "attention"):
        dtypes = (torch.float32, torch.float16, torch.bfloat16)
    for dtype in dtypes:
        cases = [case for case in precision_cases(family, dtype, hardware=True) if case.variant == variant]
        if not cases:
            raise AssertionError(f"No real-backend fixture for {family}.{variant}/{backend}/{dtype}")
        first_tensor = next(arg for arg in cases[0].args if isinstance(arg, torch.Tensor))
        _require_implementation(family, variant, backend, device_type, first_tensor.to(device))
        for case in cases:
            check_case(case, backend=backend, device=device)
