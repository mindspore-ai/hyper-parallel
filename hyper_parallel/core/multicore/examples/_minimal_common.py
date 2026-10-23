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
"""Shared command-line and comparison helpers for independent single-component examples."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

try:
    import torch_npu  # Registers NPU device support when the optional backend is installed.
except ImportError:
    torch_npu = None


def parser(description: str) -> argparse.ArgumentParser:
    """Build common input-shape options for a single-component demonstration.

    Args:
        description: Command-line description of the component example.
    """
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--device", choices=("cpu", "npu"), default="npu")
    result.add_argument("--tokens", type=int, default=128)
    result.add_argument("--hidden-size", type=int, default=512)
    return result


def device_for(name: str) -> torch.device:
    """Select one device without initializing a distributed group or other Mega components.

    Args:
        name: Requested device type, or the name of the value being compared.
    """
    if name == "cpu":
        return torch.device("cpu")
    if torch_npu is None:
        raise ImportError("NPU examples require the matching torch_npu installation")
    torch.npu.set_device(0)
    return torch.device("npu", 0)


def compare(actual: torch.Tensor, expected: torch.Tensor, name: str) -> float:
    """Check one output or gradient and report the observed maximum absolute error.

    Args:
        actual: Observed output or gradient.
        expected: Independent reference output or gradient.
        name: Requested device type, or the name of the value being compared.
    """
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-3,
                               msg=lambda message: f"{name}: {message}")
    return (actual.float() - expected.float()).abs().max().item() if actual.numel() else 0.0


def report(component: str, errors: dict[str, float], **details: Any) -> dict[str, Any]:
    """Emit one compact result after all comparisons have succeeded.

    Args:
        component: Component name written into the result.
        errors: Maximum absolute errors by checked value.
    """
    result = {"status": "passed", "component": component, "max_abs_error": errors, **details}
    print(json.dumps(result), flush=True)
    return result
