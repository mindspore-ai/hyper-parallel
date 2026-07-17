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
"""Debug logging helpers for OpDispatcher dispatch enter/exit tracing."""
import logging
from typing import Any

from hyper_parallel.platform import get_platform

logger = logging.getLogger(__name__)
_Tensor = get_platform().Tensor


def log_dispatch_enter(op_name: str, args: tuple, kwargs: dict) -> None:
    """Log debug information before dispatching an op. Caller must guard with isEnabledFor."""
    from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415
    dtensor_args_summary = []
    for i, a in enumerate(args):
        if isinstance(a, DTensor):
            dtensor_args_summary.append(
                f"args[{i}]: DTensor(shape={tuple(a.shape)}, "
                f"placements={tuple(a.placements)}, "
                f"mesh_shape={tuple(a.device_mesh.shape)})"
            )
        elif isinstance(a, _Tensor):
            dtensor_args_summary.append(
                f"args[{i}]: Tensor(shape={tuple(a.shape)}, dtype={a.dtype})"
            )
    logger.debug(
        "dispatch enter: op=%s, num_args=%d, num_kwargs=%d%s",
        op_name, len(args), len(kwargs),
        (", " + ", ".join(dtensor_args_summary)) if dtensor_args_summary else "",
    )


def log_dispatch_exit(op_name: str, result: Any) -> None:
    """Log debug information after dispatching an op. Caller must guard with isEnabledFor."""
    from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415
    result_summary = ""
    if isinstance(result, DTensor):
        result_summary = (
            f", result: DTensor(shape={tuple(result.shape)}, "
            f"placements={tuple(result.placements)}, "
            f"mesh_shape={tuple(result.device_mesh.shape)})"
        )
    elif isinstance(result, _Tensor):
        result_summary = (
            f", result: Tensor(shape={tuple(result.shape)}, "
            f"dtype={result.dtype})"
        )
    elif isinstance(result, (tuple, list)):
        type_name = type(result).__name__
        result_summary = f", result: {type_name}(len={len(result)})"
    logger.debug("dispatch exit: op=%s%s", op_name, result_summary)
