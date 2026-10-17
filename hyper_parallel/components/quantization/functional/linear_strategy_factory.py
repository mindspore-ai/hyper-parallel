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
"""Build an independent Dense Linear strategy from resolved policy data."""

from typing import Optional

from hyper_parallel.components.quantization.config import (
    LowPrecisionConfig,
    LowPrecisionDtypeScheme,
)
from hyper_parallel.components.quantization.functional.base_linear_func import (
    LinearStrategy,
)
from hyper_parallel.components.quantization.functional.hifloat8_linear_func import (
    HiFloat8LinearStrategy,
)
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import (
    MXFP8LinearStrategy,
)
from hyper_parallel.components.quantization.ops.npu_hifloat8 import (
    validate_hifloat8_runtime,
)
from hyper_parallel.components.quantization.ops.npu_mxfp8 import validate_npu_runtime


def build_linear_strategy(
    config: Optional[LowPrecisionDtypeScheme | LowPrecisionConfig],
    *,
    in_features: int,
    out_features: int,
) -> LinearStrategy:
    """Build one Dense strategy without consulting grouped-expert state.

    Args:
        config: Resolved dtype scheme, legacy global config, or ``None`` for
            the legacy MXFP8 default.
        in_features: Dense input feature count.
        out_features: Dense output feature count.

    Returns:
        A new strategy instance owned by one replacement Linear.

    Raises:
        ValueError: If the MXFP8 weight shape is not tile aligned.
        NotImplementedError: If the selected Dense policy is not implemented.
    """

    if isinstance(config, LowPrecisionConfig):
        config = config.resolve_dtype_scheme()
    if config is None:
        config = LowPrecisionDtypeScheme()

    if (
        not config.is_fake_quantize
        and config.weight_format == "mxfp8"
        and config.act_format == "mxfp8"
    ):
        if in_features % 32 or out_features % 32:
            raise ValueError(
                "MXFP8 Linear is not tile aligned: "
                f"shape=({out_features}, {in_features}) requires multiples of 32."
            )
        validate_npu_runtime()
        return MXFP8LinearStrategy()

    if (
        not config.is_fake_quantize
        and config.weight_format == "hif8"
        and config.act_format == "hif8"
    ):
        validate_hifloat8_runtime()
        return HiFloat8LinearStrategy()

    raise NotImplementedError(
        "Dense Linear policy is not implemented: "
        f"fake={config.is_fake_quantize}, weight_format={config.weight_format!r}, "
        f"act_format={config.act_format!r}."
    )


__all__ = ["build_linear_strategy"]
