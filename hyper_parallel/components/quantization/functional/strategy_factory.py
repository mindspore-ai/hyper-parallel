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
"""Build concrete grouped-linear strategies from resolved policy data."""

from typing import Optional

from hyper_parallel.components.quantization.config import (
    LowPrecisionConfig,
    LowPrecisionDtypeScheme,
    MX_BLOCK_SIZE_32,
    MX_SCALE_TILE,
    _format_family,
)
from hyper_parallel.components.quantization.functional.base_gmm_func import (
    GroupedLinear,
)
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import (
    FakeW4A8GroupedLinear,
)
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import (
    HiFloat8GroupedLinear,
)
from hyper_parallel.components.quantization.ops.npu_hifloat8 import (
    validate_hifloat8_gmm_runtime,
)
from hyper_parallel.components.quantization.ops.npu_mxfp8 import validate_npu_gmm_runtime
from hyper_parallel.components.quantization.ops.npu_w4a8 import validate_w4a8_gmm_runtime
from hyper_parallel.components.quantization.ops.npu_fake_w4a8 import (
    validate_fake_w4a8_gmm_runtime,
)


def build_low_precision_strategy(
    config: Optional[LowPrecisionDtypeScheme | LowPrecisionConfig],
    *,
    tile_shapes: tuple[tuple[int, ...], tuple[int, ...]],
) -> GroupedLinear:
    """Pick the concrete grouped-linear strategy for one replaced module.

    ``config`` is the already-validated dtype scheme (``None`` = default native
    ``mxfp8``/``mxfp8``, i.e. W8A8). Discriminators run in priority order: the
    explicit fake-QAT flag selects the independent fake W4A8 strategy, then
    format family and the weight/activation profile select MXFP8, HiFloat8,
    or native W4A8 with the corresponding runtime probe. MX strategies also
    enforce their real-weight tile-alignment contract.
    """

    # Replacement factories historically receive the global ``LowPrecisionConfig``
    # directly.  Resolve its legacy/default catalog entry at this boundary so
    # all downstream strategy selection operates on one typed scheme.  Named
    # schemes are resolved by the plan resolver before replacement and arrive
    # as ``LowPrecisionDtypeScheme`` instances.
    if isinstance(config, LowPrecisionConfig):
        config = config.resolve_dtype_scheme()

    if config is None:
        weight_format = act_format = "mxfp8"
    else:
        weight_format = config.weight_format
        act_format = config.act_format

    family = _format_family(weight_format)

    if family == "mxfp":
        if config is not None and config.is_fake_quantize:
            if weight_format == "mxfp4" and act_format == "mxfp8":
                tile_size = config.block_size
                _validate_tile_alignment("fake_w4a8", tile_size, tile_shapes)
                validate_fake_w4a8_gmm_runtime(tile_size)
                return FakeW4A8GroupedLinear(tile_size)
            raise NotImplementedError(
                "fake QAT currently supports only mxfp4/mxfp8 (fake W4A8); "
                f"got weight_format={weight_format!r}/act_format={act_format!r}."
            )
        if weight_format == "mxfp8" and act_format == "mxfp8":
            validate_npu_gmm_runtime()
            strategy = MXFP8GroupedLinear()
            _validate_tile_alignment("w8a8", MX_BLOCK_SIZE_32, tile_shapes)
        elif weight_format == "mxfp4" and act_format == "mxfp8":
            tile_size = config.block_size if config is not None else MX_BLOCK_SIZE_32
            validate_w4a8_gmm_runtime()
            strategy = W4A8GroupedLinear(tile_size)
            _validate_tile_alignment("w4a8", tile_size, tile_shapes)
        else:
            raise NotImplementedError(
                "mxfp combination weight_format="
                f"{weight_format!r}/act_format={act_format!r} is not "
                "implemented; supported: mxfp8/mxfp8 (w8a8), mxfp4/mxfp8 (w4a8)."
            )
    elif family == "hifloat":
        if config is not None and config.is_fake_quantize:
            raise NotImplementedError(
                "fake HiFloat QAT is not implemented; "
                f"got weight_format={weight_format!r}/act_format={act_format!r}."
            )
        if weight_format == "hif8" and act_format == "hif8":
            validate_hifloat8_gmm_runtime()
            return HiFloat8GroupedLinear()
        raise NotImplementedError(
            "hifloat combination weight_format="
            f"{weight_format!r}/act_format={act_format!r} is not implemented; "
            "supported: hif8/hif8."
        )
    else:
        raise NotImplementedError(
            "grouped-linear format family is not implemented: "
            f"weight_format={weight_format!r}/act_format={act_format!r}."
        )
    return strategy


def _validate_tile_alignment(
    strategy_name: str,
    tile_size: int,
    tile_shapes: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    """Require both non-expert weight dims to be whole blocks of ``tile_size``.

    Live ``[E, O, K]`` weights must have their ``-2``/``-1`` dims divisible by
    the block size (``MX_BLOCK_SIZE_32`` for both MXFP8 and W4A8), so a
    misaligned model fails at replacement time instead of leaving a partial
    block for the NPU operator. Block-32 fake W4A8 also needs one matrix axis
    divisible by 64 so the CANN converter can use it as its row axis.
    """

    if any(dimension % tile_size for shape in tile_shapes for dimension in shape[-2:]):
        raise ValueError(
            f"{strategy_name!r} is not aligned for block_size={tile_size}: "
            f"{tile_shapes}."
        )
    if strategy_name == "fake_w4a8":
        unsupported = tuple(
            shape
            for shape in tile_shapes
            if all(dimension % MX_SCALE_TILE for dimension in shape[-2:])
        )
        if unsupported:
            raise ValueError(
                "'fake_w4a8' block_size=32 requires at least one matrix "
                "dimension divisible by 64 for CANN's packed-MXFP4 "
                f"to block-MXFP8 conversion; unsupported shapes: {unsupported}."
            )


__all__ = ["build_low_precision_strategy"]
