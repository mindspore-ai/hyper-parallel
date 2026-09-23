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
"""Ascend native W4A8 operator adapter."""

from functools import lru_cache
from types import ModuleType
from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

_A5_DEVICE_MARKERS = ("Ascend950", "Ascend910_95")
_SUPPORTED_BLOCK_SIZES = (32, 128)


class W4A8CapabilityError(RuntimeError):
    """Report that the active torch_npu stack lacks native W4A8 support."""


def transform_grouped_scale(
    input_tensor: torch.Tensor,
    input_scale_mxfp8: torch.Tensor,
    group_list: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Insert expert-boundary scale rows for block-128 grouped matmul."""

    if input_scale_mxfp8.ndim != 3 or group_list.ndim != 1:
        raise ValueError("Grouped MXFP8 scales require a 3D scale and 1D group list.")
    if block_size not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"Unsupported W4A8 block size {block_size}.")
    indices = torch.arange(
        group_list.shape[0],
        dtype=group_list.dtype,
        device=group_list.device,
    )
    zero_rows = group_list // (block_size // 2) + indices
    scale_repeat = input_scale_mxfp8.reshape(-1, input_scale_mxfp8.shape[1]).repeat_interleave(4, dim=0)
    scale_packed = scale_repeat.reshape(input_scale_mxfp8.shape[0] * 4, input_scale_mxfp8.shape[1], 2)
    scale_shape = input_tensor.shape[0] // (block_size // 2) + group_list.shape[0]
    all_indices = torch.arange(scale_shape, device=scale_packed.device)
    non_zero_rows = all_indices[~torch.isin(all_indices, zero_rows)]
    grouped_scale = torch.zeros(
        scale_shape,
        input_scale_mxfp8.shape[1],
        2,
        dtype=scale_packed.dtype,
        device=scale_packed.device,
    )
    if scale_packed.shape[0] < non_zero_rows.shape[0]:
        raise ValueError(
            "Grouped MX scale rows do not cover all non-zero token blocks: "
            f"required {non_zero_rows.shape[0]}, got {scale_packed.shape[0]}."
        )
    # A final partial 256-token source tile emits four physical rows even when
    # the GMM contract only consumes complete 64-token rows; discard those
    # excess rows while inserting expert-boundary zero rows.
    grouped_scale[non_zero_rows] = scale_packed[: non_zero_rows.shape[0]]
    return grouped_scale


class W4A8NpuOps:
    """Adapt the native W4A8 torch_npu operator contract."""

    def __init__(self, torch_npu_module: ModuleType) -> None:
        """Validate and retain an imported torch_npu module."""

        required = (
            "npu_dynamic_mx_quant",
            "npu_grouped_matmul",
            "npu_format_cast",
            "float4_e2m1fn_x2",
            "float8_e4m3fn",
            "float8_e8m0fnu",
        )
        missing = [name for name in required if not hasattr(torch_npu_module, name)]
        if missing:
            version = getattr(torch_npu_module, "__version__", "unknown")
            raise W4A8CapabilityError(
                "The active torch_npu stack does not provide the native W4A8 "
                f"operator contract; missing {missing}, torch_npu={version}."
            )
        npu_handle = getattr(torch_npu_module, "npu", None)
        try:
            device_name = (
                npu_handle.get_device_name()
                if npu_handle is not None and hasattr(npu_handle, "get_device_name")
                else ""
            )
        except (RuntimeError, TypeError, OSError):
            device_name = ""
        if not any(marker in device_name for marker in _A5_DEVICE_MARKERS):
            raise W4A8CapabilityError(
                "Native W4A8 is supported only on Ascend 950PR/950DT (A5), "
                f"but the active device is {device_name or 'unknown'}."
            )
        self._torch_npu = torch_npu_module

    @property
    def e8m0_dtype(self) -> torch.dtype:
        """Return the runtime E8M0 scale dtype."""

        return self._torch_npu.float8_e8m0fnu

    @property
    def activation_dtype(self) -> torch.dtype:
        """Return the runtime MXFP8 data dtype."""

        return self._torch_npu.float8_e4m3fn

    @property
    def weight_dtype(self) -> torch.dtype:
        """Return the runtime packed MXFP4 data dtype."""

        return self._torch_npu.float4_e2m1fn_x2

    def dynamic_mx_quant(
        self,
        tensor: torch.Tensor,
        *,
        axis: int,
        block_size: int,
        quant_dtype: torch.dtype,
        round_mode: Optional[str] = None,
        scale_alg: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run dynamic MX quantization for one operand direction."""

        kwargs = {"axis": axis, "dst_type": quant_dtype, "block_size": block_size}
        if round_mode is not None:
            kwargs["round_mode"] = round_mode
        if scale_alg is not None:
            kwargs["scale_alg"] = scale_alg
        return self._torch_npu.npu_dynamic_mx_quant(tensor, **kwargs)

    def grouped_dynamic_mx_quant(
        self,
        tensor: torch.Tensor,
        group_list: torch.Tensor,
        *,
        block_size: int,
        quant_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run grouped dynamic MX quantization over expert token ranges."""

        grouped_quant = getattr(
            self._torch_npu,
            "npu_grouped_dynamic_mx_quant",
            None,
        )
        if grouped_quant is None:
            raise W4A8CapabilityError(
                "The active torch_npu stack does not provide "
                "npu_grouped_dynamic_mx_quant, which native W4A8 block-32 "
                "backward requires."
            )
        return grouped_quant(
            tensor,
            # A5 GroupedDynamicMxQuant's groupIndex contract is int32.
            group_list.to(torch.int32),
            round_mode="rint",
            dst_type=quant_dtype,
            blocksize=block_size,
        )

    def grouped_matmul(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        right_scale: torch.Tensor,
        *,
        left_scale: torch.Tensor,
        group_list: torch.Tensor,
        group_type: int,
        output_dtype: torch.dtype,
        use_e8m0_scale: bool,
        split_item: int,
        native_w4: bool = False,
    ) -> torch.Tensor:
        """Run one native grouped matrix multiplication.

        ``native_w4`` selects the A5 ``MXFP8 x Packed MXFP4`` contract.  The
        W4A8 weight scale belongs to ``antiquant_scale``; the ordinary
        ``scale`` input is reserved for the MXFP8 path used by wgrad.
        """

        # A5 GroupedMatmul's groupList contract is int64.
        group_list = group_list.to(torch.int64)

        if native_w4:
            kwargs = {
                "bias": [],
                "scale": None,
                "antiquant_scale": [right_scale],
                "per_token_scale": [left_scale],
                "group_list": group_list,
                "group_type": group_type,
                "output_dtype": output_dtype,
                "group_list_type": 0,
                "split_item": 2,
                "x_dtype": self.activation_dtype,
                "weight_dtype": self.weight_dtype,
            }
            # Block-32 dynamic MX quantization returns raw uint8 E8M0 scales;
            # block-128 activation scales are converted to FP32 by the
            # quantizer and therefore must retain the default dtype contract.
            if use_e8m0_scale:
                kwargs["per_token_scale_dtype"] = self.e8m0_dtype
            return self._torch_npu.npu_grouped_matmul([left], [right], **kwargs)[0]

        kwargs = {
            "bias": [],
            "scale": [right_scale],
            "per_token_scale": [left_scale],
            "group_list": group_list,
            "group_type": group_type,
            "output_dtype": output_dtype,
            "group_list_type": 0,
            "split_item": split_item,
        }
        if use_e8m0_scale:
            kwargs["scale_dtype"] = self.e8m0_dtype
            kwargs["per_token_scale_dtype"] = self.e8m0_dtype
        return self._torch_npu.npu_grouped_matmul([left], [right], **kwargs)[0]

    def pack_mxfp4_weight(
        self,
        tensor: torch.Tensor,
        *,
        block_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and prepare one ``[E, N, K]`` weight for A5 W4 GMM.

        The native API returns two FP4 values per byte plus raw E8M0 scales.
        A5 grouped matmul requires the packed data in its NZ layout and the
        scale metadata transposed in the matching
        ``[E, ceil(K / 64), N, 2]`` contract (the final pair represents the
        two 32-value MX blocks covered by one packed scale tile). This method
        deliberately does not materialize an intermediate MXFP8 weight.
        """

        if tensor.ndim != 3:
            raise ValueError(
                "Native MXFP4 weight preparation expects [experts, rows, K], "
                f"got shape {tuple(tensor.shape)}."
            )
        if block_size != 32:
            raise ValueError(
                "Native Packed MXFP4 weights use block_size=32, "
                f"got {block_size}."
            )
        if tensor.shape[-1] % block_size:
            raise ValueError(
                "Native MXFP4 weight K dimension must be divisible by 32, "
                f"got shape {tuple(tensor.shape)}."
            )
        packed, scale = self.dynamic_mx_quant(
            tensor,
            axis=-1,
            block_size=block_size,
            quant_dtype=self.weight_dtype,
            round_mode="rint",
            scale_alg=0,
        )
        packed_nz = self._torch_npu.npu_format_cast(
            packed,
            29,
            customize_dtype=self.activation_dtype,
            input_dtype=self.weight_dtype,
        ).transpose(-2, -1)
        expected_scale_tiles = (tensor.shape[-1] + 63) // 64
        if (
            scale.ndim != 4
            or tuple(scale.shape[:2]) != tuple(tensor.shape[:2])
            or scale.shape[2] != expected_scale_tiles
            or scale.shape[-1] != 2
        ):
            raise ValueError(
                "Native MXFP4 scale must have [E, N, ceil(K/64), 2] layout, "
                f"got shape {tuple(scale.shape)}."
            )
        return packed_nz, scale.permute(0, 2, 1, 3)


@lru_cache(maxsize=1)
def _get_w4a8_npu_ops() -> W4A8NpuOps:
    """Return the process-local native W4A8 operator adapter."""

    try:
        import torch_npu  # pylint: disable=C0415
    except ImportError as exc:
        raise W4A8CapabilityError(
            "Native W4A8 requires torch_npu, but it is not importable."
        ) from exc
    return W4A8NpuOps(torch_npu)


def validate_w4a8_gmm_runtime() -> None:
    """Fail during model setup when native W4A8 is unavailable."""

    _get_w4a8_npu_ops()


__all__ = [
    "W4A8CapabilityError",
    "W4A8NpuOps",
    "transform_grouped_scale",
    "validate_w4a8_gmm_runtime",
]
