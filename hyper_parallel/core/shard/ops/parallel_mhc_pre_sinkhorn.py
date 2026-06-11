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
"""Distributed implementation for npu_mhc_pre_sinkhorn operator."""
from typing import Tuple, Dict, Any

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
from .parallel_ops import DistributedOp

platform = get_platform()

_HC_MULT_DEFAULT = 4
_NUM_ITERS_DEFAULT = 20
_HC_EPS_DEFAULT = 1e-6
_NORM_EPS_DEFAULT = 1e-6


def _normalize_mhc_pre_sinkhorn_args(
        x,
        phi,
        alpha,
        bias,
        hc_mult=_HC_MULT_DEFAULT,
        num_iters=_NUM_ITERS_DEFAULT,
        hc_eps=_HC_EPS_DEFAULT,
        norm_eps=_NORM_EPS_DEFAULT,
        out_flag=True):
    """Normalize positional and keyword arguments into a canonical positional tuple.

    Args:
        x: Input tensor [B,S,N,C] or [T,N,C].
        phi: mHC parameter matrix [N*N+2*N, N*C].
        alpha: mHC scaling parameters [3].
        bias: mHC bias parameters [N*N+2*N].
        hc_mult: HC dimension size (currently only 4 supported).
        num_iters: Sinkhorn iteration count.
        hc_eps: H_pre sigmoid eps parameter.
        norm_eps: RmsNorm eps parameter.
        out_flag: Whether to output intermediate gradients.

    Returns:
        tuple: (positional_args_tuple, empty_kwargs_dict)
    """
    return (
        x, phi, alpha, bias,
        hc_mult, num_iters, hc_eps, norm_eps, out_flag,
    ), {}


def _normalize_mhc_pre_clamp_sinkhorn_args(
        x,
        phi,
        alpha,
        bias,
        hc_mult=_HC_MULT_DEFAULT,
        num_iters=_NUM_ITERS_DEFAULT,
        hc_eps=_HC_EPS_DEFAULT,
        norm_eps=_NORM_EPS_DEFAULT,
        out_flag=True,
        clamp_min=0.0,
        clamp_max=0.0):
    """Normalize npu_mhc_pre_clamp_sinkhorn arguments."""
    return (
        x, phi, alpha, bias,
        hc_mult, num_iters, hc_eps, norm_eps, out_flag,
        clamp_min, clamp_max,
    ), {}


# Validation rules table for npu_mhc_pre_sinkhorn
# Key: tensor_map length (format identifier)
# Value: validation rules for that format
_MHC_PRE_SINKHORN_VALIDATION_RULES: Dict[int, Dict[str, Any]] = {
    4: {
        "op_name": "npu_mhc_pre_sinkhorn",
        "forbidden_dims": {2: "N"},
        "phi_forbidden_dims": {0: "dim0", 1: "dim1"},
        "alpha_forbidden_dims": {0: "dim0"},
        "bias_forbidden_dims": {0: "dim0"},
    },
    3: {
        "op_name": "npu_mhc_pre_sinkhorn",
        "forbidden_dims": {1: "N"},
        "phi_forbidden_dims": {0: "dim0", 1: "dim1"},
        "alpha_forbidden_dims": {0: "dim0"},
        "bias_forbidden_dims": {0: "dim0"},
    },
}


def _validate_tensor_map_dims(
        tensor_map: tuple,
        op_name: str,
        forbidden_dims: Dict[int, str],
) -> None:
    """Check that specified dimensions are not sharded (replicated).

    Args:
        tensor_map: The tensor_map to check.
        op_name: Operator name for error message.
        forbidden_dims: Dict mapping dim index to dim name.

    Raises:
        ValueError: If any forbidden dimension is sharded.
    """
    for dim_idx, dim_name in forbidden_dims.items():
        dim_value = tensor_map[dim_idx]
        if dim_value != -1:
            raise ValueError(
                f"For {op_name}, {dim_name} dimension (dim {dim_idx}) of x "
                f"should be replicated, but got {dim_value}"
            )


def _validate_input_layouts_mhc_pre_sinkhorn(
        x_layout: Layout,
        phi_layout: Layout,
        alpha_layout: Layout,
        bias_layout: Layout,
) -> None:
    """Validate input layouts for npu_mhc_pre_sinkhorn operator."""
    x_tm = x_layout.tensor_map
    x_tm_len = len(x_tm)

    rules = _MHC_PRE_SINKHORN_VALIDATION_RULES.get(x_tm_len)
    if rules is None:
        raise ValueError(
            f"For npu_mhc_pre_sinkhorn, tensor_map length should be 4 or 3, but got {x_tm_len}"
        )

    _validate_tensor_map_dims(x_tm, rules["op_name"], rules["forbidden_dims"])
    _validate_tensor_map_dims(phi_layout.tensor_map, rules["op_name"], rules["phi_forbidden_dims"])
    _validate_tensor_map_dims(alpha_layout.tensor_map, rules["op_name"], rules["alpha_forbidden_dims"])
    _validate_tensor_map_dims(bias_layout.tensor_map, rules["op_name"], rules["bias_forbidden_dims"])


class NpuMhcPreSinkhornDistributedOp(DistributedOp):
    """DistributedOp for npu_mhc_pre_sinkhorn operator.

    Implements layout inference for the MHC pre-processing with Sinkhorn operation.
    Outputs 8 tensors: hin, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        norm_args, _ = _normalize_mhc_pre_sinkhorn_args(*args, **kwargs)
        dtensor_x = norm_args[0]
        dtensor_phi = norm_args[1]
        dtensor_alpha = norm_args[2]
        dtensor_bias = norm_args[3]

        if platform.platform_type == PlatformType.MINDSPORE:
            local_args = (
                dtensor_x.to_local(),
                dtensor_phi.to_local(),
                dtensor_alpha.to_local(),
                dtensor_bias.to_local(),
                norm_args[4],
                norm_args[5],
                norm_args[6],
                norm_args[7],
                norm_args[8],
            )
            local_kwargs = {}
        else:
            local_args = (
                dtensor_x.to_local(),
                dtensor_phi.to_local(),
                dtensor_alpha.to_local(),
                dtensor_bias.to_local(),
            )
            local_kwargs = {
                'hc_mult': norm_args[4],
                'num_iters': norm_args[5],
                'hc_eps': norm_args[6],
                'norm_eps': norm_args[7],
                'out_flag': norm_args[8],
            }

        cache_values = [
            dtensor_x.layout,
            dtensor_phi.layout,
            dtensor_alpha.layout,
            dtensor_bias.layout,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        x_layout, phi_layout, alpha_layout, bias_layout = cache_values

        self._check_partial_inputs([x_layout, phi_layout, alpha_layout, bias_layout])

        _validate_input_layouts_mhc_pre_sinkhorn(
            x_layout, phi_layout, alpha_layout, bias_layout
        )

        out_layouts = self._infer_output_layouts(x_layout)
        return out_layouts, None

    @staticmethod
    def _infer_output_layouts(
            x_layout: Layout,
    ) -> Tuple[Layout, Layout, Layout, Layout, Layout, Layout, Layout, Layout]:
        out_layout = Layout.from_device_mesh(x_layout.mesh)
        out_layout.set_tensor_map(x_layout.tensor_map)
        out_layout.tensor_map_to_placement()

        return (
            out_layout, out_layout, out_layout, out_layout,
            out_layout, out_layout, out_layout, out_layout,
        )


class NpuMhcPreClampSinkhornDistributedOp(DistributedOp):
    """DistributedOp for npu_mhc_pre_clamp_sinkhorn operator.

    The clamp variant follows the same input layout rules as npu_mhc_pre_sinkhorn
    and emits one additional h_res_logits output.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        norm_args, _ = _normalize_mhc_pre_clamp_sinkhorn_args(*args, **kwargs)
        dtensor_x = norm_args[0]
        dtensor_phi = norm_args[1]
        dtensor_alpha = norm_args[2]
        dtensor_bias = norm_args[3]

        if platform.platform_type == PlatformType.MINDSPORE:
            local_args = (
                dtensor_x.to_local(),
                dtensor_phi.to_local(),
                dtensor_alpha.to_local(),
                dtensor_bias.to_local(),
                norm_args[4],
                norm_args[5],
                norm_args[6],
                norm_args[7],
                norm_args[8],
                norm_args[9],
                norm_args[10],
            )
            local_kwargs = {}
        else:
            local_args = (
                dtensor_x.to_local(),
                dtensor_phi.to_local(),
                dtensor_alpha.to_local(),
                dtensor_bias.to_local(),
            )
            local_kwargs = {
                'hc_mult': norm_args[4],
                'num_iters': norm_args[5],
                'hc_eps': norm_args[6],
                'norm_eps': norm_args[7],
                'out_flag': norm_args[8],
                'clamp_min': norm_args[9],
                'clamp_max': norm_args[10],
            }

        cache_values = [
            dtensor_x.layout,
            dtensor_phi.layout,
            dtensor_alpha.layout,
            dtensor_bias.layout,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        x_layout, phi_layout, alpha_layout, bias_layout = cache_values

        self._check_partial_inputs([x_layout, phi_layout, alpha_layout, bias_layout])
        _validate_input_layouts_mhc_pre_sinkhorn(
            x_layout, phi_layout, alpha_layout, bias_layout
        )

        out_layout = Layout.from_device_mesh(x_layout.mesh)
        out_layout.set_tensor_map(x_layout.tensor_map)
        out_layout.tensor_map_to_placement()
        return (out_layout,) * 9, None
