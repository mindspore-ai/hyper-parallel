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
"""Distributed implementation for npu_mhc_pre_cmhc operator."""
from typing import Any, Dict, Tuple

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
from .parallel_ops import DistributedOp

platform = get_platform()

# Validation rules table for npu_mhc_pre_cmhc
# Key: tensor_map length (format identifier)
# Value: validation rules for that format
_MHC_PRE_CMHC_VALIDATION_RULES: Dict[int, Dict[str, Any]] = {
    4: {
        "op_name": "npu_mhc_pre_cmhc",
        "forbidden_dims": {2: "N", 3: "C"},
        "phi_forbidden_dims": {0: "dim0", 1: "dim1"},
        "alpha_forbidden_dims": {0: "dim0"},
        "bias_forbidden_dims": {0: "dim0"},
    },
    3: {
        "op_name": "npu_mhc_pre_cmhc",
        "forbidden_dims": {1: "N", 2: "C"},
        "phi_forbidden_dims": {0: "dim0", 1: "dim1"},
        "alpha_forbidden_dims": {0: "dim0"},
        "bias_forbidden_dims": {0: "dim0"},
    },
}


def _create_output_layout(mesh: Any, tensor_map: tuple) -> Layout:
    """Create an output layout with placements derived from ``tensor_map``."""
    output_layout = Layout.from_device_mesh(mesh)
    output_layout.set_tensor_map(tensor_map)
    output_layout.tensor_map_to_placement()
    return output_layout


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
                f"For {op_name}, {dim_name} dimension (dim {dim_idx}) "
                f"should be replicated, but got {dim_value}"
            )


def _validate_input_layouts_mhc_pre_cmhc(
        x_layout: Layout,
        phi_layout: Layout,
        alpha_layout: Layout,
        bias_layout: Layout,
        perm_mats_layout: Layout,
) -> None:
    """Validate input layouts for npu_mhc_pre_cmhc operator."""
    x_tm = x_layout.tensor_map
    x_tm_len = len(x_tm)

    rules = _MHC_PRE_CMHC_VALIDATION_RULES.get(x_tm_len)
    if rules is None:
        raise ValueError(
            f"For npu_mhc_pre_cmhc, tensor_map length should be 4 or 3, "
            f"but got {x_tm_len}"
        )

    _validate_tensor_map_dims(x_tm, rules["op_name"], rules["forbidden_dims"])
    _validate_tensor_map_dims(phi_layout.tensor_map, rules["op_name"], rules["phi_forbidden_dims"])
    _validate_tensor_map_dims(alpha_layout.tensor_map, rules["op_name"], rules["alpha_forbidden_dims"])
    _validate_tensor_map_dims(bias_layout.tensor_map, rules["op_name"], rules["bias_forbidden_dims"])
    # perm_mats [n!,n,n] must be fully replicated
    _validate_tensor_map_dims(
        perm_mats_layout.tensor_map, rules["op_name"],
        {0: "dim0", 1: "dim1", 2: "dim2"},
    )


def _normalize_mhc_pre_cmhc_args(
        x,
        phi,
        alpha,
        bias,
        perm_mats,
        gamma=None,
        hc_eps=1e-6,
        norm_eps=1e-6):
    """Normalize npu_mhc_pre_cmhc arguments.

    gamma is kept in the normalized positional args so preprocess can forward
    it to DFunction.forward (which needs gamma positionally for the .so aclnn
    call). The kernel ignores gamma values ((void)gamma); the wrapper
    auto-builds ones when gamma is absent.
    """
    return (
        x, phi, alpha, bias, perm_mats,
        gamma, hc_eps, norm_eps,
    ), {}


class NpuMhcPreCmhcDistributedOp(DistributedOp):
    """DistributedOp for npu_mhc_pre_cmhc operator.

    The CMHC variant has 5 tensor inputs (x, phi, alpha, bias, perm_mats)
    where x is sharded on batch dims and the rest are replicated.
    Outputs 7 tensors (h_in, h_post, h_res, inv_rms, h_mix, h_pre, coeff).
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        norm_args, _ = _normalize_mhc_pre_cmhc_args(*args, **kwargs)
        dtensor_x = norm_args[0]
        dtensor_phi = norm_args[1]
        dtensor_alpha = norm_args[2]
        dtensor_bias = norm_args[3]
        dtensor_perm_mats = norm_args[4]

        if platform.platform_type == PlatformType.MINDSPORE:
            local_args = (
                dtensor_x.to_local(),
                dtensor_phi.to_local(),
                dtensor_alpha.to_local(),
                dtensor_bias.to_local(),
                dtensor_perm_mats.to_local(),
                norm_args[5],
                norm_args[6],
                norm_args[7],
            )
            local_kwargs = {}
        else:
            local_args = (
                dtensor_x.to_local(),
                dtensor_phi.to_local(),
                dtensor_alpha.to_local(),
                dtensor_bias.to_local(),
                dtensor_perm_mats.to_local(),
            )
            local_kwargs = {
                'gamma': norm_args[5],
                'hc_eps': norm_args[6],
                'norm_eps': norm_args[7],
            }

        cache_values = [
            dtensor_x.layout,
            dtensor_phi.layout,
            dtensor_alpha.layout,
            dtensor_bias.layout,
            dtensor_perm_mats.layout,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        x_layout, phi_layout, alpha_layout, bias_layout, perm_mats_layout = cache_values

        self._check_partial_inputs([x_layout, phi_layout, alpha_layout, bias_layout, perm_mats_layout])
        _validate_input_layouts_mhc_pre_cmhc(
            x_layout, phi_layout, alpha_layout, bias_layout, perm_mats_layout
        )

        # Build per-output tensor_maps — outputs have varying ranks
        # (inv_rms is one rank lower than the others), so blindly copying
        # x's tensor_map would break get_global_shape.
        x_tm = x_layout.tensor_map
        mesh = x_layout.mesh
        if len(x_tm) == 4:
            b, s, _, c = x_tm
            maps = [
                (b, s, c),      # h_in: [B,S,D]
                (b, s, -1),     # h_post: [B,S,N]
                (b, s, -1),     # h_res: [B,S,N²]
                (b, s),         # inv_rms: [B,S]
                (b, s, -1),     # h_mix: [B,S,n!+2n]
                (b, s, -1),     # h_pre: [B,S,N]
                (b, s, -1),     # coeff: [B,S,n!]
            ]
        elif len(x_tm) == 3:
            t, _, c = x_tm
            maps = [
                (t, c),         # h_in: [T,D]
                (t, -1),        # h_post: [T,N]
                (t, -1),        # h_res: [T,N²]
                (t,),           # inv_rms: [T]
                (t, -1),        # h_mix: [T,n!+2n]
                (t, -1),        # h_pre: [T,N]
                (t, -1),        # coeff: [T,n!]
            ]
        else:
            raise ValueError(
                f"For npu_mhc_pre_cmhc, tensor_map length should be "
                f"4 or 3, but got {len(x_tm)}."
            )
        return tuple(_create_output_layout(mesh, tm) for tm in maps), None
