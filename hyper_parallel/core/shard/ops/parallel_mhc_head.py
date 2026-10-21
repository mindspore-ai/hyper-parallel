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
"""Distributed implementation for npu_mhc_head operator."""
from typing import Any, Dict, Tuple

from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
from .parallel_ops import DistributedOp

platform = get_platform()

# Validation rules table for npu_mhc_head.
# Input x is SBH 3D [s, b, n*H]; the n*H channel dim may shard (aligned with
# weight's n*H shard), but the n dimension (in weight/hc_base) must replicate.
_MHC_HEAD_VALIDATION_RULES: Dict[int, Dict[str, Any]] = {
    3: {
        "op_name": "npu_mhc_head",
        # weight [n, n*H]: dim 0 (n) must replicate; dim 1 (n*H) may shard.
        "weight_forbidden_dims": {0: "n"},
        # hc_base [n]: dim 0 must replicate.
        "hc_base_forbidden_dims": {0: "n"},
        # hc_scale [1]: dim 0 must replicate.
        "hc_scale_forbidden_dims": {0: "dim0"},
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
    """Check that specified dimensions are not sharded (replicated)."""
    for dim_idx, dim_name in forbidden_dims.items():
        dim_value = tensor_map[dim_idx]
        if dim_value != -1:
            raise ValueError(
                f"For {op_name}, {dim_name} dimension (dim {dim_idx}) "
                f"should be replicated, but got {dim_value}"
            )


def _validate_input_layouts_mhc_head(
        x_layout: Layout,
        weight_layout: Layout,
        hc_base_layout: Layout,
        hc_scale_layout: Layout,
) -> None:
    """Validate input layouts for npu_mhc_head operator."""
    x_tm = x_layout.tensor_map
    x_tm_len = len(x_tm)

    rules = _MHC_HEAD_VALIDATION_RULES.get(x_tm_len)
    if rules is None:
        raise ValueError(
            f"For npu_mhc_head, tensor_map length should be 3 (SBH), "
            f"but got {x_tm_len}"
        )

    _validate_tensor_map_dims(
        weight_layout.tensor_map, rules["op_name"], rules["weight_forbidden_dims"])
    _validate_tensor_map_dims(
        hc_base_layout.tensor_map, rules["op_name"], rules["hc_base_forbidden_dims"])
    _validate_tensor_map_dims(
        hc_scale_layout.tensor_map, rules["op_name"], rules["hc_scale_forbidden_dims"])


def _normalize_mhc_head_args(
        x,
        weight,
        hc_base,
        hc_scale,
        hc_eps=1e-6,
        norm_eps=1e-6):
    """Normalize npu_mhc_head arguments into positional form for DFunction.forward."""
    return (
        x, weight, hc_base, hc_scale,
        hc_eps, norm_eps,
    ), {}


class NpuMhcHeadDistributedOp(DistributedOp):
    """DistributedOp for npu_mhc_head operator.

    4 tensor inputs (x sharded on the channel dim, weight/hc_base/hc_scale
    replicated or weight sharded on dim 1 aligned with x). 3 tensor outputs
    (out, rms_inv, mixes), all rank-3 SBH.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        norm_args, _ = _normalize_mhc_head_args(*args, **kwargs)
        dtensor_x = norm_args[0]
        dtensor_weight = norm_args[1]
        dtensor_hc_base = norm_args[2]
        dtensor_hc_scale = norm_args[3]

        if platform.platform_type == PlatformType.MINDSPORE:
            local_args = (
                dtensor_x.to_local(),
                dtensor_weight.to_local(),
                dtensor_hc_base.to_local(),
                dtensor_hc_scale.to_local(),
                norm_args[4],
                norm_args[5],
            )
            local_kwargs = {}
        else:
            local_args = (
                dtensor_x.to_local(),
                dtensor_weight.to_local(),
                dtensor_hc_base.to_local(),
                dtensor_hc_scale.to_local(),
            )
            local_kwargs = {
                'hc_eps': norm_args[4],
                'norm_eps': norm_args[5],
            }

        cache_values = [
            dtensor_x.layout,
            dtensor_weight.layout,
            dtensor_hc_base.layout,
            dtensor_hc_scale.layout,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        x_layout, weight_layout, hc_base_layout, hc_scale_layout = cache_values

        self._check_partial_inputs(
            [x_layout, weight_layout, hc_base_layout, hc_scale_layout])
        _validate_input_layouts_mhc_head(
            x_layout, weight_layout, hc_base_layout, hc_scale_layout)

        # SBH 3D [s, b, c=n*H]. out [s,b,H] follows x's channel shard (n must
        # replicate, so sharding n*H shards H by the same mesh dim).
        # rms_inv [s,b,1] and mixes [s,b,n] keep the last dim replicated.
        x_tm = x_layout.tensor_map
        mesh = x_layout.mesh
        if len(x_tm) == 3:
            s, b, c = x_tm
            maps = [
                (s, b, c),      # out: [s, b, H]
                (s, b, -1),     # rms_inv: [s, b, 1]
                (s, b, -1),     # mixes: [s, b, n]
            ]
        else:
            raise ValueError(
                f"For npu_mhc_head, tensor_map length should be 3 (SBH), "
                f"but got {len(x_tm)}."
            )
        return tuple(_create_output_layout(mesh, tm) for tm in maps), None
