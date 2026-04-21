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
"""Distributed implementation for npu_mhc_post operator."""
from typing import Tuple, Dict, Any

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_mhc_post_args(
        x,
        h_res,
        h_out,
        h_post):
    """Normalize positional and keyword arguments into a canonical positional tuple.

    Args:
        x: Input tensor [B,S,N,D] or [T,N,D].
        h_res: mHC h_res transformation matrix [B,S,N,N] or [T,N,N].
        h_out: Attention/MLP output [B,S,D] or [T,D].
        h_post: mHC h_post transformation matrix [B,S,N] or [T,N].

    Returns:
        tuple: (positional_args_tuple, empty_kwargs_dict)
    """
    return (x, h_res, h_out, h_post), {}


# Validation rules table for npu_mhc_post
# Key: tensor_map length (format identifier)
# Value: validation rules for that format
_MHC_POST_VALIDATION_RULES: Dict[int, Dict[str, Any]] = {
    4: {
        "op_name": "npu_mhc_post",
        "forbidden_dims": {2: "N", 3: "D"},
        "dim_requirements": {
            "h_res": 4,
            "h_out": 3,
            "h_post": 3,
        },
    },
    3: {
        "op_name": "npu_mhc_post",
        "forbidden_dims": {1: "N", 2: "D"},
        "dim_requirements": {
            "h_res": 3,
            "h_out": 2,
            "h_post": 2,
        },
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


def _validate_tensor_dimensions(
        actual_dims: Dict[str, int],
        required_dims: Dict[str, int],
        op_name: str,
) -> None:
    """Validate that each input tensor has the expected number of dimensions.

    Args:
        actual_dims: Dict mapping input name to actual dimension count.
        required_dims: Dict mapping input name to expected dimension count.
        op_name: Operator name for error message.

    Raises:
        ValueError: If any input has wrong dimension count.
    """
    for input_name, required_dim in required_dims.items():
        actual_dim = actual_dims.get(input_name)
        if actual_dim != required_dim:
            raise ValueError(
                f"For {op_name}, {input_name} tensor should have {required_dim} dimensions, "
                f"but got {actual_dim}"
            )


class NpuMhcPostDistributedOp(DistributedOp):
    """DistributedOp for npu_mhc_post operator.

    Implements layout inference for the MHC post-processing operation:
    x_{l+1} = (H_l^res)^T × x_l + h_l^out ⊗ H_t^post
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        norm_args, _ = _normalize_mhc_post_args(*args, **kwargs)
        dtensor_x, dtensor_h_res, dtensor_h_out, dtensor_h_post = (
            norm_args[0], norm_args[1], norm_args[2], norm_args[3]
        )

        local_args = (
            dtensor_x.to_local(),
            dtensor_h_res.to_local(),
            dtensor_h_out.to_local(),
            dtensor_h_post.to_local(),
        )
        local_kwargs = {}

        cache_values = [
            dtensor_x.layout,
            dtensor_h_res.layout,
            dtensor_h_out.layout,
            dtensor_h_post.layout,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        x_layout, h_res_layout, h_out_layout, h_post_layout = cache_values

        self._check_partial_inputs([x_layout, h_res_layout, h_out_layout, h_post_layout])

        self._validate_input_layouts_mhc_post(
            x_layout, h_res_layout, h_out_layout, h_post_layout
        )

        out_layout = self._infer_output_layout(x_layout)
        return (out_layout,), None

    @staticmethod
    def _validate_input_layouts_mhc_post(
            x_layout: Layout,
            h_res_layout: Layout,
            h_out_layout: Layout,
            h_post_layout: Layout,
    ) -> None:
        """Validate input layouts for npu_mhc_post operator."""
        x_tm = x_layout.tensor_map
        x_tm_len = len(x_tm)

        # Get validation rules from table based on tensor_map length
        rules = _MHC_POST_VALIDATION_RULES.get(x_tm_len)
        if rules is None:
            raise ValueError(
                f"For npu_mhc_post, tensor_map length should be 4 or 3, but got {x_tm_len}"
            )

        # Validate forbidden dimensions (N, D must be replicated)
        _validate_tensor_map_dims(x_tm, rules["op_name"], rules["forbidden_dims"])

        # Validate dimension counts for each input
        actual_dims = {
            "h_res": len(h_res_layout.tensor_map),
            "h_out": len(h_out_layout.tensor_map),
            "h_post": len(h_post_layout.tensor_map),
        }
        _validate_tensor_dimensions(actual_dims, rules["dim_requirements"], rules["op_name"])

    @staticmethod
    def _infer_output_layout(x_layout: Layout) -> Layout:
        out_layout = Layout.from_device_mesh(x_layout.mesh)
        out_layout.set_tensor_map(x_layout.tensor_map)
        out_layout.tensor_map_to_placement()
        return out_layout
