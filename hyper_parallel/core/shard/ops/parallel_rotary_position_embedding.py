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
"""Distributed implementation for RotaryPositionEmbedding operator."""
import copy
from typing import Optional, Tuple

from .parallel_ops import DistributedOp


def _normalize_rpe_args(x, cos, sin, mode=0):
    """Normalize positional and keyword arguments into a canonical positional tuple.

    Args:
        x: Input tensor.
        cos: Cosine position encoding tensor.
        sin: Sine position encoding tensor.
        mode: Rotation mode. 0=rotate_half, 1=rotate_interleaved, 2=quarter,
            3=interleave-half. Defaults to 0.

    Returns:
        tuple: (positional_args_tuple, empty_kwargs_dict)
    """
    return (x, cos, sin, mode), {}


class RotaryPositionEmbeddingDistributedOp(DistributedOp):
    """Distributed operator for RotaryPositionEmbedding.

    Computes rotary position embedding element-wise:
        y = x * cos + x_rotate * sin

    where x_rotate is obtained by rotating within the last (D) dimension.
    Output shape equals x shape exactly.

    Sharding constraints:
      - D (last dim) must be replicated for x, cos, and sin: the kernel rotates
        within D and the operation is indivisible along that axis.
      - B, N, S dims are fully independent across positions and can be freely
        sharded.
      - cos/sin may have any subset of non-D dims replicated (broadcast case),
        but if cos/sin is sharded on a dimension, it must match x's sharding
        on that dimension.

    MODE does not affect layout inference: all modes produce output shape == x
    shape and leave B/N/S independence unchanged. MODE is passed through as a
    kernel parameter.

    Output:
      Single tensor with the same shape and layout as x.
    """

    @staticmethod
    def _validate_input_layouts(x_layout, cos_layout, sin_layout) -> None:
        """Validate sharding constraints for all input tensors.

        Rules (applied to both 4-D BNSD/BSND/SBND and 3-D TND layouts):
          - x's last dim (D) must be replicated.
          - cos and sin's last dim (D) must be replicated.
          - For any non-D dimension d: if cos/sin is sharded there, the mesh
            axis must equal x's mesh axis on the same dimension.

        Args:
            x_layout: Layout of the x tensor.
            cos_layout: Layout of the cos tensor.
            sin_layout: Layout of the sin tensor.

        Raises:
            ValueError: If D is sharded for any input, or if cos/sin sharding
                is inconsistent with x on any non-D dimension.
        """
        op = "rotary_position_embedding"
        x_tm = x_layout.tensor_map

        if x_tm[-1] != -1:
            raise ValueError(
                f"For {op}, D (last dim) of x must be replicated, "
                f"but got tensor_map={x_tm}"
            )

        for name, layout in [('cos', cos_layout), ('sin', sin_layout)]:
            tm = layout.tensor_map
            if tm[-1] != -1:
                raise ValueError(
                    f"For {op}, D (last dim) of {name} must be replicated, "
                    f"but got tensor_map={tm}"
                )
            for d in range(len(tm) - 1):
                x_d = x_tm[d] if d < len(x_tm) - 1 else -1
                if tm[d] != -1 and tm[d] != x_d:
                    raise ValueError(
                        f"For {op}, {name} sharding on dim {d} must match x or be replicated, "
                        f"but got x={x_d}, {name}={tm[d]}"
                    )

    def preprocess(self, args: tuple, kwargs: dict) -> Optional[tuple]:
        """Extract local tensors and build the layout cache.

        All inputs (x, cos, sin) are expected to be DTensors.

        Args:
            args: Positional arguments, may include DTensors.
            kwargs: Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values) where
                local_args = (x_local, cos_local, sin_local, mode),
                local_kwargs = {},
                cache_values = [x_layout, cos_layout, sin_layout].
        """
        norm_args, _ = _normalize_rpe_args(*args, **kwargs)
        x = norm_args[0]
        cos = norm_args[1]
        sin = norm_args[2]
        mode = norm_args[3]

        local_args = (x.to_local(), cos.to_local(), sin.to_local(), mode)
        cache_values = [x.layout, cos.layout, sin.layout]
        return local_args, {}, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer output layout for the single output tensor.

        Rules:
            1. Partial inputs are rejected.
            2. D (last dim) must be replicated for x, cos, and sin.
            3. cos/sin sharding on non-D dims must match x or be replicated.
            4. Output layout = deep copy of x_layout (output shape == x shape).

        Args:
            cache_values: [x_layout, cos_layout, sin_layout]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any input has Partial status, D is sharded,
                or cos/sin sharding is inconsistent with x.
        """
        x_layout = cache_values[0]
        cos_layout = cache_values[1]
        sin_layout = cache_values[2]

        self._check_partial_inputs([x_layout, cos_layout, sin_layout])
        self._validate_input_layouts(x_layout, cos_layout, sin_layout)
        return (copy.deepcopy(x_layout),), None
