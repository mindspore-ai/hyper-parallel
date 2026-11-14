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
"""
Distributed implementation for Conv1d operator.
"""

from typing import Callable, Optional, Tuple

from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp


def _normalize_conv1d_args(input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Normalize positional and keyword arguments into a canonical positional tuple.

    Args:
        input_tensor: Input tensor of shape (N, C_in, L).
        weight: Weight tensor of shape (C_out, C_in/groups, kernel_size).
        bias: Optional bias tensor of shape (C_out,). Defaults to None.
        stride: Stride of the convolution. Defaults to 1.
        padding: Padding of the convolution. Defaults to 0.
        dilation: Dilation of the convolution. Defaults to 1.
        groups: Number of blocked connections. Defaults to 1.

    Returns:
        tuple: (positional_args_tuple, empty_kwargs_dict)
    """
    return (input_tensor, weight, bias, stride, padding, dilation, groups), {}


def _is_group_aligned(in_map, w_map, groups):
    """Return whether complete convolution groups are sharded on aligned axes."""
    return (
        groups > 1
        and in_map[1] != "None"
        and in_map[1] == w_map[0]
        and w_map[1] == "None"
    )


class Conv1dDistributedOp(DistributedOp):
    """Distributed implementation for torch.nn.functional.conv1d.

    Supports Data Parallel (DP), Column Tensor Parallel (CP), Row Tensor
    Parallel (RP), and DP combined with either CP or RP. Group-aligned
    sharding distributes complete convolution groups across the same axes.

    Only 3D input (N, C_in, L) is supported.  Sharding on the L (spatial)
    and kernel_size dimensions is prohibited.

    cache_values = [input_layout, weight_layout, bias_layout_or_None, bias_present, groups]
    """

    @staticmethod
    def _as_axes(axis):
        """Normalize an alias axis to a tuple of axis names.

        Returns an empty tuple for replicated axes (``"None"``).
        """
        if axis == "None":
            return ()
        return axis if isinstance(axis, tuple) else (axis,)

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Extract local tensors and build the layout cache.

        Args:
            args: Positional arguments (may include DTensors).
            kwargs: Keyword arguments.

        Returns:
            tuple: (local_args, local_kwargs, cache_values) where
                local_args = (input, weight, bias, stride, padding, dilation, groups),
                local_kwargs = {},
                cache_values = [in_layout, w_layout, b_layout, bias_present, groups].
        """
        normalized_kwargs = dict(kwargs)
        if "input" in normalized_kwargs:
            normalized_kwargs["input_tensor"] = normalized_kwargs.pop("input")
        args, _ = _normalize_conv1d_args(*args, **normalized_kwargs)
        input_tensor, weight, bias, stride, padding, dilation, groups = args

        local_args = (
            input_tensor.to_local(),
            weight.to_local(),
            bias.to_local() if hasattr(bias, '_layout') else bias,
            stride,
            padding,
            dilation,
            groups,
        )
        local_kwargs = {}
        cache_values = [
            input_tensor.layout,
            weight.layout,
            bias.layout if hasattr(bias, '_layout') else None,
            bias is not None,
            groups,
        ]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer output layout for Conv1d operator.

        Rules:
            1. Input, weight, and bias must not have Partial status.
            2. Input and weight must be 3D, bias (if present) must be 1D.
            3. L dimension and kernel_size dimension must not be sharded.
            4. Row Parallelism: C_in sharding must match between input and weight;
               groups > 1 and bias are not supported unless complete groups are
               sharded with C_out on the same mesh axes.
            5. Column Parallelism: C_out sharding must match between weight and bias.
            6. Groups > 1 with C_out sharding: additional divisibility and axis
               conflict checks.
            7. Row TP and Column TP cannot be used simultaneously, except for
               group-aligned sharding.
            8. Output layout: (N from input, C_out from weight, L from input).
            9. Row Parallelism: output carries Partial("sum") on the C_in axis;
               group-aligned sharding produces complete output channel shards.

        Args:
            cache_values: [input_layout, weight_layout, bias_layout_or_None, bias_present, groups]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any sharding constraint is violated.
        """
        in_layout, w_layout, b_layout = cache_values[0], cache_values[1], cache_values[2]
        bias_present = cache_values[3]
        groups = cache_values[4]

        # 1. Required layouts and Partial validation
        if not in_layout or not w_layout:
            raise ValueError(
                f"For {self.op_name}, Requires at least input and weight layouts."
            )

        self._check_partial_inputs([in_layout, w_layout])

        if b_layout is not None:
            self._check_partial_inputs([b_layout])

        # 2. Mesh compatibility
        if in_layout.mesh_shape != w_layout.mesh_shape:
            raise ValueError(
                f"For {self.op_name}, input and weight must have the same mesh_shape, "
                f"but got input: {in_layout.mesh_shape} and weight: {w_layout.mesh_shape}"
            )
        if b_layout is not None and b_layout.mesh_shape != in_layout.mesh_shape:
            raise ValueError(
                f"For {self.op_name}, bias and input must have the same mesh_shape, "
                f"but got bias: {b_layout.mesh_shape} and input: {in_layout.mesh_shape}"
            )

        # 3. Get alias tensor maps and validate dimensionality
        in_map = in_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map

        if len(in_map) != 3 or len(w_map) != 3:
            raise ValueError(f"For {self.op_name}, Input and weight must be 3D.")

        b_map = b_layout.alias_tensor_map if b_layout is not None else None
        group_aligned = _is_group_aligned(in_map, w_map, groups)

        # 4. Reject unsupported L and kernel_size sharding
        if in_map[2] != "None":
            raise ValueError(
                f"For {self.op_name}, Sharding on L dimension is not supported."
            )
        if w_map[2] != "None":
            raise ValueError(
                f"For {self.op_name}, Sharding on kernel_size dimension is not supported."
            )

        # 5. Bias dimensionality
        if b_map is not None and len(b_map) != 1:
            raise ValueError(
                f"For {self.op_name}, Bias must be 1D, but got {len(b_map)}D"
            )

        # 6. Row Parallelism validation
        self._validate_row_parallelism(in_map, w_map, groups, bias_present, group_aligned)

        # 7. Column Parallelism validation
        self._validate_column_parallelism(w_map, b_map, bias_present)

        # 8. Grouped Column Parallelism validation
        if groups > 1 and w_map[0] != "None":
            self._validate_grouped_column_parallelism(
                in_map, w_map, groups, w_layout, group_aligned
            )

        # 9. Parallelism combination check
        self._validate_parallelism_combination(in_map, w_map, group_aligned)

        # 10. Construct output map: (N from input, C_out from weight, L from input)
        out_map = (in_map[0], w_map[0], in_map[2])

        # 11. Build output Layout
        output_layout = Layout(
            mesh_shape=in_layout.mesh_shape,
            alias_name=in_layout.alias_name,
            rank_list=in_layout.rank_list,
        )
        output_layout = output_layout(*out_map)

        # 12. Set Partial status only for contracting-dimension Row Parallelism
        if in_map[1] != "None" and not group_aligned:
            for axis in self._as_axes(in_map[1]):
                output_layout.set_partial_by_dev_axis(axis, "sum")

        return (output_layout,), None

    def get_expand_impl(self, func: Optional[Callable], infer_result: tuple,
                        cache_values: list) -> Optional[Callable]:
        """Return a custom expand implementation for Grouped Column Parallelism.

        When groups > 1 and weight C_out is sharded, each device only handles
        a subset of groups. Replicated input is sliced to the corresponding
        group range. Group-aligned input is already local and is passed through
        unchanged. Both paths call native conv1d with adjusted local groups.

        Args:
            func: Original operator callable (native conv1d).
            infer_result: ((output_layout,), None) from infer_layout.
            cache_values: [input_layout, weight_layout, bias_layout_or_None, bias_present, groups].

        Returns:
            callable | None: The expand closure, or None if not needed.
        """
        in_layout = cache_values[0]
        w_layout = cache_values[1]
        groups = cache_values[4]

        in_map = in_layout.alias_tensor_map
        w_map = w_layout.alias_tensor_map

        if w_map[0] == "None" or groups == 1:
            return None

        mesh = w_layout.mesh
        axes = self._as_axes(w_map[0])
        dev_num = 1
        local_rank = 0
        for axis_name in axes:
            axis_size = mesh.get_device_num_along_axis(axis_name)
            dev_num *= axis_size
            local_rank = local_rank * axis_size + mesh.get_local_rank(axis_name)

        local_groups = groups // dev_num

        if _is_group_aligned(in_map, w_map, groups):
            def _group_aligned_impl(
                input_tensor, weight_tensor, bias=None,
                stride=1, padding=0, dilation=1, groups=1,
            ):
                runtime_local_groups = groups // dev_num
                return func(
                    input_tensor, weight_tensor, bias,
                    stride, padding, dilation, runtime_local_groups,
                )

            return _group_aligned_impl

        start_group = local_rank * local_groups

        def _grouped_column_parallel_impl(
            input_tensor, weight_tensor, bias=None,
            stride=1, padding=0, dilation=1, groups=1,
        ):
            c_in = input_tensor.shape[1]
            c_in_per_group = c_in // groups
            start_channel = start_group * c_in_per_group
            end_channel = start_channel + local_groups * c_in_per_group

            sliced_input = input_tensor[:, start_channel:end_channel, :]
            return func(
                sliced_input, weight_tensor, bias,
                stride, padding, dilation, local_groups,
            )

        return _grouped_column_parallel_impl

    def _validate_row_parallelism(self, in_map, w_map, groups, bias_present, group_aligned):
        """Validate constraints for Row Parallelism (C_in sharding).

        Raises:
            ValueError: If non-group-aligned groups > 1 use C_in sharding,
                C_in axes mismatch, or bias is present with row parallelism.
        """
        if group_aligned:
            return

        if groups > 1 and (in_map[1] != "None" or w_map[1] != "None"):
            raise ValueError(
                f"For {self.op_name}, Sharding on C_in with groups > 1 is not supported."
            )

        if in_map[1] != w_map[1]:
            raise ValueError(
                f"For {self.op_name}, Input C_in and Weight C_in must be sharded "
                f"on the same axis."
            )

        # Keep row-parallel bias disabled until Linear/Add establish a unified strategy
        # that preserves both forward contributions and backward gradients.
        if in_map[1] != "None" and bias_present:
            raise ValueError(
                f"For {self.op_name}, Row Parallelism requires bias=None."
            )

    def _validate_column_parallelism(self, w_map, b_map, bias_present):
        """Validate constraints for Column Parallelism (weight C_out sharding).

        Always called to ensure bias C_out matches weight C_out, including the
        case where weight is replicated but bias is sharded alone.

        Raises:
            ValueError: If bias C_out sharding does not match weight C_out sharding,
                or a non-DTensor bias is used with sharded weight C_out.
        """
        if bias_present and b_map is None and w_map[0] != "None":
            raise ValueError(
                f"For {self.op_name}, bias should be a DTensor sharded with Weight C_out, "
                f"but got a non-DTensor bias."
            )

        if b_map is not None and w_map[0] != b_map[0]:
            raise ValueError(
                f"For {self.op_name}, Weight C_out and Bias C_out must be sharded "
                f"on the same axis."
            )

    def _validate_grouped_column_parallelism(
            self, in_map, w_map, groups, w_layout, group_aligned):
        """Validate constraints for Grouped Column Parallelism.

        Called when groups > 1 and w_map[0] != "None".

        Raises:
            ValueError: If groups is not divisible by tp_size, C_out mesh axis
                conflicts with output N axis, or input C_in is sharded on the
                same axis as C_out without a group-aligned layout.
        """
        c_out_axes = set(self._as_axes(w_map[0]))
        n_axes = set(self._as_axes(in_map[0]))
        c_in_axes = set(self._as_axes(in_map[1]))

        tp_size = 1
        for axis_name in c_out_axes:
            tp_size *= w_layout.mesh.get_device_num_along_axis(axis_name)

        if groups % tp_size != 0:
            raise ValueError(
                f"For {self.op_name}, groups ({groups}) "
                f"must be divisible by tp_size ({tp_size})."
            )

        if c_out_axes & n_axes:
            raise ValueError(
                f"For {self.op_name}, C_out mesh axis conflicts with output N axis "
                f"when groups > 1."
            )

        if c_out_axes & c_in_axes and not group_aligned:
            raise ValueError(
                f"For {self.op_name}, Input C_in must not be sharded on the same "
                f"axis as C_out when groups > 1."
            )

    def _validate_parallelism_combination(self, in_map, w_map, group_aligned):
        """Reject simultaneous Row TP and Column TP unless group-aligned.

        Raises:
            ValueError: If both C_in and C_out are sharded without a
                group-aligned layout.
        """
        # Non-group-aligned Row/Column TP needs a shared model-parallel autograd mapping
        # for Column-axis grad_input reduction before it can be enabled.
        if in_map[1] != "None" and w_map[0] != "None" and not group_aligned:
            raise ValueError(
                f"For {self.op_name}, Simultaneous Row and Column Parallelism "
                f"is not supported."
            )
