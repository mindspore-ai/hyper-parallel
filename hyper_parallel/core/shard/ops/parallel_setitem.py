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
Distributed implementation for __setitem__ operator (PyTorch only).

__setitem__ (tensor[key] = value) is an in-place operation.
The LHS key follows the same classification as __getitem__:
BASIC (int/slice/None/Ellipsis), ADVANCED (list/LongTensor), BOOL_MASK (not supported).
"""
from typing import Any, Tuple

from hyper_parallel.core.dtensor.dtensor import DTensor, _build_layout
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.platform import get_platform
from .parallel_getitem import (
    _BASIC,
    _BOOL_MASK,
    _broadcast_shapes,
    _key_cache_descriptor,
    _descriptor_to_expanded_actions,
    _unwrap_key_for_local,
    GetItemDistributedOp,
)
from .parallel_ops import DistributedOp

platform = get_platform()
Tensor = platform.Tensor


def _normalize___setitem___args(self_t, key, value):
    """Normalize __setitem__ arguments to canonical positional form.

    __setitem__ is always called as func(self, key, value) with empty kwargs.

    Args:
        self_t: The DTensor self argument.
        key: The indexing key.
        value: The value to assign.

    Returns:
        tuple: ((self_t, key, value), {})
    """
    return (self_t, key, value), {}


def _compute_lhs_shape(global_shape, expanded_actions, kind, op_name="__setitem__"):
    """Compute the shape of the LHS slice that __getitem__ would produce.

    This is used to validate that the RHS value can broadcast to the LHS slice.

    Args:
        global_shape: Tuple of self tensor's global shape.
        expanded_actions: Expanded key actions from _descriptor_to_expanded_actions.
        kind: "basic" or "advanced".
        op_name: Operator name for error messages.

    Returns:
        tuple: Shape of the LHS slice.
    """
    if kind == _BASIC:
        return _compute_basic_lhs_shape(global_shape, expanded_actions, op_name=op_name)
    return _compute_advanced_lhs_shape(global_shape, expanded_actions, op_name=op_name)


def _compute_basic_lhs_shape(global_shape, expanded_actions, op_name="__setitem__"):
    """Compute LHS shape for basic indexing.

    Raises ValueError if step is not None or 1.
    """
    lhs_dims = []
    for action in expanded_actions:
        action_type = action[0]
        if action_type == "newaxis":
            lhs_dims.append(1)
        elif action_type == "slice":
            start, stop, step = action[1], action[2], action[3]
            step = step if step is not None else 1
            if step != 1:
                raise ValueError(
                    f"For {op_name}, slice step must be None or 1, but got {step}."
                )
            input_dim = action[4]
            dim_size = global_shape[input_dim]
            # Compute slice length
            start_val = start if start is not None else 0
            if start_val < 0:
                start_val += dim_size
            stop_val = stop if stop is not None else dim_size
            if stop_val < 0:
                stop_val += dim_size
            # Clamp to valid range
            start_val = max(0, min(start_val, dim_size))
            stop_val = max(0, min(stop_val, dim_size))
            slice_len = max(0, (stop_val - start_val + step - 1) // step)
            lhs_dims.append(slice_len)
        # int actions are skipped (dimension removed)
    return tuple(lhs_dims) if lhs_dims else ()


def _compute_slice_len(action, global_shape, op_name="__setitem__"):
    """Compute output dim length for a slice action.

    Raises ValueError if step is not None or 1.
    """
    start, stop, step = action[1], action[2], action[3]
    step = step if step is not None else 1
    if step != 1:
        raise ValueError(
            f"For {op_name}, slice step must be None or 1, but got {step}."
        )
    input_dim = action[4]
    dim_size = global_shape[input_dim]
    start_val = start if start is not None else 0
    if start_val < 0:
        start_val += dim_size
    stop_val = stop if stop is not None else dim_size
    if stop_val < 0:
        stop_val += dim_size
    start_val = max(0, min(start_val, dim_size))
    stop_val = max(0, min(stop_val, dim_size))
    return max(0, (stop_val - start_val + step - 1) // step)


def _append_non_advanced_shape(action, global_shape, result, op_name="__setitem__"):
    """Append output dim length for a non-advanced action (newaxis, slice; skip int)."""
    if action[0] == "newaxis":
        result.append(1)
    elif action[0] == "slice":
        result.append(_compute_slice_len(action, global_shape, op_name=op_name))
    # int: dimension removed, skip


def _compute_advanced_lhs_shape(global_shape, expanded_actions, op_name="__setitem__"):
    """Compute LHS shape for advanced indexing.

    Walks expanded_actions positionally: newaxis → 1, slice → slice_len,
    int → skip, advanced block → replaced by broadcast shape B.
    """
    # Collect advanced index shapes and their positions
    advanced_shapes = []
    advanced_positions = []
    for i, action in enumerate(expanded_actions):
        if action[0] == "idx_list":
            advanced_shapes.append((len(action[1]),))
            advanced_positions.append(i)
        elif action[0] == "idx_tensor":
            advanced_shapes.append(tuple(action[1]["shape"]))
            advanced_positions.append(i)

    bcast_shape = _broadcast_shapes(advanced_shapes, op_name=op_name) if advanced_shapes else ()

    advanced_pos_set = set(advanced_positions)
    are_consecutive = (
        len(advanced_positions) > 0
        and advanced_positions == list(range(advanced_positions[0],
                                             advanced_positions[-1] + 1))
    )

    result = []

    if are_consecutive and advanced_positions:
        first_adv_pos = advanced_positions[0]
        last_adv_pos = advanced_positions[-1]

        # Actions before the advanced block
        for i in range(first_adv_pos):
            _append_non_advanced_shape(expanded_actions[i], global_shape, result, op_name=op_name)

        # B dims replace the advanced block
        result.extend(bcast_shape)

        # Actions after the advanced block
        for i in range(last_adv_pos + 1, len(expanded_actions)):
            _append_non_advanced_shape(expanded_actions[i], global_shape, result, op_name=op_name)
    else:
        # Non-consecutive: B goes at position 0
        result.extend(bcast_shape)

        for i, action in enumerate(expanded_actions):
            if i in advanced_pos_set:
                continue
            _append_non_advanced_shape(action, global_shape, result)

    return tuple(result) if result else ()


def _compute_lhs_layout(self_layout, expanded_actions, kind, op_name="__setitem__"):
    """Compute the LHS layout (getitem output layout) for the given key.

    Args:
        self_layout: Layout of self tensor.
        expanded_actions: Expanded key actions.
        kind: "basic" or "advanced".
        op_name: Operator name for error messages.

    Returns:
        Layout: The layout that __getitem__ would produce for this key.
    """
    if kind == _BASIC:
        return GetItemDistributedOp._infer_basic_output_layout(  # pylint: disable=W0212
            self_layout, expanded_actions
        )
    return GetItemDistributedOp._infer_advanced_output_layout(  # pylint: disable=W0212
        self_layout, expanded_actions, op_name=op_name
    )


def _build_broadcast_value_layout(mesh, lhs_layout, value_shape, lhs_shape):
    """Build a layout for the value tensor with broadcast dims forced to Replicate.

    When a value tensor has a smaller shape that needs to broadcast to the LHS
    slice shape, any dimension where value size is 1 and LHS size > 1 must be
    Replicate (cannot be sharded). Non-broadcast dims inherit LHS placements.

    Args:
        mesh: Device mesh.
        lhs_layout: Expected LHS layout (getitem output layout).
        value_shape: Shape of the value tensor.
        lhs_shape: Expected LHS slice shape.

    Returns:
        Layout: Value layout adjusted for broadcasting.
    """
    ndim_diff = len(lhs_shape) - len(value_shape)
    padded_value_shape = (1,) * ndim_diff + value_shape

    placements = []
    for v_size, l_size, lhs_placement in zip(
        padded_value_shape, lhs_shape, lhs_layout.alias_placements
    ):
        if v_size == 1 and l_size > 1:
            placements.append(Replicate())
        else:
            placements.append(lhs_placement)

    return _build_layout(mesh, tuple(placements), len(placements))


class SetItemDistributedOp(DistributedOp):
    """Distributed implementation for tensor.__setitem__.

    __setitem__ (tensor[key] = value) is an in-place operation.
    The LHS key follows the same validation rules as __getitem__.
    RHS value can be a scalar, Tensor, or DTensor.

    Sharding constraints:
      - Same LHS constraints as __getitem__ (sharded dims cannot be indexed).
      - BoolTensor masks are rejected.
      - RHS DTensor must have mesh consistent with self and broadcast-compatible
        layout (non-broadcast dims must match LHS layout, broadcast dims must be
        Replicate).
      - RHS Tensor/scalar must be broadcastable to LHS slice shape.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Preprocess arguments for __setitem__.

        Args:
            args: (self_tensor, key, value)
            kwargs: Empty dict for __setitem__.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        norm_args = _normalize___setitem___args(*args, **kwargs)[0]
        self_t, key, value = norm_args

        local_key, key_cache, lhs_shape, lhs_layout = self._preprocess_setitem_key(
            self_t, key, self.op_name
        )

        local_value, value_desc = self._preprocess_setitem_value(
            value, key_cache[3], lhs_shape, lhs_layout, key_cache[0].mesh, self.op_name,
        )

        cache_values = key_cache + [value_desc, lhs_shape]
        return ((self_t.to_local(), local_key, local_value), {}, cache_values)

    @staticmethod
    def _preprocess_setitem_key(self_t, key, op_name):
        """Process key and compute LHS metadata for __setitem__ preprocess.

        Args:
            self_t: The self DTensor.
            key: Raw indexing key.
            op_name: Operator name for error messages.

        Returns:
            tuple: (local_key, key_cache, lhs_shape, lhs_layout)
                key_cache is [self_layout, key_desc, global_shape, kind].
        """
        self_layout = self_t.layout
        global_shape = tuple(self_t.shape)
        key_desc, kind = _key_cache_descriptor(key, op_name=op_name)
        local_key = _unwrap_key_for_local(key)
        key_cache = [self_layout, key_desc, global_shape, kind]

        if kind == _BOOL_MASK:
            return local_key, key_cache, None, None

        expanded_actions = _descriptor_to_expanded_actions(
            key_desc, len(global_shape), op_name=op_name
        )
        lhs_shape = _compute_lhs_shape(global_shape, expanded_actions, kind, op_name=op_name)
        lhs_layout = _compute_lhs_layout(self_layout, expanded_actions, kind, op_name=op_name)

        return local_key, key_cache, lhs_shape, lhs_layout

    @staticmethod
    def _preprocess_setitem_value(value, kind, lhs_shape, lhs_layout, self_mesh, op_name):
        """Process the value argument for __setitem__ preprocess.

        Handles DTensor unwrapping, plain-tensor broadcast validation and sharding,
        and scalar passthrough.

        Args:
            value: The raw value tensor (DTensor, Tensor, or scalar).
            kind: "basic", "advanced", or "bool_mask".
            lhs_shape: Computed LHS slice shape (tuple).
            lhs_layout: Computed LHS slice Layout.
            self_mesh: DeviceMesh of the self tensor.
            op_name: Operator name for error messages.

        Returns:
            (local_value, value_desc): Processed local tensor and cache descriptor.
        """
        if isinstance(value, DTensor):
            value_shape = tuple(value.shape)
            local_value = value.to_local()
            if not value_shape:
                return local_value, None  # 0-D DTensor treated as scalar
            return local_value, ("dtensor", value.layout, value_shape)

        if isinstance(value, Tensor):
            value_desc = ("plain_tensor", tuple(value.shape))
            if value.ndim > 0 and kind != _BOOL_MASK:
                _validate_value_broadcast(op_name, tuple(value.shape), lhs_shape)
                # pylint: disable=C0415
                from hyper_parallel.core.dtensor.layout import _get_slice_tensor_by_layout
                value_layout = _build_broadcast_value_layout(
                    self_mesh, lhs_layout, tuple(value.shape), lhs_shape
                )
                ndim_diff = len(lhs_shape) - value.ndim
                if ndim_diff > 0:
                    padded_value = value.reshape((1,) * ndim_diff + tuple(value.shape))
                else:
                    padded_value = value
                local_value = _get_slice_tensor_by_layout(padded_value, value_layout)
                return local_value, value_desc
            return value, value_desc

        return value, None

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """Infer output layout for __setitem__.

        __setitem__ is in-place, so output layout equals self layout.
        Validates LHS key constraints and RHS value compatibility.

        Rules:
            1. self must not be in Partial status.
            2. BoolTensor mask indexing is not supported.
            3. Any dimension being written must be replicated.
            4. slice step must be None or 1.
            5. Advanced index tensors must be replicated.
            6. RHS DTensor must have mesh consistent with self and
               broadcast-compatible layout (non-broadcast dims match LHS
               layout, broadcast dims must be Replicate).
            7. RHS Tensor/scalar must be broadcastable to LHS slice shape.
            8. Output layout equals self_layout (in-place).

        Args:
            cache_values: [self_layout, key_desc, global_shape, kind, value_desc, lhs_shape]

        Returns:
            tuple: ((self_layout,), None)

        Raises:
            ValueError: If any constraint is violated.
        """
        self_layout = cache_values[0]
        key_desc = cache_values[1]
        global_shape = cache_values[2]
        kind = cache_values[3]
        value_desc = cache_values[4]
        lhs_shape = cache_values[5]

        # key_desc -> expanded_actions (for LHS validation)
        expanded_actions = _descriptor_to_expanded_actions(
            key_desc, len(global_shape), op_name=self.op_name
        )

        if not self._allow_partial_inputs:
            self._check_partial_inputs([self_layout])
        self._validate_input_layouts(
            self_layout, expanded_actions, global_shape, kind, value_desc, lhs_shape
        )
        return ((self_layout,), None)

    @staticmethod
    def _validate_input_layouts(self_layout, expanded_actions, global_shape, kind, value_desc, lhs_shape):
        """Validate sharding constraints for __setitem__."""
        op_name = "__setitem__"

        GetItemDistributedOp._validate_input_layouts(  # pylint: disable=W0212
            self_layout, expanded_actions, global_shape, kind, op_name=op_name
        )

        if value_desc is None:
            return

        val_kind = value_desc[0]
        if val_kind == "plain_tensor":
            _validate_value_broadcast(op_name, value_desc[1], lhs_shape)
        elif val_kind == "dtensor":
            SetItemDistributedOp._validate_dtensor_value(
                self_layout, expanded_actions, kind, value_desc[1], value_desc[2], lhs_shape, op_name,
            )
        else:
            raise ValueError(
                f"For {op_name}, unsupported value descriptor kind: {val_kind}."
            )

    @staticmethod
    def _validate_dtensor_value(
            self_layout, expanded_actions, kind, val_layout, val_shape, lhs_shape, op_name):
        """Validate DTensor RHS mesh and layout."""
        _validate_value_broadcast(op_name, val_shape, lhs_shape)

        if val_layout.mesh.to_hash() != self_layout.mesh.to_hash():
            raise ValueError(
                f"For {op_name}, value DTensor must be on same mesh as self, "
                f"but got mesh {val_layout.mesh_shape} "
                f"vs {self_layout.mesh_shape}."
            )

        expected_layout = _compute_lhs_layout(
            self_layout, expanded_actions, kind, op_name=op_name
        )
        SetItemDistributedOp._validate_value_layout(
            val_layout, expected_layout, val_shape, lhs_shape, op_name,
        )

    @staticmethod
    def _validate_value_layout(val_layout, expected_layout, val_shape, lhs_shape, op_name):
        """Validate RHS DTensor placement against expected LHS placement."""
        ndim_diff = len(lhs_shape) - len(val_shape)
        val_alias = val_layout.alias_tensor_map
        expected_alias = expected_layout.alias_tensor_map

        for i, v_size in enumerate(val_shape):
            lhs_index = ndim_diff + i
            e_size = lhs_shape[lhs_index]
            v_alias = val_alias[i]
            e_alias = expected_alias[lhs_index]

            if v_size == 1 and e_size > 1:
                if v_alias != "None":
                    raise ValueError(
                        f"For {op_name}, value broadcast dim {i} (size {v_size}) "
                        f"must be Replicate, but got {v_alias}."
                    )
            elif v_alias != e_alias:
                raise ValueError(
                    f"For {op_name}, value layout mismatch at dim {i}: "
                    f"expected {e_alias}, but got {v_alias}."
                )

    def wrap_output(self, py_output: Any, output_layouts: tuple) -> None:
        """Override wrap_output for __setitem__.

        __setitem__ is in-place and returns None in PyTorch.
        The default wrap_output would try to wrap None into a DTensor and fail.

        Args:
            py_output: The output from the local function call (None).
            output_layouts: Output layouts from infer_layout.

        Returns:
            None: __setitem__ has no return value.
        """
        return None


def _validate_value_broadcast(op_name, val_shape, lhs_shape):
    """Validate that value shape can broadcast to LHS slice shape.

    Args:
        op_name: Operator name for error messages.
        val_shape: Shape of RHS value.
        lhs_shape: Expected LHS slice shape.

    Raises:
        ValueError: If shapes are not broadcastable.
    """
    if len(val_shape) > len(lhs_shape):
        raise ValueError(
            f"For {op_name}, value shape {val_shape} cannot broadcast "
            f"to target shape {lhs_shape}."
        )

    # Pad val_shape with leading 1s
    padded = (1,) * (len(lhs_shape) - len(val_shape)) + val_shape
    for v, t in zip(padded, lhs_shape):
        if v not in (1, t):
            raise ValueError(
                f"For {op_name}, value shape {val_shape} cannot broadcast "
                f"to target shape {lhs_shape}."
            )
