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
Distributed implementation for __getitem__ operator (PyTorch only).

Supports basic indexing (int, slice, None, Ellipsis) which produces views,
and advanced indexing (list, LongTensor) which produces copies.
BoolTensor masks are not supported because the output shape is data-dependent.
"""
from typing import Tuple

from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from .parallel_ops import DistributedOp

platform = get_platform()
Tensor = platform.Tensor

_BASIC = "basic"
_ADVANCED = "advanced"
_BOOL_MASK = "bool_mask"


def _normalize___getitem___args(self_t, key):
    """Normalize __getitem__ arguments to canonical positional form.

    __getitem__ is always called as func(self, key) with empty kwargs.

    Args:
        self_t: The DTensor self argument.
        key: The indexing key (int, slice, None, Ellipsis, tuple, list, Tensor).

    Returns:
        tuple: ((self_t, key), {})
    """
    return (self_t, key), {}


def _is_long_tensor(obj) -> bool:
    """Check if obj is a non-bool Tensor (LongTensor or similar)."""
    return isinstance(obj, Tensor) and not _is_bool_tensor(obj)


def _is_bool_tensor(obj) -> bool:
    """Check if obj is a BoolTensor."""
    if not isinstance(obj, Tensor):
        return False
    if not hasattr(obj, 'dtype'):
        return False
    return str(obj.dtype).rsplit('.', maxsplit=1)[-1] in ('bool', 'bool_')


def _is_advanced_elem(k) -> bool:
    """Check if key element triggers advanced indexing."""
    if isinstance(k, list):
        return True
    if isinstance(k, Tensor):
        # 0-D long tensor is treated as basic (equivalent to int)
        if k.ndim == 0 and not _is_bool_tensor(k):
            return False
        return True
    return False


def _build_key_element_descriptor(k, op_name="__getitem__"):
    """Build a descriptor tuple for a single key element.

    Args:
        k: A single key element (int, slice, None, Ellipsis, list, Tensor).
        op_name: Operator name for error messages.

    Returns:
        tuple: A descriptor tuple for this element.

    Raises:
        ValueError: If the key element type is unsupported.
    """
    if k is None:
        return ("none",)
    if k is Ellipsis:
        return ("ellipsis",)
    if isinstance(k, int):
        return ("int", k)
    if isinstance(k, slice):
        return ("slice", k.start, k.stop, k.step)
    if isinstance(k, list):
        return ("idx_list_len", len(k))
    if isinstance(k, Tensor) and k.ndim == 0 and not _is_bool_tensor(k):
        return ("int", int(k.item()))
    if isinstance(k, Tensor):
        shape = tuple(k.shape)
        if isinstance(k, DTensor):
            alias = tuple(k.layout.alias_tensor_map)
            return ("idx_tensor", shape, alias)
        return ("idx_tensor", shape)
    raise ValueError(
        f"For {op_name}, unsupported index type: {type(k).__name__}."
    )


def _key_cache_descriptor(key, op_name="__getitem__"):
    """Convert raw key to hashable (desc, kind) for cache_values.

    The descriptor captures only what affects layout derivation:
    int value, slice bounds, list length, and tensor shape.
    Exact list/tensor contents are not preserved — only shape matters for layout.

    Args:
        key: Raw key from __getitem__ call (int, slice, None, Ellipsis, tuple, list, Tensor).
        op_name: Operator name for error messages.

    Returns:
        (desc, kind): desc is a tuple of hashable descriptor tuples,
        kind is "basic", "advanced", or "bool_mask".

    Raises:
        ValueError: If key contains unsupported types.
    """
    if not isinstance(key, tuple):
        key = (key,)

    # First pass: detect BoolTensor anywhere in the key
    for k in key:
        if _is_bool_tensor(k):
            mask_shape = tuple(k.shape) if hasattr(k, 'shape') else ()
            return (("bool_mask", mask_shape),), _BOOL_MASK

    # Check for advanced indexing elements (list or non-0D tensor)
    has_advanced = any(_is_advanced_elem(k) for k in key)
    kind = _ADVANCED if has_advanced else _BASIC

    desc = [_build_key_element_descriptor(k, op_name) for k in key]
    return tuple(desc), kind


def _desc_action_to_expanded(action_type, d, input_dim):
    """Convert a single descriptor element to expanded action.

    Args:
        action_type: Type string from descriptor (none, int, slice, etc.).
        d: Descriptor tuple for this element.
        input_dim: Current input dimension index.

    Returns:
        tuple: (expanded_entries, input_dim_delta)
            expanded_entries: list of action tuples to append.
            input_dim_delta: how much to advance input_dim.
    """
    if action_type == "none":
        return [("newaxis",)], 0
    if action_type == "int":
        return [("int", d[1], input_dim)], 1
    if action_type == "slice":
        return [("slice", d[1], d[2], d[3], input_dim)], 1
    if action_type == "idx_list_len":
        return [("idx_list", (0,) * d[1], input_dim)], 1
    if action_type == "idx_tensor":
        shape = d[1]
        alias = d[2] if len(d) > 2 else None
        idx_info = {"shape": shape, "alias": alias}
        return [("idx_tensor", idx_info, input_dim)], 1
    raise ValueError(
        f"Unsupported descriptor type: {action_type}."
    )


def _descriptor_to_expanded_actions(desc, ndim, op_name="__getitem__"):
    """Reconstruct expanded_actions from key descriptor for layout derivation.

    Args:
        desc: Key descriptor tuple from _key_cache_descriptor.
        ndim: Number of input tensor dimensions.
        op_name: Operator name for error messages.

    Returns:
        list: Expanded actions, each a tuple (action_type, *args).

    Raises:
        ValueError: On invalid descriptor (multiple Ellipsis, too many indices).
    """
    if desc is None:
        return []

    # Pass through bool_mask descriptor
    if len(desc) == 1 and desc[0][0] == "bool_mask":
        return [("bool_mask", desc[0][1])]

    non_none = [d for d in desc if d[0] != "none"]
    n_ellipsis = sum(1 for d in non_none if d[0] == "ellipsis")

    if n_ellipsis > 1:
        raise ValueError(
            f"For {op_name}, an index can only have a single Ellipsis ('...')."
        )

    n_specified_dims = len(non_none) - n_ellipsis
    n_fill = ndim - n_specified_dims

    if n_fill < 0:
        raise ValueError(
            f"For {op_name}, too many indices for tensor of dimension {ndim}, "
            f"but got {len(non_none)} non-None indices."
        )

    expanded = []
    input_dim = 0

    for d in desc:
        action_type = d[0]

        if action_type == "ellipsis":
            for _ in range(n_fill):
                expanded.append(("slice", None, None, None, input_dim))
                input_dim += 1
        else:
            entries, dim_delta = _desc_action_to_expanded(action_type, d, input_dim)
            expanded.extend(entries)
            input_dim += dim_delta

    # Implicitly add full slices for unspecified trailing dimensions
    while input_dim < ndim:
        expanded.append(("slice", None, None, None, input_dim))
        input_dim += 1

    return expanded


def _unwrap_key_for_local(key):
    """Convert key for local execution: DTensors to local.

    Args:
        key: Raw key from __getitem__ call.

    Returns:
        Unwrapped key suitable for passing to local func.
    """
    if isinstance(key, DTensor):
        return key.to_local()
    if isinstance(key, tuple):
        return tuple(_unwrap_key_for_local(k) for k in key)
    return key


def _copy_partial_state(src_layout, dst_layout):
    """Copy partial state from src_layout to dst_layout using public API.

    Args:
        src_layout: Source Layout to copy partials from.
        dst_layout: Destination Layout to copy partials to.
    """
    for dev_idx, op in enumerate(src_layout.partial):
        if op is not None:
            dst_layout.set_partial_by_dev_axis(dst_layout.alias_name[dev_idx], op)



class GetItemDistributedOp(DistributedOp):
    """Distributed implementation for tensor.__getitem__.

    Supports basic indexing (int, slice, None, Ellipsis) which produces views,
    and advanced indexing (list, LongTensor) which produces copies.
    BoolTensor masks are rejected because they produce data-dependent shapes.

    Sharding constraints:
      - Any dimension indexed by int, non-full slice, or advanced index must
        be replicated.
      - Advanced index tensors must themselves be replicated.
      - Input must not have Partial status.
      - slice step != 1 is not supported.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Preprocess arguments for __getitem__.

        Converts raw key to hashable descriptor for cache, and unwraps
        DTensors to local tensors for execution.

        Args:
            args: (self_tensor, key)
            kwargs: Empty dict for __getitem__.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
        """
        norm_args, _ = _normalize___getitem___args(*args, **kwargs)
        self_t, key = norm_args

        self_layout = self_t.layout
        global_shape = tuple(self_t.shape)

        # raw key -> hashable descriptor for cache
        key_desc, kind = _key_cache_descriptor(key, op_name=self.op_name)

        # Unwrap DTensor in key to local for local execution
        local_key = _unwrap_key_for_local(key)
        local_args = (self_t.to_local(), local_key)
        local_kwargs = {}

        cache_values = [self_layout, key_desc, global_shape, kind]
        return local_args, local_kwargs, cache_values

    @staticmethod
    def _reject_bool_mask(expanded_actions, op_name="__getitem__"):
        """Raise ValueError for bool_mask key kind."""
        mask_shape = "unknown"
        for action in expanded_actions:
            if action[0] == "bool_mask":
                mask_shape = action[1]
                break
        raise ValueError(
            f"For {op_name}, boolean-mask indexing has data-dependent "
            f"output shape and is not supported in DTensor. "
            f"Got mask of shape {mask_shape}."
        )

    @staticmethod
    def _validate_int_action(action, alias_map, global_shape, op_name="__getitem__"):
        """Validate an int indexing action."""
        input_dim = action[-1]
        idx = action[1]
        if idx < -global_shape[input_dim] or idx >= global_shape[input_dim]:
            raise ValueError(
                f"For {op_name}, index {idx} is out of range "
                f"for dimension {input_dim} with size {global_shape[input_dim]}."
            )
        if alias_map[input_dim] != "None":
            raise ValueError(
                f"For {op_name}, indexing with int on non-replicate "
                f"dim {input_dim} is not supported, "
                f"but got sharding {alias_map[input_dim]} on dim {input_dim}."
            )

    @staticmethod
    def _validate_slice_action(action, alias_map, global_shape, op_name="__getitem__"):
        """Validate a slice indexing action."""
        input_dim = action[-1]
        start, stop, step = action[1], action[2], action[3]
        step = step if step is not None else 1

        if step != 1:
            raise ValueError(
                f"For {op_name}, slice step should be 1 or None, "
                f"but got {step}."
            )

        is_full = (start is None or start == 0) and (
            stop is None or stop >= global_shape[input_dim]
        )

        if not is_full and alias_map[input_dim] != "None":
            raise ValueError(
                f"For {op_name}, non-full slice on non-replicate "
                f"dim {input_dim} is not supported, "
                f"but got sharding {alias_map[input_dim]} on dim {input_dim}."
            )

    @staticmethod
    def _validate_advanced_action(action, alias_map, op_name="__getitem__"):
        """Validate an advanced indexing action (idx_list or idx_tensor)."""
        input_dim = action[-1]
        if alias_map[input_dim] != "None":
            raise ValueError(
                f"For {op_name}, advanced indexing on non-replicate "
                f"dim {input_dim} is not supported, "
                f"but got sharding {alias_map[input_dim]} on dim {input_dim}."
            )
        # For idx_tensor, the index tensor itself must be replicated
        if action[0] == "idx_tensor":
            idx_alias = action[1].get("alias")
            if idx_alias and any(x != "None" for x in idx_alias):
                raise ValueError(
                    f"For {op_name}, advanced index tensor must be "
                    f"replicated, but got layout with sharding {idx_alias}."
                )

    @staticmethod
    def _validate_input_layouts(self_layout, expanded_actions, global_shape, kind,
                                 op_name="__getitem__"):
        """Validate sharding constraints for __getitem__.

        Rules:
            1. BoolTensor mask indexing is not supported.
            2. Any dimension indexed by int, non-full slice, or advanced index
               must be replicated.
            3. slice step must be None or 1.
            4. Advanced index tensors must be replicated.
            5. int indices must be in range.

        Args:
            self_layout: Layout of self tensor.
            expanded_actions: Expanded key actions from _descriptor_to_expanded_actions.
            global_shape: Global shape of self tensor.
            kind: "basic" or "advanced".
            op_name: Operator name for error messages (default "__getitem__").

        Raises:
            ValueError: If any constraint is violated.
        """
        alias_map = self_layout.alias_tensor_map

        if kind == _BOOL_MASK:
            GetItemDistributedOp._reject_bool_mask(expanded_actions, op_name)

        for action in expanded_actions:
            action_type = action[0]

            if action_type == "newaxis":
                continue
            if action_type == "int":
                GetItemDistributedOp._validate_int_action(
                    action, alias_map, global_shape, op_name
                )
            elif action_type == "slice":
                GetItemDistributedOp._validate_slice_action(
                    action, alias_map, global_shape, op_name
                )
            elif action_type in ("idx_list", "idx_tensor"):
                GetItemDistributedOp._validate_advanced_action(
                    action, alias_map, op_name
                )

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:  # pylint: disable=W0221
        """Infer output layout for __getitem__.

        Rules:
            1. Input must not have Partial status.
            2. BoolTensor mask indexing is not supported.
            3. Any dimension indexed by int, non-full slice, or advanced index
               must be replicated.
            4. slice step must be None or 1.
            5. Advanced index tensors must be replicated.
            6. BASIC: Output alias_tensor_map is derived by removing int-indexed
               dims, inserting Replicate for newaxis, and preserving sharding
               for full-slice dims.
            7. ADVANCED: Advanced indices broadcast shape B is inserted at
               position p (consecutive: p = first advanced dim; non-consecutive:
               p = 0). B dims are all Replicate. Other dims preserve sharding.
            8. Partial state is copied to the output layout for preserved dims.

        Args:
            cache_values: [self_layout, key_desc, global_shape, kind]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If any constraint is violated.
        """
        self_layout = cache_values[0]
        key_desc = cache_values[1]
        global_shape = cache_values[2]
        kind = cache_values[3]

        # key_desc -> expanded_actions
        expanded_actions = _descriptor_to_expanded_actions(
            key_desc, len(global_shape), op_name=self.op_name
        )

        if not self._allow_partial_inputs:
            self._check_partial_inputs([self_layout])
        self._validate_input_layouts(self_layout, expanded_actions, global_shape, kind)

        if kind == _BASIC:
            out_layout = self._infer_basic_output_layout(self_layout, expanded_actions)
        elif kind == _ADVANCED:
            out_layout = self._infer_advanced_output_layout(
                self_layout, expanded_actions, op_name=self.op_name
            )
        else:
            # _BOOL_MASK is rejected by _validate_input_layouts above,
            # so we should never reach this branch.
            raise ValueError(
                f"For __getitem__, unexpected kind: {kind}. "
                f"Expected 'basic' or 'advanced'."
            )

        return ((out_layout,), None)

    @staticmethod
    def _infer_basic_output_layout(self_layout, expanded_actions):
        """Derive output layout for basic indexing.

        Walk through expanded_actions:
          - int: remove the dimension from alias_tensor_map
          - full slice (None, None, 1): keep the dimension with same sharding
          - non-full slice: keep the dimension, must be replicated (validated earlier)
          - newaxis: insert Replicate dimension

        Args:
            self_layout: Layout of self tensor.
            expanded_actions: Expanded key actions.

        Returns:
            Layout: Output layout.
        """
        alias_map = self_layout.alias_tensor_map
        mesh = self_layout.mesh

        out_alias = []
        for action in expanded_actions:
            action_type = action[0]
            if action_type == "newaxis":
                out_alias.append("None")
            elif action_type == "slice":
                input_dim = action[4]
                out_alias.append(alias_map[input_dim])
            # "int" actions are skipped (dimension removed)

        out_layout = Layout.from_device_mesh(mesh)
        out_layout = out_layout(*out_alias)

        # Copy partial state from input
        _copy_partial_state(self_layout, out_layout)
        out_layout.tensor_map_to_placement()
        out_layout.update_compact_str()
        return out_layout

    @staticmethod
    def _compute_advanced_index_info(expanded_actions, op_name):
        """Analyze advanced indexing actions for output layout derivation.

        Returns:
            (advanced_positions, advanced_pos_set, are_consecutive, b_ndim)
        """
        advanced_actions = []
        for i, action in enumerate(expanded_actions):
            if action[0] in ("idx_list", "idx_tensor"):
                advanced_actions.append((i, action))

        index_shapes = []
        for _, action in advanced_actions:
            if action[0] == "idx_list":
                index_shapes.append((len(action[1]),))
            elif action[0] == "idx_tensor":
                index_shapes.append(tuple(action[1]["shape"]))

        bcast_shape = _broadcast_shapes(index_shapes, op_name=op_name)
        b_ndim = len(bcast_shape) if bcast_shape else 0

        advanced_positions = [pos for pos, _ in advanced_actions]
        advanced_pos_set = set(advanced_positions)
        are_consecutive = (
            len(advanced_positions) > 0
            and advanced_positions == list(range(advanced_positions[0],
                                                 advanced_positions[-1] + 1))
        )
        return advanced_positions, advanced_pos_set, are_consecutive, b_ndim

    @staticmethod
    def _infer_advanced_output_layout(self_layout, expanded_actions, op_name="__getitem__"):
        """Derive output layout for advanced indexing.

        Advanced indices are on specific input dims L. Other input dims K
        (including newaxis) are preserved. The broadcast shape B of the
        index tensors is inserted at position p:
          - consecutive advanced indices: p = position of first advanced action
          - non-consecutive: p = 0 (B precedes all other dims)

        Args:
            self_layout: Layout of self tensor.
            expanded_actions: Expanded key actions.
            op_name: Operator name for error messages.

        Returns:
            Layout: Output layout.
        """
        alias_map = self_layout.alias_tensor_map
        mesh = self_layout.mesh

        advanced_positions, advanced_pos_set, are_consecutive, b_ndim = \
            GetItemDistributedOp._compute_advanced_index_info(expanded_actions, op_name)

        def _append_non_advanced(action):
            """Append output alias for a non-advanced action (newaxis, slice, int)."""
            if action[0] == "newaxis":
                out_alias.append("None")
            elif action[0] == "slice":
                out_alias.append(alias_map[action[4]])
            # int: dimension removed, skip

        out_alias = []

        if are_consecutive and advanced_positions:
            first_adv_pos = advanced_positions[0]
            last_adv_pos = advanced_positions[-1]

            for i in range(first_adv_pos):
                _append_non_advanced(expanded_actions[i])
            for _ in range(b_ndim):
                out_alias.append("None")
            for i in range(last_adv_pos + 1, len(expanded_actions)):
                _append_non_advanced(expanded_actions[i])
        else:
            for _ in range(b_ndim):
                out_alias.append("None")
            for i, action in enumerate(expanded_actions):
                if i in advanced_pos_set:
                    continue
                _append_non_advanced(action)

        out_layout = Layout.from_device_mesh(mesh)
        out_layout = out_layout(*out_alias)

        _copy_partial_state(self_layout, out_layout)
        out_layout.tensor_map_to_placement()
        out_layout.update_compact_str()
        return out_layout



def _broadcast_shapes(shapes, op_name="__getitem__"):
    """Compute broadcast shape for a list of shapes.

    Args:
        shapes: List of shape tuples.
        op_name: Operator name for error messages.

    Returns:
        tuple: Broadcast shape.
    """
    if not shapes:
        return ()
    result = list(shapes[0])
    for shape in shapes[1:]:
        ndim_diff = len(result) - len(shape)
        if ndim_diff < 0:
            result = [1] * (-ndim_diff) + result
            ndim_diff = 0
        for i, d2 in enumerate(shape):
            d1 = result[ndim_diff + i]
            if d1 == 1:
                result[ndim_diff + i] = d2
            elif d2 not in (1, d1):
                raise ValueError(
                    f"For {op_name}, advanced index shapes {shapes} "
                    f"cannot be broadcast together."
                )
    return tuple(result)
