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
Distributed implementation for torch.vstack operator.
"""

from typing import Tuple

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from .parallel_concat import ConcatDistributedOp


def _normalize_vstack_args(tensors, *, out=None):
    """Normalize torch.vstack arguments.

    vstack takes a single positional argument (sequence of tensors)
    and an optional keyword-only ``out``.
    """
    return (tensors,), {"out": out}


def _promote_tensor_map(alias_tensor_map):
    """Apply atleast_2d semantic promotion to a tensor_map tuple.

    Returns the promoted alias_tensor_map.
    """
    ndim = len(alias_tensor_map)
    if ndim == 0:
        return ("None", "None")
    if ndim == 1:
        return ("None",) + alias_tensor_map
    return alias_tensor_map


class VstackDistributedOp(ConcatDistributedOp):
    """Distributed implementation for torch.vstack().

    vstack = cat(atleast_2d(*tensors), dim=0).

    Inherits from ConcatDistributedOp to reuse cat's layout validation
    (Partial check, dim-0 Replicate constraint, same-layout requirement).
    Only the atleast_2d promotion and DTensor enforcement are added.
    """

    def preprocess(self, args: tuple, kwargs: dict) -> tuple:
        """Preprocess arguments for vstack operator.

        Responsibilities:
          - Normalize args/kwargs
          - Enforce all-DTensor and out=None
          - Extract local tensors
          - Cache original layouts (no promotion — that's infer_layout's job)

        Args:
            args: Raw positional args from call site.
            kwargs: Raw keyword args from call site.

        Returns:
            tuple: (local_args, local_kwargs, cache_values)
                cache_values = [original_layout_0, ..., original_layout_n-1]
        """
        args, kwargs = _normalize_vstack_args(*args, **kwargs)
        tensors = args[0]
        out = kwargs["out"]

        if out is not None:
            raise ValueError(
                f"For {self.op_name}, out keyword is not supported. "
                f"vstack currently only supports out=None."
            )

        # Enforce all-DTensor policy
        for i, t in enumerate(tensors):
            if not isinstance(t, DTensor):
                raise ValueError(
                    f"For {self.op_name}, all inputs must be DTensor, "
                    f"but input {i} is {type(t).__name__}."
                )

        local_tensors = tuple(t.to_local() for t in tensors)

        local_args = (local_tensors,)
        local_kwargs = {}
        cache_values = [t.layout for t in tensors]
        return local_args, local_kwargs, cache_values

    def infer_layout(self, cache_values: list) -> Tuple[tuple, None]:
        """Infer output layout for vstack operator.

        1. Check Partial on original layouts (before promotion)
        2. Promote each layout via atleast_2d
        3. Delegate to parent ConcatDistributedOp with promoted layouts + [0]

        Args:
            cache_values: [original_layout_0, ..., original_layout_n-1]

        Returns:
            tuple: ((output_layout,), None)

        Raises:
            ValueError: If layout constraints are violated.
        """
        original_layouts = cache_values

        # Check Partial on original layouts before promotion
        if not self._allow_partial_inputs:
            self._check_partial_inputs(original_layouts)

        # Apply atleast_2d promotion to each layout
        promoted_layouts = []
        for layout in original_layouts:
            promoted_map = _promote_tensor_map(layout.alias_tensor_map)
            if promoted_map == layout.alias_tensor_map:
                # ndim >= 2, unchanged → reuse original
                promoted_layouts.append(layout)
            else:
                promoted = Layout(
                    mesh_shape=layout.mesh_shape,
                    alias_name=layout.alias_name,
                    rank_list=layout.rank_list,
                )
                promoted = promoted(*promoted_map)
                promoted_layouts.append(promoted)

        # Delegate to parent: partial already checked, promoted layouts + dim=0
        return super().infer_layout(promoted_layouts + [0])

    # get_expand_impl not overridden — returns None from parent.
