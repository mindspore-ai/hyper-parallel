# Copyright 2025 Huawei Technologies Co., Ltd
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
"""shard"""

import queue
from typing import Callable, Tuple, Optional
import mindspore as ms

import dist_parallel
from .layout import Layout

def custom_shard(
        func: Callable,
        out_layouts: Tuple[Layout, ...],
        in_layouts: Optional[Tuple[Optional[Layout], ...]] = None,
        redistribute_inputs: bool = True,
) -> Callable:
    """
    Wraps a function to handle distributed tensor conversions.

    Args:
        func (Callable): The function to be wrapped.
        out_layouts (Tuple[Layout, ...]): Layouts for each output tensor.
        in_layouts (Optional[Tuple[Optional[Layout], ...]], optional):
            Layouts for each input argument. None entries indicate non-tensor inputs.
        redistribute_inputs (bool): Whether to redistribute inputs to required layouts.

    Returns:
        Callable: Wrapped function that handles distributed tensors.
    """
    def wrapped(*args, **kwargs):
        if in_layouts is not None:
            assert len(in_layouts) == len(args), (
                f"in_placements length {len(in_layouts)} does not match "
                f"the number of input args {len(args)}!"
            )

        local_args = []
        contain_distributed_arg = False

        args_layout = queue.Queue(len(args))
        for i, arg in enumerate(args):
            if isinstance(arg, dist_parallel.DTensor):
                if in_layouts is None:
                    raise RuntimeError("Found Tensor input but in_layouts is None")

                required_in_layout = in_layouts[i]
                if required_in_layout is None:
                    raise TypeError(
                        f"Tensor input at position {i} requires Layout, "
                        "but corresponding in_layouts entry is None!"
                    )

                if redistribute_inputs:
                    arg = arg.redistribute(required_in_layout)

                args_layout.put(arg.layout)
                local_tensor = arg.to_local()
                local_args.append(local_tensor)
                contain_distributed_arg = True

            else:
                if in_layouts is not None and in_layouts[i] is not None:
                    raise TypeError(
                        f"Non-DTensor input at position {i} requires None in_layouts, "
                        f"but received {in_layouts[i]}!"
                    )
                local_args.append(arg)

        out = func(*local_args, **kwargs)

        if not contain_distributed_arg:
            return out

        out_is_tuple = isinstance(out, tuple)
        out_tuple = (out,) if not out_is_tuple else out

        assert len(out_tuple) == len(out_layouts), (
            f"Output count {len(out_tuple)} does not match "
            f"out_layouts count {len(out_layouts)}!"
        )

        dist_output = []
        for item, out_layout in zip(out_tuple, out_layouts):
            if isinstance(item, ms.Tensor):
                if out_layout is None:
                    raise TypeError(
                        "Tensor output requires non-None out_layout!"
                    )
                dist_output.append(dist_parallel.DTensor.from_local(item, out_layout))
            else:
                if out_layout is not None:
                    raise TypeError(
                        f"Non-tensor output requires None out_layout, got {out_layout}!"
                    )
                dist_output.append(item)

        return dist_output[0] if not out_is_tuple else tuple(dist_output)

    return wrapped
