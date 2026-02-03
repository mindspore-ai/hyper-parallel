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
from hyper_parallel.core.layout import DeviceMesh
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.placement_types import Placement
from hyper_parallel.platform import get_platform
platform = get_platform()
Tensor = platform.Tensor

def custom_shard(
        func: Callable,
        device_mesh: DeviceMesh,
        out_placements: Tuple[Tuple[Placement, ...], ...],
        in_placements: Optional[Tuple[Optional[Tuple[Placement, ...]], ...]] = None,
        redistribute_inputs: bool = True,
) -> Callable:
    """
    Wraps a function to handle distributed tensor conversions.

    Args:
        func (Callable): The function to be wrapped.
        device_mesh (DeviceMesh): The device mesh for sharding.
        out_placements (Tuple[Tuple[Placement, ...], ...]): Placements for each output tensor.
        in_placements (Optional[Tuple[Optional[Tuple[Placement, ...]], ...]], optional):
            Placements for each input argument. None entries indicate non-tensor inputs.
        redistribute_inputs (bool): Whether to redistribute inputs to required placements.

    Returns:
        Callable: Wrapped function that handles distributed tensors.

    Examples:
        >>> mesh = DeviceMesh("npu", (2, 2), nesh_dim_names=("dp", "tp"))
        >>> @custom_shard(
        ...     device_mesh=mesh,
        ...     out_placements=((Shard(0), Replicate()),),
        ...     in_placements=((Shard(0), Replicate()), (Replicate(), Shard(1)))
        ... )
        ... def my_func(x, y):
        ...     return x + y
    """
    def wrapped(*args, **kwargs):
        if in_placements is not None:
            assert len(in_placements) == len(args), (
                f"in_placements length {len(in_placements)} does not match "
                f"the number of input args {len(args)}!"
            )

        local_args = []
        contain_distributed_arg = False

        args_layout = queue.Queue(len(args))
        for i, arg in enumerate(args):
            if isinstance(arg, DTensor):
                if in_placements is None:
                    raise RuntimeError("Found Tensor input but in_placements is None")

                required_in_placement = in_placements[i]
                if required_in_placement is None:
                    raise TypeError(
                        f"Tensor input at position {i} requires Placement, "
                        "but corresponding in_placements entry is None!"
                    )

                if redistribute_inputs:
                    arg = arg.redistribute(device_mesh, required_in_placement)

                args_layout.put(arg.layout)
                local_tensor = arg.to_local()
                local_args.append(local_tensor)
                contain_distributed_arg = True

            else:
                if in_placements is not None and in_placements[i] is not None:
                    raise TypeError(
                        f"Non-DTensor input at position {i} requires None in_placements, "
                        f"but received {in_placements[i]}!"
                    )
                local_args.append(arg)

        out = func(*local_args, **kwargs)

        if not contain_distributed_arg:
            return out

        out_is_tuple = isinstance(out, tuple)
        out_tuple = (out,) if not out_is_tuple else out

        assert len(out_tuple) == len(out_placements), (
            f"Output count {len(out_tuple)} does not match "
            f"out_placements count {len(out_placements)}!"
        )

        dist_output = []
        for item, out_placement in zip(out_tuple, out_placements):
            if isinstance(item, Tensor):
                if out_placement is None:
                    raise TypeError(
                        "Tensor output requires non-None out_placements!"
                    )
                dist_output.append(
                    DTensor.from_local(item, device_mesh=device_mesh, placements=out_placement)
                )
            else:
                if out_placement is not None:
                    raise TypeError(
                        f"Non-tensor output requires None out_placements, got {out_placement}!"
                    )
                dist_output.append(item)

        return dist_output[0] if not out_is_tuple else tuple(dist_output)

    return wrapped
