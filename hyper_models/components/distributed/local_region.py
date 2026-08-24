# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""local_region: a DTensor -> local -> DTensor local-compute-region wrapper.

Built on the ``core.shard.custom_shard`` skeleton, with three enhancements for
the 05 dual-mode DTensor design (the MoE local_map / CP attention internal
regions in validate mode):

1. **Named-parameter binding**: ``in_placements`` is a
   ``dict[str, placements]``, aligned with the dict contract of
   ``ModuleShardingSpec.in_dst``; positional args are mapped to parameter
   names via ``inspect.signature``, and kwargs are natively supported (HF
   forwards are predominantly kwargs-based).
2. **Tolerant passthrough**: inputs that are not DTensors are passed through
   as-is (the production path where parameters are already unwrapped); outputs
   that are already DTensors are not re-wrapped; when no input is a DTensor,
   outputs are not wrapped either.

**Why there is no backward stitching (important)**: hyper_parallel's DTensor
is an in-house **forward-only** placement/dispatch system; the backward pass
does not go through DTensor (there is no DTensor autograd). Therefore this
function only performs forward unwrap/wrap and contains -- and needs -- no
autograd.Function stitching or gradient-placement declarations (unlike
PyTorch ``local_map`` / Titan ``LocalMapConfig.in_grad_placements``, which
exist because torch DTensor has backward semantics). Backward inside the
region is plain autograd on local tensors, with gradients landing directly on
the local parameter shards, consistent with production mode.

Relationship with production mode: production's forward wrapper
(``_wrap_local_region_forward``) permanently unwraps parameters to plain
tensors at build time, and boundary communication is executed by
``PrecompiledBoundary``; it does not use this function. This function serves
**validate mode** (parameters stay DTensors and the region boundary needs
DTensor-contract stitching) and standalone use.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Sequence

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Placement

# Placements of a single tensor: tuple[Placement, ...], aligned with the mesh dims
Placements = Sequence[Placement]


def _bind_arg_names(func: Callable) -> Dict[str, int]:
    """Map positional-argument positions to parameter names (for name-based
    lookup under the dict contract).

    When the signature cannot be introspected (C extensions, etc.), an empty
    dict is returned -- in that case only kwargs-passed arguments can be
    matched by in_placements, and all positional arguments pass through.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    return {
        name: idx
        for idx, (name, p) in enumerate(sig.parameters.items())
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    }


def _normalize_out_placements(out_placements, num_outputs: int):
    """Normalize out_placements into a per-output
    tuple[tuple[Placement, ...] | None, ...].

    Accepted forms:
      - flat single-output form ``(Partial(), Replicate())`` (all elements
        are Placements);
      - per-output form ``((Partial(), Replicate()), None, (Shard(1), Replicate()))``
        (any element is a tuple or None; length must equal the output count).
    """
    if len(out_placements) == 0:
        raise ValueError("out_placements must not be empty")
    per_output = any(p is None or isinstance(p, tuple) for p in out_placements)
    if per_output:
        if len(out_placements) != num_outputs:
            raise ValueError(
                f"out_placements count {len(out_placements)} does not match "
                f"output count {num_outputs}!"
            )
        return tuple(out_placements)
    if num_outputs != 1:
        raise ValueError(
            f"flat out_placements only valid for single-output functions, "
            f"got {num_outputs} outputs!"
        )
    return (tuple(out_placements),)


def local_region(
    func: Optional[Callable] = None,
    *,
    device_mesh: DeviceMesh,
    in_placements: Optional[Dict[str, Optional[Placements]]] = None,
    out_placements: Optional[Sequence[Optional[Placements]]] = None,
    redistribute_inputs: bool = False,
) -> Callable:
    """Wrap func into a DTensor -> local -> DTensor local compute region (forward).

    Args:
        func: the function to wrap (a forward or any callable). Can also be
            used as a decorator factory.
        device_mesh: the mesh used for DTensor construction / redistribution.
        in_placements: ``{arg_name: placements}`` -- the expected placements
            of each DTensor input at the region entry. Inputs left as None
            (value) or not listed are not redistributed; non-DTensor inputs
            always pass through.
        out_placements: placement declarations for the outputs at the region
            exit. A single output may be written flat as
            ``(Partial(), Replicate())``; multiple outputs are written
            position-by-position, with None as the placeholder for non-tensor
            outputs. When None, outputs are not wrapped (returned as-is).
        redistribute_inputs: whether to redistribute inputs to the placements
            declared in in_placements at the entry. In dual-mode scenarios the
            boundary communication is already done by PrecompiledBoundary, so
            pass False (default); pass True for standalone use.

    Returns:
        The wrapped function, with the same signature as func.

    Example:
        >>> # validate-mode MoE module: boundary DTensor contract preserved,
        >>> # internal local all-to-all
        >>> wrapped = local_region(
        ...     moe.forward, device_mesh=mesh,
        ...     in_placements={"hidden_states": (Replicate(), Replicate())},
        ...     out_placements=(Partial(), Replicate()),
        ... )

        >>> # decorator form (standalone use, entry redistributes by itself)
        >>> @local_region(device_mesh=mesh,
        ...               in_placements={"x": (Shard(0),)},
        ...               out_placements=((Shard(0),),),
        ...               redistribute_inputs=True)
        ... def my_fn(x, bias=None):
        ...     return x + bias
    """
    def decorator(fn: Callable) -> Callable:
        """Build the local-region wrapper around one concrete function."""
        name_to_idx = None  # lazily cached signature mapping

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            """Unwrap DTensor inputs to local, run fn, and re-wrap outputs."""
            nonlocal name_to_idx
            if name_to_idx is None:
                name_to_idx = _bind_arg_names(fn)

            args = list(args)
            saw_dtensor = False

            if in_placements:
                for name, placements in in_placements.items():
                    if name in kwargs:
                        from_kwargs, idx = True, None
                        value = kwargs[name]
                    else:
                        from_kwargs = False
                        idx = name_to_idx.get(name)
                        if idx is None or idx >= len(args):
                            continue
                        value = args[idx]

                    if not isinstance(value, DTensor):
                        # non-DTensor input (already unwrapped by production /
                        # non-tensor argument) -> passthrough
                        continue
                    saw_dtensor = True

                    dt = value
                    if (redistribute_inputs and placements is not None
                            and tuple(dt.placements) != tuple(placements)):
                        dt = dt.redistribute(device_mesh, placements)

                    local_value = dt.to_local()
                    if from_kwargs:
                        kwargs[name] = local_value
                    else:
                        args[idx] = local_value

            out = fn(*args, **kwargs)

            if not saw_dtensor or out_placements is None:
                return out

            single = not isinstance(out, tuple)
            out_items = (out,) if single else out
            placements_items = _normalize_out_placements(out_placements, len(out_items))

            wrapped_out = []
            for item, placements in zip(out_items, placements_items):
                if isinstance(item, DTensor):
                    # already wrapped inside the region -> do not re-wrap
                    wrapped_out.append(item)
                elif isinstance(item, torch.Tensor):
                    if placements is None:
                        raise TypeError(
                            "Tensor output requires non-None out_placements entry!"
                        )
                    wrapped_out.append(
                        DTensor.from_local(item, device_mesh, tuple(placements))
                    )
                else:
                    wrapped_out.append(item)

            return wrapped_out[0] if single else tuple(wrapped_out)

        return wrapped

    if func is not None:
        return decorator(func)
    return decorator
