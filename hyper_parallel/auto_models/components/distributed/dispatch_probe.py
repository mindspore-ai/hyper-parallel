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
"""dispatch_probe: a development-time tool for deciding whether
region_dispatch should be True or False.

The axiom requires "region_dispatch to be filled when declaring an
injection", and answering it correctly requires knowing whether the
injected function can be fully dispatched through DTensor dispatch. This
tool moves the trial-and-error from apply time to development time: it
dry-runs the injected function with DTensors, records the dispatch trace,
reports the first failing operator, and suggests what to fill in.

Usage::

    from hyper_parallel.auto_models.components.distributed.dispatch_probe import (
        check_dispatchable)

    report = check_dispatchable(my_compute_fn, example_inputs, mesh)
    print(report)   # dispatchable=True/False + suggestion + failing op

Decision criteria (consistent with the axiom):
- No exception during the whole dry run -> the injected function uses only
  standard operators, so ``region_dispatch=True`` may be filled;
- Any operator failing during DTensor dispatch/propagation (unsupported
  operator, communication primitive acting on a DTensor, data-dependent
  branching, etc.) -> fill ``False`` (black-box hosting).

Note: the tool only verifies "whether dispatch works", not numerical or
layout correctness -- after a True result, the validate mode's out_src
verification is still required as the safety net.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate

logger = logging.getLogger(__name__)


@dataclass
class DispatchProbeReport:
    """The verdict report of check_dispatchable (readable via print).

    Attributes:
        dispatchable: True means the whole dry run dispatched through
            DTensor without exceptions (region_dispatch=True may be filled).
        ops: The dispatch trace (all aten ops, including DTensor-internal
            redistribution).
        failed_op: The first failing operator (provided when
            dispatchable=False).
        error: Summary of the failure exception (provided when
            dispatchable=False).
        recommendation: Suggested value for region_dispatch.
    """

    dispatchable: bool

    ops: List[str] = field(default_factory=list)

    failed_op: Optional[str] = None

    error: Optional[str] = None

    recommendation: str = ""

    def __str__(self) -> str:
        """Render the report as a human-readable multi-line summary."""
        lines = [
            "=== check_dispatchable report ===",
            f"dispatchable: {self.dispatchable}",
            f"dispatch trace: {len(self.ops)} ops"
            + (f" (tail: {', '.join(self.ops[-3:])})" if self.ops else ""),
        ]
        if not self.dispatchable:
            lines.append(f"first failing op: {self.failed_op}")
            lines.append(f"exception: {self.error}")
        lines.append(f"recommendation: {self.recommendation}")
        return "\n".join(lines)


class _OpRecorder(TorchDispatchMode):
    """Record the dispatch trace and then let ops through unchanged (the
    DTensor subclass propagation logic runs as usual).

    ``in_flight`` keeps the chain of operators that entered but never
    returned normally -- on an exception its tail is the operator that
    actually failed (the tail of ``ops`` may only be the last operator that
    completed before the failure).
    """

    def __init__(self) -> None:
        """Initialize empty dispatch-trace and in-flight stacks."""
        self.ops: List[str] = []
        self.in_flight: List[str] = []

    def __torch_dispatch__(self, func, _types, args=(), kwargs=None):
        name = str(func).replace("aten.", "")
        self.ops.append(name)
        self.in_flight.append(name)
        result = func(*args, **(kwargs or {}))   # on exception, in_flight keeps the scene
        self.in_flight.pop()
        return result


def check_dispatchable(
    fn: Any,
    example_inputs: Sequence[Any],
    mesh: Any,
    *,
    placements: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> DispatchProbeReport:
    """Dry-run an injected function with DTensors to decide whether
    region_dispatch should be True or False.

    Args:
        fn: The injected object to judge -- the pure-function form of a
            regional compute fn ``fn(module, *args)``, a plain callable, or
            an nn.Module (its forward is called).
        example_inputs: Local example inputs of this rank (each tensor item
            is wrapped via ``DTensor.from_local``; non-tensor items pass
            through unchanged).
        mesh: The target DeviceMesh (a single dp-slice consistent with the
            plan's coordinate system).
        placements: The placements used for wrapping; defaults to all
            Replicate (the most permissive entry; pass explicitly, e.g.
            ``(Shard(1),)``, when simulating real entry layouts).
        kwargs: Keyword arguments forwarded to fn (tensor values are wrapped
            in the same way).

    Returns:
        DispatchProbeReport -- ``dispatchable=True`` suggests
        ``region_dispatch=True``; ``False`` comes with the first failing
        operator and an exception summary, and suggests
        ``region_dispatch=False``.
    """
    if placements is None:
        placements = tuple(Replicate() for _ in range(mesh.ndim))

    def wrap(value: Any) -> Any:
        """Wrap a plain tensor into a DTensor; pass other values through."""
        if isinstance(value, torch.Tensor) and not isinstance(value, DTensor):
            return DTensor.from_local(value, mesh, tuple(placements))
        return value

    dt_args = [wrap(v) for v in example_inputs]
    dt_kwargs = {k: wrap(v) for k, v in (kwargs or {}).items()}
    target = fn
    swapped_params = []
    if isinstance(fn, torch.nn.Module):
        # Module form: parameters are also dispatch participants -- wrap
        # them into DTensors temporarily (restored after the dry run),
        # otherwise "DTensor input x plain parameter" would fail with a
        # mixing error at the very first operator, which does not mean the
        # injected object itself is not dispatchable
        target = fn.forward
        for submodule in fn.modules():
            for key, param in list(submodule._parameters.items()):
                if param is None or isinstance(param, DTensor):
                    continue
                swapped_params.append((submodule, key, param))
                submodule._parameters[key] = DTensor.from_local(
                    param.detach(), mesh, tuple(placements))

    recorder = _OpRecorder()
    try:
        with recorder:
            target(*dt_args, **dt_kwargs)
    except Exception as exc:  # noqa: BLE001 -- the probe's duty: classify every failure as not dispatchable
        if recorder.in_flight:
            failed = recorder.in_flight[-1]
        else:
            # Operators going through the __torch_function__ channel (e.g.
            # c10d communication) do not pass through this mode; extract the
            # operator name from the exception text ("Operator all_gather
            # does not ...")
            m = re.search(r"Operator '?([A-Za-z0-9_.]+)'?", str(exc))
            failed = (m.group(1) if m
                      else (recorder.ops[-1] if recorder.ops
                            else "(no operator trace -- failed before the call)"))
        return DispatchProbeReport(
            dispatchable=False,
            ops=recorder.ops,
            failed_op=failed,
            error=f"{type(exc).__name__}: {exc}"[:500],
            recommendation=(
                "region_dispatch=False -- the injected object contains "
                "operators/communication/data-dependent logic that cannot be "
                "dispatched (skeleton black-box hosting: to_local -> local "
                "execution -> declarative re-wrapping)"),
        )
    finally:
        for submodule, key, param in swapped_params:
            submodule._parameters[key] = param   # restore plain parameters
    return DispatchProbeReport(
        dispatchable=True,
        ops=recorder.ops,
        recommendation=(
            "region_dispatch=True may be filled -- the whole dry run "
            "dispatched through DTensor without exceptions (pure standard "
            "operators; the validate mode's out_src verification is still "
            "required as the safety net for layout correctness)"),
    )


__all__ = ["DispatchProbeReport", "check_dispatchable"]
