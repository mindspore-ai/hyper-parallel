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
"""Run real MindSpore public APIs on a local RaggedShard DTensor."""
import json
import os
import sys

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

# The backend must be selected before importing hyper_parallel.
# pylint: disable=wrong-import-position
import mindspore as ms
from mindspore import mint, ops

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import RaggedShard
from hyper_parallel.core.shard._op_dispatch import _debug_mode_observer


class _DispatchObserver:
    """Record canonical names without replacing the dispatched callable."""

    def __init__(self) -> None:
        """Initialize an empty dispatch-name list."""
        self.names = []

    def on_op_dispatch_enter(
        self, op_name: str, op_call: object, args: tuple, kwargs: dict
    ) -> None:
        """Record the name already resolved by the platform dispatcher."""
        del op_call, args, kwargs
        self.names.append(op_name)

    def on_op_dispatch_exit(self, op_name: str, result: object) -> None:
        """Satisfy the observer protocol without changing the result."""
        del op_name, result


def _make_dtensor(values) -> DTensor:
    """Create a one-rank CPU RaggedShard DTensor."""
    local = ms.Tensor(values, dtype=ms.float32)
    layout = Layout((1,), ("ragged",), init_backend=False)
    layout.set_placements((RaggedShard(dims=(0,), local_units=(1,)),))
    layout.placement_to_tensor_map(2)
    return DTensor.from_local_with_layout(local, layout, shape=(2, 2))


def _run(api_name, expected_op_name, function, x, y, expected_error=None):
    """Execute one real public API and return its observed dispatcher name."""
    observer = _DispatchObserver()
    token = _debug_mode_observer.set(observer)
    output = None
    try:
        try:
            output = function(x, y)
        except RuntimeError as error:
            if expected_error is None or expected_error not in str(error):
                raise
    finally:
        _debug_mode_observer.reset(token)
    if observer.names != [expected_op_name]:
        raise AssertionError(
            f"{api_name} dispatch mismatch: expected={[expected_op_name]!r}, got={observer.names!r}"
        )
    if output is not None and tuple(output.placements) != tuple(x.placements):
        raise AssertionError(
            f"{api_name} placements mismatch: expected={tuple(x.placements)!r}, "
            f"got={tuple(output.placements)!r}"
        )
    return {"api": api_name, "op_name": expected_op_name}


def main() -> None:
    """Run the complete requested MindSpore whitelist interface matrix."""
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    x = _make_dtensor([1.0, 2.0, 3.0, 4.0])
    y = _make_dtensor([2.0, 3.0, 4.0, 5.0])
    # Lambdas normalize unary, binary, and reverse APIs to one test signature.
    # pylint: disable=unnecessary-lambda
    regular_cases = (
        ("mint.abs(x)", "Abs", lambda a, _: mint.abs(a)),
        ("ops.absolute(x)", "Abs", lambda a, _: ops.absolute(a)),
        ("mint.cos(x)", "Cos", lambda a, _: mint.cos(a)),
        ("mint.exp(x)", "Exp", lambda a, _: mint.exp(a)),
        ("ops.GeLU()(x)", "GeLU", lambda a, _: ops.GeLU()(a)),
        ("mint.isinf(x)", "IsInf", lambda a, _: mint.isinf(a)),
        ("ops.isnan(x)", "IsNan", lambda a, _: ops.isnan(a)),
        ("mint.log(x)", "Log", lambda a, _: mint.log(a)),
        ("mint.neg(x)", "Neg", lambda a, _: mint.neg(a)),
        ("mint.negative(x)", "Neg", lambda a, _: mint.negative(a)),
        ("ops.relu(x)", "ReLU", lambda a, _: ops.relu(a)),
        ("mint.rsqrt(x)", "Rsqrt", lambda a, _: mint.rsqrt(a)),
        ("mint.sigmoid(x)", "Sigmoid", lambda a, _: mint.sigmoid(a)),
        ("mint.nn.functional.silu(x)", "SiLU", lambda a, _: mint.nn.functional.silu(a)),
        ("mint.sin(x)", "Sin", lambda a, _: mint.sin(a)),
        ("mint.sqrt(x)", "Sqrt", lambda a, _: mint.sqrt(a)),
        ("mint.square(x)", "Square", lambda a, _: mint.square(a)),
        ("ops.add(x, y)", "Add", lambda a, b: ops.add(a, b)),
        ("mint.add(x, y)", "AddExt", lambda a, b: mint.add(a, b)),
        ("mint.div(x, y)", "Div", lambda a, b: mint.div(a, b)),
        ("mint.mul(x, y)", "Mul", lambda a, b: mint.mul(a, b)),
        ("mint.pow(x, y)", "Pow", lambda a, b: mint.pow(a, b)),
        ("ops.RealDiv()(x, y)", "RealDiv", lambda a, b: ops.RealDiv()(a, b)),
        ("ops.sub(x, y)", "Sub", lambda a, b: ops.sub(a, b)),
        ("mint.sub(x, y)", "SubExt", lambda a, b: mint.sub(a, b)),
        ("2.0 - x", "Sub", lambda a, _: 2.0 - a),
        ("2.0 ** x", "Pow", lambda a, _: 2.0 ** a),
        ("ops.true_divide(x, y)", "Div", lambda a, b: ops.true_divide(a, b)),
    )
    kernel_limited_cases = {
        "clone": (
            "mint.clone(x)", "Clone", lambda a, _: mint.clone(a),
            "kernel Clone unregistered",
        ),
        "gelu_ext": (
            "mint.nn.functional.gelu(x)", "GeluExt",
            lambda a, _: mint.nn.functional.gelu(a),
            "kernel GeluExt unregistered",
        ),
    }
    selection = sys.argv[1] if len(sys.argv) > 1 else "regular"
    if selection == "regular":
        records = [
            _run(api, op_name, function, x, y)
            for api, op_name, function in regular_cases
        ]
    elif selection in kernel_limited_cases:
        api, op_name, function, expected_error = kernel_limited_cases[selection]
        records = [_run(api, op_name, function, x, y, expected_error)]
    else:
        raise ValueError(f"Unknown case selection: {selection!r}")
    print("RAGGED_OP_RECORDS=" + json.dumps(records, sort_keys=True))


if __name__ == "__main__":
    main()
