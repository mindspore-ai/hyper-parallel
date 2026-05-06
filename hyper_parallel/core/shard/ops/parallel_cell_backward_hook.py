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
"""CellBackwardHook distributed op."""

from .parallel_tuple_elementwise import TupleElementWiseDistributedOp


class CellBackwardHookDistributedOp(TupleElementWiseDistributedOp):
    """Distributed op for MindSpore CellBackwardHook.

    Hook outputs may contain a mix of DTensor-backed slots and plain local
    Tensor slots. The local slots should be passed through unchanged.
    """

    def wrap_output(self, py_output, output_layouts):
        """Wrap outputs while preserving local Tensor slots."""
        # pylint: disable=C0415
        from hyper_parallel.core.dtensor.dtensor import DTensor

        if isinstance(py_output, (tuple, list)):
            if len(py_output) != len(output_layouts):
                raise RuntimeError(
                    f"Output tuple size ({len(py_output)}) "
                    f"does not match layout tuple size ({len(output_layouts)})")
            output = ()
            for item, layout in zip(py_output, output_layouts):
                if layout is None:
                    output += (item,)
                else:
                    output += (DTensor.from_local(item, layout.mesh, layout.alias_placements),)
            return output

        if output_layouts[0] is None:
            return py_output
        return DTensor.from_local(
            py_output, output_layouts[0].mesh, output_layouts[0].alias_placements
        )
