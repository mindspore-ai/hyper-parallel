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
"""Torch function override"""
from torch.nn.modules import _functions
from torch.nn.modules._functions import BackwardHookFunction


class DTensorBackwardHookFunction(BackwardHookFunction):
    """override BackwardHookFunction for dtensor"""

    @classmethod
    def apply(cls, *args, **kwargs):
        """override apply function"""
        # pylint: disable=C0415
        from hyper_parallel import DTensor

        input_args = []
        input_layouts = []

        for arg in args:
            if arg is None:
                input_layouts.append(None)
                input_args.append(arg)
                continue

            if not hasattr(arg, "_layout"):
                input_layouts.append(None)
                input_args.append(arg)
            else:
                layout = arg.layout
                input_layouts.append(layout)
                input_args.append(arg.to_local())

        origin_output = BackwardHookFunction.apply(*input_args, **kwargs)

        if len(origin_output) != len(input_args):
            raise RuntimeError("number of output should equal to number of input")

        if isinstance(origin_output, (tuple, list)):
            output = ()
            for i, output_item in enumerate(origin_output):
                if input_layouts[i] is None:
                    output += (output_item,)
                else:
                    output += (DTensor.from_local(output_item, input_layouts[i]),)
            return output
        return origin_output


def override_functions():
    _functions.BackwardHookFunction = DTensorBackwardHookFunction
