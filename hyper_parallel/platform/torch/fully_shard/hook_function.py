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
"""add post backward hook function"""
import torch


class PostBackwardFunction(torch.autograd.Function):
    """Post backward hook function"""

    @staticmethod
    def forward(ctx, hsdp_scheduler, *inputs):
        ctx.hsdp_scheduler = hsdp_scheduler
        return inputs

    @staticmethod
    def backward(ctx, *grads):
        # pylint: disable=W0212
        ctx.hsdp_scheduler._backward_hook()
        return (None,) + grads

    @classmethod
    def apply(cls, *args, **kwargs):
        """Override apply function to handle DTensor inputs"""
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

        origin_output = super().apply(*input_args, **kwargs)

        if len(origin_output) != len(input_args) - 1:
            raise RuntimeError("number of output should equal to number of input minus 1")

        if isinstance(origin_output, (tuple, list)):
            output = ()
            for i, output_item in enumerate(origin_output):
                item_layout = input_layouts[i+1]
                if item_layout is None:
                    output += (output_item,)
                else:

                    output += (DTensor.from_local(output_item, item_layout.mesh, item_layout.placements),)
            return output
        return origin_output
