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
"""Post-backward autograd hook that preserves DTensor wrapper types."""
import torch


class PostBackwardFunction(torch.autograd.Function):
    """Post backward hook function"""

    @staticmethod
    def forward(ctx, hsdp_scheduler, *inputs):
        """Save the scheduler reference and pass inputs through unchanged."""
        ctx.hsdp_scheduler = hsdp_scheduler
        return inputs

    @staticmethod
    def backward(ctx, *grads):
        """Trigger the scheduler's backward hook and pass gradients through unchanged."""
        # pylint: disable=W0212
        ctx.hsdp_scheduler._backward_hook()
        return (None,) + grads

    @classmethod
    def apply(cls, *args, **kwargs):
        """Override apply function to handle DTensor inputs"""
        input_args = []
        input_wrappers = []
        for arg in args:
            if arg is None:
                input_wrappers.append(None)
                input_args.append(arg)
                continue
            if not hasattr(arg, "layout") or not callable(getattr(arg, "to_local", None)):
                input_wrappers.append(None)
                input_args.append(arg)
            else:
                layout = arg.layout
                from_local = getattr(type(arg), "from_local", None)
                if not callable(from_local):
                    # Legacy DTensor-like wrappers carried only ``_layout``;
                    # retain their historical reconstruction path while new
                    # extension wrappers use their own class constructor.
                    from hyper_parallel import DTensor  # pylint: disable=import-outside-toplevel
                    from_local = DTensor.from_local
                if not hasattr(layout, "mesh") or not hasattr(layout, "alias_placements"):
                    raise TypeError(
                        f"{type(arg).__name__}.layout must expose mesh and alias_placements."
                    )
                input_wrappers.append((from_local, layout))
                input_args.append(arg.to_local())

        origin_output = super().apply(*input_args, **kwargs)

        if len(origin_output) != len(input_args) - 1:
            raise RuntimeError("number of output should equal to number of input minus 1")

        if isinstance(origin_output, (tuple, list)):
            output = ()
            for i, output_item in enumerate(origin_output):
                wrapper = input_wrappers[i + 1]
                if wrapper is None:
                    output += (output_item,)
                else:
                    from_local, layout = wrapper
                    output += (
                        from_local(output_item, layout.mesh, layout.alias_placements),
                    )
            return output
        return origin_output
