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
"""HSDP gradient hook"""
from mindspore import ops
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from hyper_parallel.core.hsdp.hsdp_grad_hook import HSDPGradHook


class MindSporeHSDPGradHook(HSDPGradHook):
    """MindSpore HSDP gradient hook"""

    def __init__(self, config, platform):
        """init"""
        super().__init__(config, platform)
        if not self.use_eager_hook:
            self.requires_grad_sync = Parameter(Tensor(False), name="hsdp_requires_grad_sync", requires_grad=False)

    def _cast_hook(self, hook, grad):
        """add cast before and after reduce hook"""
        if self.reduce_dtype is None:
            return hook(grad)
        origin_dtype = ops.dtype(grad)
        grad_cast = ops.cast(grad, self.reduce_dtype)
        output = hook(grad_cast)
        output = ops.cast(output, origin_dtype)
        return output

    def _get_final_grad_hook(self, param, grad_hook, no_cast=False):
        """add cast and scale grad"""
        final_hook = super()._get_final_grad_hook(param, grad_hook, no_cast)
        def set_grad_hook(grad):
            grad = final_hook(grad)
            param.grad = grad
            return grad

        if self.use_eager_hook:
            return set_grad_hook
        return final_hook

    def set_requires_grad_sync(self, requires_grad_sync):
        """set requires grad sync flag to control gradient sync."""
        if self.use_eager_hook:
            self.requires_grad_sync = requires_grad_sync
        else:
            ops.assign(self.requires_grad_sync, Tensor(requires_grad_sync))
    