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
"""HSDP async gradient hook"""
from hyper_parallel.core.hsdp.hsdp_async_grad_hook import HSDPAsyncGradHook


class MindSporeHSDPAsyncGradHook(HSDPAsyncGradHook):
    """HSDP gradient hook with async communication op"""

    def _get_final_grad_hook(self, param, grad_hook, post_hook=None):
        """add cast and scale grad"""
        final_hook = super()._get_final_grad_hook(param, grad_hook, post_hook)
        def set_grad_hook(grad):
            grad = final_hook(grad)
            param.grad = grad
            return grad

        return set_grad_hook
