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
"""Torch HSDP async gradient hook"""
from hyper_parallel.core.fully_shard.hsdp_async_grad_hook import HSDPAsyncGradHook


class TorchHSDPAsyncGradHook(HSDPAsyncGradHook):
    """
    Torch HSDP gradient hook for handling gradient operations in hybrid sharded data parallel training.
    
    This class extends the base HSDPAsyncGradHook to provide PyTorch-specific async gradient hook functionality,
    including gradient scaling and parameter gradient management.
    """

    def _get_final_async_grad_hook(self, param, async_hook, post_hook=None):
        """
        Create a final async gradient hook with post hook.
        
        Args:
            param: The parameter tensor to apply the gradient hook to
            async_hook: The async gradient hook function
            post_hook: post-processing hook, defaults to None
            
        Returns:
            function: A async gradient hook function that processes gradients with scaling,
                    and parameter gradient management
        """
        final_hook = super()._get_final_async_grad_hook(param, async_hook, post_hook)
        def set_grad_hook(grad):
            final_grad = final_hook(grad)
            grad.data = final_grad
            param.grad = None
            return grad

        return set_grad_hook
