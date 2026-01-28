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
"""Torch HSDP gradient hook"""
from hyper_parallel.core.hsdp_refactor.hsdp_grad_hook import HSDPGradHook


class TorchHSDPGradHook(HSDPGradHook):
    """
    Torch HSDP gradient hook for handling gradient operations in hybrid sharded data parallel training.
    
    This class extends the base HSDPGradHook to provide PyTorch-specific gradient hook functionality,
    including gradient casting, scaling and parameter gradient management.
    """

    def _get_final_grad_hook(self, param, grad_hook, no_cast=False):
        """
        Create a final gradient hook that adds casting and scaling operations.
        
        Args:
            param: The parameter tensor to apply the gradient hook to
            grad_hook: The base gradient hook function
            no_cast (bool): Whether to skip gradient casting operations, defaults to False
            
        Returns:
            function: A gradient hook function that processes gradients with casting, scaling,
                    and parameter gradient management
        """
        final_hook = super()._get_final_grad_hook(param, grad_hook, no_cast)
        def set_grad_hook(grad):
            final_grad = final_hook(grad)
            grad.data = final_grad
            param.grad = None
            return grad

        return set_grad_hook
