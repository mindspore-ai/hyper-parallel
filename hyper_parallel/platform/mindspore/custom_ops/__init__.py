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
"""MindSpore platform custom operations.

.. warning::
    This is an experimental API that subject to change or deletion.

Custom operators are routed through the DFunction distributed dispatch
framework.  When inputs are plain ``mindspore.Tensor`` objects the call goes
directly to the Ascend NPU custom C++ kernel.  When inputs are ``DTensor``
objects the call is automatically routed through the registered
``DistributedOp`` for layout inference and re-distribution.
"""

from hyper_parallel.platform.mindspore.custom_ops.custom_ops import MindSporeCustomOps  # noqa: F401

__all__ = ["MindSporeCustomOps"]
