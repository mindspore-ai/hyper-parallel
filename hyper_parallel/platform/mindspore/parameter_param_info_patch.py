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
"""Patch MindSpore ``Parameter.param_info`` to avoid the ``ParamInfo.obj`` cycle."""
# pylint: disable=protected-access,forbidden-backend-import

from mindspore._c_expression import TensorPy as Tensor_
from mindspore.common.parameter import Parameter


def patch_mindspore_parameter_param_info_cycle_if_needed() -> None:
    """Break the ``Parameter <-> ParamInfo.obj`` Python reference cycle.
    MindSpore's default ``Parameter.param_info`` setter stores ``param_info.obj = self``.
    That back-reference is only needed for Python-side bookkeeping and can keep Parameter
    objects alive longer than necessary. hyper_parallel does not rely on that backlink, so
    reset it to ``None`` when binding new ParamInfo objects.
    """
    if getattr(Parameter, "_hyper_parallel_param_info_cycle_patched", False):
        return

    original_param_info = Parameter.param_info

    def _patched_setter(self, param_info_):
        Tensor_.wait_pipeline(self)
        # Break the cycle by clearing the back-reference to self in the new ParamInfo object.
        param_info_.obj = None
        self._param_info = param_info_
        Tensor_.set_param_info(self, param_info_)

    Parameter.param_info = property(
        original_param_info.fget,
        _patched_setter,
        original_param_info.fdel,
        original_param_info.__doc__,
    )
    Parameter._hyper_parallel_param_info_cycle_patched = True
