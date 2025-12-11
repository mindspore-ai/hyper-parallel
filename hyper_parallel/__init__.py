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
"hyper parallel interface"
from hyper_parallel.platform import set_platform, get_platform
from hyper_parallel.core.hsdp.hsdp_api import hsdp, hsdp_wait_grad_handle, HSDPCell
from hyper_parallel.core.layout import Layout
from hyper_parallel.core.dtensor import DTensor, SkipDTensorDispatch
from hyper_parallel.core.parameter_init import init_parameters
from hyper_parallel.core.tensor_parallel.shard import shard
from hyper_parallel.core.tensor_parallel.shard import parallelize_value_and_grad
from hyper_parallel.core.tensor_parallel.local_func import custom_shard


__all__ = ["set_platform", "get_platform", "hsdp", "hsdp_wait_grad_handle", "HSDPCell", "Layout", "DTensor",
           "init_parameters", "shard", "custom_shard", "parallelize_value_and_grad", "SkipDTensorDispatch"]
