# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Llama3-style TP demo on MindSpore (mirrors ``examples/torch/llama3`` layout).

Import from the ``examples/mindspore/llama3`` directory (see ``tensor_parallel_example.py``).
"""
from model import Llama3DemoConfig, Llama3Model
from parallelize import (
    broadcast_state_dict_from_rank0,
    build_dp_tp_mesh,
    build_tp_mesh,
    parallelize_llama3,
)

__all__ = [
    "Llama3DemoConfig",
    "Llama3Model",
    "broadcast_state_dict_from_rank0",
    "build_dp_tp_mesh",
    "build_tp_mesh",
    "parallelize_llama3",
]
