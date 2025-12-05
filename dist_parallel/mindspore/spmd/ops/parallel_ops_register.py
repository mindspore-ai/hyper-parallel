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
"""
Parallel layout decorator
"""
_DISTRIBUTED_OPS = {}

def register_distributed_op(op_name, op_class):
    """
    Register a distributed operator implementation.

    Args:
        op_name (str): Name of the operator
        op_class (class): Distributed operator implementation class
    """
    global _DISTRIBUTED_OPS
    _DISTRIBUTED_OPS[op_name] = op_class

def get_distributed_op(op_name):
    """
    Get distributed operator implementation by operator name.

    Args:
        op_name (str): Name of the operator

    Returns:
        object: Distributed operator instance or None if not found
    """
    global _DISTRIBUTED_OPS
    return _DISTRIBUTED_OPS.get(op_name, None)
