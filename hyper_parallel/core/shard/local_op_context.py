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
"""Dynamic executor for the local operator inside a distributed expansion.

The dispatcher consumes the slot while expanding an operator, so operations in
the expansion do not inherit its executor. No framework or checkpoint dependency
is needed here.
"""
from contextvars import ContextVar
from typing import Any, Callable, Optional


LOCAL_OP_EXECUTOR: ContextVar[Optional[Callable[..., Any]]] = ContextVar(
    "hyper_parallel_local_op_executor", default=None
)
