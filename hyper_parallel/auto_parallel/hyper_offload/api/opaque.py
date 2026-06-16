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
"""Public API: skip_offload decorator for opaque offload regions.

Delegates all interval recording or fast-path switching to the active
:class:`~offload.execution.base.BaseExecutor` via
:meth:`execute_opaque_op`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from hyper_parallel.auto_parallel.hyper_offload.api.session import OffloadSession

F = TypeVar("F", bound=Callable[..., Any])


def skip_offload(fn: F) -> F:
    """Decorate a callable as a virtual op offload region.

    The decorator wraps the function execution into a single "virtual op"
    in the execution trace, while suspending fine-grained tracing for
    internal operations.
    """
    if not callable(fn):
        raise TypeError("skip_offload only supports decorator usage")

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrap the function to skip offloading."""
        session: OffloadSession | None = OffloadSession.get_active()
        if session is None:
            return fn(*args, **kwargs)

        return session.executor.execute_opaque_op(fn.__name__, fn, args, kwargs)

    return wrapper
