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
"""Region state for trainer-managed local compute execution.

Keeps only the ``_LOCAL_COMPUTE_ACTIVE`` ContextVar and the
``local_compute_context`` manager; the descendant forward adapters that
consult this state live in ``distributed/_builder/forward_rewriter.py``
(the single forward-assignment site, AST-gated per 05 §15.2.3).
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_LOCAL_COMPUTE_ACTIVE = ContextVar("trainer_local_compute_active", default=False)


@contextmanager
def local_compute_context() -> Iterator[None]:
    """Mark execution of a trainer-managed local computation."""
    token = _LOCAL_COMPUTE_ACTIVE.set(True)
    try:
        yield
    finally:
        _LOCAL_COMPUTE_ACTIVE.reset(token)
