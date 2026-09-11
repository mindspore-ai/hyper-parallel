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
"""Tracer for the collective communication operations of core.dtensor."""
import threading
from typing import Callable, Dict

from hyper_parallel.core.dtensor import _utils

# Collective functions in hyper_parallel.core.dtensor._utils to intercept.
_COLLECTIVE_METHODS = (
    "differentiable_all_gather_concat",
    "differentiable_all_to_all",
    "differentiable_all_reduce",
    "differentiable_reduce_scatter",
    "differentiable_all_to_all_single",
    "differentiable_all_to_all_single_async",
)


class CollectiveTracer:
    """Intercepts core.dtensor collective operations via monkey-patching.

    Args:
        on_collective_call: Callback invoked after each collective with
            ``(method_name, args, kwargs, result)``.
    """

    _patch_lock = threading.Lock()

    def __init__(self, on_collective_call: Callable):
        self._callback = on_collective_call
        self._originals: Dict[str, object] = {}

    def install(self):
        """Replace the ``_utils`` collective functions with tracing wrappers."""
        with self._patch_lock:
            for name in _COLLECTIVE_METHODS:
                if not hasattr(_utils, name):
                    continue
                # Save the raw function for exact restoration.
                original_func = getattr(_utils, name)
                self._originals[name] = original_func

                callback = self._callback
                method_name = name

                def _make_wrapper(orig, cb, mname):
                    def wrapper(*args, **kwargs):
                        result = orig(*args, **kwargs)
                        try:
                            cb(mname, args, kwargs, result)
                        except Exception:  # pylint: disable=W0703
                            pass  # Never break production logic
                        return result
                    return wrapper

                wrapper = _make_wrapper(original_func, callback, method_name)
                setattr(_utils, name, wrapper)

    def uninstall(self):
        """Restore the original ``_utils`` collective functions."""
        with self._patch_lock:
            for name, original_func in self._originals.items():
                setattr(_utils, name, original_func)
            self._originals.clear()
