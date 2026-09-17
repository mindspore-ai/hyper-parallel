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
from typing import Any, Callable, Dict

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

    def __init__(self, on_collective_call: Callable) -> None:
        """Record the callback and start with nothing patched.

        Args:
            on_collective_call: Callback invoked after each collective with
                ``(method_name, args, kwargs, result)``.
        """
        self._callback = on_collective_call
        self._originals: Dict[str, object] = {}
        self._wrappers: Dict[str, object] = {}

    def install(self) -> None:
        """Replace the ``_utils`` collective functions with tracing wrappers.

        Names in :data:`_COLLECTIVE_METHODS` that ``_utils`` does not define are
        skipped, so a missing collective degrades to no tracing rather than an
        error. Installations are idempotent per name: a second tracer replaces
        the first tracer's wrapper and saves that wrapper as its own original.
        """
        with self._patch_lock:
            for name in _COLLECTIVE_METHODS:
                if not hasattr(_utils, name):
                    continue
                # Save the raw function for exact restoration.
                original_func = getattr(_utils, name)
                self._originals[name] = original_func

                callback = self._callback
                method_name = name

                def _make_wrapper(orig: Callable, cb: Callable, mname: str) -> Callable:
                    def wrapper(*args: Any, **kwargs: Any) -> Any:
                        """Call the original collective, then notify the callback.

                        A callback failure must never break the collective, so it
                        is swallowed and the original result returned untouched.
                        """
                        result = orig(*args, **kwargs)
                        try:
                            cb(mname, args, kwargs, result)
                        except Exception:  # pylint: disable=W0703
                            pass  # Never break production logic
                        return result
                    return wrapper

                wrapper = _make_wrapper(original_func, callback, method_name)
                self._wrappers[name] = wrapper
                setattr(_utils, name, wrapper)

    def uninstall(self) -> None:
        """Restore the original ``_utils`` collective functions.

        Only restores the attributes this tracer installed. If an inner tracer
        patched the same name afterwards, restoring our saved original would
        silently strip that tracer's wrapper, so it is left in place.
        """
        with self._patch_lock:
            for name, original_func in self._originals.items():
                if getattr(_utils, name, None) is self._wrappers.get(name):
                    setattr(_utils, name, original_func)
            self._originals.clear()
            self._wrappers.clear()
