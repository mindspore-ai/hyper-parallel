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
"""Platform monkey-patch tracer for collective communication operations."""
import threading
from typing import Callable, Dict, Optional

from hyper_parallel.platform import get_platform

# Collective methods to intercept on the platform class.
_COLLECTIVE_METHODS = (
    "differentiable_all_gather_concat",
    "differentiable_all_to_all",
    "differentiable_all_reduce",
    "differentiable_reduce_scatter",
    "differentiable_all_to_all_single",
    "differentiable_all_to_all_single_async",
)


class CollectiveTracer:
    """Intercepts platform collective operations via monkey-patching.

    Args:
        on_collective_call: Callback invoked after each collective with
            ``(method_name, args, kwargs, result)``.
    """

    _patch_lock = threading.Lock()

    def __init__(self, on_collective_call: Callable):
        self._callback = on_collective_call
        self._originals: Dict[str, object] = {}
        self._platform_cls: Optional[type] = None

    def install(self):
        """Replace platform collective methods with tracing wrappers."""
        with self._patch_lock:
            platform = get_platform()
            self._platform_cls = type(platform)
            cls = self._platform_cls

            for name in _COLLECTIVE_METHODS:
                if name not in cls.__dict__:
                    continue
                # Save the raw descriptor (staticmethod wrapper) for exact restoration.
                original_descriptor = cls.__dict__[name]
                self._originals[name] = original_descriptor

                # Unwrap staticmethod to get the underlying function.
                if isinstance(original_descriptor, staticmethod):
                    original_func = original_descriptor.__func__
                else:
                    original_func = original_descriptor

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
                setattr(cls, name, staticmethod(wrapper))

    def uninstall(self):
        """Restore original platform collective methods."""
        with self._patch_lock:
            if self._platform_cls is None:
                return
            cls = self._platform_cls
            for name, original_descriptor in self._originals.items():
                # Use type.__setattr__ to precisely restore the original descriptor.
                type.__setattr__(cls, name, original_descriptor)
            self._originals.clear()
            self._platform_cls = None
