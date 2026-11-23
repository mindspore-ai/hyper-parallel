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
"""Optional native metadata copy for ordinary saved tensors."""

from functools import lru_cache
import logging
from importlib import import_module

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_native_saved_tensor() -> type | None:
    """Load the saved-tensor helper once, outside replay's per-tensor path.

    Returns:
        A native saved-value type, or None when an optional extension
        binary is unavailable.
    """
    try:
        module = import_module(
            "hyper_parallel.auto_parallel.hyper_offload.execution._native_saved_tensor"
        )
    except (ImportError, OSError, RuntimeError) as error:
        logger.warning("Native saved-tensor helper unavailable; using regular detach: %s", error)
        return None
    return module.SavedNativeValue
