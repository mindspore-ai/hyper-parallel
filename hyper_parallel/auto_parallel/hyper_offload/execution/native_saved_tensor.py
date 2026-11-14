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
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_native_saved_tensor() -> type | None:
    """Load the saved-tensor helper once, outside replay's per-tensor path.

    Returns:
        A native saved-value type, or None when an optional extension
        toolchain is unavailable.
    """
    try:
        # Building the optional extension must not make importing HyperOffload
        # depend on a compiler or import the build toolchain on the hot path.
        from torch.utils.cpp_extension import load  # pylint: disable=import-outside-toplevel

        module = load(
            name="hyperoffload_saved_tensor",
            sources=[str(Path(__file__).with_suffix(".cpp"))],
            extra_cflags=["-O2"],
            verbose=False,
        )
    except (ImportError, OSError, RuntimeError) as error:
        logger.warning("Native saved-tensor helper unavailable; using regular detach: %s", error)
        return None
    return module.SavedNativeValue
