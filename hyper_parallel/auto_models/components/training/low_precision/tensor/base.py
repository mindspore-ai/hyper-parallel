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
"""Storage contract for typed low-precision tensors."""

from typing import Any


class QuantizedTensorStorage:
    """Describe the physical storage owned by a quantized tensor format."""

    def update_usage(self, rowwise: bool = True, colwise: bool = True) -> None:
        """Release directional representations that are no longer required."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement update_usage()."
        )

    def is_rowwise(self) -> bool:
        """Return whether row-wise quantized data is available."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement is_rowwise()."
        )

    def is_colwise(self) -> bool:
        """Return whether column-wise quantized data is available."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement is_colwise()."
        )

    def get_metadata(self) -> dict[str, Any]:
        """Return the data needed to rebuild the quantized tensor."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_metadata()."
        )
