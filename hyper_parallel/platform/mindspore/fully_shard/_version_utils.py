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
"""Helpers for storage-preserving tensor updates in MindSpore fully_shard."""

import mindspore as ms


def copy_without_bumping_version(dst: ms.Tensor, src: ms.Tensor) -> None:
    """Copy through ``dst.data`` so the destination tensor version is preserved."""
    dst.data.copy_(src)
