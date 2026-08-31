# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""torch dtensor base (re-export shim)

``DTensorBase`` now lives in ``hyper_parallel.core.dtensor._dtensor_base`` so
that the MindSpore patch can rebind it before ``class DTensor(DTensorBase)`` is
defined.  This module keeps the legacy import path working for existing
importers by re-exporting the same class object and the ``Tensor`` type (which
is patched by unit tests through this module path).
"""
from torch import Tensor
from hyper_parallel.core.dtensor._dtensor_base import DTensorBase

__all__ = ["DTensorBase", "Tensor"]
