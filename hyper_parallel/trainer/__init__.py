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
"""HyperParallel Trainer module."""

__all__ = ["ParallelDims"]

# Importing utils first installs ``info_rank0`` / ``warning_rank0`` /
# ``info_once`` / ``warning_once`` on ``logging.Logger`` so every
# downstream module that does ``logger = logging.getLogger(__name__)``
# can use them without explicit setup.
from hyper_parallel.trainer import utils  # noqa: F401
from hyper_parallel.trainer.parallel_dims import ParallelDims
