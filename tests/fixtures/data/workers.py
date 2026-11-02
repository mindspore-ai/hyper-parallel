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
"""Spawn callbacks kept separate from test modules and their Trainer imports."""


def initialize_worker(worker_id: int) -> None:
    """Provide a picklable initializer without importing the test suite.

    Args:
        worker_id: DataLoader worker index, unused by this no-op fixture.
    """
    del worker_id
