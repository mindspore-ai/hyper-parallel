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
"""Tests for dependencies used by the VLM batch pipeline."""

from types import SimpleNamespace

from hyper_parallel.auto_models.components.datasets.parallel.batch_context import (
    create_batch_parallel_context,
)
from hyper_parallel.auto_models.components.datasets.vlm.get_batch import VLMBatchProcessor


def test_vlm_batch_pipeline_dependencies_are_available() -> None:
    """The VLM processor should import with its parallel batch dependencies."""
    assert VLMBatchProcessor is not None

    context = create_batch_parallel_context(
        SimpleNamespace(tp_rank=0, tp_size=1, cp_rank=0, cp_size=1)
    )
    assert context.reads_data()
