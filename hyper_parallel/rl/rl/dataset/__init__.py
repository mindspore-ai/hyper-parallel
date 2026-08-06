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
"""Prompt adapters, data sources, and canonical batch construction."""

from rl.dataset.adapters import PROMPT_INSTRUCTION, extract_ground_truth, format_prompt
from rl.dataset.batch_builder import ExperienceBuilder, build_experience_batch
from rl.dataset.data_source import (
    PromptDataset,
    build_padded_evaluation_batches,
    collate_prompt_samples,
)

__all__ = [
    "PROMPT_INSTRUCTION",
    "ExperienceBuilder",
    "PromptDataset",
    "build_experience_batch",
    "build_padded_evaluation_batches",
    "collate_prompt_samples",
    "extract_ground_truth",
    "format_prompt",
]
