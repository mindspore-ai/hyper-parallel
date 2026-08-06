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
"""Stateless adapters from source records to canonical prompt fields."""

PROMPT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def format_prompt(prompt_source: str) -> str:
    """Append the configured answer-format instruction to one source prompt."""
    normalized = prompt_source.strip()
    if not normalized:
        raise ValueError("Prompt source must not be empty")
    return f"{normalized} {PROMPT_INSTRUCTION}"


def extract_ground_truth(answer: str) -> str:
    """Extract and normalize the final value after the last ``####``."""
    if "####" not in answer:
        raise ValueError("Source answer must contain a '####' delimiter")
    ground_truth = answer.rsplit("####", maxsplit=1)[-1].strip()
    ground_truth = ground_truth.replace(",", "").replace("$", "")
    if not ground_truth:
        raise ValueError("Source answer after '####' must not be empty")
    return ground_truth
