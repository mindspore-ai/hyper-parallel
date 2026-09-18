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
"""OpenCompass 0.5.4 MMLU PPL configuration for HyperParallel."""

import copy
import os

from opencompass.configs.datasets.mmlu.mmlu_ppl_ac766d import mmlu_datasets

from hyper_parallel.integration.opencompass import HyperOpenCompassModel


def _required_environment(name: str) -> str:
    """Read a required nonempty environment variable."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must name a local checkpoint or immutable model identifier")
    return value


def _positive_or_zero_environment(name: str, default: int) -> int:
    """Read a nonnegative integer environment variable."""
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def _positive_environment(name: str, default: int) -> int:
    """Read a positive integer environment variable."""
    value = _positive_or_zero_environment(name, default)
    if value == 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _select_datasets() -> list[dict]:
    """Select a reproducible smoke subset or the complete MMLU suite."""
    subject = os.environ.get("HP_MMLU_SUBJECT", "college_computer_science").strip()
    selected = copy.deepcopy(mmlu_datasets)
    if subject and subject.lower() != "all":
        expected_abbr = f"lukaemon_mmlu_{subject}"
        selected = [dataset for dataset in selected if dataset["abbr"] == expected_abbr]
        if not selected:
            raise ValueError(f"unknown HP_MMLU_SUBJECT: {subject!r}")
    sample_limit = _positive_or_zero_environment("HP_MMLU_SAMPLE_LIMIT", 32)
    if sample_limit:
        for dataset in selected:
            dataset["reader_cfg"]["test_range"] = f"[0:{sample_limit}]"
    return selected


checkpoint = _required_environment("HP_CHECKPOINT")
tokenizer = os.environ.get("HP_TOKENIZER", checkpoint)
trust_remote_code = os.environ.get("HP_TRUST_REMOTE_CODE", "0") == "1"

models = [
    dict(
        type=HyperOpenCompassModel,
        abbr="hyper-mmlu-ppl",
        path=checkpoint,
        tokenizer_path=tokenizer,
        max_seq_len=_positive_environment("HP_MAX_SEQ_LEN", 4096),
        batch_size=_positive_environment("HP_EVAL_BATCH_SIZE", 4),
        batch_padding=True,
        tokenizer_kwargs=dict(
            padding_side="right",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        ),
        model_kwargs=dict(
            torch_dtype=os.environ.get("HP_EVAL_DTYPE", "bfloat16"),
            trust_remote_code=trust_remote_code,
        ),
        run_cfg=dict(num_gpus=_positive_or_zero_environment("HP_OPENCOMPASS_NUM_GPUS", 1)),
    )
]

datasets = _select_datasets()
