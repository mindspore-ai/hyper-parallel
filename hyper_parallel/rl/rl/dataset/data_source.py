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
"""Stateful prompt sources and deterministic evaluation partitioning."""
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
import pandas as pd
from hyper_parallel import get_platform
from rl.dataset.contracts import Message, PromptRecord
platform = get_platform()
PROMPT_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
_PROMPT_COLUMN_CANDIDATES = ("prompt", "question", "problem", "input_text")
_ANSWER_COLUMN_CANDIDATES = ("extra_info", "answer", "solution")
_MAPPING_ANSWER_KEYS = ("ground_truth", "answer", "solution")
def _to_builtin(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value
def _pick_column(
    columns: set[str], configured: Optional[str], candidates: Sequence[str],
    label: str, required: bool,
) -> Optional[str]:
    """Resolve an explicit or inferred parquet column."""
    if configured is not None:
        column = configured.strip()
        if not column:
            raise ValueError(f"{label}_column must be non-empty when provided")
        if column in columns:
            return column
        raise ValueError(
            "Prompt parquet is missing required columns: "
            f"{label}_column={column!r}; available={sorted(columns)}"
        )
    column = next((candidate for candidate in candidates if candidate in columns), None)
    if column is not None or not required:
        return column
    raise ValueError(
        f"Prompt parquet is missing required columns: no {label} column found; "
        f"expected one of {list(candidates)}, available={sorted(columns)}"
    )
def _normalize_prompt(prompt_source: Any, index: int) -> tuple[str, str]:
    """Normalize one raw or structured prompt source."""
    prompt_source = _to_builtin(prompt_source)
    if isinstance(prompt_source, str):
        source_prompt = prompt_source.strip()
        if not source_prompt:
            raise ValueError("Prompt source must not be empty")
        prompt = (
            source_prompt
            if PROMPT_INSTRUCTION in source_prompt
            else f"{source_prompt} {PROMPT_INSTRUCTION}"
        )
        return source_prompt, prompt
    if isinstance(prompt_source, Sequence) and not isinstance(prompt_source, (str, bytes)):
        messages = list(prompt_source)
        if len(messages) != 1 or not isinstance(messages[0], Mapping):
            raise ValueError("Structured prompt sources must contain exactly one chat message")
        role = messages[0].get("role")
        content = messages[0].get("content")
        if role != "user" or not isinstance(content, str) or not content.strip():
            raise ValueError("Structured prompt sources must contain one non-empty user message")
        source_prompt = content.strip()
        return source_prompt, source_prompt
    raise ValueError(
        f"Unsupported prompt source type at sample {index}: {type(prompt_source)!r}"
    )
def _normalize_ground_truth(answer_source: Any, index: int) -> str:
    """Normalize one mapping or string answer into an exact-match target."""
    answer_source = _to_builtin(answer_source)
    if isinstance(answer_source, Mapping):
        for key in _MAPPING_ANSWER_KEYS:
            value = answer_source.get(key)
            if value is not None:
                return _normalize_ground_truth(value, index)
        raise ValueError(
            f"Answer mapping at sample {index} must contain one of {_MAPPING_ANSWER_KEYS}"
        )
    if not isinstance(answer_source, str):
        raise ValueError(f"Answer source at sample {index} must be text or a supported mapping")
    ground_truth = answer_source.strip()
    if "####" in ground_truth:
        ground_truth = ground_truth.rsplit("####", maxsplit=1)[-1].strip()
    ground_truth = ground_truth.replace(",", "").replace("$", "").strip()
    if not ground_truth:
        raise ValueError(f"Ground truth at sample {index} must not be empty")
    return ground_truth
def _select_answer_source(record: Mapping[str, Any], answer_column: Optional[str], index: int) -> Any:
    reward_model = _to_builtin(record.get("reward_model"))
    if isinstance(reward_model, Mapping) and reward_model.get("ground_truth") is not None:
        return reward_model["ground_truth"]
    if answer_column is not None:
        return record[answer_column]
    raise ValueError(
        f"Sample {index} is missing reward_model.ground_truth and fallback answer"
    )
class PromptDataset:
    """Read tokenized prompt records from supported text parquet layouts."""
    def __init__(
        self,
        parquet_path: str,
        tokenizer: Any,
        max_prompt_length: int,
        prompt_column: Optional[str] = None,
        answer_column: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        path = Path(parquet_path)
        if not path.is_file():
            raise ValueError(f"Prompt parquet file does not exist: {path}")
        if max_prompt_length <= 0:
            raise ValueError(
                f"max_prompt_length must be positive, got {max_prompt_length}"
            )
        frame = pd.read_parquet(path)
        if frame.empty:
            raise ValueError(f"Prompt parquet contains no rows: {path}")
        columns = set(str(column) for column in frame.columns)
        self._prompt_column = _pick_column(
            columns, prompt_column, _PROMPT_COLUMN_CANDIDATES, "prompt", True
        )
        self._answer_column = _pick_column(
            columns, answer_column, _ANSWER_COLUMN_CANDIDATES, "answer", False
        )
        if self._answer_column is None and "reward_model" not in columns:
            raise ValueError(
                "Prompt parquet is missing required columns: no ground-truth source "
                "found; expected reward_model.ground_truth or one of "
                f"{list(_ANSWER_COLUMN_CANDIDATES)}, available={sorted(columns)}"
            )
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError(
                    f"max_samples must be positive or null, got {max_samples}"
                )
            frame = frame.iloc[:max_samples]
        self._records = frame.to_dict("records")
        self._tokenizer = tokenizer
        self._max_prompt_length = max_prompt_length
    def __len__(self) -> int:
        """Return the number of prompt samples."""
        return len(self._records)
    def _tokenize(self, prompt: str, index: int) -> tuple[Any, Any]:
        """Tokenize one prompt and enforce its configured length bound."""
        encoded = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            truncation=True,
            max_length=self._max_prompt_length,
            return_dict=True,
            tokenizer_kwargs={"return_attention_mask": True},
        )
        input_ids = platform.tensor(encoded["input_ids"], dtype=platform.tensor_dtype.long)
        attention_mask = platform.tensor(
            encoded.get("attention_mask", [1] * len(encoded["input_ids"])),
            dtype=platform.tensor_dtype.long,
        )
        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError(
                f"Tokenizer returned invalid input_ids for sample {index}: "
                f"shape={input_ids.shape}"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"Tokenizer attention_mask shape must equal input_ids shape for sample {index}: "
                f"input_ids={input_ids.shape}, "
                f"attention_mask={attention_mask.shape}"
            )
        return input_ids, attention_mask
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one formatted and tokenized prompt sample."""
        record = self._records[index]
        source_prompt, prompt = _normalize_prompt(record[self._prompt_column], index)
        answer_source = _select_answer_source(record, self._answer_column, index)
        input_ids, attention_mask = self._tokenize(prompt, index)
        return {
            "sample_index": index,
            "source_prompt": source_prompt,
            "prompt": prompt,
            "ground_truth": _normalize_ground_truth(answer_source, index),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
def build_padded_evaluation_batches(
    dataset_size: int, num_replicas: int, rank: int, batch_size: int,
    max_samples: Optional[int] = None,
) -> list[list[tuple[int, bool]]]:
    """Partition evaluation rows and pad the final distributed batch."""
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}")
    if num_replicas <= 0:
        raise ValueError(f"num_replicas must be positive, got {num_replicas}")
    if rank < 0 or rank >= num_replicas:
        raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be positive or null, got {max_samples}")
    sample_count = dataset_size if max_samples is None else min(dataset_size, max_samples)
    global_batch_size = num_replicas * batch_size
    num_batches = (sample_count + global_batch_size - 1) // global_batch_size
    batches: list[list[tuple[int, bool]]] = []
    for batch_index in range(num_batches):
        start = batch_index * global_batch_size + rank * batch_size
        indices = range(start, start + batch_size)
        batches.append(
            [
                (index if index < sample_count else 0, index < sample_count)
                for index in indices
            ]
        )
    return batches
def build_prompt_records(
    batch: Mapping[str, Any],
    input_ids: Any,
    attention_mask: Any,
) -> tuple[PromptRecord, ...]:
    """Attach unpadded input tokens to prompt records consumed by rollout."""
    return tuple(
        PromptRecord(
            prompt_id=str(int(batch["sample_indices"][index])),
            messages=(Message("user", batch["prompts"][index]),),
            ground_truth=batch["ground_truths"][index],
            metadata={
                "input_ids": input_ids[index][attention_mask[index].bool()].detach()
            },
        )
        for index in range(input_ids.shape[0])
    )
def _left_pad(tensor: Any, length: int, value: int) -> Any:
    padding = platform.full((length - int(tensor.numel()),), value, dtype=tensor.dtype)
    return platform.cat((padding, tensor), dim=0)
def collate_prompt_samples(samples: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_prompt_samples requires at least one sample")
    max_length = max(int(sample["input_ids"].numel()) for sample in samples)
    input_ids = [_left_pad(sample["input_ids"], max_length, pad_token_id) for sample in samples]
    attention_masks = [_left_pad(sample["attention_mask"], max_length, 0) for sample in samples]
    return {
        "input_ids": platform.cat(tuple(tensor.unsqueeze(0) for tensor in input_ids), dim=0),
        "attention_mask": platform.cat(
            tuple(tensor.unsqueeze(0) for tensor in attention_masks),
            dim=0,
        ),
        "sample_indices": [int(sample["sample_index"]) for sample in samples],
        "source_prompts": [str(sample["source_prompt"]) for sample in samples],
        "prompts": [str(sample["prompt"]) for sample in samples],
        "ground_truths": [str(sample["ground_truth"]) for sample in samples],
    }
