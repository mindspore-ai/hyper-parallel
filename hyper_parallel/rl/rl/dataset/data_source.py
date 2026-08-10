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
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from rl.dataset.adapters import extract_ground_truth, format_prompt
from hyper_parallel import get_platform

platform = get_platform()


class PromptDataset:
    """Map-style prompt dataset backed by one local parquet file.

    Args:
        parquet_path: Path to a parquet file containing prompt and answer columns.
        tokenizer: Hugging Face-compatible tokenizer.
        max_prompt_length: Maximum number of prompt tokens.
        prompt_column: Parquet column containing raw prompts.
        answer_column: Parquet column containing expected answers.
        prompt_formatter: Converts a raw prompt into the rollout prompt.
        ground_truth_extractor: Converts a raw answer into its canonical value.
    """

    def __init__(
        self,
        parquet_path: str,
        tokenizer: Any,
        max_prompt_length: int,
        prompt_column: str = "question",
        answer_column: str = "answer",
        prompt_formatter: Callable[[str], str] = format_prompt,
        ground_truth_extractor: Callable[[str], str] = extract_ground_truth,
    ) -> None:
        """Initialize and validate the local parquet-backed dataset."""
        path = Path(parquet_path)
        if not path.is_file():
            raise ValueError(f"Prompt parquet file does not exist: {path}")
        if max_prompt_length <= 0:
            raise ValueError(f"max_prompt_length must be positive, got {max_prompt_length}")
        if not prompt_column or not answer_column:
            raise ValueError("prompt_column and answer_column must be non-empty")
        frame = pd.read_parquet(path)
        missing = {prompt_column, answer_column} - set(frame.columns)
        if missing:
            raise ValueError(f"Prompt parquet is missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"Prompt parquet contains no rows: {path}")
        self._prompt_sources = frame[prompt_column].astype(str).tolist()
        self._answer_sources = frame[answer_column].astype(str).tolist()
        self._tokenizer = tokenizer
        self._max_prompt_length = max_prompt_length
        self._prompt_formatter = prompt_formatter
        self._ground_truth_extractor = ground_truth_extractor

    def __len__(self) -> int:
        """Return the number of prompt samples."""
        return len(self._prompt_sources)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return one formatted and tokenized prompt sample.

        Args:
            index: Stable row index.

        Returns:
            Prompt text, token tensors, ground truth, and stable sample index.
        """
        prompt_source = self._prompt_sources[index]
        prompt = self._prompt_formatter(prompt_source)
        messages = [{"role": "user", "content": prompt}]
        encoded = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            truncation=True,
            max_length=self._max_prompt_length,
            return_dict=True,
            tokenizer_kwargs={"return_attention_mask": True},
        )
        input_ids = platform.tensor(encoded["input_ids"], dtype=platform.tensor_dtype.long)
        encoded_attention_mask = encoded.get("attention_mask")
        if encoded_attention_mask is None:
            encoded_attention_mask = [1] * len(encoded["input_ids"])
        attention_mask = platform.tensor(encoded_attention_mask, dtype=platform.tensor_dtype.long)
        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError(f"Tokenizer returned invalid input_ids for sample {index}: shape={input_ids.shape}")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"Tokenizer attention_mask shape must equal input_ids shape for sample {index}: "
                f"input_ids={input_ids.shape}, attention_mask={attention_mask.shape}"
            )
        return {
            "sample_index": index,
            "source_prompt": prompt_source,
            "prompt": prompt,
            "ground_truth": self._ground_truth_extractor(self._answer_sources[index]),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def build_padded_evaluation_batches(
    dataset_size: int,
    num_replicas: int,
    rank: int,
    batch_size: int,
    max_samples: Optional[int] = None,
) -> List[List[tuple[int, bool]]]:
    """Partition evaluation rows without duplicating samples in the metrics.

    Every rank receives the same number of fixed-size batches so FSDP executes
    the same collective sequence. Positions past the requested sample count
    reuse row zero for generation and are marked invalid for metric accounting.

    Args:
        dataset_size: Total number of rows in the evaluation dataset.
        num_replicas: Number of FSDP data-parallel ranks.
        rank: Current data-parallel rank.
        batch_size: Number of prompts generated per rank and iteration.
        max_samples: Optional prefix size used for a bounded evaluation.

    Returns:
        Per-rank batches of ``(dataset_index, is_valid)`` entries.
    """
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
    batches: List[List[tuple[int, bool]]] = []
    for batch_index in range(num_batches):
        rank_offset = batch_index * global_batch_size + rank * batch_size
        batch = []
        for local_offset in range(batch_size):
            sample_index = rank_offset + local_offset
            is_valid = sample_index < sample_count
            batch.append((sample_index if is_valid else 0, is_valid))
        batches.append(batch)
    return batches


def collate_prompt_samples(samples: Sequence[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    """Left-pad prompts and preserve their string metadata.

    Args:
        samples: Non-empty sequence of dataset samples.
        pad_token_id: Token ID used for left padding.

    Returns:
        Batched tensors plus prompt, answer, and index lists.

    Raises:
        ValueError: If no samples are supplied.
    """
    if not samples:
        raise ValueError("collate_prompt_samples requires at least one sample")
    max_length = max(int(sample["input_ids"].numel()) for sample in samples)
    input_ids: List[platform.Tensor] = []
    attention_masks: List[platform.Tensor] = []
    for sample in samples:
        pad_length = max_length - int(sample["input_ids"].numel())
        id_padding = platform.full(
            (pad_length,),
            pad_token_id,
            dtype=sample["input_ids"].dtype,
        )
        mask_padding = platform.full(
            (pad_length,),
            0,
            dtype=sample["attention_mask"].dtype,
        )
        input_ids.append(platform.cat((id_padding, sample["input_ids"]), dim=0))
        attention_masks.append(platform.cat((mask_padding, sample["attention_mask"]), dim=0))
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
