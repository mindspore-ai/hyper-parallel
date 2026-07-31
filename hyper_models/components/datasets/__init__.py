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
"""YAML-callable dataset factories."""

from dataclasses import dataclass
from typing import Any, Optional

from hyper_models.data.build_dataloader import DummyDataset


@dataclass
class DatasetBuildResult:
    """Dataset runtime state returned to the trainer."""

    dataset: Any
    train_steps: int


def build_dummy_dataset(
    *,
    model_config: Any,
    seed: Optional[int],
    dp_rank: int,
    train_steps: int,
    num_samples: int = 100,
    seq_len: int = 32,
    vocab_size: Optional[int] = None,
    data_type: str = "mapping",
    chat_template: Optional[str] = None,
) -> DatasetBuildResult:
    """Build the temporary rank-aware dataset used by Trainer.

    Args:
        model_config: Runtime model configuration used for the vocabulary size.
        seed: Base random seed.
        dp_rank: Data-parallel rank.
        train_steps: Configured number of optimizer steps.
        num_samples: Number of generated samples.
        seq_len: Token sequence length.
        vocab_size: Optional vocabulary-size override.
        data_type: Dataset type retained for TextTrainer compatibility.
        chat_template: Optional chat-template name retained for compatibility.

    Returns:
        Dataset and derived training-step state.
    """
    del data_type, chat_template
    resolved_seed = 42 if seed is None else seed
    model_vocab_size = getattr(model_config, "vocab_size", 1000)
    dataset = DummyDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        vocab_size=vocab_size or model_vocab_size,
        seed=resolved_seed + dp_rank,
    )
    return DatasetBuildResult(dataset=dataset, train_steps=train_steps)


__all__ = ["DatasetBuildResult", "build_dummy_dataset"]
