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
"""Standalone evaluation configuration hosted beside Trainer-owned data types.

The evaluation runtime receives plain statistics and does not import these
Trainer configuration classes.
"""

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

from hyper_parallel.trainer.config.data import DataLoaderConfig, DatasetConfig
from hyper_parallel.trainer.config.target import Target, _serialize_config_value


@dataclass
class EvaluationConfig:
    """Bounded standalone evaluation options.

    Args:
        mode: ``loss`` evaluates existing text with a forward pass and no KV
            cache, without generating new tokens. ``generation`` would generate
            answers before scoring them; it is reserved and currently rejected
            by the run-mode validator.
        split: Dataset split to evaluate. ``valid`` and ``test`` select the
            matching member of a train/valid/test triple; ``dataset`` selects a
            single unsplit Dataset. The training split is never selected
            implicitly.
        batch_size: Source batch size per logical DP lane, independent of
            training batch sizes and gradient accumulation. ``TokenBatchLoader``
            derives its default token budget as ``batch_size * max_seq_len``;
            the number of encoded samples in each forward batch can vary.
            An explicit dataloader ``token_budget`` overrides that budget.
        max_samples: Upper bound on source dataset records, applied to the
            globally ordered range before DP partitioning and sample encoding.
            ``None`` means the whole selected finite split.
        max_batches: Maximum forward rounds per logical DP lane, not a global
            sample count. ``None`` means no round limit.
        dataset: Optional evaluation dataset override; falls back to the
            top-level ``dataset`` section.
        dataloader: Optional evaluation dataloader override; falls back to the
            top-level ``dataloader`` section.
        metrics: Metric targets evaluated in configured order.
    """

    mode: Literal["loss", "generation"] = "loss"
    split: Literal["valid", "test", "dataset"] = "valid"
    batch_size: int = 1
    max_samples: Optional[int] = None
    max_batches: Optional[int] = None
    dataset: Optional[DatasetConfig] = None
    dataloader: Optional[DataLoaderConfig] = None
    metrics: List[Target[Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject a range that cannot describe a finite evaluation pass."""
        if self.mode not in ("loss", "generation"):
            raise ValueError(
                f"evaluation.mode must be one of: loss, generation; got {self.mode!r}"
            )
        if self.split not in ("valid", "test", "dataset"):
            raise ValueError(
                f"evaluation.split must be one of: valid, test, dataset; got {self.split!r}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"evaluation.batch_size must be positive, got {self.batch_size}"
            )
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError(
                "evaluation.max_samples must be positive or null, "
                f"got {self.max_samples}"
            )
        if self.max_batches is not None and self.max_batches < 1:
            raise ValueError(
                "evaluation.max_batches must be positive or null, "
                f"got {self.max_batches}"
            )
        # Whether metrics are required depends on the run mode, which this
        # section does not know: training builds this default section without
        # any metric, and the run-mode validator enforces the requirement for
        # ``run_mode: eval``. Checking here would reject every training config.

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evaluation options in their compact YAML shape."""
        config: dict[str, Any] = {
            "mode": self.mode,
            "split": self.split,
            "batch_size": self.batch_size,
            "max_samples": self.max_samples,
            "max_batches": self.max_batches,
            "metrics": [_serialize_config_value(metric) for metric in self.metrics],
        }
        if self.dataset is not None:
            config["dataset"] = _serialize_config_value(self.dataset)
        if self.dataloader is not None:
            config["dataloader"] = _serialize_config_value(self.dataloader)
        return config
