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
"""Standalone evaluation configuration section.

The section describes one bounded evaluation pass over a fixed data range; it
does not reuse the training-loop fields. ``evaluation.batch_size`` is a logical
per-DP-lane forward batch, ``max_samples`` bounds the global ordered sample
range, and ``max_batches`` bounds the forward rounds executed by every lane.
"""

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

from hyper_parallel.trainer.config.data import DataLoaderConfig, DatasetConfig
from hyper_parallel.trainer.config.target import Target, _serialize_config_value


@dataclass
class EvaluationConfig:
    """Bounded standalone evaluation options.

    Args:
        mode: ``loss`` runs teacher-forcing forward-only evaluation;
            ``generation`` is reserved for the generation-based evaluator and
            is rejected by the P0 run-mode validator.
        split: Dataset split to evaluate. ``valid`` and ``test`` select the
            matching member of a train/valid/test triple; ``dataset`` selects a
            single unsplit Dataset. The training split is never selected
            implicitly.
        batch_size: Samples per logical DP lane per forward. Independent of the
            training micro/global batch and gradient accumulation.
        max_samples: Upper bound on the globally ordered sample range, applied
            before sharding. ``None`` means the whole selected finite split.
        max_batches: Upper bound on the forward rounds executed by all lanes
            together. ``None`` means no round limit.
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


@dataclass
class PredictionConfig:
    """Offline batch-generation options (schema only until P1).

    The section is declared now so ``run_mode: predict`` fails on a concrete
    missing capability instead of an unknown configuration field; the Predictor
    implementation lands in P1.
    """

    input: Optional[str] = None
    output_file: Optional[str] = None
    batch_size: int = 1
    max_batch_tokens: Optional[int] = None
    bucket_by_length: bool = False
    overwrite: bool = False
    generation: Optional[Any] = None

    def __post_init__(self) -> None:
        """Reject batch sizes the static P1 request planner cannot shard."""
        if self.batch_size < 1:
            raise ValueError(
                f"prediction.batch_size must be positive, got {self.batch_size}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction options in their compact YAML shape."""
        return {
            "input": self.input,
            "output_file": self.output_file,
            "batch_size": self.batch_size,
            "max_batch_tokens": self.max_batch_tokens,
            "bucket_by_length": self.bucket_by_length,
            "overwrite": self.overwrite,
            "generation": _serialize_config_value(self.generation),
        }
