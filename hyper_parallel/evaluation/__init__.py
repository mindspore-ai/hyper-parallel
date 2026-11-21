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
"""Standalone evaluation: bounded loop, additive statistics, metrics.

The package is a leaf and must not import ``hyper_parallel.trainer``: the
Trainer injects a step callable that owns batching, loss adapters, and token
counting, and consumes the result. Dependencies point one way only.
"""

from hyper_parallel.evaluation.aggregation import ReducedStatistics, reduce_statistics
from hyper_parallel.evaluation.evaluator import EvaluationReport, Evaluator
from hyper_parallel.evaluation.metrics import (
    LossMetric,
    Metric,
    PerplexityMetric,
    build_metrics,
)
from hyper_parallel.evaluation.types import (
    KIND_LM_NLL,
    KIND_OBJECTIVE,
    EvalStepResult,
    LossStatistics,
    ReductionContext,
)

__all__ = [
    "KIND_LM_NLL",
    "KIND_OBJECTIVE",
    "EvalStepResult",
    "EvaluationReport",
    "Evaluator",
    "LossMetric",
    "LossStatistics",
    "Metric",
    "PerplexityMetric",
    "ReducedStatistics",
    "ReductionContext",
    "build_metrics",
    "reduce_statistics",
]
