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
"""Blend multiple datasets with per-source weights.

Given ``N`` sub-datasets and ``N`` non-negative weights ``w_i``, a sample at
global index ``g`` is read from sub-dataset ``dataset_index[g]`` at local
index ``dataset_sample_index[g]``. Both indices are precomputed once (and
kept in memory) using a deterministic deficit scheduler, so the same
``(weights, num_samples)`` tuple always yields the same blend. ``seed`` is
accepted only for API compatibility.

Use this to mix corpora — e.g. 30 % code, 70 % web — while keeping the
downstream :class:`GPTDataset` API intact (each sub-dataset can be a
GPTDataset itself, so blending stacks cleanly).
"""
import logging
from typing import Any, Dict, List, Sequence

import numpy as np
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


def _build_blend_indices(
    dataset_sizes: Sequence[int],
    weights: Sequence[float],
    num_samples: int,
    seed: int,  # pylint: disable=W0613
) -> tuple:
    """Return ``(dataset_index, dataset_sample_index)`` for a weighted blend.

    At every global slot ``g``, pick the source whose realised fraction lags
    its target the most (i.e. ``target * (g+1) - realised`` is maximal).
    This guarantees the realised blend tracks ``weights`` exactly up to a
    one-sample rounding gap, with no run-to-run variance.

    ``seed`` is accepted for API parity but unused; the algorithm is
    deterministic in ``(weights, num_samples)``.
    """
    if len(dataset_sizes) != len(weights):
        raise ValueError(
            f"dataset_sizes ({len(dataset_sizes)}) and weights ({len(weights)}) "
            f"must have the same length"
        )
    if any(w < 0 for w in weights):
        raise ValueError(f"weights must be non-negative, got {list(weights)}")
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("weights must contain at least one non-zero value")
    norm_w = np.asarray([w / total_w for w in weights], dtype=np.float64)
    sizes = np.asarray([int(s) for s in dataset_sizes], dtype=np.int64)
    # A zero-length source with a non-zero weight would be selected by the
    # greatest-error rule and then hit ``counters[d] % sizes[d]`` → divide by
    # zero. Reject up-front (a source contributing samples must have samples).
    empty_weighted = [i for i, (s, w) in enumerate(zip(sizes, weights)) if s == 0 and w > 0]
    if empty_weighted:
        raise ValueError(
            f"BlendableDataset sources {empty_weighted} have a non-zero weight "
            f"but zero length; an empty source cannot contribute samples"
        )

    dataset_index = np.zeros(num_samples, dtype=np.int32)
    dataset_sample_index = np.zeros(num_samples, dtype=np.int64)
    realised = np.zeros(len(weights), dtype=np.float64)
    counters = np.zeros(len(weights), dtype=np.int64)
    for g in range(num_samples):
        # error[d] = how far behind source d's realised fraction is —
        # picking the argmax keeps every source within 1 sample of its
        # target share at every prefix (Megatron's invariant).
        error = norm_w * (g + 1) - realised
        d = int(np.argmax(error))
        dataset_index[g] = d
        # Modulo wrap so a small sub-dataset reused inside a long blend
        # cycles through its samples in order instead of indexing OOB.
        dataset_sample_index[g] = counters[d] % sizes[d]
        realised[d] += 1.0
        counters[d] += 1
    return dataset_index, dataset_sample_index


class BlendableDataset(Dataset):
    """Weighted blend over a list of datasets.

    Args:
        datasets: Concrete ``torch.utils.data.Dataset`` objects; each must
            implement ``__len__`` and ``__getitem__``. Typically these are
            :class:`GPTDataset` instances.
        weights: One non-negative weight per dataset (auto-normalised).
        num_samples: Total length exposed by the blend.
        seed: Accepted for API compatibility; index construction is
            deterministic and does not use random sampling.

    Note:
        At every prefix, each source's realised count tracks its weighted
        target within the scheduler's one-sample rounding bound.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: Sequence[float],
        num_samples: int,
        seed: int = 1234,
    ) -> None:
        if not datasets:
            raise ValueError("BlendableDataset requires at least one dataset")
        self.datasets = datasets
        sizes = [len(d) for d in datasets]
        self.num_samples = int(num_samples)
        self.dataset_index, self.dataset_sample_index = _build_blend_indices(
            sizes, weights, self.num_samples, seed,
        )
        logger.info(
            "BlendableDataset built: %d sub-datasets, weights=%s, total samples=%d",
            len(datasets), list(weights), self.num_samples,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d = int(self.dataset_index[idx])
        s = int(self.dataset_sample_index[idx])
        return self.datasets[d][s]
