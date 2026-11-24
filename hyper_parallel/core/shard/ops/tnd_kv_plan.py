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
"""Pure document-local causal intervals shared by folded KV consumers and transport."""
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from functools import lru_cache
from numbers import Integral


@dataclass(frozen=True)
class TNDCompactBlock:
    """Immutable natural-coordinate query/KV interval and matching document ends."""

    query_interval: tuple
    key_interval: tuple
    query_ends: tuple
    key_ends: tuple


@lru_cache(maxsize=256)
def plan_tnd_kv(ends: tuple, full_len: int, queries: tuple) -> tuple:
    """Plan independent causal documents for any nonempty natural Q intervals.

    Args:
        ends: Global, nondecreasing document ends covering ``full_len``. Repeated
            ends represent empty documents or metadata padding.
        full_len: Complete natural-order KV length.
        queries: Immutable tuple of nonempty half-open Q intervals. No shard
            count, equal-size split, or two-block/folding assumption is needed.

    Returns:
        Tuple of TNDCompactBlock, preserving query interval order.

    Raises:
        ValueError: Metadata does not cover the sequence or an interval is invalid.

    Note:
        This API requires causal, document-isolated attention. Bidirectional or
        cross-document masks need a different interval policy.
    """
    if (not isinstance(full_len, Integral) or isinstance(full_len, bool)
            or any(not isinstance(end, Integral) or isinstance(end, bool) for end in ends)
            or not ends or full_len <= 0 or ends[-1] != full_len
            or any(a > b for a, b in zip((0,) + ends, ends))):
        raise ValueError("Compact TND KV requires monotonic document ends covering the full sequence.")
    documents = tuple(dict.fromkeys(end for end in ends if end > 0))
    result = []
    for start, stop in queries:
        if (any(not isinstance(edge, Integral) or isinstance(edge, bool) for edge in (start, stop))
                or not 0 <= start < stop <= full_len):
            raise ValueError("Compact TND query intervals must satisfy 0 <= start < stop <= full_len.")
        first, last = bisect_right(documents, start), bisect_left(documents, stop)
        origin = documents[first - 1] if first else 0
        selected = tuple(min(end, stop) for end in documents[first:last + 1])
        result.append(TNDCompactBlock((start, stop), (origin, stop),
                                      tuple(end - start for end in selected),
                                      tuple(end - origin for end in selected)))
    return tuple(result)
