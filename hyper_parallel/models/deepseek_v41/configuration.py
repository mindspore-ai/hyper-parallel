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
"""Pure-Python configuration helpers for the DeepSeek-V4.1 validation crop."""

from __future__ import annotations

import math
from typing import Iterable


def _is_prime(value: int) -> bool:
    """Return whether ``value`` is prime."""
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    limit = math.isqrt(value)
    return all(value % divisor for divisor in range(3, limit + 1, 2))


def _next_unused_prime(start: int, seen: set[int]) -> int:
    """Return the first unused prime strictly greater than ``start``."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_scaled_engram_buckets(
        layer_ids: Iterable[int],
        *,
        bucket_base: int,
        max_ngram_size: int,
        num_heads: int,
) -> tuple[list[list[list[int]]], list[int]]:
    """Build synchronized prime buckets and table sizes for scaled Engram.

    Args:
        layer_ids: Decoder layers that contain an Engram module.
        bucket_base: Approximate rows assigned to each n-gram/head bucket.
        max_ngram_size: Largest n-gram included in the hash.
        num_heads: Independent Engram hash heads.

    Returns:
        Nested prime moduli ``[layer][ngram][head]`` and the exact number of
        embedding rows required by every layer.

    Raises:
        ValueError: If a shape argument cannot describe a useful table.
    """
    layer_ids = list(layer_ids)
    if not layer_ids:
        raise ValueError("scaled Engram requires at least one layer")
    if bucket_base < 2:
        raise ValueError("engram bucket_base must be at least 2")
    if max_ngram_size < 2:
        raise ValueError("engram max_ngram_size must be at least 2")
    if num_heads <= 0:
        raise ValueError("engram num_heads must be positive")

    seen: set[int] = set()
    all_primes: list[list[list[int]]] = []
    table_sizes: list[int] = []
    for _ in layer_ids:
        layer_primes: list[list[int]] = []
        for _ in range(max_ngram_size - 1):
            head_primes: list[int] = []
            current = bucket_base - 1
            for _ in range(num_heads):
                current = _next_unused_prime(current, seen)
                seen.add(current)
                head_primes.append(current)
            layer_primes.append(head_primes)
        all_primes.append(layer_primes)
        table_sizes.append(sum(prime for row in layer_primes for prime in row))
    return all_primes, table_sizes


__all__ = ["build_scaled_engram_buckets"]
