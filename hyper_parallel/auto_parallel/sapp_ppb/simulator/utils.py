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
"""Helpers used by the pipeline simulator: numeric coercion, colouring, decorators."""
import time
from functools import wraps
from typing import Any, Callable, List, Tuple, Union

import numpy as np
from matplotlib import colors

from hyper_parallel.auto_parallel.sapp_ppb.utils.logger import logger

ScalarOrMatrix = Union[int, float, List[List[Union[int, float]]], Tuple[Tuple[Union[int, float], ...], ...]]


def format_2d_inputs(a: ScalarOrMatrix, raw: int, col: int) -> np.ndarray:
    """Coerce ``a`` into a 2-D :class:`numpy.ndarray` of shape ``(raw, col)``.

    Args:
        a: Scalar broadcast to ``(raw, col)``, a flat sequence treated as one row,
            or a nested sequence interpreted as a 2-D matrix.
        raw: Number of rows when broadcasting a scalar.
        col: Number of columns when broadcasting a scalar.

    Returns:
        A 2-D array matching the supplied data.

    Raises:
        ValueError: If ``a`` does not match any of the supported shapes.
    """
    if isinstance(a, (int, float)):
        return np.broadcast_to(a, (raw, col))
    if isinstance(a, (list, tuple)):
        if all(isinstance(item, (list, tuple)) for item in a):
            return np.array(a)
        if all(isinstance(item, (int, float)) for item in a):
            return np.array([a])
        raise ValueError(f"Unsupported inputs: {a}")
    raise ValueError(f"Unsupported inputs: {a}")


def apply_color(target_list: list, c: List[str]) -> list:
    """Wrap each element of ``target_list`` with an ANSI colour escape from ``c``.

    Args:
        target_list: Values to colour (floats are formatted to four decimals).
        c: One ANSI colour code per target element.

    Returns:
        The same list with each element wrapped in the matching colour escape.
    """
    for i, target in enumerate(target_list):
        target = f'{target:.4f}' if isinstance(target, float) else target
        target_list[i] = f"\033[{c[i]}m{target}\033[0m"
    return target_list


def apply_format(target_list: list) -> str:
    """Join a sequence of pre-coloured values into the single-line bubble report.

    Args:
        target_list: Coloured strings produced by :func:`apply_color`.

    Returns:
        The formatted single-line string.
    """
    s = f'{target_list[0]:^22}'
    symbol = ['=', '+', '+', '+', '+', '+']
    for i in range(len(target_list) - 1):
        s = f'{s}{symbol[i]}{target_list[i + 1]:^22}'
    return s


def color_mix(c1: Any, c2: Any, w1: float = 0.5, w2: float = 0.5) -> Tuple[float, float, float, float]:
    """Blend two matplotlib colours with weights ``w1`` and ``w2``.

    Args:
        c1: First colour in any format understood by :func:`matplotlib.colors.to_rgba`.
        c2: Second colour.
        w1: Weight for ``c1``. Default: 0.5.
        w2: Weight for ``c2``. Default: 0.5.

    Returns:
        A ``(r, g, b, a)`` tuple with values in ``[0, 1]``.
    """
    rgb = (np.array(colors.to_rgba(c1, 1)) * w1 + np.array(colors.to_rgba(c2, 1)) * w2) / (w1 + w2)
    return colors.to_rgba(rgb)


def dfs_builder(comm: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Build a decorator that guards a DFS visit against re-entry and unmet dependencies.

    Args:
        comm: When ``True``, use the communication-aware ``depend_pre``/``depend_left``
            attributes; otherwise use the compute-only ``pre``/``left`` attributes.

    Returns:
        A decorator wrapping a DFS visit method on :class:`BlockSim`-like objects.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Attach the DFS visit guards to ``func``."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Run ``func`` exactly once per node after asserting dependencies."""
            self = args[0]
            pre, left = (self.depend_pre, self.depend_left) if comm else (self.pre, self.left)
            if self.finish:
                return None
            if pre is None or left is None:
                raise NotImplementedError
            if self.in_queue:
                raise ValueError("Dependency loop detected during DFS traversal")
            self.in_queue = True
            res = func(*args, **kwargs)
            self.finish = True
            self.in_queue = False
            return res
        return wrapper

    return decorator


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Log the wall-clock time a function takes.

    Args:
        func: Callable to time.

    Returns:
        A wrapper that logs the elapsed time at INFO level after ``func`` returns.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Time one call to ``func`` and log the elapsed wall clock."""
        t0 = time.time()
        res = func(*args, **kwargs)
        t1 = time.time() - t0
        logger.info("function `%s` time used: %.4f s", func.__name__, t1)
        return res

    return wrapper
