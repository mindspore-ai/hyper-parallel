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
"""Pytest fixtures and shared utilities for ``sapp_nd`` tests.

Provides a cross-file search-result cache so that expensive
``Parallelize.run_generation_to_ordering`` calls are executed only once per
unique combination of (device, n_devices, dims, swap_os, mppb, gbs).
"""
import os
import tempfile
from typing import List, Optional
from unittest.mock import patch

WORK_PATH = os.path.dirname(os.path.abspath(__file__))
ND_PATH = os.path.join(WORK_PATH, "nd")
CONFIG_PATH = os.path.join(ND_PATH, "deepseek.yaml")

_SEARCH_CACHE: dict = {}


def shared_search(
    device: str,
    n_devices: Optional[int],
    dims: List[str],
    swap_os: bool = False,
    mppb: bool = False,
    top_k: int = 5,
    gbs: Optional[int] = None,
) -> list:
    """Run and cache a full ND search; same parameters returns cached result.

    Args:
        device: Device type string (e.g. ``"A2"``).
        n_devices: Number of devices, or ``None`` for auto.
        dims: List of dimension name strings (e.g. ``["DP", "TP", "PP"]``).
        swap_os: Whether optimizer sharding swap is enabled.
        mppb: Whether manual pipeline balance is enabled.
        top_k: Hint for display; the full scored list is always returned.
        gbs: Global batch size, or ``None``.

    Returns:
        The scored search-space list produced by
        ``Parallelize.run_generation_to_ordering``.
    """
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory  # pylint: disable=C0415
    from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim  # pylint: disable=C0415
    from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par  # pylint: disable=C0415
    from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware as Hard  # pylint: disable=C0415
    from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level  # pylint: disable=C0415

    key = (device, n_devices, tuple("TP" if d == "MP" else d for d in dims), swap_os, mppb, gbs)
    if key in _SEARCH_CACHE:
        return _SEARCH_CACHE[key]
    with tempfile.TemporaryDirectory() as mpl_tmp, \
            patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
        set_verbose_level(0)
        Dim.TP.reset_bound()
        machine = Hard.Machine(n_devices, device)
        dim_objs = Dim.get_dims(dims)
        runner = Par.Parallelize(
            "mindformers", CONFIG_PATH, machine,
            global_batch_size=gbs, dimensions=dim_objs,
            swap_os=swap_os, mppb=mppb,
            model=None, max_mem=None,
            mem_for_ppb=Memory.from_string("0GB"),
        )
        scored = runner.run_generation_to_ordering(
            None, threads_num=None, top_num=top_k, cache_file=None,
        )
    Dim.TP.reset_bound()
    _SEARCH_CACHE[key] = scored
    return scored


def suspend_coverage():
    """Temporarily suspend coverage tracing."""
    import sys  # pylint: disable=C0415
    original_trace = sys.gettrace()
    if original_trace is None:
        return None
    sys.settrace(None)
    cov = None
    try:
        cov = __import__("coverage").current()
    except (ImportError, AttributeError):
        pass

    def _resume():
        sys.settrace(original_trace)
        if cov is not None:
            try:
                cov.collect()
            except Exception:  # pylint: disable=W0718
                pass

    return _resume
