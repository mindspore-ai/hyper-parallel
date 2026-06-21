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
"""Small stateless helpers — no platform deps."""
import contextlib
import fnmatch
import re
import socket
from typing import Iterable, Optional

import numpy as np

from tests.shard_ops.framework.case_spec import InputSpec

# Env-var keys (parent -> child protocol).
ENV_CASES_PKG = "HYPER_PARALLEL_SHARD_CASES_PKG"
ENV_CASE_NAMES = "HYPER_PARALLEL_SHARD_CASE_NAMES"   # comma-separated list
ENV_MESH_SHAPE = "HYPER_PARALLEL_SHARD_MESH_SHAPE"
ENV_MESH_NAMES = "HYPER_PARALLEL_SHARD_MESH_NAMES"
ENV_FRAMEWORK = "HYPER_PARALLEL_SHARD_FRAMEWORK"
ENV_DEVICE_TYPE = "HYPER_PARALLEL_SHARD_DEVICE_TYPE"
ENV_REPORT_DIR = "HYPER_PARALLEL_SHARD_REPORT_DIR"
ENV_CASE_MODE = "HYPER_PARALLEL_SHARD_CASE_MODE"     # "single" -> activate G6
ENV_FAIL_FAST = "HYPER_PARALLEL_SHARD_FAIL_FAST"     # "1" -> break on first fail

_SLUG_RE = re.compile(r"[^A-Za-z0-9._\-\[\]]+")


def slugify(s: str) -> str:
    """Lossy filesystem-safe slug. Brackets kept for parametrized case ids."""
    return _SLUG_RE.sub("_", s).strip("_") or "unnamed"


def match_filter(
        case_name: str,
        case_tags: Iterable[str],
        exact: Optional[Iterable[str]] = None,
        glob_pat: Optional[str] = None,
        tag: Optional[str] = None,
) -> bool:
    """A case is selected when *all* provided filters match."""
    if exact:
        if case_name not in set(exact):
            return False
    if glob_pat:
        if not fnmatch.fnmatchcase(case_name, glob_pat):
            return False
    if tag:
        if tag not in set(case_tags):
            return False
    return True


def build_numpy(spec: InputSpec) -> np.ndarray:
    """Construct a numpy array from an ``InputSpec`` with deterministic seed.

    When ``spec.data`` is provided it is returned directly (after a shape
    sanity check) — all other fields are ignored.
    """
    if spec.data is not None:
        if spec.data.shape != spec.shape:
            raise ValueError(
                f"InputSpec.data.shape={spec.data.shape} != "
                f"InputSpec.shape={spec.shape}"
            )
        return spec.data.astype(np.dtype(spec.dtype))
    rng = np.random.RandomState(spec.seed if spec.seed is not None else 0)
    dtype = np.dtype(spec.dtype)
    if spec.init == "randn":
        return rng.standard_normal(spec.shape).astype(dtype)
    if spec.init == "uniform":
        lo, hi = (0.0, 1.0) if spec.range is None else spec.range
        return rng.uniform(lo, hi, size=spec.shape).astype(dtype)
    if spec.init == "ones":
        return np.ones(spec.shape, dtype=dtype)
    if spec.init == "zeros":
        return np.zeros(spec.shape, dtype=dtype)
    if spec.init == "arange":
        return np.arange(int(np.prod(spec.shape)), dtype=dtype).reshape(spec.shape)
    raise ValueError(f"unknown InputSpec.init={spec.init!r}")


def find_free_port() -> int:
    """Return a TCP port that the OS reports as free *right now*.

    Uses the classic ``bind(("", 0))`` + ``getsockname()`` trick (same
    pattern as ``3rdparty/shmem/.../test_fusion_matmul_allreduce.py``).
    The kernel's free-port picker is authoritative — we don't rely on a
    counter file that can hand out a still-occupied port.

    There is a tiny race window after we close the socket and before the
    launcher binds it; the launcher's own retry logic handles that.
    """
    with contextlib.closing(
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", 0))
        return sock.getsockname()[1]


def parse_int_tuple(env_val: str) -> tuple:
    """Parse '2,2' -> (2, 2)."""
    return tuple(int(x) for x in env_val.split(",") if x.strip())


def parse_str_tuple(env_val: str) -> tuple:
    """Parse 'dp,tp' -> ('dp', 'tp')."""
    return tuple(x.strip() for x in env_val.split(",") if x.strip())
