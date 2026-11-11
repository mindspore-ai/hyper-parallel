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
"""Framework-free launch configuration for DeepSeek HyperMulticore integration."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.common.distributed_launcher import torchrun_case

EXTENDED_LEVEL1 = os.getenv("HP_MEGA_MOE_LEVEL1_EXTENDED", "0") == "1"


def activate_multicore_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Activate the installed package before starting framework workers."""
    spec = importlib.util.find_spec("hyper_multicore")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("Install HyperMulticore in the test Python environment")
    script = Path(spec.origin).resolve().parent / "lib" / "set_env.bash"
    result = subprocess.run(
        ["bash", "-c", 'source "$1" >&2 && env -0', "multicore-activation", str(script)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"HyperMulticore activation failed: {script}\n{result.stderr}")
    environment = dict(entry.split("=", 1) for entry in result.stdout.split("\0") if entry)
    for name in ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH"):
        monkeypatch.setenv(name, environment[name])


@contextmanager
def without_inherited_rank_environment() -> Iterator[None]:
    """Keep parent rank metadata out of locally launched workers."""
    names = ("RANK_TABLE_FILE", "RANK_ID", "RANK_SIZE")
    previous = {name: os.environ.pop(name) for name in names if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(previous)


def run_acceptance_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str, cards: int, worker: str,
    variant: str | None = None,
) -> dict:
    """Run a model worker and require fresh numerical evidence from every rank."""
    mode = os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    result_dir = Path(os.getenv("HP_MEGA_MOE_EVIDENCE_DIR", str(tmp_path))) / case / mode / (variant or "default")
    result_path = result_dir / f"{case}.json"
    result_path.unlink(missing_ok=True)
    monkeypatch.setenv("HP_DYNAMIC_DISPATCH", mode)
    monkeypatch.setenv("HP_MEGA_MOE_LEVEL1_RESULT", str(result_path))
    monkeypatch.delenv("HYPER_MULTICORE_SHMEM_HEAP_SIZE", raising=False)
    activate_multicore_environment(monkeypatch)
    with without_inherited_rank_environment():
        torchrun_case(str(Path(__file__).with_name(worker)), case, num_proc=cards)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(result["ranks"]) == cards, f"expected {cards} rank records, got {len(result['ranks'])}."
    return result
