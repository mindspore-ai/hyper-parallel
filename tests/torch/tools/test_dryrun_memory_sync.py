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
"""Dry-run vs real-training memory reconciliation ST (launcher, torch-free).

Runs the same DeepSeek-V4.1 validation-crop recipe twice — once through the
dry-run simulator (single process, FakeTensors, no device) and once through
torchrun on 8 cards with allocator snapshots — then reconciles the two memory
traces: the logical peak must land inside a tolerance band of the real trace
peak, and every large real allocation call site must be present in the
dry-run trace (see ``dryrun_sync_common.reconcile``).

Model/tokenizer/data assets are environment-supplied (the released config is
not redistributable). Set:

    HP_DRYRUN_ST_MODEL_PATH    DeepSeek-V4.1-Flash config/tokenizer directory
    HP_DRYRUN_ST_ENGRAM_PATH   engram_depth_preserving_d8.json
    HP_DRYRUN_ST_DATA_PATH     online_4k.jsonl

The test skips when any is unset or missing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark

_ASSET_ENV = {
    "model": "HP_DRYRUN_ST_MODEL_PATH",
    "engram": "HP_DRYRUN_ST_ENGRAM_PATH",
    "data": "HP_DRYRUN_ST_DATA_PATH",
}

#: Scenario matrix: (name, extra CLI overrides). The base recipe is EP8/FSDP8.
_SCENARIOS = (
    ("cp2_ep4_fsdp4", (
        "--accelerator.cp_size=2",
        "--accelerator.ep_size=4",
        "--fsdp_config.dp_shard_size=4",
    )),
)


def _assets() -> dict:
    """Resolve env-supplied assets or skip the ST."""
    resolved = {}
    for key, env_name in _ASSET_ENV.items():
        value = os.environ.get(env_name)
        if not value or not Path(value).exists():
            pytest.skip(f"{env_name} is unset or missing; dry-run sync ST needs local V4.1 assets")
        resolved[key] = value
    return resolved


def _write_recipe(directory: Path, assets: dict) -> Path:
    """Materialize the reconciliation recipe with env-supplied asset paths."""
    template = Path(__file__).with_name("dryrun_sync_recipe.yaml").read_text(encoding="utf-8")
    recipe = (
        template
        .replace("@MODEL_PATH@", assets["model"])
        .replace("@ENGRAM_PATH@", assets["engram"])
        .replace("@DATA_PATH@", assets["data"])
        .replace("@OUTPUT_ROOT@", str(directory))
    )
    path = directory / "recipe.yaml"
    path.write_text(recipe, encoding="utf-8")
    return path


def _run(command: list, **kwargs) -> None:
    """Run one child process, surfacing its tail on failure."""
    result = subprocess.run(command, capture_output=True, text=True, check=False, **kwargs)
    if result.returncode != 0:
        tail = (result.stdout + result.stderr)[-4000:]
        raise AssertionError(f"{' '.join(command[:6])}... failed:\n{tail}")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dryrun_memory_reconciles_with_real_training():
    """
    Feature: Dry-run logical memory vs real allocator traces.
    Description: Run the V4.1 validation crop through the dry-run simulator and
        through an 8-card real training step with allocator snapshots; compare
        peak memory and per-call-site large allocations.
    Expectation: Peak ratio inside the tolerance band; every large real
        allocation site is present in the dry-run trace.
    """
    assets = _assets()
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        recipe = _write_recipe(directory, assets)
        for name, overrides in _SCENARIOS:
            dry_dir = directory / f"dry_{name}"
            real_dir = directory / f"real_{name}"
            _run([
                sys.executable, "-m", "hyper_parallel.tools.dryrun", str(recipe),
                *overrides,
                f"--dry_run.snapshot.path={dry_dir}",
            ])
            _run([
                sys.executable, "-m", "torch.distributed.run",
                "--standalone", "--nproc_per_node=8",
                "--module", "examples.training_demo.train_text", str(recipe),
                *overrides,
                f"--memory.save_path={real_dir}",
            ])
            report = subprocess.run(
                [sys.executable, "-c", (
                    "import sys, json; sys.path.insert(0, sys.argv[1]);"
                    "import dryrun_sync_common as c;"
                    "dry = c.load_snapshot(sys.argv[2] + '/rank_0/memory.pickle');"
                    "real = c.latest_snapshot(sys.argv[3]);"
                    "print(json.dumps(c.reconcile(dry, real)))"
                ), str(Path(__file__).parent), str(dry_dir), str(real_dir)],
                capture_output=True, text=True, check=True,
            )
            problems = json.loads(report.stdout.strip().splitlines()[-1])
            assert not problems, f"scenario {name} diverged:\n" + "\n".join(problems)
