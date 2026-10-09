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
"""Explicit feature ST launchers; collection never imports training frameworks."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


HERE = Path(__file__).resolve().parent


def _result_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep rank logs and completion evidence under one caller-owned directory."""
    output = Path(os.environ.get("RL_ST_RESULT_DIR", str(tmp_path))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("completed.json", "result.json")):
        pytest.fail("Use a fresh RL_ST_RESULT_DIR; stale evidence must not satisfy this run")
    monkeypatch.setenv("RL_ST_RESULT_DIR", str(output))
    monkeypatch.chdir(output)
    return output


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_agent_dp_padding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Feature: agent DP padding.

    Description: compare two real CPU ranks with an independent unpadded objective.
    Expectation: padding has zero gradient and both updates match the reference.
    """
    output = _result_directory(tmp_path, monkeypatch)
    torchrun_case(str(HERE / "_agent_dp.py"), "test_padding", num_proc=2)
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "passed" and result["world_size"] == 2 and result["steps"] == 2


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
@pytest.mark.parametrize("feature", ["moe", "code", "agent"])
def test_feature_training(feature: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Feature: MoE, code and external agent production training.

    Description: launch the selected feature worker with an explicit local recipe.
    Expectation: two updates, real parameter changes and feature contracts pass.
    """
    variable = f"RL_ST_{feature.upper()}_CONFIG"
    config = os.environ.get(variable)
    if not config:
        if os.environ.get("RL_ST_REQUIRED") == "1":
            pytest.fail(f"Required system test needs {variable}")
        pytest.skip(f"Set {variable} and RL_ST_WORLD_SIZE to run this real-NPU case")
    config_path = Path(config).resolve()
    if not config_path.is_file():
        pytest.fail(f"Recipe does not exist: {config_path}")
    world = int(os.environ.get("RL_ST_WORLD_SIZE", "0"))
    if world < 1:
        pytest.fail("RL_ST_WORLD_SIZE must explicitly select the recipe's device count")
    monkeypatch.setenv("RL_ST_CONFIG", str(config_path))
    output = _result_directory(tmp_path, monkeypatch)
    torchrun_case(str(HERE / f"_{feature}_train.py"), "test_training", num_proc=world)
    result = json.loads((output / "completed.json").read_text(encoding="utf-8"))
    assert result["status"] == "passed" and result["steps"] >= 2


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_code_sandbox() -> None:
    """Feature: real remote Python judging.

    Description: score known programs through the configured SandboxFusion service.
    Expectation: correct, wrong, runtime, timeout and output-limit results stay distinct.
    """
    required = ("RL_ST_SANDBOX_ENDPOINT", "RL_ST_SANDBOX_RUNTIME_VERSION")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        if os.environ.get("RL_ST_REQUIRED") == "1":
            pytest.fail(f"Required sandbox test needs {missing}")
        pytest.skip(f"Configure the real sandbox first: {missing}")
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "_code_sandbox.py") + "::test_contract"],
                   check=True)
