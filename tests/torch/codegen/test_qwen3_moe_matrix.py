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
"""Opt-in Qwen3-MoE integration cases with real CLI and Trainer-sidecar entry points.

Set HP_CODEGEN_QWEN_CONFIG to a locally runnable Qwen3-MoE training recipe.
Set HP_CODEGEN_QWEN_NPU=1 and HP_CODEGEN_QWEN_BASELINE to run the eight-card
loss comparison against a baseline checkout with the same runtime and config.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "scripts/train_lm.py"
WORKER = Path(__file__).with_name("_qwen3_moe_entry.py")
CONFIG = os.environ.get("HP_CODEGEN_QWEN_CONFIG")
pytestmark = pytest.mark.skipif(not CONFIG, reason="set HP_CODEGEN_QWEN_CONFIG for the Qwen3-MoE matrix")


@pytest.fixture(name="recipes")
def make_recipes(tmp_path):
    """Derive test YAMLs without modifying the model's tracked training configuration."""
    full = yaml.safe_load(Path(CONFIG).read_text(encoding="utf-8"))
    full["accelerator"].update(tp_size=2, cp_size=2, ep_size=2, pp_size=1)
    full["fsdp_config"].update(dp_shard_size=4, edp_shard_size=4)
    single = deepcopy(full)
    single["accelerator"].update(pp_size=2, cp_size=1, ep_size=1)
    single["fsdp_config"].update(dp_shard_size=2, edp_shard_size=2)
    attention = next(rule for rule in full["plan_overrides"] if rule["match"] == "*.self_attn")
    single["plan_overrides"] = [{**attention, "match": "model.layers.0.self_attn"}]
    single["plan_overrides"].append({
        "match": "*.mlp",
        "params": {name: {"tp": "replicate"} for name in ("gate.weight", "experts.gate_up_proj", "experts.down_proj")},
        "out_src": {"output": {"tp": "replicate"}},
    })
    one_rule = deepcopy(full)
    one_rule["plan_overrides"] = [
        rule for rule in one_rule["plan_overrides"]
        if rule["match"] in ("*.self_attn", "*.mlp.forward")
    ]
    empty = deepcopy(full)
    empty["plan_overrides"] = []
    audit = deepcopy(empty)
    audit["accelerator"].update(tp_size=1, cp_size=1, ep_size=1, sequence_parallel=False, loss_parallel=False)
    audit["fsdp_config"].update(dp_shard_size=1, edp_shard_size=1)
    unmatched = deepcopy(single)
    unmatched["plan_overrides"][0]["match"] = "model.layers.999.self_attn"
    invalid = deepcopy(full)
    invalid["plan_overrides"][0]["replace"]["implementation"]["_target_"] = "missing_codegen_factory.no_such_module"
    paths = {}
    for name, config in (("full", full), ("empty", empty), ("audit", audit),
                         ("single", single), ("one_rule", one_rule),
                         ("unmatched", unmatched), ("invalid", invalid)):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths


def _run(command, log, *, tree=ROOT, timeout=200):
    with log.open("w", encoding="utf-8") as stream:
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(tree), os.getenv("PYTHONPATH", "")))}
        result = subprocess.run(
            command, cwd=tree, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False,
        )
    return result.returncode, log.read_text(encoding="utf-8", errors="replace")


def _cli(config, output, *prefix):
    return [sys.executable, *prefix, "-m", "hyper_parallel.codegen", "--verbose", "generate",
            "--config", str(config), "--output", str(output)]


def _train(config, output, ranks, *, stop=True, bad_child=None, overrides=()):
    events = output.with_suffix(".events")
    events.mkdir()
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ranks}",
               str(WORKER), "--events", str(events), "--output", str(output)]
    if stop:
        command.append("--stop-after-sidecar")
    if bad_child:
        command.extend(("--child-config", str(bad_child)))
    return [*command, str(ENTRY), str(config), *overrides], events


def _bundle(path):
    files = {file.relative_to(path).as_posix(): sha256(file.read_bytes()).hexdigest()
             for file in path.rglob("*") if file.is_file()}
    assert len(files) == 3 and "__init__.py" in files, files
    source, = path.glob("modeling_*_gen_npu.py")
    diff, = path.glob("modeling_*_gen_npu.py.diff")
    assert diff.read_text(encoding="utf-8").strip(), diff
    return files, ast.parse(source.read_text(encoding="utf-8"))


def _prepared(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "__hp_prepared_components__" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return ()


def _assert_success(result):
    status, log = result
    assert status == 0, log[-8000:]


def test_single_module_replacement_and_pipeline_topology(recipes, tmp_path):
    """Generate into an empty directory and check layout degrees and the exact replacement FQN."""
    output = tmp_path / "cli"
    output.mkdir()
    _assert_success(_run(_cli(recipes["single"], output), tmp_path / "cli.log"))
    _, tree = _bundle(output)
    assert [record["primary_fqn"] for record in _prepared(tree)] == ["model.layers.0.self_attn"]
    installer = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "install_prepared_components")
    degrees = ast.literal_eval(installer.args[2])
    assert {key: degrees[key] for key in ("dp_size", "tp_size", "pp_size")} == {
        "dp_size": 2, "tp_size": 2, "pp_size": 2,
    }


def test_empty_patch_and_active_cp_contract(recipes, tmp_path):
    """Publish a source audit for a true empty patch and preserve the active-CP planner contract."""
    output = tmp_path / "audit"
    _assert_success(_run(_cli(recipes["audit"], output), tmp_path / "audit.log"))
    _, tree = _bundle(output)
    assert not _prepared(tree), "empty patch installed an unexpected component"
    status, log = _run(_cli(recipes["empty"], tmp_path / "empty"), tmp_path / "empty.log")
    assert status != 0 and "has no CP-aware forward replacement" in log, log[-4000:]
    assert not (tmp_path / "empty").exists(), "invalid CP config published an artifact"


@pytest.mark.parametrize("ranks, recipe", [(8, "single"), (8, "full"), (4, "full"), (1, "full")])
def test_real_entry_single_child_and_parity(recipes, tmp_path, ranks, recipe):
    """Use the actual Trainer entry at each launcher size and compare every bundle byte."""
    cli_output = tmp_path / "cli"
    _assert_success(_run(_cli(recipes[recipe], cli_output), tmp_path / "cli.log"))
    sidecar_output = tmp_path / "sidecar"
    command, events = _train(recipes[recipe], sidecar_output, ranks)
    _assert_success(_run(command, tmp_path / "launcher.log"))
    observed = [json.loads(path.read_text(encoding="utf-8")) for path in events.glob("*.json")]
    assert sorted(event["rank"] for event in observed) == list(range(ranks)), observed
    assert [event["rank"] for event in observed if event["pid"]] == [0], observed
    assert _bundle(cli_output)[0] == _bundle(sidecar_output)[0]


def test_sidecar_forwards_explicit_override(recipes, tmp_path):
    """Use the same nondefault setting through both entry points."""
    override = "--accelerator.sequence_parallel=false"
    cli = tmp_path / "cli"
    _assert_success(_run([*_cli(recipes["full"], cli), override], tmp_path / "cli.log"))
    sidecar = tmp_path / "sidecar"
    command, _ = _train(recipes["full"], sidecar, 1, overrides=(override,))
    _assert_success(_run(command, tmp_path / "sidecar.log"))
    expected, tree = _bundle(cli)
    assert expected == _bundle(sidecar)[0]
    installer = next(node for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "install_prepared_components")
    assert ast.literal_eval(installer.args[2])["sequence_parallel"] is False


def _wait_for_publishers(logs):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if all(path.exists() and "virtual-EP mesh" in path.read_text(encoding="utf-8", errors="replace")
               for path in logs):
            return
        time.sleep(0.1)
    pytest.fail("both generators did not reach native planning under the publication lock")


def test_rebuild_and_concurrent_publish(recipes, tmp_path):
    """Force a clean rebuild, then allow a CLI and real sidecar to contend for one bundle."""
    output = tmp_path / "shared"
    command = _cli(recipes["full"], output)
    _assert_success(_run(command, tmp_path / "first.log"))
    hashes, _ = _bundle(output)
    (output / "old-file").write_text("stale", encoding="utf-8")
    _assert_success(_run(command, tmp_path / "second.log"))
    assert hashes == _bundle(output)[0]
    sidecar, events = _train(recipes["full"], output, 1)
    logs = [tmp_path / "concurrent-cli.log", tmp_path / "concurrent-sidecar.log"]
    lock = tmp_path / ".shared.codegen.lock"
    lock.write_text(str(os.getpid()), encoding="ascii")
    with ThreadPoolExecutor(max_workers=2) as pool:
        cli_job = pool.submit(_run, command, logs[0])
        sidecar_job = pool.submit(_run, sidecar, logs[1])
        try:
            _wait_for_publishers(logs)
            assert not cli_job.done() and not sidecar_job.done()
            assert hashes == _bundle(output)[0]
        finally:
            lock.unlink()
        _assert_success(cli_job.result())
        _assert_success(sidecar_job.result())
    assert json.loads((events / "0.json").read_text(encoding="utf-8"))["exit"] == 0
    assert hashes == _bundle(output)[0]
    assert not list(tmp_path.glob(".codegen_*")) and not list(tmp_path.glob(".*.codegen.lock"))


@pytest.mark.parametrize("name, diagnostic", [
    ("unmatched", "matched no compatible module"),
    ("invalid", "missing_codegen_factory.no_such_module"),
])
def test_failed_replacement_does_not_publish(recipes, tmp_path, name, diagnostic):
    """Keep the prior complete bundle when a replacement cannot be applied."""
    output = tmp_path / "published"
    _assert_success(_run(_cli(recipes["full"], output), tmp_path / "success.log"))
    previous, _ = _bundle(output)
    status, log = _run(_cli(recipes[name], output), tmp_path / f"{name}.log")
    assert status != 0 and diagnostic in log, log[-4000:]
    assert previous == _bundle(output)[0]


def test_torchrun_cli_is_rejected(recipes, tmp_path):
    """Reject distributed CLI launch before publication."""
    output = tmp_path / "unsupported"
    launcher = ("-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2")
    status, log = _run(_cli(recipes["full"], output, *launcher), tmp_path / "torchrun.log")
    assert status != 0 and "torchrun" in log and not output.exists(), log[-4000:]


def test_native_match_and_world_size_contracts(recipes, tmp_path):
    """Observe native validation with inactive capture and no process group."""
    worker = """
import sys
from hyper_parallel.codegen.build import create_offline_distributed_setup
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.config import normalize_distributed_setup_overrides
from hyper_parallel.trainer.runtime.distributed import create_distributed_setup_from_config

mode = sys.argv[1]
config = parse_training_args(sys.argv[2:])
if mode == "topology":
    create_distributed_setup_from_config(config)
else:
    setup = create_offline_distributed_setup(config)
    normalize_distributed_setup_overrides(setup, config)
    config.model.build(distributed_setup=setup)
"""
    for mode, config, expected in (
        ("topology", "full", "world_size 1 is not divisible"),
        ("match", "unmatched", "matched no compatible module"),
    ):
        status, log = _run([sys.executable, "-c", worker, mode, str(recipes[config])], tmp_path / f"{mode}.log")
        assert status != 0 and expected in log, log[-4000:]


@pytest.mark.parametrize("recipe", ["one_rule", "full"])
def test_parallel_strategy_and_replacement_sites(recipes, tmp_path, recipe):
    """Compare generated replacement sites and CP/EP calls with the observed Trainer plan."""
    output = tmp_path / "full"
    _assert_success(_run(_cli(recipes[recipe], output), tmp_path / "cli.log"))
    _, tree = _bundle(output)
    worker = """
import json
import sys
from unittest.mock import patch
from hyper_parallel.codegen.build import generate_from_native_build
from hyper_parallel.codegen.cli import _load_config

class Captured(Exception):
    pass

def observe(capture):
    model = capture.model
    print("OBSERVED " + json.dumps({
        "replaced": sorted(model._hp_replaced_module_fqns),
        "parallel": {fqn: binding.binding_kind for fqn, binding in capture.plan.parallel_modules.items()},
        "dims": {key: getattr(capture.mesh, key) for key in ("dp_size", "tp_size", "cp_size", "ep_size")},
        "compute": sorted(fqn for fqn, spec in capture.plan.modules.items() if spec.local_compute_fn),
    }), flush=True)
    raise Captured

try:
    with patch("hyper_parallel.codegen.build.project_model_build", side_effect=observe):
        generate_from_native_build(_load_config(sys.argv[1], []))
except Captured:
    pass
"""
    result = _run([sys.executable, "-c", worker, str(recipes[recipe])], tmp_path / "native-plan.log")
    _assert_success(result)
    native = json.loads(next(line.removeprefix("OBSERVED ") for line in result[1].splitlines()
                             if line.startswith("OBSERVED ")))
    records = _prepared(tree)
    assert sorted(record["primary_fqn"] for record in records) == native["replaced"]
    if recipe == "one_rule":
        assert len({record["rule_id"].split(":replace:", 1)[0] for record in records}) == 1
    assert native["dims"] == {"dp_size": 2, "tp_size": 2, "cp_size": 2, "ep_size": 2}
    assert len(native["compute"]) == 4 and all(name.endswith(".mlp") for name in native["compute"])
    cp_sites = {name for name, kind in native["parallel"].items() if kind == "cp"}
    assert cp_sites == {f"model.layers.{layer}.self_attn" for layer in range(4)}
    generated = ast.unparse(tree)
    for term in ("self.exchange_attention(", "self.restore_attention(", "self._hp_tp.all_gather(",
                 "self.qwen3moe_ep_compute_fn(", "self._hp_boundary.redistribute_outputs("):
        assert term in generated, term


@pytest.mark.skipif(os.getenv("HP_CODEGEN_QWEN_NPU") != "1", reason="requires eight free Ascend devices")
def test_native_loss_and_sidecar_failure_isolation(recipes, tmp_path):
    """Compare the actual Trainer run with a baseline and an independently failing child."""
    baseline = Path(os.environ["HP_CODEGEN_QWEN_BASELINE"]).resolve()
    entry = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8"]
    native = _run([*entry, "scripts/train_lm.py", str(recipes["full"])], tmp_path / "baseline.log",
                  tree=baseline, timeout=600)
    sidecar, _ = _train(recipes["full"], tmp_path / "sidecar", 8, stop=False)
    current = _run(sidecar, tmp_path / "current.log", timeout=600)
    bad_child, _ = _train(recipes["full"], tmp_path / "failed", 8, stop=False, bad_child=recipes["invalid"])
    injected = _run(bad_child, tmp_path / "injected.log", timeout=600)
    for result in (native, current, injected):
        _assert_success(result)
    assert "native training continues" in injected[1]
    losses = [re.findall(r"total_loss=([0-9.eE+-]+)", result[1]) for result in (native, current, injected)]
    assert len(losses[0]) == yaml.safe_load(recipes["full"].read_text(encoding="utf-8"))["training"]["train_iters"]
    assert losses[0] == losses[1] == losses[2], losses
