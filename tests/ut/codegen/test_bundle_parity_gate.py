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
"""End-to-end CPU gate for the CLI and the real rank-zero sidecar."""

import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from hyper_parallel.codegen.emit.family_forwards import verify_parallel_family_forwards
from hyper_parallel.codegen.emit.modeling import _preinstalled_calls, _verify_replacement_records
from hyper_parallel.codegen.train_sidecar import start_codegen_sidecar
from scripts.check_codegen_bundle_parity import bundle_hashes
from tests.common.mla_cp_utils import make_deepseek_model


def test_cli_and_rank_zero_sidecar_publish_identical_bundles(tmp_path, monkeypatch):
    """Run both production entry points with a small offline checkpoint config."""
    root = Path(__file__).resolve().parents[3]
    recipe = root / "hyper_parallel/models/deepseek_v3/recipes/mla_ulysses.yaml"
    checkpoint = tmp_path / "checkpoint"
    make_deepseek_model().config.save_pretrained(checkpoint)
    cli = tmp_path / "cli"
    sidecar = tmp_path / "sidecar"
    overrides = [
        f"--model.pretrained_model_name_or_path={checkpoint}",
        "--accelerator.cp_size=2",
        "--fsdp_config.dp_shard_size=2",
    ]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    result = subprocess.run(
        [sys.executable, "-m", "hyper_parallel.codegen", "generate",
         "--config", str(recipe), "--output", str(cli), *overrides],
        check=False, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"CLI exited {result.returncode}. {result.stderr}"
    process = start_codegen_sidecar(recipe, ["--output", str(sidecar), *overrides])
    assert process is not None, "Rank zero did not start the Codegen sidecar"
    try:
        exit_code = process.wait(timeout=120)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    assert exit_code == 0, f"Trainer-sidecar exited {exit_code}"
    expected, actual = bundle_hashes(cli), bundle_hashes(sidecar)
    assert actual == expected, f"CLI bundle {expected}, sidecar bundle {actual}"
    modeling, = cli.glob("modeling_*_gen_npu.py")
    tree = ast.parse(modeling.read_text(encoding="utf-8"))
    prepared, emitted = _preinstalled_calls(tree)
    name = "CPMultiLatentAttentionModule"
    assert prepared == emitted
    assert emitted["model.layers.0.self_attn"] == "build_prepared_family_replacement"
    verify_parallel_family_forwards(tree, {
        "model.layers.0.self_attn": {
            "type": name,
            "kind": "cp",
            "stages": ("exchange_attention", "restore_attention", "project_output"),
        },
    })


def test_replacement_metadata_without_an_executable_call_is_rejected():
    """Record-only artifacts must not pass the replacement source gate."""
    tree = ast.parse("""
def constructor(self):
    self.__hp_prepared_components__ = (
        {"primary_fqn": "layer.attention", "replay": {"constructor": "demo:ParallelAttention"}},
    )
    install_prepared_components(self, self.__hp_prepared_components__, {}, globals())
""")
    record = SimpleNamespace(module_fqn="layer.attention", replacement_type=object)
    missing = _verify_replacement_records(tree, {}, (record,), [])
    assert missing == ["layer.attention (missing executable constructor 'ParallelAttention')"], (
        f"Metadata-only replacement should be rejected, got {missing}"
    )
