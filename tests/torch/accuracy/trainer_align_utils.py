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
"""Shared helpers for the **trainer-driven** real-weight accuracy STs.

Unlike the standalone ``_test_*_accuracy.py`` workers, these drive the real
training entry point (``scripts/train_lm.py`` / ``scripts/train_vl.py``)
end-to-end, so the comparison validates exactly the path production training
takes: a reduced-layer real HF checkpoint, deterministic data + fixed seed,
and the per-step ``loss`` logged by the trainer compared across configs.
Tests skip (exit 0) when the checkpoint or the required NPUs are absent.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

import yaml

# Per-step loss absolute-error bar between the single-card and the parallel run.
ATOL = 2e-3
STEPS = 20
_LOSS_RE = re.compile(r"step=(\d+)\s*\|\s*loss=([0-9.]+)")


def repo_root() -> str:
    """Return the hyper-parallel repo root (three levels up from this file)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def checkpoint_present(path: str) -> bool:
    """``True`` if ``path`` is an on-disk HF checkpoint dir with an index."""
    return os.path.isdir(path) and os.path.isfile(
        os.path.join(path, "model.safetensors.index.json")
    )


def base_config(model_name: str, ckpt: str, num_layers: int, vl: bool) -> dict:
    """Build the shared single-card training config dict.

    Deterministic, fp32 end-to-end, gentle warmup-cosine lr, synthetic data
    (``dummy`` text or ``vl_dummy`` multimodal). The ``accelerator`` block is
    filled per-case by the caller (``dp_shard=1`` for the single-card baseline,
    ``ep`` / ``tp`` / ``cp`` / ``pp`` / ``dp_shard`` for the parallel variants).
    """
    overrides: dict = {"num_hidden_layers": num_layers}
    model: dict = {
        "name": model_name,
        "weights_path": ckpt,
        "tokenizer_path": ckpt,
        "config_overrides": overrides,
    }
    if vl:
        overrides.clear()
        overrides.update({"vl": True, "text_config": {"num_hidden_layers": num_layers}})
        model["freeze_modules"] = ["model.visual"]
        data = {"type": "vl_dummy", "max_seq_len": 64,
                "vl_grid_t": 2, "vl_grid_h": 2, "vl_grid_w": 2}
    else:
        data = {"type": "dummy", "max_seq_len": 64}
    return {
        "model": model,
        "data": data,
        "train": {
            "max_steps": STEPS,
            "num_train_epochs": 1,
            "global_batch_size": 4,
            "micro_batch_size": 1,
            "seed": 1234,
            "backend": "torch",
            "init_device": "meta",
            "accelerator": {"comm_fusion": True},
            "optimizer": {
                "type": "adamw", "lr": 1.0e-4, "lr_min": 0.0,
                "lr_decay_style": "cosine", "lr_warmup_ratio": 0.1,
                "max_grad_norm": 1.0e9, "weight_decay": 0.0,
                "loss_aggregation": "token_weighted", "foreach": False,
            },
            "mixed_precision": {
                "enabled": True, "param_dtype": "float32",
                "reduce_dtype": "float32", "output_dtype": "float32",
            },
            "gradient_checkpointing": {"activation_checkpoint": "none"},
            "checkpoint": {"output_dir": "outputs/_trainer_align",
                           "save_steps": 0, "save_hf_weights": False},
            "logging": {"log_steps": 1, "report_throughput": False},
            "debug": {"deterministic": True},
        },
    }


def run_trainer_losses(config: dict, nproc: int, master_port: int, vl: bool) -> list:
    """Run one trainer process group on ``nproc`` cards and return per-step loss.

    Writes ``config`` to a temp yaml, launches
    ``torchrun --nproc-per-node=nproc scripts/train_{lm,vl}.py`` pinned to cards
    ``0..nproc-1``, and parses the ``step=N | loss=X`` log line per step. Raises
    ``AssertionError`` with the captured output if the run produced no losses.
    """
    root = repo_root()
    script = os.path.join(root, "scripts", "train_vl.py" if vl else "train_lm.py")
    env = os.environ.copy()
    env["HYPER_PARALLEL_PLATFORM"] = "torch"
    env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
    # Honour an externally-pinned card list (CI / multi-suite runs) — take its
    # first ``nproc`` ids — else default to cards ``0..nproc-1``.
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    cards = visible.split(",")[:nproc] if visible else [str(i) for i in range(nproc)]
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(cards)
    env["HCCL_IF_BASE_PORT"] = str(master_port + 4000)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        cfg_path = handle.name
    try:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc-per-node={nproc}", "--master_addr=127.0.0.1",
            f"--master_port={master_port}", script, cfg_path,
        ]
        proc = subprocess.run(cmd, env=env, cwd=root, capture_output=True,
                              text=True, check=False)
    finally:
        os.unlink(cfg_path)
    losses = {int(m.group(1)): float(m.group(2))
              for m in _LOSS_RE.finditer(proc.stdout + proc.stderr)}
    if not losses:
        output = proc.stdout + proc.stderr
        dump_dir = os.path.join(root, "outputs", "_trainer_align")
        os.makedirs(dump_dir, exist_ok=True)
        dump = os.path.join(dump_dir, f"failed_run_port{master_port}.log")
        with open(dump, "w", encoding="utf-8") as handle:
            handle.write(output)
        # torchrun's elastic footer ends the output; the worker's own traceback
        # comes earlier — surface the first one instead of the raw tail.
        first_tb = output.find("Traceback (most recent call last)")
        excerpt = output[first_tb:first_tb + 3000] if first_tb != -1 else output[-2000:]
        raise AssertionError(
            f"trainer produced no losses (nproc={nproc}); "
            f"full output at {dump}\n{excerpt}")
    return losses


def assert_trajectories_match(case: str, single: dict, parallel: dict) -> None:
    """Assert the parallel per-step loss tracks single-card within :data:`ATOL`.

    Also asserts the single-card loss is finite and decreases (real weights +
    real data = real descent), so a degenerate flat run cannot pass vacuously.
    """
    common = sorted(set(single) & set(parallel))
    if not common:
        raise AssertionError(f"{case}: no overlapping steps to compare.")
    worst = max(abs(single[i] - parallel[i]) for i in common)
    if worst > ATOL:
        offenders = [
            f"step {i}: single={single[i]:.6f} parallel={parallel[i]:.6f} "
            f"|Δ|={abs(single[i] - parallel[i]):.6f}"
            for i in common if abs(single[i] - parallel[i]) > ATOL
        ]
        raise AssertionError(
            f"{case}: parallel loss drifts from single-card beyond atol={ATOL} "
            f"(max|Δ|={worst:.6f}):\n  " + "\n  ".join(offenders[:5])
        )
    first, last = single[common[0]], single[common[-1]]
    if not last < first:
        raise AssertionError(
            f"{case}: single-card loss did not decrease "
            f"(first={first:.6f}, last={last:.6f})."
        )
    print(f"[trainer-align] {case} ok: max|Δ|={worst:.6e} over {len(common)} steps "
          f"(single {first:.4f}->{last:.4f}).")
