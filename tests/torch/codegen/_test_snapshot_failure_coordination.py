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
"""Distributed worker for owner/non-owner sidecar failure coordination."""

import os
from pathlib import Path
import shutil
import tempfile

import torch.distributed as dist

from hyper_parallel.codegen import manager
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.codegen.snapshot import CodegenStatus


def test_rank0_generation_failure_releases_non_owner(monkeypatch):
    """Every rank records failure when the owner cannot render the bundle."""
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    shared_root = Path(tempfile.gettempdir()) / (
        f"hp_codegen_failure_{os.environ['MASTER_PORT']}"
    )
    source = shared_root / "modeling_toy.py"
    artifact_dir = shared_root / "generated"
    if rank == 0:
        shared_root.mkdir(parents=True, exist_ok=True)
        source.write_text("class ToyModel:\n    pass\n", encoding="utf-8")
    dist.barrier()

    session = CodegenBuildSession()
    session.record_source({
        "model_type": "toy",
        "architecture": "ToyModel",
        "file_path": str(source),
        "is_hf_model": True,
    })
    session.record_replacements()
    session.record_plan({
        "param_plan": {"block": {"params": {}}},
        "parallel_dims": {"tp_size": 2},
    })
    session.record_plan_application({"applied_boundaries": ["block"]})

    if rank == 0:
        monkeypatch.setattr(
            manager,
            "emit_bundle",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected render failure")),
        )

    snapshot = session.finish(
        lambda derived: manager.ensure_codegen_artifact_from_snapshot(
            derived,
            str(artifact_dir),
            timeout_s=10.0,
        )
    )

    assert snapshot.status is CodegenStatus.FAILED, (
        f"rank {rank} should observe failed sidecar publication, got={snapshot.status}"
    )
    dist.barrier()
    if rank == 0:
        shutil.rmtree(shared_root)
    dist.destroy_process_group()
