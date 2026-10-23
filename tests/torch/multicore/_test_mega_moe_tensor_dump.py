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
"""Two-rank Torch worker for one MegaMoe forward L0 Tensor dump."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.diagnostics import schedule
from hyper_parallel.core.multicore.diagnostics import dump as tensor_dump
from tests.torch.multicore import _test_mega_moe as common


_DUMP_TENSORS = (
    "target",
    "src",
    "up_proj_weight",
    "up_proj_glist",
    "up_proj_y",
    "swiglu_out",
    "down_proj_weight",
    "down_proj_glist",
    "down_proj_y",
    "combine_out",
)


def _build_layer(shape: common.MoeShape) -> MegaMoeExperts:
    """Construct the small representative two-rank MegaMoe layer."""
    return MegaMoeExperts(
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        dispatch_mode=os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push"),
        ep_size=shape.ep_size,
        ep_group=dist.group.WORLD,
    ).to(device=common.DEVICE, dtype=torch.bfloat16)


def test_mega_moe_forward_tensor_dump_worker() -> None:
    """Feature: MegaKernel diagnostics.
    Description: Warm up, synchronize ranks and dump one small forward invocation.
    Expectation: Mega moe forward tensor dump worker.
    """
    shape = common.MoeShape()
    topk_ids, topk_weights, tokens_per_expert = common.make_balanced_route(shape)
    hidden_states, _ = common.make_data(shape)
    layer = _build_layer(shape)
    result_dir = os.getenv("HP_MEGA_MOE_DUMP_RESULT_DIR")
    if not result_dir:
        raise ValueError("HP_MEGA_MOE_DUMP_RESULT_DIR must name the dump output directory")
    config = tensor_dump.TensorDumpConfig(
        result_dir, kernel_names="mega_moe", tensor_names=_DUMP_TENSORS, directions="forward", ranks=0,
    )
    capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=1)
    try:
        with torch.no_grad():
            layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
            torch.npu.synchronize()
            dist.barrier()
            with tensor_dump.tensor_dump(config, schedule=capture_schedule) as dumper:
                layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
                dumper.step()
        if common.RANK == 0:
            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tensors = manifest["invocations"][0]["tensors"]
            names = {item["name"] for item in tensors}
            assert names == set(_DUMP_TENSORS), (
                f"unexpected dump tensors: expected={set(_DUMP_TENSORS)}, got={names}"
            )
            assert manifest["schemaVersion"] == 4, (
                f"unexpected manifest schema: expected=4, got={manifest['schemaVersion']}"
            )
            assert manifest["tensorSerialization"] == "torch.save", (
                f"unexpected Tensor serialization: {manifest['tensorSerialization']}"
            )
            assert manifest["tensorFileLayout"] == "stage_only", (
                f"unexpected Tensor file layout: {manifest['tensorFileLayout']}"
            )
            assert manifest["invocations"][0]["kernelName"] == "mega_moe", (
                f"unexpected kernel name: {manifest['invocations'][0]['kernelName']}"
            )
            swiglu_ports = [
                (port["stage"], port["io"], port["index"])
                for item in tensors
                if item["name"] == "swiglu_out"
                for port in item["ports"]
            ]
            expected_swiglu_ports = [("SwiGLU", "output", 0), ("GMM2", "input", 0)]
            assert swiglu_ports == expected_swiglu_ports, (
                f"unexpected swiglu ports: expected={expected_swiglu_ports}, got={swiglu_ports}"
            )
            invocation_dir = manifest_path.parent / "invocation_000000_forward"
            assert not (invocation_dir / "before").exists(), f"unexpected before directory: {invocation_dir}"
            assert not (invocation_dir / "after").exists(), f"unexpected after directory: {invocation_dir}"
            for metadata in tensors:
                primary_path = manifest_path.parent / metadata["file"]
                assert metadata["file"] == metadata["ports"][0]["file"], (
                    f"primary file does not reference the first stage port: metadata={metadata}"
                )
                for port in metadata["ports"]:
                    stage_path = manifest_path.parent / port["file"]
                    assert stage_path.is_file(), f"missing stage file: path={stage_path}"
                    assert primary_path.samefile(stage_path), (
                        f"stage ports do not share storage: primary={primary_path}, stage={stage_path}"
                    )
                payload = torch.load(primary_path, map_location="cpu", weights_only=True)
                assert list(payload.shape) == metadata["actualShape"], (
                    f"shape mismatch for {metadata['name']}: "
                    f"expected={metadata['actualShape']}, got={list(payload.shape)}"
                )
                assert torch.isfinite(payload).all().item(), (
                    f"non-finite dump for {metadata['name']}: payload={payload}"
                )
                assert str(payload.dtype).removeprefix("torch.") == metadata["storedDtype"], (
                    f"dtype mismatch for {metadata['name']}: "
                    f"expected={metadata['storedDtype']}, got={payload.dtype}"
                )
            print(f"MEGA_KERNEL_TENSOR_DUMP={manifest_path}")
    finally:
        layer.close()
    dist.barrier()


def test_mega_moe_forward_l1_tensor_dump_worker() -> None:
    """Feature: MegaKernel diagnostics.
    Description: Dump task-local SwiGLU inputs and outputs from one forward invocation.
    Expectation: Mega moe forward l1 tensor dump worker.
    """
    shape = common.MoeShape()
    topk_ids, topk_weights, tokens_per_expert = common.make_balanced_route(shape)
    hidden_states, _ = common.make_data(shape)
    layer = _build_layer(shape)
    result_dir = os.getenv("HP_MEGA_MOE_DUMP_RESULT_DIR")
    if not result_dir:
        raise ValueError("HP_MEGA_MOE_DUMP_RESULT_DIR must name the dump output directory")
    config = tensor_dump.TensorDumpConfig(
        result_dir,
        level="L1",
        kernel_names="mega_moe",
        stages="SwiGLU",
        directions="forward",
        ranks=0,
        io=("input", "output"),
        device_memory_budget_mb=32,
    )
    capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=1)
    try:
        with torch.no_grad():
            layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
            torch.npu.synchronize()
            dist.barrier()
            with tensor_dump.tensor_dump(config, schedule=capture_schedule) as dumper:
                layer(hidden_states, topk_ids, topk_weights, tokens_per_expert=tokens_per_expert)
                dumper.step()
        if common.RANK == 0:
            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            invocation = manifest["invocations"][0]
            tasks = invocation["tasks"]
            assert manifest["level"] == "L1", f"unexpected dump level: {manifest['level']}"
            assert manifest["schemaVersion"] == 4, (
                f"unexpected manifest schema: expected=4, got={manifest['schemaVersion']}"
            )
            assert manifest["tensorSerialization"] == "torch.save", (
                f"unexpected Tensor serialization: {manifest['tensorSerialization']}"
            )
            assert manifest["tensorFileLayout"] == "stage_only", (
                f"unexpected Tensor file layout: {manifest['tensorFileLayout']}"
            )
            assert manifest["capturePhaseSemantics"] == "task_boundary", (
                f"unexpected capture semantics: {manifest['capturePhaseSemantics']}"
            )
            assert tasks, "L1 dump did not export any SwiGLU tasks"
            assert len({task["taskId"] for task in tasks}) == len(tasks), (
                f"L1 task IDs are not unique: tasks={tasks}"
            )
            for task in tasks:
                assert task["stage"] == "SwiGLU", f"unexpected L1 stage: task={task}"
                assert task["coreType"] == "AIV", f"unexpected L1 core type: task={task}"
                assert (manifest_path.parent / task["metadataFile"]).is_file(), (
                    f"missing task metadata: task={task}"
                )
                tensors = {(item["io"], item["index"]): item for item in task["tensors"]}
                assert set(tensors) == {("input", 0), ("output", 0)}, (
                    f"unexpected SwiGLU task tensors: tensors={tensors}"
                )
                task_values = {}
                for (io, _), metadata in tensors.items():
                    payload = torch.load(
                        manifest_path.parent / metadata["file"],
                        map_location="cpu",
                        weights_only=True,
                    )
                    task_values[io] = payload
                    assert list(payload.shape) == metadata["storedShape"], (
                        f"stored shape mismatch: metadata={metadata}, shape={payload.shape}"
                    )
                    assert torch.isfinite(payload).all().item(), (
                        f"non-finite L1 payload: metadata={metadata}, payload={payload}"
                    )
                    assert str(payload.dtype).removeprefix("torch.") == metadata["storedDtype"], (
                        f"stored dtype mismatch: metadata={metadata}, dtype={payload.dtype}"
                    )
                    expected_width = shape.intermediate_size * (2 if io == "input" else 1)
                    assert payload.shape[1] == expected_width, (
                        f"unexpected SwiGLU {io} width: expected={expected_width}, got={payload.shape[1]}"
                    )
                gate, up = task_values["input"].float().chunk(2, dim=-1)
                torch.testing.assert_close(task_values["output"].float(), torch.nn.functional.silu(gate) * up,
                                           rtol=2e-2, atol=5e-3,
                                           msg=f"rank={common.RANK}: incorrect L1 SwiGLU task snapshot: {task}")
            print(f"MEGA_KERNEL_L1_TENSOR_DUMP={manifest_path}")
    finally:
        layer.close()
    dist.barrier()
