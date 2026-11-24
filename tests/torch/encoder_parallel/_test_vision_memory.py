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
"""Distributed workers for host-sharded visual input and device memory checks."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

try:
    import torch_npu  # noqa: F401 pylint: disable=unused-import
except ImportError:
    torch_npu = None

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.data.batching.get_batch import OmniParallelBatch
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime
from hyper_parallel.distributed.encoder_parallel.transport import EncoderParallelBatchTransport
from hyper_parallel.models.deepseek_v41.components.vision_transformer import (
    get_deepseek_v41_encoder_parallel_spec,
    replace_vision_attention,
)
from hyper_parallel.models.deepseek_v41.vision import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionAttention,
    DeepseekV41VisionTower,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)


def _init_backend(backend):
    """Initialize a CPU or NPU worker without selecting unrelated devices."""
    if backend == "hccl":
        if torch_npu is None:
            pytest.skip("torch_npu is required for the HCCL worker")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.npu.set_device(local_rank)
        device = torch.device("npu", local_rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend)
    return device


def _make_runtime(device, cp_size, tp_size, patch_size=2):
    """Build native visual modules and the requested encoder/decoder topology."""
    torch.manual_seed(41)
    config = SimpleNamespace(
        v41_vision_hidden_size=16, v41_vision_patch_size=patch_size, v41_vision_num_attention_heads=4,
        v41_vision_rope_theta=10000.0, v41_vision_intermediate_size=32,
        v41_vision_num_hidden_layers=2, v41_vision_downsample_ratio=1, hidden_size=16,
        model_type="deepseek_v41",
    )
    model = nn.Module()
    model.config = config
    model.model = nn.Module()
    model.model.vision = DeepseekV41VisionTower(config)
    model.model.aligner = DeepseekV41VisionAligner(config)
    rule = ModuleReplacementSpec(match=("model.vision.blocks.*.attn",), factory=replace_vision_attention,
                                 module_type=DeepseekV41VisionAttention)
    apply_module_replacements(model, compile_module_replacements(model, (rule,)), weights_mapping=[])
    model.to(device)
    world_size = dist.get_world_size()
    mesh = init_device_mesh(device.type, (world_size // (cp_size * tp_size), cp_size, tp_size),
                            mesh_dim_names=("dp", "cp", "tp"), rank_list=tuple(range(world_size)), init_backend=True)
    parent_rank = dist.get_rank() % (cp_size * tp_size)
    context = SimpleNamespace(
        device_mesh=mesh, cp_size=cp_size, tp_size=tp_size,
        cp_rank=parent_rank // tp_size, tp_rank=parent_rank % tp_size,
        dp_size=world_size // (cp_size * tp_size), dp_rank=dist.get_rank() // (cp_size * tp_size),
        pp_size=1, sequence_parallel=tp_size > 1, loss_parallel=False,
    )
    runtime = EncoderParallelRuntime(
        config=EncoderParallelConfig(enabled=True, encoder_cp_size=2), mesh_context=context,
        model=model, spec=get_deepseek_v41_encoder_parallel_spec(),
        execution_options={"micro_batch_size": 1, "compile_enabled": False, "validate_placement": False},
    )
    return model, runtime


def _image_batch(grids=((3, 5), (2, 4), (1, 3)), starts=(24, 48, 4), patch_size=2, sequence_length=64):
    """Create reproducible CPU image metadata with odd patches and crossing token spans."""
    batch = {
        "input_ids": torch.zeros(1, sequence_length, dtype=torch.long),
        "labels": torch.zeros(1, sequence_length, dtype=torch.long),
        "token_types": torch.zeros(1, sequence_length, dtype=torch.long),
        "cu_seq_lens": torch.tensor([0, sequence_length], dtype=torch.int32),
    }
    if grids:
        offsets = [0]
        for height, width in grids:
            offsets.append(offsets[-1] + height * width)
        generator = torch.Generator().manual_seed(42)
        batch.update(
            pixel_values=torch.randn(offsets[-1], 3, patch_size, patch_size, generator=generator),
            image_patch_offsets=torch.tensor(offsets), image_vit_grid_hw=torch.tensor(grids),
            image_llm_grid_hw=torch.tensor(grids), image_batch_indices=torch.zeros(len(grids), dtype=torch.long),
            image_token_starts=torch.tensor(starts),
        )
    return batch


def _encode_loss(plan, inputs):
    """Return routed records and a loss touching every real and dummy visual branch."""
    records, anchor = plan.encode_and_route(inputs)
    return records, sum((feature.square().sum() for _, feature in records), anchor)


def _compare_transport(model, runtime, source):
    """Compare real Omni transport outputs and gradients against the previous device path."""
    get_batch = OmniParallelBatch(mesh_context=runtime.mesh_context, device=next(model.parameters()).device)
    local = get_batch.cp_sharder.shard(source) if runtime.mesh_context.tp_rank == 0 else None
    original = get_batch.tp_broadcaster.broadcast(local, broadcast_all_fields=True)
    reference_plan = runtime.finalize_plan(*runtime.preflight_batch(original))
    reference_records, reference_loss = _encode_loss(reference_plan, original)
    reference_loss.backward()
    expected = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    get_batch.batch_transport = EncoderParallelBatchTransport(runtime=runtime, chunk_bytes=256)
    prepared, _ = get_batch(None, external_batch=source)
    assert "pixel_values" not in prepared, f"complete image payload reached forward: keys={tuple(prepared)}"
    actual_records, actual_loss = _encode_loss(prepared["encoder_parallel_plan"], prepared["encoder_parallel_inputs"])
    assert [item.item_id for item, _ in actual_records] == [item.item_id for item, _ in reference_records], (
        f"actual_items={[item.item_id for item, _ in actual_records]}, "
        f"expected_items={[item.item_id for item, _ in reference_records]}"
    )
    for (_, actual), (_, reference) in zip(actual_records, reference_records):
        torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)
    actual_loss.backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], atol=2e-4, rtol=2e-4,
                                   msg=lambda message, parameter_name=name: f"{parameter_name}: {message}")
    model.zero_grad(set_to_none=True)


def _run_transport(backend):
    """Exercise TP/CP reshapes, empty buckets, and a text-only DP participant."""
    device = _init_backend(backend)
    try:
        for cp_size, tp_size in ((2, 2), (2, 1), (1, 4)):
            model, runtime = _make_runtime(device, cp_size, tp_size)
            _compare_transport(model, runtime, _image_batch())
            source = _image_batch(grids=(), starts=()) if runtime.mesh_context.dp_rank else _image_batch(
                grids=((3, 5),), starts=(24,),
            )
            _compare_transport(model, runtime, source)
        if dist.get_rank() == 0:
            print(f"{backend}: CPU-sharded vision transport forward/backward passed", flush=True)
    finally:
        dist.destroy_process_group()


def test_cpu_transport():
    """Feature: CPU image sharding across TP and CP.

    Description: Compare new Omni input transport with the previous full-device path.
    Expectation: Routed features and every visual gradient match on four Gloo ranks.
    """
    _run_transport("gloo")


def test_npu_transport():
    """Feature: HCCL image sharding across TP and CP.

    Description: Execute the same transport matrix on four NPU ranks.
    Expectation: Features and gradients match the previous complete-device path.
    """
    _run_transport("hccl")


def test_npu_input_memory():
    """Feature: bounded device input memory with Encoder-CP.

    Description: Measure a 96 MiB CPU image split across two NPU ranks.
    Expectation: Input peak drops by at least one third without a full device image.
    """
    device = _init_backend("hccl")
    try:
        _, runtime = _make_runtime(device, 1, 2, patch_size=32)
        source = _image_batch(grids=((64, 128),), starts=(0,), patch_size=32, sequence_length=16384)
        get_batch = OmniParallelBatch(mesh_context=runtime.mesh_context, device=device)
        local = source if runtime.mesh_context.tp_rank == 0 else None
        torch.npu.synchronize()
        baseline = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        original = get_batch.tp_broadcaster.broadcast(local, broadcast_all_fields=True)
        torch.npu.synchronize()
        original_peak = torch.npu.max_memory_allocated() - baseline
        del original
        torch.npu.synchronize()
        baseline = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        get_batch.batch_transport = EncoderParallelBatchTransport(runtime=runtime)
        prepared, _ = get_batch(None, external_batch=source)
        torch.npu.synchronize()
        local_peak = torch.npu.max_memory_allocated() - baseline
        full_bytes = source["pixel_values"].numel() * source["pixel_values"].element_size()
        assert prepared["encoder_parallel_inputs"].patches.shape[0] == 4096, (
            f"local_shape={prepared['encoder_parallel_inputs'].patches.shape}, expected_patches=4096"
        )
        assert original_peak - local_peak > full_bytes // 3, (
            f"original_peak={original_peak}, local_peak={local_peak}, full_bytes={full_bytes}"
        )
        print(f"rank={dist.get_rank()} input peak bytes: original={original_peak}, local={local_peak}", flush=True)
    finally:
        dist.destroy_process_group()
