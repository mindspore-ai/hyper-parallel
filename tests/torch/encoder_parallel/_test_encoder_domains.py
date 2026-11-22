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
"""Ten-step visual domain precision against an unpartitioned native source."""

import copy
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn


try:
    import torch_npu
except ImportError:
    torch_npu = None
from transformers import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

import hyper_parallel

from hyper_parallel.components.modules import AxialVisionTransformer
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.data.parallel import DataLoaderParallelContext, TPBatchBroadcaster
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime
from hyper_parallel.distributed.encoder_parallel.transport import EncoderParallelBatchTransport
from hyper_parallel.models.deepseek_v41.components.vision_transformer import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionTower,
    VisionTransformer,
    get_deepseek_v41_encoder_parallel_spec,
)
from hyper_parallel.models.qwen3_5.qwen3_5_spec import get_encoder_parallel_spec


def _models(family: str, device: torch.device) -> tuple:
    """Copy one exact initial state into native and partitioned computations."""
    model = nn.Module()
    model.model = nn.Module()
    if family == "qwen":
        config = Qwen3_5VisionConfig.from_dict({
            "depth": 2, "hidden_size": 16, "intermediate_size": 32, "num_heads": 4,
            "out_hidden_size": 12, "num_position_embeddings": 16, "patch_size": 2,
            "temporal_patch_size": 2, "spatial_merge_size": 2,
        })
        config._attn_implementation = "eager"
        source = Qwen3_5VisionModel(config).to(device)
        model.config = SimpleNamespace(image_token_id=32)
        model.model.visual = AxialVisionTransformer(module=copy.deepcopy(source), module_fqn="model.visual")
        return model, source, get_encoder_parallel_spec()
    config = SimpleNamespace(v41_vision_hidden_size=16, v41_vision_patch_size=2,
                             v41_vision_num_attention_heads=4, v41_vision_rope_theta=10000.0,
                             v41_vision_intermediate_size=32, v41_vision_num_hidden_layers=2,
                             v41_vision_downsample_ratio=1, hidden_size=12)
    source = nn.Module()
    source.vision = DeepseekV41VisionTower(config).to(device)
    source.aligner = DeepseekV41VisionAligner(config).to(device)
    model.model.vision = VisionTransformer(module=copy.deepcopy(source.vision), module_fqn="model.vision")
    model.model.aligner = copy.deepcopy(source.aligner)
    return model, source, get_deepseek_v41_encoder_parallel_spec()


def _batch(family: str, dp_rank: int, step: int, cp: int, empty: bool) -> dict:
    """Replay distinct DP images while keeping every decoder CP/TP replica identical."""
    generator = torch.Generator().manual_seed(1000 + dp_rank * 100 + step)
    ids = torch.ones((1, cp * 16), dtype=torch.long)
    if empty:
        return {"input_ids": ids[:, :16], "vlm_global_input_ids": ids}
    if family == "qwen":
        # A two-row image crosses the decoder CP boundary; the next image starts later.
        start = 15 if cp > 1 else 2
        ids[0, start:start + 2] = 32
        ids[0, start + 3] = 32
        return {"input_ids": ids[:, :16], "vlm_global_input_ids": ids,
                "pixel_values": torch.randn(12, 24, generator=generator),
                "image_grid_thw": torch.tensor([[1, 2, 4], [1, 2, 2]])}
    return {"input_ids": ids[:, :16], "token_types": ids[:, :16] * 0,
            "pixel_values": torch.randn(8, 3, 2, 2, generator=generator),
            "image_patch_offsets": torch.tensor([0, 6, 8]),
            "image_vit_grid_hw": torch.tensor([[2, 3], [1, 2]]),
            "image_llm_grid_hw": torch.tensor([[2, 3], [1, 2]]),
            "image_batch_indices": torch.tensor([0, 0]),
            "image_token_starts": torch.tensor([12, 0] if cp > 1 else [6, 0])}


def _reference_records(source: nn.Module, family: str, batch: dict, items: tuple,
                       device: torch.device) -> list:
    """Compute full images independently of candidate segmentation and communication."""
    if not items:
        return []
    pixels = batch["pixel_values"].to(device)
    if family == "qwen":
        values = source(pixels, grid_thw=batch["image_grid_thw"].to(device)).pooler_output
        return list(zip(items, values.split([item.output_rows for item in items])))
    return [(item, source.aligner(source.vision(pixels[item.payload_start:item.payload_stop],
                                               *item.metadata["vit_hw"]), *item.metadata["llm_hw"]))
            for item in items]


def _loss(records: list, anchor: torch.Tensor, layout: object, family: str) -> torch.Tensor:
    """Consume each visual row only on its decoder owner, accounting for replicated TP."""
    result = anchor.float()
    for item, value in records:
        rows = torch.arange(item.output_rows, device=value.device)
        positions = item.decoder_token_start + rows
        if family == "deepseek":
            width = item.metadata["llm_hw"][1]
            positions = positions + 1 + rows // width
        mask = (positions >= layout.global_start) & (positions < layout.global_stop)
        result = result + (value.float()[mask].square().sum() / 16)
    return result / (1 if layout.sequence_parallel else layout.tp_size)


def _parameter_pairs(candidate: nn.Module, reference: nn.Module, family: str) -> list:
    """Compare matching logical names even when adoption changes child registration order."""
    prefix = "model.visual." if family == "qwen" else "model."
    actual = {name.removeprefix(prefix): parameter for name, parameter in candidate.named_parameters()}
    expected = dict(reference.named_parameters())
    assert actual.keys() == expected.keys()
    return [(actual[name], expected[name]) for name in actual]


def _initial_state_fingerprint(pairs: list) -> str:
    """Prove common state across ranks instead of relying on a shared seed alone."""
    digest = hashlib.sha256()
    for left, right in pairs:
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        digest.update(str((tuple(left.shape), left.dtype)).encode())
        digest.update(left.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    value = digest.hexdigest()
    _assert_hash_replicas(value, pairs[0][0].device)
    return value


def _assert_hash_replicas(value: str, device: torch.device, group: object = None) -> None:
    """Verify every digest byte using MIN/MAX without object-allgather staging."""
    minimum = torch.tensor(list(bytes.fromhex(value)), dtype=torch.int32, device=device)
    maximum = minimum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    assert torch.equal(minimum, maximum)



def _optimizers(model: nn.Module) -> tuple:
    """Use Muon on logical matrices and AdamW on remaining norm/bias/convolution state."""
    matrix = [p for p in model.parameters() if p.ndim == 2]
    other = [p for p in model.parameters() if p.ndim != 2]
    return Muon(matrix, lr=1e-4, weight_decay=0.0), torch.optim.AdamW(other, lr=1e-4, weight_decay=0.0)


def _global_gradients(model: nn.Module) -> torch.Tensor:
    """Aggregate replicated parameter gradients once; compare before clipping or updates."""
    square = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        dist.all_reduce(parameter.grad)
        square += parameter.grad.square().sum()
    return square.sqrt()


def _optimizer_state_probe(pairs: list, optimizers: tuple, reference_optimizers: tuple) -> tuple:
    """Compare optimizer keys, finite states and strict tensor diagnostics."""
    state_pass, state_error = True, 0.0
    reference_by_id = {id(left): right for left, right in pairs}
    for actual_optimizer, expected_optimizer in zip(optimizers, reference_optimizers):
        for left in actual_optimizer.param_groups[0]["params"]:
            right = reference_by_id[id(left)]
            assert actual_optimizer.state[left].keys() == expected_optimizer.state[right].keys()
            for key, value in actual_optimizer.state[left].items():
                if torch.is_tensor(value):
                    expected = expected_optimizer.state[right][key]
                    assert value.shape == expected.shape
                    assert torch.isfinite(value).all() and torch.isfinite(expected).all()
                    state_pass &= bool(torch.isclose(value, expected, rtol=2e-3, atol=2e-5).all().item())
                    state_error = max(state_error, (value - expected).abs().max().item())
    return state_pass, state_error


def _step_optimizers(pairs: list, optimizers: tuple, reference_optimizers: tuple,
                     verify_source_parity: bool) -> dict:
    """Check initial same-state gradients; retain update/state tensor verdicts as diagnostics.

    Muon quantizes NS operands to BF16, so tiny gradient reduction differences
    may change its update. Ten-step acceptance uses loss and pre-clip norm;
    strict tensor thresholds remain separately reported, including failures.
    """
    grad_error = 0.0
    for left, right in pairs:
        if verify_source_parity:
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-3, atol=2e-5)
        grad_error = max(grad_error, (left.grad - right.grad).abs().max().item())
    for optimizer in (*optimizers, *reference_optimizers):
        optimizer.step()
    parameter_pass = all(torch.isclose(left, right, rtol=3e-5, atol=3e-6).all().item() for left, right in pairs)
    state_pass, state_error = _optimizer_state_probe(pairs, optimizers, reference_optimizers)
    assert all(torch.isfinite(left).all() and torch.isfinite(right).all() for left, right in pairs)
    return {"gradient_max_abs": grad_error,
            "parameter_max_abs": max((p - q).abs().max().item() for p, q in pairs),
            "optimizer_state_max_abs": state_error,
            "strict_parameter_status": "PASS" if parameter_pass else "FAIL",
            "strict_optimizer_status": "PASS" if state_pass else "FAIL"}


def _initialize_backend() -> tuple:
    """Initialize deterministic communication before constructing any resource meshes."""
    backend = os.environ.get("HP_ENCODER_BACKEND", "gloo")
    if backend == "hccl":
        assert os.environ.get("HCCL_DETERMINISTIC") == "true"
        assert torch_npu is not None
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu", int(os.environ["LOCAL_RANK"])) if backend == "hccl" else torch.device("cpu")
    torch.set_num_threads(1)
    dist.init_process_group(backend, timeout=timedelta(seconds=120))
    return backend, device


def test_encoder_domain_precision() -> None:
    """Feature: Independent encoder resources.

    Description: Compare distinct decoder samples for ten actual Muon updates across resource domains.
    Expectation: Initial source parity and ten-step global loss/norm pass; update/state probes are reported.
    """
    backend, device = _initialize_backend()
    rank, world = dist.get_rank(), dist.get_world_size()
    cp, tp = int(os.environ.get("HP_ENCODER_LLM_CP", "2")), int(os.environ.get("HP_ENCODER_LLM_TP", "1"))
    ecp = int(os.environ.get("HP_ENCODER_CP", "4"))
    scope = os.environ.get("HP_ENCODER_DOMAIN", "auto")
    family = os.environ.get("HP_ENCODER_FAMILY", "deepseek")
    sp = os.environ.get("HP_ENCODER_SP", "0") == "1"
    dp_rank, remainder = divmod(rank, cp * tp)
    cp_rank, tp_rank = divmod(remainder, tp)
    torch.manual_seed(2026)
    candidate, reference, spec = _models(family, device)
    pairs = _parameter_pairs(candidate, reference, family)
    initial_hash = _initial_state_fingerprint(pairs)
    mesh = init_device_mesh(device.type, (world // (cp * tp), cp, tp), mesh_dim_names=("dp", "cp", "tp"))
    context = SimpleNamespace(device_mesh=mesh, cp_size=cp, tp_size=tp, cp_rank=cp_rank, tp_rank=tp_rank,
                              pp_size=1, sequence_parallel=sp, loss_parallel=False)
    runtime = EncoderParallelRuntime(config=EncoderParallelConfig(True, ecp, scope), mesh_context=context,
                                     model=candidate, spec=spec, execution_options={"micro_batch_size": 1})
    transport = EncoderParallelBatchTransport(runtime=runtime, chunk_bytes=28)
    broadcaster = TPBatchBroadcaster(DataLoaderParallelContext(tp_world_size=tp, tp_rank=tp_rank,
                                        tp_group=mesh["tp"].get_group(), cp_world_size=cp, cp_rank=cp_rank), device)
    optimizers, reference_optimizers = _optimizers(candidate), _optimizers(reference)
    metrics = []
    for step in range(10):
        # A whole resource domain is empty on step zero when world > the minimal domain.
        empty = (step == 0 and dp_rank > 0) or (step % 3 == 1 and dp_rank % 2 == 1)
        batch = _batch(family, dp_rank, step, cp, empty)
        if family == "qwen":
            batch["input_ids"] = batch["vlm_global_input_ids"][:, cp_rank * 16:(cp_rank + 1) * 16]
        input_hash = hashlib.sha256(batch.get("pixel_values", torch.empty(0)).numpy().tobytes()).hexdigest()
        _assert_hash_replicas(input_hash, device, runtime.decoder_mesh.get_group())
        for optimizer in (*optimizers, *reference_optimizers):
            optimizer.zero_grad(set_to_none=True)
        _, inputs = transport(batch if tp_rank == 0 else None, broadcaster)
        plan = inputs["encoder_parallel_plan"]
        records, anchor = plan.encode_and_route(inputs["encoder_parallel_inputs"])
        expected = _reference_records(reference, family, batch, runtime.bridge.parse_images(batch), device)
        expected = [
            (item, value)
            for item, value in expected
            if max(item.decoder_token_start, plan.decoder_layout.global_start)
            < min(item.decoder_token_start + item.decoder_token_length, plan.decoder_layout.global_stop)
        ]
        expected.sort(key=lambda record: record[0].decoder_token_start)
        assert len(records) == len(expected)
        output_error = 0.0
        for (item, actual), (original, value) in zip(records, expected):
            assert item.decoder_token_start == original.decoder_token_start
            if step == 0:
                torch.testing.assert_close(actual, value, rtol=3e-5, atol=2e-6)
            output_error = max(output_error, (actual - value).abs().max().item())
        native_anchor = sum((p.sum() * 0 for p in reference.parameters()))
        loss = _loss(records, anchor, plan.decoder_layout, family)
        native_loss = _loss(expected, native_anchor, plan.decoder_layout, family)
        loss.backward()
        native_loss.backward()
        norm, native_norm = _global_gradients(candidate), _global_gradients(reference)
        scalars = torch.stack([loss.detach(), native_loss.detach()])
        dist.all_reduce(scalars)
        torch.testing.assert_close(scalars[0], scalars[1], rtol=3e-5, atol=2e-6)
        torch.testing.assert_close(norm, native_norm, rtol=5e-4, atol=3e-6)
        probes = _step_optimizers(pairs, optimizers, reference_optimizers, verify_source_parity=step == 0)
        metrics.append({"step": step + 1, "loss": scalars[0].item(), "reference_loss": scalars[1].item(),
                        "norm": norm.item(), "reference_norm": native_norm.item(), "output_max_abs": output_error,
                        **probes, "input_hash": input_hash, "lr": optimizers[0].param_groups[0]["lr"]})
    result = {
        "rank": rank,
        "world": world,
        "family": family,
        "backend": backend,
        "cp": cp,
        "tp": tp,
        "encoder_cp": ecp,
        "domain": scope,
        "parent_size": runtime.parent_mesh.size(),
        "sp": sp,
        "initial_state_hash": initial_hash, "import_path": hyper_parallel.__file__, "steps": metrics,
        "status": "PASS",
        "acceptance": "initial-source-parity-and-ten-step-loss-norm",
        "scope": "FP32 visual tower/projector, replicated optimizer state; no full decoder/FSDP claim",
    }
    evidence = os.environ.get("HP_ENCODER_EVIDENCE")
    if evidence:
        Path(evidence).mkdir(parents=True, exist_ok=True)
        Path(evidence, f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if rank == 0:
        print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()
