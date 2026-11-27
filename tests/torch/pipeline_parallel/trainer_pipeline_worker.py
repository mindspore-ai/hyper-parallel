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
"""Exercise the Trainer loss path and both schedules against one unsplit DS4.1 model."""

import copy
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import distributed as dist

from hyper_parallel.components.checkpoint.load_groups import LoadReport
from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context
from hyper_parallel.components.losses.model_output import ModelOutputLoss
from hyper_parallel.trainer.callbacks.moe_monitor_callback import MoEMonitorCallback
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.data.batching.get_batch import OmniParallelBatch
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.utils.clip_grad import clip_grad_norm_
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41ForCausalLM
from hyper_parallel.models.deepseek_v41.adapter.data.image_processor import (
    IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START,
)
from hyper_parallel.models.pipeline import PipelineConfig, partition_model
from hyper_parallel.models.deepseek_v41.adapter.pipeline import get_pipeline_spec
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.runtime.distributed import destroy_process_group
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.model_integration import DisabledModelIntegrationSession
from tests.common.deepseek_v41_pipeline import make_config


@pytest.fixture(scope="module", autouse=True)
def pipeline_group() -> Iterator[None]:
    """Keep the worker process group alive across all parametrized cases."""
    dist.init_process_group("gloo")
    try:
        yield
    finally:
        destroy_process_group()


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("indexer_loss", [0.0, 0.001])
def test_pipeline_gradients(tmp_path: Path, indexer_loss: float, multimodal: bool) -> None:
    """Use unequal valid-token counts and a cut downstream of the candidate publisher.

    Args:
        tmp_path: Per-test directory for model assets.
        indexer_loss: Coefficient controlling gradients through the shared Indexer key.
        multimodal: Include the vision tower and DSpark on their owning endpoints.
    """
    torch.manual_seed(81)
    model_config = make_config(str(tmp_path))
    model_config.v41_indexer_loss_coeff = indexer_loss
    if multimodal:
        for name, value in {
            "v41_vision_enabled": True, "v41_vision_hidden_size": 16,
            "v41_vision_intermediate_size": 24, "v41_vision_patch_size": 2,
            "v41_vision_num_attention_heads": 2, "v41_vision_rope_theta": 10000.0,
            "v41_vision_num_hidden_layers": 2, "v41_vision_downsample_ratio": 1,
            "v41_dspark_depth": 1, "v41_dspark_target_layer_ids": [37, 38, 39],
            "v41_dspark_block_size": 2, "v41_dspark_window": 4,
            "v41_dspark_markov_rank": 8, "v41_dspark_noise_token_id": 63,
            "v41_dspark_n_routed_experts": 8, "v41_dspark_top_k": 3,
            "v41_dspark_confidence_coeff": 0.1,
        }.items():
            setattr(model_config, name, value)
    reference = DeepseekV41ForCausalLM(model_config)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.mesh = MeshContext(pp_size=2, pp_rank=dist.get_rank())
    trainer.mesh.pp_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("pp",), rank_list=(0, 1))
    trainer.device = torch.device("cpu")
    trainer.loss_fn = ModelOutputLoss()
    trainer.model_fwd_context = trainer.model_bwd_context = nullcontext()
    trainer.model_integration = DisabledModelIntegrationSession()
    trainer._callbacks = []
    trainer.model = reference
    trainer.moe_monitor = MoEMonitorCallback(reference)
    batches = []
    expected_loss = 0.0
    expected_router_loss = 0.0
    for ignored in (0, 3, 1):
        ids = torch.randint(0, 64, (1, 8))
        labels = ids.clone()
        labels[:, :ignored] = -100
        inputs = {"input_ids": ids, "labels": labels}
        if multimodal:
            inputs.update({
                "pixel_values": torch.randn(1, 3, 2, 2),
                "token_types": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]]),
                "image_patch_offsets": torch.tensor([0, 1]),
                "image_vit_grid_hw": torch.tensor([[1, 1]]),
                "image_llm_grid_hw": torch.tensor([[1, 1]]),
                "image_batch_indices": torch.tensor([0]),
                "image_token_starts": torch.tensor([0]),
            })
        loss_inputs = {"labels": labels}
        batches.append((inputs, loss_inputs))
        with aux_loss_scale_context():
            outputs = reference(**inputs, use_cache=False)
            loss, metrics = trainer.postforward(outputs, labels, current_token_counts=count_loss_token(loss_inputs))
            expected_router_loss += metrics["router_aux_loss"].item()
            expected_loss += loss.item()
            loss.backward()
    trainer.state = SimpleNamespace()
    trainer.train_dataloader = Mock()
    active_channels = []
    source_names = set(dict(reference.named_parameters())) | {"unexpected.source"}

    @contextmanager
    def _track_channel(_state, inputs):
        active_channels.append(inputs["channel_id"])
        try:
            yield
        finally:
            active_channels.pop()

    def _assert_channel(_model, _args, kwargs):
        assert active_channels and "channel_id" not in kwargs

    trainer.channel_loss_callback = SimpleNamespace(
        micro_step_context=_track_channel, strip_model_inputs=lambda inputs: inputs.pop("channel_id"),
    )
    pipeline_batches = [({**inputs, "channel_id": i}, labels) for i, (inputs, labels) in enumerate(batches)]
    parameters = dict(reference.named_parameters())
    for schedule in ("1f1b", "gpipe", "interleaved_1f1b"):
        interleaved = schedule == "interleaved_1f1b"
        cuts = [10, 22, 30] if interleaved else [22]
        config = PipelineConfig(schedule=schedule, split_points=cuts)
        chunks = tuple(copy.deepcopy(reference) for _ in range(config.local_stage_count))
        for index, chunk in enumerate(chunks):
            partition_model(chunk, trainer.mesh, config, chunk_index=index)
            chunk.register_forward_pre_hook(_assert_channel, with_kwargs=True)
            if interleaved:
                keys = dict(chunk.named_parameters()).keys()
                chunk._hp_checkpoint_load_report = LoadReport(tuple(keys), (), tuple(source_names - keys))
        model = chunks[0]
        if interleaved:
            model = get_pipeline_spec().combine_chunks(chunks, trainer.mesh.pp_mesh.get_group())
            report = model._hp_checkpoint_load_report
            assert set(report.loaded_keys) == dict(model.named_parameters()).keys()
            assert report.unexpected_keys == ("unexpected.source",)
            all_names = [None, None]
            dist.all_gather_object(all_names, list(dict(model.named_parameters())))
            assert not set(all_names[0]).intersection(all_names[1])
            assert set(all_names[0]).union(all_names[1]) == parameters.keys()
            owned_stages = [part._hp_pipeline_partition.stage_index for part in chunks]
            assert owned_stages == [dist.get_rank(), dist.get_rank() + 2]
        trainer.model = model
        trainer.moe_monitor = MoEMonitorCallback(model)
        trainer.config = SimpleNamespace(
            pipeline=config, training=SimpleNamespace(empty_cache_before_backward=True),
        )
        trainer.pipeline_runtime = None
        model.zero_grad(set_to_none=True)
        trainer.begin_loss_step()
        trainer.train_dataloader.reset_mock()
        with patch("hyper_parallel.trainer.runtime.pipeline.empty_cache") as release_cache:
            actual_loss, actual_metrics = trainer.pipeline_forward_backward(
                iter(pipeline_batches), len(pipeline_batches),
            )
        assert release_cache.call_count == len(batches) * len(chunks)
        assert trainer.train_dataloader.prefetch_plan.call_count == len(batches)
        assert trainer.train_dataloader.prefetch.call_count == len(batches)
        assert trainer.step_token_counts["foundation_tokens"] == sum(
            count_loss_token(labels)["foundation_tokens"] for _, labels in batches
        )
        assert not active_channels
        torch.testing.assert_close(actual_loss, expected_loss)
        torch.testing.assert_close(actual_metrics["router_aux_loss"], expected_router_loss)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.grad, parameters[name].grad, atol=2e-5, rtol=1e-5, msg=name)
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    actual_norm = clip_grad_norm_(model, 1.0, pp_group=trainer.mesh.pp_mesh.get_group())
    torch.testing.assert_close(actual_norm, expected_norm)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.grad, parameters[name].grad, atol=2e-5, rtol=1e-5, msg=name)


def test_pipeline_hsdp_gradients(tmp_path: Path) -> None:
    """Run 1F1B with fully_shard-wrapped stage roots and compare against the unsplit model."""
    torch.manual_seed(81)
    model_config = make_config(str(tmp_path))
    reference = DeepseekV41ForCausalLM(model_config)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.mesh = MeshContext(pp_size=2, pp_rank=dist.get_rank())
    trainer.mesh.pp_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("pp",), rank_list=(0, 1))
    trainer.device = torch.device("cpu")
    trainer.loss_fn = ModelOutputLoss()
    trainer.model_fwd_context = trainer.model_bwd_context = nullcontext()
    trainer.model_integration = DisabledModelIntegrationSession()
    trainer._callbacks = []
    trainer.model = reference
    trainer.moe_monitor = MoEMonitorCallback(reference)
    batches = []
    expected_loss = 0.0
    expected_router_loss = 0.0
    for ignored in (0, 3, 1):
        ids = torch.randint(0, 64, (1, 8))
        labels = ids.clone()
        labels[:, :ignored] = -100
        inputs = {"input_ids": ids, "labels": labels}
        batches.append((inputs, {"labels": labels}))
        with aux_loss_scale_context():
            outputs = reference(**inputs, use_cache=False)
            loss, metrics = trainer.postforward(outputs, labels, current_token_counts=count_loss_token({"labels": labels}))
            expected_router_loss += metrics["router_aux_loss"].item()
            expected_loss += loss.item()
            loss.backward()
    trainer.state = SimpleNamespace()
    trainer.train_dataloader = Mock()
    trainer.channel_loss_callback = None
    parameters = dict(reference.named_parameters())
    config = PipelineConfig(schedule="1f1b", split_points=[22])
    model = copy.deepcopy(reference)
    partition_model(model, trainer.mesh, config)
    dp_mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",), rank_list=(dist.get_rank(),))
    with patch("hyper_parallel.core.fully_shard.api._get_device_from_mesh", return_value=torch.device("cpu")):
        fully_shard(model, mesh=dp_mesh, reshard_after_forward=True)
    model.set_reduce_op_type("sum")
    trainer.model = model
    trainer.moe_monitor = MoEMonitorCallback(model)
    trainer.config = SimpleNamespace(pipeline=config, training=SimpleNamespace(empty_cache_before_backward=False))
    trainer.pipeline_runtime = None
    model.zero_grad(set_to_none=True)
    trainer.begin_loss_step()
    actual_loss, actual_metrics = trainer.pipeline_forward_backward(iter(batches), len(batches))
    torch.testing.assert_close(actual_loss, expected_loss)
    torch.testing.assert_close(actual_metrics["router_aux_loss"], expected_router_loss)
    for name, param in model.named_parameters():
        grad = param.grad
        grad = grad.to_local() if hasattr(grad, "to_local") else grad
        torch.testing.assert_close(grad, parameters[name].grad, atol=2e-5, rtol=1e-5, msg=name)


def _vision_microbatch(width: int) -> dict:
    """Build one packed image span or a text-only example when width is zero."""
    ids = torch.randint(1, 60, (1, 12))
    labels = ids.clone()
    token_types = torch.full_like(ids, -1)
    raw = {"input_ids": ids, "labels": labels, "token_types": token_types}
    if width:
        token_types[0, 1:4 + width] = torch.tensor([IMAGE_START, *([IMAGE] * width), IMAGE_NEW_LINE, IMAGE_END])
        labels[0, 1:4 + width] = -100
        raw.update(pixel_values=torch.randn(4 * width, 3, 2, 2), image_patch_offsets=torch.tensor([0, 4 * width]),
                   image_vit_grid_hw=torch.tensor([[2, width * 2]]), image_llm_grid_hw=torch.tensor([[1, width]]),
                   image_batch_indices=torch.tensor([0]), image_token_starts=torch.tensor([1]))
    return raw


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("chunks", [2, 3])
def test_mpipe_vision_gradients(tmp_path: Path, frozen: bool, chunks: int) -> None:
    """Compare visual injection, aligned decoder losses and gradients with one uncut VLM.

    Args:
        tmp_path: Directory for the small reference model assets.
        frozen: Whether to freeze both the ViT and its aligner.
        chunks: Decoder chunks per physical rank.
    """
    torch.manual_seed(81)
    cfg = make_config(str(tmp_path))
    cfg.v41_vision_enabled = True
    cfg.v41_vision_hidden_size = 16
    cfg.v41_vision_intermediate_size = 32
    cfg.v41_vision_patch_size = 2
    cfg.v41_vision_num_attention_heads = 2
    cfg.v41_vision_rope_theta = 10000.0
    cfg.v41_vision_num_hidden_layers = 2
    cfg.v41_vision_downsample_ratio = 2
    reference = DeepseekV41ForCausalLM(cfg)
    if frozen:
        reference.model.vision.requires_grad_(False)
        reference.model.aligner.requires_grad_(False)
    rank = dist.get_rank()
    mesh = MeshContext(pp_size=2, pp_rank=rank)
    mesh.pp_mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("pp",), rank_list=(0, 1))
    config = PipelineConfig(schedule="mpipe", virtual_stages_per_rank=chunks,
                            split_points=[10, 22, 30] if chunks == 2 else [])
    parts = tuple(copy.deepcopy(reference) for _ in range(chunks))
    for index, part in enumerate(parts):
        partition_model(part, mesh, config, chunk_index=index)
    model = get_pipeline_spec().combine_chunks(parts, mesh.pp_mesh.get_group())
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.mesh, trainer.device = mesh, torch.device("cpu")
    trainer.model_fwd_context = trainer.model_bwd_context = nullcontext()
    trainer.model_integration = DisabledModelIntegrationSession()
    trainer.loss_fn = ModelOutputLoss()
    trainer.config = SimpleNamespace(pipeline=config, training=SimpleNamespace(empty_cache_before_backward=False))
    trainer.model = reference
    trainer.moe_monitor = MoEMonitorCallback(reference)
    trainer._callbacks = []
    trainer.state = SimpleNamespace()
    trainer.train_dataloader = Mock()
    transport = OmniParallelBatch(mesh_context=mesh, device=trainer.device, mpipe=True)
    batches = []
    expected = 0.0
    expected_hidden, actual_hidden = [], []
    reference.model.layers[0].register_forward_pre_hook(
        lambda _m, args: expected_hidden.append(args[0].detach().clone()))
    if rank == 0:
        parts[0].model.layers["0"].register_forward_pre_hook(
            lambda _m, args: actual_hidden.append(args[0].detach().clone()))
    for index, width in enumerate((1, 2, 0)):
        raw = _vision_microbatch(width)
        labels = raw["labels"]
        with aux_loss_scale_context():
            outputs = reference(**raw, use_cache=False)
            loss, _ = trainer.postforward(outputs, labels, current_token_counts=count_loss_token({"labels": labels}))
            expected += loss.item()
            loss.backward()
        # A different sample on the non-owner must never contribute pixels, labels or metadata.
        local = raw if index % 2 == rank else {
            "input_ids": torch.zeros_like(raw["input_ids"]), "labels": torch.zeros_like(labels),
        }
        batches.append(transport(iter([local]), mpipe_owner=index % 2))
    trainer.model, trainer.moe_monitor = model, MoEMonitorCallback(model)
    trainer.begin_loss_step()
    result, _ = trainer.pipeline_forward_backward(iter(batches), len(batches))
    torch.testing.assert_close(result, expected)
    if rank == 0:
        torch.testing.assert_close(actual_hidden, expected_hidden)
    params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.grad, params[name].grad, atol=2e-5, rtol=1e-5, msg=name)
    if not frozen and rank == 0:
        for submodule in (model.model.vision, model.model.aligner):
            assert any(p.grad is not None and p.grad.abs().max() > 0 for p in submodule.parameters())
