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
"""Native HP image-text samples routed across DP with original VLM collation."""

from datetime import timedelta
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hyper_parallel.data.batching import build_dataloader
from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.data.vlm.collator import VLMCollator
from hyper_parallel.data.vlm.get_batch import VLMGetBatch
from hyper_parallel.data.vlm.metadata import vlm_sample_metadata
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.runtime.metrics import mean_global_loss
from tests.common.vlm_fixtures import build_image_corpus, vlm_loader_target


def _build_loader(dataset: object, mesh: object, double_buffer: bool, workers: int) -> object:
    loaders, _ = build_dataloader(
        vlm_loader_target(num_workers=workers, prefetch_factor=2 if workers else None,
                          persistent_workers=bool(workers)),
        datasets=(dataset, None, None), collate_fn=VLMCollator(),
        mesh_context=SimpleNamespace(dp_rank=dist.get_rank(), dp_size=2, device_mesh=mesh),
        training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=8, seed=17),
        data_config={"source_type": "online", "load_balance": "native_batch_sampler",
                     "distributed_dataloader": {"double_buffer": double_buffer}},
        max_seq_len=64, metadata_fn=vlm_sample_metadata,
    )
    return loaders[0]


def _compare_batches(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected), f"Model fields changed: actual={set(actual)}, expected={set(expected)}"
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def _gradient(batch: dict, mesh: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Use HP token normalization with a deterministic image-conditioned toy loss."""
    model_inputs, loss_inputs = VLMGetBatch(mesh_context=SimpleNamespace(), device="cpu")(iter([batch]))
    parameter = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    image_features, offset = [], 0
    for modal_tokens in model_inputs["mm_token_type_ids"].sum(dim=1).tolist():
        patches = int(modal_tokens) * 4
        values = model_inputs["pixel_values"][offset:offset + patches]
        image_features.append(values.double().mean() if patches else parameter.new_zeros(()))
        offset += patches
    prediction = parameter * (model_inputs["input_ids"][:, :-1].double() / 100 +
                              torch.stack(image_features)[:, None])
    targets = model_inputs["labels"][:, 1:]
    valid = targets != -100
    raw_loss = (prediction - targets.double() / 100).square()[valid].mean()
    counts = count_loss_token(loss_inputs)
    context = SimpleNamespace(dp_cp_mesh=mesh["dp"], dp_size=2, cp_size=1, sequence_parallel=False)
    with patch("hyper_parallel.trainer.runtime.distributed.get_device_type", return_value="cpu"):
        loss = mean_global_loss(raw_loss, counts, {key: value * 2 for key, value in counts.items()}, context)
    weighted = loss["foundation_loss"]
    weighted.backward()
    gradient = parameter.grad.clone()
    dist.all_reduce(gradient)
    return weighted.detach(), gradient / 2


def _run_case(dataset: object, mesh: object, *, double_buffer: bool, workers: int) -> None:
    sampler = build_dataset_batch_sampler(
        total_samples=len(dataset), micro_batch_size=2, global_batch_size=8, dp_rank=dist.get_rank(), dp_world_size=2,
    )
    reference = list(sampler)
    loader = _build_loader(dataset, mesh, double_buffer, workers)
    delivered, checkpoint = [], None
    actual_step_gradient = expected_step_gradient = torch.tensor(0.0, dtype=torch.float64)
    for round_idx, indices in enumerate(reference):
        batch = next(loader)
        delivered.append(batch)
        actual_ids = (batch["input_ids"][:, 0] - 100).tolist()
        gathered = [None, None]
        dist.all_gather_object(gathered, actual_ids)
        expected_ids = list(range(round_idx * 4, (round_idx + 1) * 4))
        assert sorted(gathered[0] + gathered[1]) == expected_ids, (
            f"Native round membership changed: actual={gathered}, expected={expected_ids}"
        )
        _compare_batches(batch, VLMCollator()([dataset[index] for index in actual_ids]))
        actual_loss, actual_gradient = _gradient(batch, mesh)
        expected_loss, expected_gradient = _gradient(VLMCollator()([dataset[index] for index in indices]), mesh)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-6)
        actual_step_gradient = actual_step_gradient + actual_gradient
        expected_step_gradient = expected_step_gradient + expected_gradient
        if round_idx % 2 == 1:
            torch.testing.assert_close(actual_step_gradient, expected_step_gradient, rtol=1e-12, atol=1e-12)
            actual_step_gradient = expected_step_gradient = torch.tensor(0.0, dtype=torch.float64)
        if round_idx == 0:
            plan = loader.last_plan
            costs = [constructor.cost.encoder for constructor in plan.constructors]
            assert max(costs) < 144, f"Expected encoder imbalance below native [144, 4], got {costs}"
            checkpoint = loader.state_dict()
            cursor = checkpoint["dataset_reader"]["sampler"]["consumed_samples"]
            assert cursor == 4, f"Prefetch leaked into checkpoint: cursor={cursor}, expected=4"
    assert not list(loader), f"Expected exhaustion after {len(reference)} native rounds"
    resumed = _build_loader(dataset, mesh, double_buffer, workers)
    resumed.load_state_dict(checkpoint)
    remaining = list(resumed)
    assert len(remaining) == len(delivered) - 1, f"Replay count mismatch: got={len(remaining)}, expected=3"
    for actual, expected in zip(remaining, delivered[1:]):
        _compare_batches(actual, expected)


def test_native_vlm_dp2_gloo() -> None:
    """Verify actual HP Dataset/transform/collator with multi-image payload A2A."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp",))
        with TemporaryDirectory() as directory:
            dataset = build_image_corpus(directory)
            for double_buffer, workers in ((False, 0), (True, 0), (True, 2)):
                _run_case(dataset, mesh, double_buffer=double_buffer, workers=workers)
        dist.barrier()
    finally:
        dist.destroy_process_group()
