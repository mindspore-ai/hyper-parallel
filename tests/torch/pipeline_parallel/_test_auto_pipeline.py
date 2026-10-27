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
"""Real Qwen3 PP/VPP communication and optimizer parity against a full model."""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

from hyper_parallel import DTensor, init_device_mesh
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.models._transformers.model_builder import apply_model_infrastructure
from hyper_parallel.models._transformers.pipeline import AutoPipeline, PipelineConfig
from hyper_parallel.models.build_options import FSDP2Config
from tests.torch.utils import get_device_type, init_backend


def _sum_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Use an additive loss to compare every micro-batch with a full batch."""
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")


def _run_case(mesh, device: torch.device, schedule: str, split: list,
              overlap: bool, multi_stream: bool, *,
              mesh_context: MeshContext | None = None, comm_fusion: bool = False) -> None:
    """Compare two complete optimizer steps, including every local gradient."""
    torch.manual_seed(42)
    reference = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=10,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=16, use_cache=False, _attn_implementation="eager",
    )).to(device).eval()
    config = PipelineConfig(
        enabled=True, schedule=schedule, split=split,
        pp_microbatch_size=1, p2p_overlap=overlap, p2p_multi_stream=multi_stream,
        loss_fn=_sum_cross_entropy,
    )
    if mesh_context is None:
        pipeline = AutoPipeline(mesh, config).build(reference, batch_size=4)
    else:
        with torch.device("meta"):
            model = Qwen3ForCausalLM(reference.config)
        pipeline = apply_model_infrastructure(
            model, mesh=mesh_context, is_meta_device=True, is_hf_model=True, device=device,
            fsdp2_manager=FSDP2Manager(FSDP2Config(comm_fusion=comm_fusion), mesh_context),
            distributed_setup=SimpleNamespace(pipeline_config=config, pipeline_batch_size=4),
        )
        assert all(not parameter.is_meta for parameter in pipeline.parameters())
        assert all(stage.device.type == device.type for stage in pipeline.info.stages)
        for part in pipeline.parts:
            keys = set(part.state_dict())
            part.load_state_dict({name: value for name, value in reference.state_dict().items() if name in keys})
            torch.testing.assert_close(part.model.rotary_emb.inv_freq, reference.model.rotary_emb.inv_freq)

    def local_reference(tensor):
        if mesh_context is None:
            return tensor
        return tensor.chunk(mesh_context.dp_shard_size, dim=0)[mesh_context.dp_rank]

    optimizer = torch.optim.SGD(pipeline.parameters(), lr=0.001)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(123)
    for _ in range(2):
        input_ids = torch.randint(0, 32, (4, 6), generator=generator).to(device)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[0, -1] = -100
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = _sum_cross_entropy(reference(input_ids, attention_mask=attention_mask).logits, labels)
        reference_loss.backward()

        losses = pipeline.step(input_ids, target=labels, attention_mask=attention_mask)
        actual_loss = sum((loss.detach() for loss in losses), torch.zeros((), device=device))
        dist.all_reduce(actual_loss, group=pipeline.pp_mesh.get_group())
        torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-4, atol=1e-4)
        reference_parameters = dict(reference.named_parameters())
        for part in pipeline.parts:
            for name, param in part.named_parameters():
                gradient = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
                torch.testing.assert_close(gradient, local_reference(reference_parameters[name].grad),
                                           rtol=2e-3, atol=2e-4)

        optimizer.step()
        reference_optimizer.step()
        for part in pipeline.parts:
            for name, param in part.named_parameters():
                parameter = param.to_local() if isinstance(param, DTensor) else param
                torch.testing.assert_close(parameter, local_reference(reference_parameters[name]),
                                           rtol=1e-4, atol=2e-5)
    dist.barrier()


def test_qwen3_pipeline_splits_match_reference() -> None:
    """Exercise ten-layer ordinary PP and VPP on four ranks in all P2P modes."""
    device_type = get_device_type()
    init_backend(device_type)
    device = torch.device(device_type, dist.get_rank()) if device_type != "cpu" else torch.device("cpu")
    if dist.get_world_size() != 4:
        raise ValueError("the ten-layer AutoPipeline parity test requires four ranks")
    mesh = init_device_mesh(device_type, (4,), mesh_dim_names=("pp",))
    try:
        for schedule, split in [
            ("gpipe", [2, 3, 3, 2]),
            ("1f1b", [2, 3, 3, 2]),
            ("interleaved1f1b", [[1, 1, 1, 1], [1, 2, 2, 1]]),
        ]:
            for overlap, multi_stream in [(False, False), (True, False), (True, True)]:
                _run_case(mesh, device, schedule, split, overlap, multi_stream)
    finally:
        dist.destroy_process_group()


def test_qwen3_meta_pipeline_fsdp_matches_reference() -> None:
    """Materialize after PP/FSDP wrapping and compare PP/VPP with a full model."""
    device_type = get_device_type()
    init_backend(device_type)
    device = torch.device(device_type, dist.get_rank()) if device_type != "cpu" else torch.device("cpu")
    if dist.get_world_size() != 4:
        raise ValueError("the meta PP/FSDP parity test requires four ranks")
    context = MeshContext(pp_size=2, dp_size=2, dp_shard_size=2, dp_rank=dist.get_rank() % 2)
    context.build_meshs(device_type, dist.get_world_size())
    try:
        for schedule, split in [("1f1b", [5, 5]),
                                ("interleaved1f1b", [[2, 3], [3, 2]])]:
            for comm_fusion in (False, True):
                _run_case(context.pp_mesh, device, schedule, split, True, False,
                          mesh_context=context, comm_fusion=comm_fusion)
    finally:
        dist.destroy_process_group()
