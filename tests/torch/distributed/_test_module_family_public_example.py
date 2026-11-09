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
"""Numerical acceptance of the actual documented model and public builder."""

import os
from unittest.mock import patch

import torch
import torch.distributed as dist

from examples.torch.module_family import DecoderConfig, FamilyDecoder
from examples.torch.module_family.train import build_training_setup
from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.distributed.mesh import MeshContext
from tests.common.training_state import capture_full_training_state


def _check_training(cp_size: int, ep_size: int, *, direct: bool) -> None:
    """Compare direct and facade parallel model updates with a dense AdamW reference."""
    setup = build_training_setup(cp_size, ep_size)
    config = DecoderConfig()
    if direct:
        with torch.device("npu"):
            model = FamilyDecoder(config, mesh=setup.mesh_context)
        model = FSDP2Manager(setup.strategy_config, setup.mesh_context).parallelize(model)
    else:
        model = HyperAutoModelForCausalLM.from_config(
            config, model_factory=FamilyDecoder, distributed_setup=setup, torch_dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    initial = capture_full_training_state(model, optimizer, global_step=0, data_progress={})
    reference = FamilyDecoder(config).to("npu")
    reference.load_state_dict(initial["model"])
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.003)
    expert_weight = initial["model"]["experts.gate_up"]
    if torch.equal(expert_weight[:2], expert_weight[2:]):
        raise AssertionError("public distributed initialization repeated expert shards")
    rank = dist.get_rank()
    dp_size = dist.get_world_size() // cp_size
    all_inputs = torch.arange(dp_size * cp_size * 2 * 8, device="npu", dtype=torch.float32)
    all_inputs = (all_inputs.reshape(dp_size, cp_size * 2, 8) / 100).sin()
    inputs = all_inputs[rank // cp_size:rank // cp_size + 1,
                        (rank % cp_size) * 2:(rank % cp_size + 1) * 2].contiguous()
    for step in range(2):
        actual = model(inputs)
        expected = reference(all_inputs)
        torch.testing.assert_close(actual.cpu(), expected[rank // cp_size:rank // cp_size + 1,
                                   (rank % cp_size) * 2:(rank % cp_size + 1) * 2].cpu(),
                                   rtol=3e-4, atol=3e-5)
        actual.square().mean().backward()
        expected.square().mean().backward()
        with SkipDTensorDispatch(no_skip={torch.zeros_like}):
            optimizer.step()
        reference_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        state = capture_full_training_state(model, optimizer, global_step=step + 1, data_progress={})
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(state["model"][name].cpu(), parameter.cpu(), rtol=5e-4, atol=4e-5)
            for key, value in reference_optimizer.state[parameter].items():
                torch.testing.assert_close(state["optimizer"]["state"][name][key].cpu(), value.cpu(),
                                           rtol=5e-4, atol=4e-5)


def test_public_example_cp_ep_fsdp_npu() -> None:
    """Compare CP, EP and combined jobs against the same global dense algorithm."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        for cp_size, ep_size in ((2, 1), (1, 2), (2, 2)):
            for direct in (False, True):
                _check_training(cp_size, ep_size, direct=direct)
    finally:
        dist.destroy_process_group()


def test_direct_mesh_construction_cpu() -> None:
    """Verify constructor-owned CP/EP transport against dense outputs and input gradients."""
    dist.init_process_group("gloo")
    try:
        for cp_size, ep_size in ((2, 1), (1, 2), (2, 2)):
            world = dist.get_world_size()
            mesh = MeshContext(dp_size=world // cp_size, cp_size=cp_size, ep_size=ep_size,
                               dp_shard_size=world, edp_shard_size=world // ep_size)
            mesh.build_meshs("cpu", world)
            config = DecoderConfig()
            torch.manual_seed(71)
            reference = FamilyDecoder(config)
            torch.manual_seed(71)
            with patch("torch.distributed.new_group", side_effect=AssertionError("component created a group")):
                model = FamilyDecoder(config, mesh=mesh)
            if model.experts.gate_up.shape[0] != config.num_experts // ep_size:
                raise AssertionError("constructor did not apply expert ownership")
            rank = dist.get_rank()
            tokens = torch.arange(cp_size * 2 * config.hidden_size, dtype=torch.float32)
            tokens = (tokens.reshape(1, cp_size * 2, config.hidden_size) / 31 + rank // cp_size).sin()
            reference_inputs = tokens.clone().requires_grad_()
            local = tokens[:, (rank % cp_size) * 2:(rank % cp_size + 1) * 2].clone().requires_grad_()
            actual = model(local)
            expected = reference(reference_inputs)
            selection = slice((rank % cp_size) * 2, (rank % cp_size + 1) * 2)
            torch.testing.assert_close(actual, expected[:, selection], atol=3e-6, rtol=3e-5)
            actual.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(local.grad, reference_inputs.grad[:, selection], atol=3e-6, rtol=3e-5)
    finally:
        dist.destroy_process_group()
