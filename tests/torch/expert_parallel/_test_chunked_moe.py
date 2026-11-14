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
"""NPU complete-gradient oracle for common MoE chunks and native EP shards."""
import copy
import json
import os

import torch
import torch.distributed as dist
# Register the NPU backend only in worker processes.
import torch_npu  # pylint: disable=unused-import

from hyper_parallel import DeviceMesh, SkipDTensorDispatch, fully_shard
from hyper_parallel.components.modules.moe import FeedForward, MoE, MoEAuxLossAutoScaler
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel, _generate_permute_indices
from hyper_parallel.core.expert_parallel.row_permutation import invert_permutation, permute_rows
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy


def _relative(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-8))


def _close(actual, expected, label):
    error = _relative(actual, expected)
    assert error < 0.03, f"{label}: norm-relative={error}, required < 0.03"
    return error


def _run(ep_size, chunk_size=64):
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    mesh = DeviceMesh("npu", torch.arange(world).reshape(world // ep_size, ep_size),
                      mesh_dim_names=("replica", "ep"))
    ep_mesh = mesh["ep"]
    replica_group = mesh["replica"].get_group()
    old_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
    MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(0.25, device="npu")
    experts, top_k = (48, 4) if ep_size == 2 else (8, 2)
    for empty_owner in ((False, True) if chunk_size is not None else (False,)):
        torch.manual_seed(23)
        reference = MoE(128, 256, experts, top_k=top_k, score_before_experts=False, use_grouped_mm=True,
                        load_balance_coeff=0.07, shared_expert=FeedForward(128, 128),
                        router_kwargs={"score_func": "sigmoid", "route_scale": 1.7}).to("npu", dtype=torch.bfloat16)
        if empty_owner:
            reference.expert_bias[:top_k] = 100
        reference.experts.w1.requires_grad_(not empty_owner)
        actual = copy.deepcopy(reference)
        actual.sequence_partition_group = dist.group.WORLD
        if ep_size > 1:
            ExpertParallel().apply(actual.experts, ep_mesh)
        identities = {name: id(parameter) for name, parameter in actual.named_parameters()}
        actual.chunk_size = chunk_size
        torch.manual_seed(90)
        full = torch.randn(1, world * 129, 128).to("npu", dtype=torch.bfloat16).requires_grad_()
        cotangent = torch.randn(1, world * 129, 128).to("npu", dtype=torch.bfloat16)
        local = full[:, rank * 129:(rank + 1) * 129].detach().clone().requires_grad_()
        ref_ids, actual_ids = [], []
        ref_hook = reference.router.register_forward_hook(lambda _m, _a, out: ref_ids.append(out[1].detach()))
        act_hook = actual.router.register_forward_hook(lambda _m, _a, out: actual_ids.append(out[1].detach()))
        expected = reference(full)
        output = actual(local)
        ref_hook.remove()
        act_hook.remove()
        expected.backward(cotangent)
        output.backward(cotangent[:, rank * 129:(rank + 1) * 129])
        expected_ids = ref_ids[0][rank * 129:(rank + 1) * 129]
        observed_ids = torch.cat(actual_ids)
        assert torch.equal(observed_ids, expected_ids), f"rank {rank}: original routing IDs differ"
        errors = {"output": _close(output, expected[:, rank * 129:(rank + 1) * 129], "output"),
                  "input": _close(local.grad, full.grad[:, rank * 129:(rank + 1) * 129], "input")}
        with SkipDTensorDispatch():
            for name, parameter in actual.named_parameters():
                assert id(parameter) == identities[name], f"{name}: parameter identity changed"
                if not parameter.requires_grad:
                    assert parameter.grad is None, f"{name}: frozen parameter acquired a gradient"
                    continue
                target = reference.get_parameter(name).grad
                if name.startswith("experts.") and ep_size > 1:
                    if world > ep_size:
                        work = dist.all_reduce(parameter.grad, group=replica_group, async_op=True)
                        work.wait()
                    start = ep_mesh.get_local_rank() * (experts // ep_size)
                    target = target[start:start + experts // ep_size]
                else:
                    work = dist.all_reduce(parameter.grad, async_op=True)
                    work.wait()
                errors[name] = _close(parameter.grad, target, name)
        counts = actual.tokens_per_expert.clone()
        work = dist.all_reduce(counts, async_op=True)
        work.wait()
        torch.testing.assert_close(counts, reference.tokens_per_expert)
        auxiliary = actual.last_aux_loss.clone()
        work = dist.all_reduce(auxiliary, async_op=True)
        work.wait()
        torch.testing.assert_close(auxiliary, reference.last_aux_loss, rtol=1e-5, atol=1e-6)
        torch.npu.synchronize()
        print(json.dumps({"rank": rank, "ep": ep_size, "empty_owner": empty_owner,
                          "errors": errors, "pass": True}), flush=True)
    MoEAuxLossAutoScaler.main_loss_backward_scale = old_scale
    dist.destroy_process_group()


def test_receive_indices_npu():
    """Check receive-size selection, empty segments, short tails and row gradients."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    for ranks, experts, rows in ((2, 12, 0), (2, 12, 257), (4, 24, 8191),
                                  (4, 24, 8192), (8, 32, 8193), (1, 1, 16385)):
        for dtype in (torch.int32, torch.int64):
            counts = torch.zeros(ranks * experts, dtype=dtype)
            counts[0] = rows // 3
            counts[-1] += rows - rows // 3
            starts = (counts.cumsum(0) - counts).tolist()
            lengths = counts.tolist()
            expected = []
            for expert in range(experts):
                for rank in range(ranks):
                    index = rank * experts + expert
                    expected.extend(range(starts[index], starts[index] + lengths[index]))
            indices, local_counts = _generate_permute_indices(counts.npu(), experts, ranks, rows)
            torch.testing.assert_close(indices.long().cpu(), torch.tensor(expected, dtype=torch.int64),
                                       rtol=0, atol=0)
            torch.testing.assert_close(local_counts.cpu(), counts.view(ranks, experts).sum(0), rtol=0, atol=0)
            indices = indices.long()
            inverse = invert_permutation(indices)
            values = torch.randn(rows, 3, device="npu", requires_grad=True)
            cotangent = torch.randn_like(values)
            permute_rows(values, indices, inverse).backward(cotangent)
            expected_gradient = torch.zeros_like(values).index_add_(0, indices, cotangent)
            torch.testing.assert_close(values.grad, expected_gradient, rtol=0, atol=0)


def test_chunked_moe_single():
    """Check the NPU minimal VJP, shared/aux and a one-token tail."""
    _run(1)


def test_chunked_moe_ep4():
    """Check native EP4 shards, empty owners and all trainable gradients."""
    _run(4)


def test_chunked_moe_ep2_replicas():
    """Check two EP2 replicas and the separate expert-gradient SUM domain."""
    _run(2)


def test_score_after_ep4():
    """Keep the original whole-input EP baseline's probabilities aligned."""
    _run(4, chunk_size=None)


def test_chunked_moe_fsdp2_ep2():
    """Check two real reshard/update cycles at a parent FSDP boundary."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    mesh = DeviceMesh("npu", torch.arange(4).reshape(2, 2), mesh_dim_names=("fsdp", "ep"))
    torch.manual_seed(34)
    reference = MoE(128, 256, 8, top_k=2, score_before_experts=False, use_grouped_mm=True,
                    shared_expert=FeedForward(128, 128)).to("npu")
    actual = copy.deepcopy(reference)
    actual.chunk_size = 64
    modules = (reference, actual)
    for module in modules:
        ExpertParallel().apply(module.experts, mesh["ep"])
        fully_shard(module, mesh=mesh["fsdp"], reshard_after_forward=True,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                   cast_forward_inputs=False))
    optimizers = [torch.optim.SGD(module.parameters(), lr=0.01) for module in modules]
    for step in range(2):
        torch.manual_seed(100 + dist.get_rank() + step * 4)
        hidden = torch.randn(1, 129, 128, device="npu", dtype=torch.bfloat16)
        outputs, gradients, input_gradients = [], [], []
        for module, optimizer in zip(modules, optimizers):
            optimizer.zero_grad(set_to_none=True)
            inputs = hidden.clone().requires_grad_()
            output = module(inputs)
            (output.float().square().mean() / 2).backward()
            with SkipDTensorDispatch():
                for name, parameter in module.named_parameters():
                    if not name.startswith("experts."):
                        dist.all_reduce(parameter.grad, group=mesh["ep"].get_group(), async_op=True).wait()
            gradients.append({name: p.grad.to_local().clone() for name, p in module.named_parameters()})
            outputs.append(output.detach())
            input_gradients.append(inputs.grad.detach())
            with SkipDTensorDispatch():
                optimizer.step()
        _close(outputs[1], outputs[0], "FSDP output")
        _close(input_gradients[1], input_gradients[0], "FSDP input")
        for name, gradient in gradients[0].items():
            _close(gradients[1][name], gradient, name)
    dist.destroy_process_group()
