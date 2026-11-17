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
"""Explicit HCCL CP attention parity, packing, empty-rank and gradient checks."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu

from rl.unified_execution.core.batch_invariant import initialize_batch_invariant
from rl.unified_execution.ops.attention.flash_attention_npu_v4 import dense_attention
from rl.unified_execution.runtimes.hyper_parallel.attention import prepare_cp_attention_metadata, run_training_attention
from hyper_parallel import init_device_mesh


def _reference(states, mask, documents):
    outputs = []
    for batch in range(mask.shape[0]):
        row = states[0][batch].transpose(0, 1) * 0
        offset = 0
        for length in documents or (mask.shape[1],):
            indices = mask[batch, offset:offset + length].nonzero().flatten() + offset
            if indices.numel():
                parts = [state[batch].transpose(0, 1).index_select(0, indices).unsqueeze(0) for state in states]
                output = dense_attention(*parts).squeeze(0)
                row = row.index_copy(0, indices, output)
            offset += length
        outputs.append(row)
    return torch.stack(outputs)


def check_case(mesh, length, *, packed=False, empty=False):
    """Compare actual autograd all-gather with independent per-document CP1 calls."""
    rank, size = mesh.get_local_rank(), mesh.size()
    total = ((length + size - 1) // size) * size
    batch = 1 if packed or empty else 2
    torch.manual_seed(1907 + length)
    masks = torch.arange(total)[None].expand(batch, -1) < length
    if batch == 2:
        masks[1, :min(length // 3, 19)] = False
    if empty:
        masks[:] = False
        masks[0, 0] = True
    masks = masks.npu()
    documents = (3, 126, total - 129) if packed else None
    states = [(torch.randn(batch, heads, total, 128) * .25).bfloat16().npu() for heads in (8, 1, 1)]
    full = [state.detach().requires_grad_() for state in states]
    reference = _reference(full, masks, documents)
    cotangent = (torch.randn(reference.shape) * .25).bfloat16().npu() * masks[:, :, None, None]
    reference.backward(cotangent)
    torch.npu.synchronize()
    dist.barrier()
    local = [state.chunk(size, dim=2)[rank].contiguous().detach().requires_grad_() for state in states]
    metadata = prepare_cp_attention_metadata(masks, batch, total // size, local[0].device, mesh, documents)
    output, _ = run_training_attention(SimpleNamespace(unified_cp_mesh=mesh), *local, None,
                                       unified_attention_metadata=metadata)
    expected = reference.chunk(size, dim=1)[rank]
    valid = masks.chunk(size, dim=1)[rank].cpu()
    bits = torch.equal(output.detach().cpu()[valid].contiguous().view(torch.uint8),
                       expected.detach().cpu()[valid].contiguous().view(torch.uint8))
    output.backward(cotangent.chunk(size, dim=1)[rank].contiguous())
    errors = []
    for actual, original in zip(local, full):
        target = original.grad.chunk(size, dim=2)[rank].float()
        difference = (actual.grad.float() - target).norm()
        errors.append(float(difference / target.norm().clamp_min(1e-12)))
        torch.testing.assert_close(actual.grad.float(), target, atol=.02, rtol=.02)
    record = {'length': length, 'cp': size, 'rank': rank, 'packed': packed, 'empty': empty,
              'exact': bits, 'gradient_l2': errors}
    flags = torch.tensor(int(bits and max(errors) < .01), device='npu')
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    if not bool(flags):
        raise AssertionError(f"CP forward bytes or gradients failed: {record}")
    return record


def main():
    """Run only on devices explicitly allocated by the launcher."""
    rank = int(os.environ['LOCAL_RANK'])
    torch_npu.npu.set_device(rank)
    torch.npu.init()
    initialize_batch_invariant({})
    dist.init_process_group('hccl')
    mesh = init_device_mesh('npu', (dist.get_world_size(),), mesh_dim_names=('cp',))
    # Match model execution: initialize HCCL before loading custom AICPU attention kernels.
    warmup = torch.zeros(1, device=f'npu:{rank}')
    dist.all_reduce(warmup, group=mesh.get_group())
    dist.all_reduce(warmup)
    torch.npu.synchronize()
    records = [check_case(mesh, n) for n in (128, 1, 3, 7, 127, 129, 255, 256, 257, 511, 512, 513)]
    records += [check_case(mesh, 257, packed=True), check_case(mesh, 16, empty=True)]
    output = Path(os.environ['CP_TEST_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    (output / f'rank{rank}.json').write_text(json.dumps(records, indent=2) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
