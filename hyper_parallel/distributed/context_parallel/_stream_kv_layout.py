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

"""KV panel transport and causal scheduling for continuous and mirrored owners.

Inputs and FA operands use BNSD. Transport packs tokens first and K/V as the
second dimension. Mirrored FA panels are half-major; ReduceScatter buffers
remain owner-major. The explicit StreamKV autograd function owns every VJP;
these synchronous internal collectives must not be used as standalone autograd
operations.
"""

from functools import lru_cache

import torch
import torch.distributed as dist


def _peer_halves(source, target, rank):
    sends = [[index for index, chunk in enumerate(source[rank]) if chunk in peer] for peer in target]
    received = [[chunk for chunk in peer if chunk in target[rank]] for peer in source]
    return sends, received


@lru_cache(maxsize=128)
def _exchange_plan(degree, rank, inverse):
    contiguous = [(2 * owner, 2 * owner + 1) for owner in range(degree)]
    mirrored = [(owner, 2 * degree - 1 - owner) for owner in range(degree)]
    source, target = (mirrored, contiguous) if inverse else (contiguous, mirrored)
    sends, received = _peer_halves(source, target, rank)
    received_ids = [chunk for chunks in received for chunk in chunks]
    return ([index for indices in sends for index in indices], [len(indices) for indices in sends],
            [received_ids.index(chunk) for chunk in target[rank]], [len(chunks) for chunks in received])


def _redistribute(tensor, mesh, inverse=False):
    """Exchange two half-shards per owner while preserving the input dtype."""
    if mesh.size() == 1:
        return tensor
    half = tensor.shape[2] // 2
    send_order, send_counts, recv_order, recv_counts = _exchange_plan(mesh.size(), mesh.get_local_rank(), inverse)
    # Only two half-shards move; zero peer splits avoid an all-gather-sized buffer.
    send = torch.cat([tensor[:, :, index * half:(index + 1) * half] for index in send_order], 2)
    send = send.permute(2, 0, 1, 3).contiguous()
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, output_split_sizes=[count * half for count in recv_counts],
                           input_split_sizes=[count * half for count in send_counts], group=mesh.get_group())
    del send
    return torch.cat([received[index * half:(index + 1) * half] for index in recv_order], 0).permute(
        1, 2, 0, 3).contiguous()


def _panel_bounds(length, budget, balanced):
    """Yield local offsets; a mirrored owner's token budget covers both halves."""
    extent, step = (length // 2, budget // 2) if balanced else (length, budget)
    for start in range(0, extent, step):
        yield start, min(start + step, extent)


def _pieces(owner, degree, width, start, length, block, balanced):
    """Yield visible KV intervals and their matching query suffixes.

    Each tuple is (key_begin, key_end, query_begin, query_end, causal,
    next_tokens). FULL ranges contain only strictly earlier global halves.
    Own-half ranges use right-aligned causal FA with an explicit offset; the
    query suffix excludes rows with no visible keys in that block. Future
    halves are omitted, but every rank still participates in every collective.
    """
    half = length // 2 if balanced else length
    for query_half in range(2 if balanced else 1):
        q_base, q_end = query_half * half, (query_half + 1) * half
        full_ranges = [(0, owner * width)] if query_half == 0 else [
            (0, degree * width), ((degree + owner + 1) * width, 2 * degree * width)]
        for left, right in full_ranges:
            for begin in range(left, right, block):
                yield begin, min(begin + block, right), q_base, q_end, False, 2**31 - 1
        own = (query_half * degree + owner) * width
        for offset in range(0, width, block):
            finish = min(offset + block, width)
            yield own + offset, own + finish, q_base + start + offset, q_end, True, half - start - finish


def _gather(key, value, mesh, start, end, balanced):
    """Gather one owner stripe, or paired half-stripes, and unpack BNSD K/V."""
    if balanced:
        half = key.shape[2] // 2
        key = torch.cat((key[:, :, start:end], key[:, :, half + start:half + end]), 2)
        value = torch.cat((value[:, :, start:end], value[:, :, half + start:half + end]), 2)
    else:
        key, value = key[:, :, start:end], value[:, :, start:end]
    send = torch.stack((key, value), 2).permute(3, 2, 0, 1, 4).contiguous()
    if mesh.size() == 1:
        packed = send
    else:
        packed = torch.empty((mesh.size() * send.shape[0], *send.shape[1:]), device=key.device, dtype=key.dtype)
        dist.all_gather_into_tensor(packed, send, group=mesh.get_group())
    if balanced:
        # FA consumes half-major stripes; ReduceScatter below still uses owner-major storage.
        packed = packed.view(mesh.size(), 2, end - start, *send.shape[1:]).transpose(0, 1).flatten(0, 2)
    return tuple(packed[:, component].permute(1, 2, 0, 3).contiguous() for component in range(2))


def _add_partial(send, gradient, begin, width, component, kv_head, balanced):
    """Add one FA dK/dV result into the owner-major FP32 reduction panel."""
    values = gradient.permute(2, 0, 1, 3)
    if not balanced:
        send[begin:begin + values.shape[0], component, :, kv_head:kv_head + 1].add_(values)
        return
    degree = send.shape[0] // (2 * width)
    kv_half, relative = divmod(begin, degree * width)
    owner, offset = divmod(relative, width)
    destination = send.view(degree, 2, width, *send.shape[1:])[:, kv_half, :, component, :, kv_head:kv_head + 1]
    # Each scheduled FULL/CAUSAL interval stays in one half. Batch all interior
    # owners into one add, leaving at most two edges; avoid O(Pg) kernel launches.
    consumed = 0
    if offset:
        consumed = min(width - offset, values.shape[0])
        destination[owner, offset:offset + consumed].add_(values[:consumed])
        owner += 1
    count, tail = divmod(values.shape[0] - consumed, width)
    if count:
        destination[owner:owner + count].add_(values[consumed:consumed + count * width].view(
            count, width, *values.shape[1:]))
        consumed += count * width
        owner += count
    if tail:
        destination[owner, :tail].add_(values[consumed:])


def _reduce_panel(owned, send, mesh, start, end, balanced):
    """SUM all query-owner contributions and store the local owner's KV stripes."""
    width = end - start
    received = torch.empty((width * (2 if balanced else 1), *owned.shape[1:]),
                           device=owned.device, dtype=owned.dtype) if balanced else owned[start:end]
    if mesh.size() == 1:
        received.copy_(send)
    else:
        dist.reduce_scatter_tensor(received, send, op=dist.ReduceOp.SUM, group=mesh.get_group())
    if balanced:
        half = owned.shape[0] // 2
        owned[start:end].copy_(received[:width])
        owned[half + start:half + end].copy_(received[width:])
