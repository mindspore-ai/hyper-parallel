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
"""Native PCP KV gathering and paged V4 against independent packed CP1 attention."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from vllm import EngineArgs
from vllm.config import set_current_vllm_config
from vllm.distributed import get_pcp_group, init_distributed_environment, initialize_model_parallel
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager

from rl.unified_execution.core.batch_invariant import initialize_batch_invariant
from rl.unified_execution.ops.attention.flash_attention_npu_v4 import varlen_attention
from rl.unified_execution.runtimes.vllm.attention.ascend_v4 import (
    UnifiedAscendAttentionImpl, UnifiedAttentionMetadataBuilder,
)


def offsets(lengths: list[int], device: torch.device) -> torch.Tensor:
    """Return zero-prefixed device int32 request offsets."""
    return torch.tensor([0, *np.cumsum(lengths).tolist()], dtype=torch.int32, device=device)


def check_case(config, query_lengths: list[int], prefixes: list[int], prefilling: list[bool], *, graph: bool) -> dict:
    """Reuse native chunk ownership and cache writes while keeping an independent CP1 oracle."""
    size, rank = get_pcp_group().world_size, get_pcp_group().rank_in_group
    device = torch.device(f"npu:{int(os.environ['LOCAL_RANK'])}")
    torch.manual_seed(1729 + sum(prefixes) + sum(query_lengths))
    lengths = [prefix + query for prefix, query in zip(prefixes, query_lengths)]
    query = (torch.randn(sum(query_lengths), 8, 128) * .25).to(device, torch.bfloat16)
    keys = [(torch.randn(length, 1, 128) * .25).to(device, torch.bfloat16) for length in lengths]
    values = [(torch.randn(length, 1, 128) * .25).to(device, torch.bfloat16) for length in lengths]
    manager = AscendPCPManager(size, rank, device)
    manager.vllm_config = config
    chunks = [list(manager._iter_rank_chunks(  # pylint: disable=protected-access
        owner, np.array(query_lengths), np.array(prefilling))) for owner in range(size)]
    padded = max(sum(length for _, _, length in owned) for owned in chunks)
    cache = tuple(torch.zeros(len(lengths) * 6, 128, 1, 128, dtype=torch.bfloat16, device=device)
                  for _ in range(2))
    tables = torch.arange(len(lengths) * 6, dtype=torch.int32, device=device).reshape(-1, 6).flip(1)
    slots_by_request = [(table[(torch.arange(length, device=device) // 128).long()] * 128
                         + torch.arange(length, device=device) % 128).long()
                        for table, length in zip(tables, lengths)]
    for target, source in zip(cache, (keys, values)):
        for slots, full, prefix in zip(slots_by_request, source, prefixes):
            target.flatten(0, 1)[slots[:prefix]] = full[:prefix]
    slots = torch.full((size, padded), -1, dtype=torch.int64, device=device)
    local_query = torch.zeros(padded, 8, 128, dtype=torch.bfloat16, device=device)
    local_key = torch.zeros(padded, 1, 128, dtype=torch.bfloat16, device=device)
    local_value = torch.zeros_like(local_key)
    query_starts = [0, *np.cumsum(query_lengths).tolist()]
    local_indices, segment_lengths, segment_ends, segment_requests = [], [], [], []
    for owner, owned in enumerate(chunks):
        start = 0
        for request, offset, count in owned:
            begin = prefixes[request] + offset
            slots[owner, start:start + count] = slots_by_request[request][begin:begin + count]
            if owner == rank:
                indices = list(range(query_starts[request] + offset, query_starts[request] + offset + count))
                local_query[start:start + count] = query[indices]
                local_key[start:start + count] = keys[request][begin:begin + count]
                local_value[start:start + count] = values[request][begin:begin + count]
                local_indices.extend(indices)
                segment_lengths.append(count)
                segment_ends.append(begin + count)
                segment_requests.append(request)
            start += count
    local_offsets = offsets(segment_lengths, device)
    local_lengths = torch.tensor(segment_ends, dtype=torch.int32, device=device)
    common = AscendCommonAttentionMetadata(
        query_start_loc=local_offsets, query_start_loc_cpu=local_offsets.cpu(),
        seq_lens=local_lengths, seq_lens_cpu=local_lengths.cpu(), _seq_lens_cpu=local_lengths.cpu(),
        num_reqs=len(segment_lengths), num_actual_tokens=len(local_indices), num_input_tokens=padded,
        max_query_len=max(segment_lengths, default=0), max_seq_len=max(lengths),
        block_table_tensor=tables[segment_requests], slot_mapping=slots.flatten(), causal=True,
        attn_state=AscendAttentionState.ChunkedPrefill,
        is_prefilling=torch.tensor(prefilling, dtype=torch.bool)[segment_requests],
    )
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    builder = UnifiedAttentionMetadataBuilder(spec, ["probe"], config, device)
    metadata = builder.build(0, common)
    impl = UnifiedAscendAttentionImpl(
        num_heads=8, head_size=128, scale=128 ** -.5, num_kv_heads=1,
        alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto", logits_soft_cap=None,
        attn_type=AttentionType.DECODER, kv_sharing_target_layer_name=None,
    )
    layer = type("ProbeLayer", (), {"layer_name": "probe", "_k_scale_float": 1., "_v_scale_float": 1.})()
    output = torch.empty_like(local_query)
    if graph:
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                impl.forward(layer, local_query, local_key, local_value, cache, metadata, output)
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        dist.barrier()
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            impl.forward(layer, local_query, local_key, local_value, cache, metadata, output)
        query.mul_(.5)
        local_query.mul_(.5)
        local_key.mul_(.5)
        local_value.mul_(.5)
        for source in (keys, values):
            for full, prefix in zip(source, prefixes):
                full[prefix:].mul_(.5)
        captured.replay()
    else:
        impl.forward(layer, local_query, local_key, local_value, cache, metadata, output)
    reference = varlen_attention(
        query, torch.cat(keys), torch.cat(values), cu_seqlens_q=offsets(query_lengths, device),
        cu_seqlens_k=offsets(lengths, device), max_seqlen_q=max(query_lengths), max_seqlen_k=max(lengths),
    )
    torch.npu.synchronize()
    expected = reference[local_indices]
    actual = output[:len(local_indices)]
    exact = torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    cache_exact = all(torch.equal(target.flatten(0, 1)[slots], full)
                      for target, source in zip(cache, (keys, values))
                      for slots, full in zip(slots_by_request, source))
    passed = torch.tensor(int(exact and cache_exact), device=device)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    record = {"pcp": size, "rank": rank, "query_lengths": query_lengths, "prefixes": prefixes,
              "local_segments": segment_lengths, "local_kv_ends": segment_ends,
              "output_exact": exact, "cache_exact": cache_exact, "direct_graph_replay": graph}
    if not bool(passed):
        raise AssertionError(f"Native PCP cache or full-prefix V4 differed from CP1: {record}")
    return record


def main() -> None:
    """Run actual native PCP groups on explicitly allocated NPU ranks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    initialize_batch_invariant({})
    config = EngineArgs(
        model=args.model, dtype="bfloat16", enforce_eager=True, max_model_len=768,
        max_num_seqs=4, max_num_batched_tokens=768, prefill_context_parallel_size=size,
        enable_prefix_caching=True, enable_chunked_prefill=True, block_size=128,
        hf_overrides={"hidden_size": 1024, "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 128},
    ).create_engine_config()
    current_platform.import_kernels()
    with torch.inference_mode(), set_current_vllm_config(config):
        init_distributed_environment(world_size=size, rank=rank, distributed_init_method="env://",
                                     local_rank=int(os.environ["LOCAL_RANK"]), backend="hccl")
        initialize_model_parallel(prefill_context_model_parallel_size=size, backend="hccl")
        dist.all_reduce(torch.zeros(1, device=f"npu:{int(os.environ['LOCAL_RANK'])}"))
        cases = [([129, 7], [0, 0], [True, True]), ([1, 129], [511, 113], [False, True]),
                 ([1], [0], [True]), ([1, 1], [129, 257], [False, False]), ([513], [128], [True])]
        rows = [check_case(config, *case, graph=args.graph) for case in cases]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{rank}.json").write_text(json.dumps(rows, indent=2) + "\n")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
