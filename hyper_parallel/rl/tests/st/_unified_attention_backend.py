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
"""Real paged cache writes and fixed-request NPU graph replay for the V4 backend."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch_npu  # pylint: disable=unused-import
from vllm import EngineArgs
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

from rl.unified_execution.runtimes.vllm.attention_backend import (
    UnifiedAscendAttentionImpl, UnifiedAttentionMetadataBuilder,
)


def _reference(query: torch.Tensor, cache: tuple[torch.Tensor, ...], tables: torch.Tensor,
               query_lengths: list[int], lengths: list[int]) -> torch.Tensor:
    """Reconstruct logical history only in the independent CPU float64 oracle."""
    query, keys, values, tables = (tensor.detach().cpu() for tensor in (query, *cache, tables))
    outputs, start = [], 0
    for index, (query_length, length) in enumerate(zip(query_lengths, lengths)):
        if length == 0:
            outputs.append(torch.zeros(query_length, query.shape[1], query.shape[2], dtype=torch.float64))
            start += query_length
            continue
        key = keys[tables[index].long()].flatten(0, 1)[:length].double()
        value = values[tables[index].long()].flatten(0, 1)[:length].double()
        repeats = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeats, dim=1).transpose(0, 1)
        value = value.repeat_interleave(repeats, dim=1).transpose(0, 1)
        part = query[start:start + query_length].double().transpose(0, 1)
        scores = part @ key.transpose(-2, -1) * query.shape[-1] ** -0.5
        allowed = torch.arange(length)[None, :] <= torch.arange(query_length)[:, None] + length - query_length
        outputs.append((scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ value).transpose(0, 1))
        start += query_length
    return torch.cat(outputs)


class BackendProbe:
    """Exercise the installed native lifecycle with mutable runner-like buffers."""

    def __init__(self, config: Any, device: torch.device, query_lengths: list[int], lengths: list[int]) -> None:
        """Initialize real native metadata and cache with nonconsecutive physical pages."""
        self.device = device
        self.query_lengths = query_lengths
        self.lengths = lengths
        self.layer = SimpleNamespace(layer_name="v4_backend_probe", _k_scale_float=1.0, _v_scale_float=1.0)
        self.impl = UnifiedAscendAttentionImpl(
            num_heads=4, head_size=64, scale=64 ** -0.5, num_kv_heads=2,
            alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto", logits_soft_cap=None,
            attn_type=AttentionType.DECODER, kv_sharing_target_layer_name=None,
        )
        if self.impl.use_bnsd_kv_cache:
            raise ValueError("This direct cache-write probe requires the default NHD cache layout")
        self.cache = tuple((torch.randn(24, 128, 2, 64) * 0.25).to(device, torch.bfloat16) for _ in range(2))
        self.tables = torch.tensor([
            [17, 2, 11, 5, 19, 8], [13, 4, 22, 6, 1, 20], [0, 0, 0, 0, 0, 0],
        ], dtype=torch.int32, device=device)[:len(lengths)]
        count = sum(query_lengths)
        self.query = (torch.randn(count, 4, 64) * 0.25).to(device, torch.bfloat16)
        self.key = (torch.randn(count, 2, 64) * 0.25).to(device, torch.bfloat16)
        self.value = (torch.randn(count, 2, 64) * 0.25).to(device, torch.bfloat16)
        self.output = torch.empty_like(self.query)
        offsets = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32)
        seq_lens = torch.tensor(lengths, dtype=torch.int32)
        self.common = AscendCommonAttentionMetadata(
            query_start_loc=offsets.to(device), query_start_loc_cpu=offsets,
            seq_lens=seq_lens.to(device), seq_lens_cpu=seq_lens, _seq_lens_cpu=seq_lens,
            num_reqs=len(lengths), num_actual_tokens=count, max_query_len=max(query_lengths),
            max_seq_len=max(lengths), block_table_tensor=self.tables,
            slot_mapping=torch.empty(count, dtype=torch.int64, device=device), causal=True,
            attn_state=AscendAttentionState.ChunkedPrefill,
        )
        spec = FullAttentionSpec(block_size=128, num_kv_heads=2, head_size=64, dtype=torch.bfloat16)
        self.builder = UnifiedAttentionMetadataBuilder(spec, [self.layer.layer_name], config, device)
        self.metadata = self.builder.build(0, self.common)
        self.update(query_lengths, lengths)

    def update(self, query_lengths: list[int], lengths: list[int]) -> None:
        """Refresh offsets, logical lengths and physical writes without replacing storage."""
        if sum(query_lengths) != self.query.shape[0] or len(lengths) != len(self.lengths):
            raise ValueError("Direct graph replay requires fixed token and request capacities")
        offsets = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32)
        self.common.query_start_loc.copy_(offsets)
        self.common.query_start_loc_cpu.copy_(offsets)
        self.common.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        self.common.seq_lens_cpu.copy_(torch.tensor(lengths, dtype=torch.int32))
        tables = self.tables.cpu().tolist()
        slots = []
        for table, query_length, length in zip(tables, query_lengths, lengths):
            for position in range(length - query_length, length):
                slots.append(-1 if length == 0 else table[position // 128] * 128 + position % 128)
        self.common.slot_mapping.copy_(torch.tensor(slots, dtype=torch.int64))
        self.query_lengths, self.lengths = query_lengths, lengths
        # Model runners rebuild metadata outside capture before each replay.
        self.metadata = self.builder.build(0, self.common)

    def forward(self) -> torch.Tensor:
        """Run inherited native slot writes followed by the shared V4 numerical entry."""
        return self.impl.forward(self.layer, self.query, self.key, self.value, self.cache, self.metadata, self.output)

    def check(self) -> dict[str, Any]:
        """Validate both current slot writes and attention over complete logical history."""
        torch.npu.synchronize()
        slots = self.common.slot_mapping.cpu()
        active = slots >= 0
        for cache, current in zip(self.cache, (self.key, self.value)):
            actual = cache.flatten(0, 1)[slots[active].to(self.device)].cpu()
            torch.testing.assert_close(actual, current.cpu()[active], atol=0, rtol=0)
        expected = _reference(self.query, self.cache, self.tables, self.query_lengths, self.lengths)
        actual = self.output.cpu().double()
        valid_rows = torch.repeat_interleave(torch.tensor(self.lengths) > 0, torch.tensor(self.query_lengths))
        torch.testing.assert_close(actual[valid_rows], expected[valid_rows], atol=0.02, rtol=0.02)
        error = (actual[valid_rows] - expected[valid_rows]).abs()
        return {"query_lengths": self.query_lengths, "kv_lengths": self.lengths,
                "max_abs": float(error.max()), "mean_abs": float(error.mean()), "cache_writes_exact": True}


def _graph_check(probe: BackendProbe, updates: list[tuple[list[int], list[int]]]) -> list[dict[str, Any]]:
    """Capture actual cache writes plus attention, then mutate all runtime device lengths."""
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            probe.forward()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        probe.forward()
    results = []
    for query_lengths, lengths in updates:
        probe.update(query_lengths, lengths)
        graph.replay()
        torch.npu.synchronize()
        captured = probe.output.clone()
        result = probe.check()
        probe.forward()
        torch.npu.synchronize()
        if not torch.equal(captured, probe.output):
            raise AssertionError("V4 graph replay differs from eager execution with the same live metadata")
        result["graph_eager_exact"] = True
        results.append(result)
    return results


def main() -> None:
    """Run on an explicitly selected NPU without loading model weights or an engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model config directory for native vLLM setup")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(6108)
    device = torch.device(f"npu:{args.device}")
    config = EngineArgs(
        model=args.model, dtype="bfloat16", enforce_eager=True, max_model_len=768,
        max_num_seqs=3, max_num_batched_tokens=768, enable_prefix_caching=True,
        enable_chunked_prefill=True, block_size=128,
        hf_overrides={"hidden_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64},
    ).create_engine_config()
    current_platform.import_kernels()
    with torch.inference_mode(), set_current_vllm_config(config):
        mixed = BackendProbe(config, device, [2, 3], [113, 257])
        mixed.forward()
        report = {"mixed_eager": mixed.check()}
        report["fixed_request_mixed_graph"] = _graph_check(mixed, [([2, 3], [113, 257]), ([3, 2], [641, 514])])
        decode = BackendProbe(config, device, [1, 1], [113, 127])
        report["decode_graph"] = _graph_check(decode, [([1, 1], [113, 127]), ([1, 1], [641, 514])])
        padded = BackendProbe(config, device, [1, 1, 1], [113, 514, 0])
        padded.forward()
        report["padded_decode"] = padded.check()
    report["scope"] = "Native cache/metadata and fixed-request direct NPUGraph; not variable-request engine FULL graphs"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
