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
"""End-to-end context-parallel generate check."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.infer import GenerationConfig, generate, get_sequence_shard_info


def _init_weights(module: nn.Module, vocab_size: int) -> None:
    with torch.no_grad():
        module.embedding.weight.zero_()
        module.lm_head.weight.zero_()
        for token_id in range(vocab_size):
            scale = 20.0 if token_id == 7 else 1.0
            module.embedding.weight[token_id, token_id] = scale
            module.lm_head.weight[token_id, token_id] = 1.0


class FullTinyAttentionLM(nn.Module):
    """Single-process tiny attention model used as the baseline."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, vocab_size, bias=False)
        _init_weights(self, vocab_size)

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        **kwargs,
    ):
        del position_ids, kwargs
        hidden = self.embedding(input_ids)
        key = hidden.unsqueeze(1)
        value = key
        if past_key_values is not None:
            past_key, past_value = past_key_values[0]
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        logits = self._attention_logits(hidden, key, value, attention_mask)
        past = [(key.detach(), value.detach())] if use_cache else None
        return {"logits": logits, "past_key_values": past}

    def _attention_logits(self, hidden, key, value, attention_mask):
        query = torch.ones_like(hidden).unsqueeze(1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_size)
        if attention_mask is not None:
            scores = scores + attention_mask.to(dtype=scores.dtype)
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, value).squeeze(1)
        return self.lm_head(context)


class ContextParallelTinyAttentionLM(FullTinyAttentionLM):
    """Tiny attention model whose KV cache is sharded by sequence across ranks."""

    def __init__(self, vocab_size: int, rank: int, world_size: int):
        super().__init__(vocab_size)
        self.rank = rank
        self.world_size = world_size

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        sequence_shard_info=None,
        global_seq_len=None,
        **kwargs,
    ):
        del position_ids, kwargs
        hidden = self.embedding(input_ids)
        new_key = hidden.unsqueeze(1)
        new_value = new_key
        if past_key_values is None:
            full_key = new_key
            full_value = new_value
        else:
            full_key, full_value = self._gather_full_cache(
                past_key_values,
                sequence_shard_info,
            )
            full_key = torch.cat([full_key, new_key], dim=-2)
            full_value = torch.cat([full_value, new_value], dim=-2)
        if global_seq_len is not None and full_key.shape[-2] != global_seq_len:
            raise ValueError("global_seq_len does not match rebuilt CP cache")
        logits = self._attention_logits(hidden, full_key, full_value, attention_mask)
        shard_info = get_sequence_shard_info(full_key.shape[-2], self.rank, self.world_size)
        local_key = full_key.narrow(-2, shard_info.start, shard_info.local_seq_len).contiguous()
        local_value = full_value.narrow(-2, shard_info.start, shard_info.local_seq_len).contiguous()
        past = [(local_key.detach(), local_value.detach())] if use_cache else None
        return {
            "logits": logits,
            "past_key_values": past,
            "sequence_shard_info": shard_info,
        }

    def _gather_full_cache(self, past_key_values, sequence_shard_info):
        if sequence_shard_info is None:
            raise ValueError("sequence_shard_info is required for CP cached decode")
        local_key, local_value = past_key_values[0]
        start = torch.tensor([sequence_shard_info.start], device=local_key.device)
        local_len = torch.tensor([local_key.shape[-2]], device=local_key.device)
        starts = [torch.empty_like(start) for _ in range(self.world_size)]
        lengths = [torch.empty_like(local_len) for _ in range(self.world_size)]
        dist.all_gather(starts, start)
        dist.all_gather(lengths, local_len)

        max_len = int(torch.stack(lengths).max().item())
        padded_key = _pad_sequence_shard(local_key, max_len)
        padded_value = _pad_sequence_shard(local_value, max_len)
        gathered_keys = [torch.empty_like(padded_key) for _ in range(self.world_size)]
        gathered_values = [torch.empty_like(padded_value) for _ in range(self.world_size)]
        dist.all_gather(gathered_keys, padded_key)
        dist.all_gather(gathered_values, padded_value)

        gathered = sorted(
            (
                int(rank_start.item()),
                key.narrow(-2, 0, int(rank_len.item())),
                value.narrow(-2, 0, int(rank_len.item())),
            )
            for rank_start, rank_len, key, value in zip(
                starts,
                lengths,
                gathered_keys,
                gathered_values,
            )
        )
        full_key = torch.cat([item[1] for item in gathered], dim=-2)
        full_value = torch.cat([item[2] for item in gathered], dim=-2)
        return full_key, full_value


def _pad_sequence_shard(tensor: torch.Tensor, max_len: int) -> torch.Tensor:
    pad_len = max_len - tensor.shape[-2]
    if pad_len == 0:
        return tensor.contiguous()
    pad_shape = list(tensor.shape)
    pad_shape[-2] = pad_len
    padding = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, padding], dim=-2).contiguous()


def _gather_token_outputs(output: torch.Tensor, world_size: int):
    gathered = [torch.empty_like(output) for _ in range(world_size)]
    dist.all_gather(gathered, output)
    return [item.cpu().tolist() for item in gathered]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    if not dist.is_initialized():
        dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise ValueError("CP generate check requires world_size >= 2")

    input_ids = torch.tensor([[7, 7, 1], [0, 7, 1]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.long)
    config_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "eos_token_id": None,
        "pad_token_id": 0,
    }
    cp_output = generate(
        ContextParallelTinyAttentionLM(args.vocab_size, rank, world_size),
        input_ids,
        GenerationConfig(
            **config_kwargs,
            use_cache=True,
            context_parallel_cache=True,
            context_process_group=dist.group.WORLD,
            context_logits_rank=world_size - 1,
        ),
        attention_mask=attention_mask,
    )
    gathered_outputs = _gather_token_outputs(cp_output, world_size)

    prefix_ids = torch.tensor([[7, 2], [3, 7]], dtype=torch.long)
    prefix_mask = torch.ones_like(prefix_ids)
    suffix_ids = torch.tensor([[1], [1]], dtype=torch.long)
    suffix_mask = torch.ones_like(suffix_ids)
    prefix_outputs = FullTinyAttentionLM(args.vocab_size)(
        input_ids=prefix_ids,
        attention_mask=torch.zeros(prefix_ids.size(0), 1, prefix_ids.size(1), prefix_ids.size(1)),
        use_cache=True,
    )
    cp_prefix_output = generate(
        ContextParallelTinyAttentionLM(args.vocab_size, rank, world_size),
        suffix_ids,
        GenerationConfig(
            **config_kwargs,
            use_cache=True,
            prefix_past_key_values=prefix_outputs["past_key_values"],
            prefix_attention_mask=prefix_mask,
            context_parallel_cache=True,
            context_process_group=dist.group.WORLD,
            context_logits_rank=world_size - 1,
        ),
        attention_mask=suffix_mask,
    )
    gathered_prefix_outputs = _gather_token_outputs(cp_prefix_output, world_size)

    if rank == 0:
        baseline_output = generate(
            FullTinyAttentionLM(args.vocab_size),
            input_ids,
            GenerationConfig(**config_kwargs, use_cache=True),
            attention_mask=attention_mask,
        )
        prefix_baseline_output = generate(
            FullTinyAttentionLM(args.vocab_size),
            torch.cat([prefix_ids, suffix_ids], dim=-1),
            GenerationConfig(**config_kwargs, use_cache=True),
            attention_mask=torch.cat([prefix_mask, suffix_mask], dim=-1),
        )
        expected_prefix_output = prefix_baseline_output[:, prefix_ids.shape[-1]:]
        result = {
            "world_size": world_size,
            "max_new_tokens": args.max_new_tokens,
            "cp_outputs": gathered_outputs,
            "baseline_output": baseline_output.tolist(),
            "all_ranks_match_baseline": all(
                output == baseline_output.tolist()
                for output in gathered_outputs
            ),
            "cp_prefix_outputs": gathered_prefix_outputs,
            "prefix_baseline_output": expected_prefix_output.tolist(),
            "prefix_all_ranks_match_baseline": all(
                output == expected_prefix_output.tolist()
                for output in gathered_prefix_outputs
            ),
        }
        print(json.dumps(result, indent=2))
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not result["all_ranks_match_baseline"] or not result["prefix_all_ranks_match_baseline"]:
            raise SystemExit(1)
    dist.barrier()


if __name__ == "__main__":
    main()
