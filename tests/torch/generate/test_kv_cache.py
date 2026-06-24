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
"""KV cache tests."""
import pytest
import torch
from torch.nn import functional as F

from hyper_parallel.infer import (
    GenerationConfig,
    KVCache,
    generate,
)
from hyper_parallel.infer.kv_cache import (
    ContextParallelKVCache,
    SequenceShardInfo,
    get_sequence_shard_info,
    shard_past_key_values,
)
from tests.torch.generate.stub_model import CacheLengthLM, NoCacheLengthLM


def _past(seq_len: int, requires_grad: bool = False):
    key = torch.ones(2, 3, seq_len, 4, requires_grad=requires_grad)
    value = torch.ones(2, 3, seq_len, 4, requires_grad=requires_grad)
    return [(key, value)]


class OpaqueCache:
    """HF-style cache object with its own sequence-length API."""

    def get_seq_length(self):
        return 1


class TinyAttentionCacheLM(torch.nn.Module):
    """Small causal-attention model with KV cache support."""

    def __init__(self, vocab_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, self.hidden_size)
        self.lm_head = torch.nn.Linear(self.hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.lm_head.weight.zero_()
            for token_id in range(vocab_size):
                scale = 20.0 if token_id == 7 else 1.0
                self.embedding.weight[token_id, token_id] = scale
                self.lm_head.weight[token_id, token_id] = 1.0

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
        query = torch.ones_like(hidden).unsqueeze(1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask.to(dtype=scores.dtype)
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, value).squeeze(1)
        projected = self.lm_head(context)
        logits = input_ids.new_full(
            input_ids.shape + (self.vocab_size,),
            -1000,
            dtype=torch.float32,
        )
        logits.copy_(projected)
        past = [(key.detach(), value.detach())] if use_cache else None
        return {"logits": logits, "past_key_values": past}


def test_cache_update_detaches_tensors():
    """
    Feature: KV cache update
    Description: Cache update stores detached key/value tensors.
    Expectation: Cached tensors do not require grad.
    """
    cache = KVCache()
    cache.update(_past(5, requires_grad=True))

    assert not cache.is_empty
    key, value = cache.past_key_values[0]
    assert key.shape[-2] == 5
    assert value.shape[-2] == 5
    assert key.requires_grad is False
    assert value.requires_grad is False


def test_cache_update_preserves_opaque_cache_object():
    """
    Feature: KV cache update
    Description: Some model backends return cache objects instead of K/V lists.
    Expectation: Opaque cache objects are preserved for the next model call.
    """
    opaque = OpaqueCache()
    cache = KVCache()

    cache.update(opaque)

    assert cache.past_key_values is opaque


def test_cache_update_normalizes_empty_list_to_empty_cache():
    """
    Feature: KV cache update
    Description: Some model backends may return an empty cache list.
    Expectation: Empty lists are treated as no available cache.
    """
    cache = KVCache()

    cache.update([])

    assert cache.is_empty
    assert cache.past_key_values is None


def test_cache_merge_appends_sequence_dimension():
    """
    Feature: KV cache merge
    Description: Incremental K/V tensors append on the sequence dimension.
    Expectation: Sequence length grows after merge.
    """
    cache = KVCache()
    cache.update(_past(5))
    cache.merge(_past(1))

    key, value = cache.past_key_values[0]
    assert key.shape[-2] == 6
    assert value.shape[-2] == 6


def test_cache_clear_and_shape_validation():
    """
    Feature: KV cache validation
    Description: Invalid cache entries should fail fast.
    Expectation: Clear empties cache; invalid K/V rank raises ValueError.
    """
    cache = KVCache()
    cache.update(_past(2))
    cache.clear()
    assert cache.is_empty

    with pytest.raises(ValueError, match="shape"):
        cache.update([(torch.ones(2, 3), torch.ones(2, 3))])


def test_generate_cache_and_no_cache_outputs_match():
    """
    Feature: cache and no-cache generation consistency
    Description: Run the same deterministic prompt with and without KV cache.
    Expectation: Generated token ids match exactly.
    """
    input_ids = torch.tensor([[7, 8, 9]])
    config = GenerationConfig(max_new_tokens=4, do_sample=False, eos_token_id=None)

    cache_output = generate(CacheLengthLM(vocab_size=32), input_ids, config)
    no_cache_output = generate(NoCacheLengthLM(vocab_size=32), input_ids, config)

    assert torch.equal(cache_output, no_cache_output)


def test_left_padded_attention_cache_decode_matches_no_cache_decode():
    """
    Feature: left-padded KV cache decode
    Description: Cached attention decode should keep padding keys masked after prefill.
    Expectation: Cache and no-cache outputs match for left-padded batches.
    """
    input_ids = torch.tensor([[7, 7, 1], [0, 7, 1]])
    attention_mask = torch.tensor([[0, 0, 1], [0, 1, 1]])
    base_config = {
        "max_new_tokens": 3,
        "do_sample": False,
        "eos_token_id": None,
        "pad_token_id": 0,
    }

    cache_output = generate(
        TinyAttentionCacheLM(vocab_size=64),
        input_ids,
        GenerationConfig(**base_config, use_cache=True),
        attention_mask=attention_mask,
    )
    no_cache_output = generate(
        TinyAttentionCacheLM(vocab_size=64),
        input_ids,
        GenerationConfig(**base_config, use_cache=False),
        attention_mask=attention_mask,
    )

    assert torch.equal(cache_output, no_cache_output)


def test_cache_and_no_cache_decode_logits_cosine_similarity():
    """
    Feature: cache and no-cache decode consistency
    Description: Compare next-step logits from cached and full-sequence decode.
    Expectation: Cosine similarity is close to 1.0 for deterministic logits.
    """
    input_ids = torch.tensor([[7, 8, 9]])
    next_token = torch.tensor([[3]])

    cache_model = CacheLengthLM(vocab_size=32)
    no_cache_model = NoCacheLengthLM(vocab_size=32)

    prefill = cache_model(input_ids=input_ids, use_cache=True)
    cached_decode = cache_model(
        input_ids=next_token,
        past_key_values=prefill["past_key_values"],
        use_cache=True,
    )
    no_cache_decode = no_cache_model(
        input_ids=torch.cat([input_ids, next_token], dim=-1),
        use_cache=False,
    )

    cached_logits = cached_decode["logits"][:, -1, :]
    no_cache_logits = no_cache_decode["logits"][:, -1, :]
    similarity = F.cosine_similarity(cached_logits, no_cache_logits, dim=-1)

    assert torch.all(similarity > 0.9999)


def test_sequence_shard_info_handles_uneven_lengths():
    """
    Feature: CP sequence shard metadata
    Description: Split a global sequence across CP ranks with uneven remainder.
    Expectation: Shards are contiguous, non-overlapping, and cover the full sequence.
    """
    shards = [get_sequence_shard_info(5, rank, 2) for rank in range(2)]

    assert [(shard.start, shard.end) for shard in shards] == [(0, 3), (3, 5)]
    assert sum(shard.local_seq_len for shard in shards) == 5
    with pytest.raises(ValueError, match="rank"):
        get_sequence_shard_info(5, rank=2, world_size=2)


def test_shard_past_key_values_slices_sequence_dimension():
    """
    Feature: CP full-cache sharding
    Description: Convert full prefill K/V cache into this rank's local sequence shard.
    Expectation: The returned tensors match the contiguous global sequence slice.
    """
    key = torch.arange(2 * 3 * 5 * 4).view(2, 3, 5, 4)
    value = key + 1000

    sharded, shard_info = shard_past_key_values([(key, value)], rank=1, world_size=2)

    assert shard_info == SequenceShardInfo(
        rank=1,
        world_size=2,
        start=3,
        end=5,
        global_seq_len=5,
    )
    assert torch.equal(sharded[0][0], key[:, :, 3:5, :])
    assert torch.equal(sharded[0][1], value[:, :, 3:5, :])


def test_context_parallel_cache_update_full_and_clear():
    """
    Feature: CP KV cache prefill
    Description: Store only the local shard of a full prefill cache.
    Expectation: Cache metadata tracks this rank's global sequence range.
    """
    cache = ContextParallelKVCache(rank=0, world_size=2)
    cache.update_full(_past(5))

    key, value = cache.past_key_values[0]
    assert cache.shard_info.start == 0
    assert cache.shard_info.end == 3
    assert key.shape[-2] == 3
    assert value.shape[-2] == 3

    cache.clear()
    assert cache.is_empty
    assert cache.shard_info.global_seq_len == 0


def test_context_parallel_cache_merge_local_validates_shard_growth():
    """
    Feature: CP KV cache local decode
    Description: Append local decode K/V only when this rank's sequence shard grows.
    Expectation: Valid local growth updates metadata; invalid growth raises ValueError.
    """
    cache = ContextParallelKVCache(rank=1, world_size=2)
    cache.update_full(_past(5))
    cache.merge_local(_past(1), global_seq_len=6)

    key, value = cache.past_key_values[0]
    assert cache.shard_info.start == 3
    assert cache.shard_info.end == 6
    assert key.shape[-2] == 3
    assert value.shape[-2] == 3

    cache = ContextParallelKVCache(rank=0, world_size=2)
    cache.update_full(_past(5))
    before = cache.past_key_values[0][0].clone()
    with pytest.raises(ValueError, match="growth"):
        cache.merge_local(_past(1), global_seq_len=6)
    assert torch.equal(cache.past_key_values[0][0], before)


def test_context_parallel_initial_local_merge_requires_global_seq_len():
    """
    Feature: CP KV cache local decode
    Description: Initial local cache merge cannot infer global length in CP mode.
    Expectation: Missing global_seq_len raises ValueError when world_size > 1.
    """
    cache = ContextParallelKVCache(rank=0, world_size=2)

    with pytest.raises(ValueError, match="global_seq_len"):
        cache.merge_local(_past(3))


def test_context_parallel_cache_update_local_validates_metadata():
    """
    Feature: CP KV cache local update
    Description: Accept K/V tensors that are already local to the current CP rank.
    Expectation: Mismatched shard metadata or local length raises ValueError.
    """
    cache = ContextParallelKVCache(rank=1, world_size=2)
    cache.update_local(
        _past(2),
        SequenceShardInfo(rank=1, world_size=2, start=3, end=5, global_seq_len=5),
    )

    assert cache.past_key_values[0][0].shape[-2] == 2
    with pytest.raises(ValueError, match="does not match"):
        cache.update_local(
            _past(2),
            SequenceShardInfo(rank=0, world_size=2, start=0, end=3, global_seq_len=5),
        )
    with pytest.raises(ValueError, match="sequence length"):
        cache.update_local(
            _past(1),
            SequenceShardInfo(rank=1, world_size=2, start=3, end=5, global_seq_len=5),
        )
