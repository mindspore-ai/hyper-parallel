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

from hyper_parallel.infer import (
    ContextParallelKVCache,
    GenerationConfig,
    KVCache,
    SequenceShardInfo,
    generate,
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
