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
"""Generate tests."""
import pytest
import torch

from hyper_parallel.infer import (
    GenerationConfig,
    GenerateMixin,
    apply_repetition_penalty,
    build_causal_mask,
    build_position_ids,
    gather_context_parallel_logits,
    gather_tensor_parallel_logits,
    generate,
    greedy_sample,
    top_k_sample,
    top_p_sample,
)
from hyper_parallel.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM
from hyper_parallel.models.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
)
from tests.torch.generate.stub_model import CacheLengthLM, NoCacheLengthLM


class MixinLengthLM(GenerateMixin, CacheLengthLM):
    """CacheLengthLM with a model.generate method."""


class MixinQwen3_5ForCausalLM(GenerateMixin, Qwen3_5ForCausalLM):
    """Project Qwen3.5 model with a model.generate method."""


class MixinQwen3_5MoeForCausalLM(GenerateMixin, Qwen3_5MoeForCausalLM):
    """Project Qwen3.5-MoE model with a model.generate method."""


class StrictNoCacheLM(NoCacheLengthLM):
    """No-cache model without cache-related forward kwargs."""

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        del position_ids, attention_mask
        return super().forward(input_ids, use_cache=False)


class InternalTypeErrorLM(CacheLengthLM):
    """Model that raises TypeError from inside forward."""

    def forward(self, *args, **kwargs):
        raise TypeError("internal model bug")


def test_generation_config_validation():
    """
    Feature: generation config validation
    Description: Invalid generation options fail early.
    Expectation: ValueError identifies invalid fields.
    """
    assert GenerationConfig().max_new_tokens == 256
    with pytest.raises(ValueError, match="temperature"):
        GenerationConfig(temperature=0)
    with pytest.raises(ValueError, match="top_p"):
        GenerationConfig(top_p=0)
    with pytest.raises(ValueError, match="top_k"):
        GenerationConfig(top_k=-1)
    with pytest.raises(ValueError, match="logits_processor"):
        GenerationConfig(logits_processor=[object()])
    with pytest.raises(ValueError, match="mask_dtype"):
        GenerationConfig(mask_dtype="float32")


def test_position_ids_and_causal_mask_support_left_padding():
    """
    Feature: prefill helpers
    Description: Position ids and masks respect left padding.
    Expectation: Padding columns and future tokens are masked.
    """
    input_ids = torch.tensor([[0, 0, 5, 6]])
    attention_mask = torch.tensor([[0, 0, 1, 1]])

    position_ids = build_position_ids(input_ids, attention_mask)
    mask = build_causal_mask(input_ids, attention_mask)

    assert position_ids.tolist() == [[0, 0, 0, 1]]
    assert torch.isneginf(mask[0, 0, 2, 1])
    assert torch.isneginf(mask[0, 0, 2, 3])
    assert torch.isfinite(mask[0, 0, 0]).all()
    assert mask.shape == (1, 1, 4, 4)


def test_sampler_helpers():
    """
    Feature: sampler helpers
    Description: Greedy/top-k sampling and repetition penalty operate on logits.
    Expectation: Output ids have shape (batch, 1) and penalties modify seen ids.
    """
    logits = torch.tensor([[0.0, 1.0, 4.0, 3.0]])
    assert greedy_sample(logits).tolist() == [[2]]

    torch.manual_seed(1)
    sampled = top_k_sample(logits, top_k=2)
    assert sampled.shape == (1, 1)
    assert sampled.item() in {2, 3}

    torch.manual_seed(1)
    nucleus = top_p_sample(logits, top_p=0.7)
    assert nucleus.shape == (1, 1)
    assert nucleus.item() == 2

    adjusted = apply_repetition_penalty(
        logits,
        input_ids=torch.tensor([[2, 2, 3]]),
        penalty=2.0,
    )
    assert adjusted[0, 2] == 2.0
    assert adjusted[0, 3] == 1.5


def test_generate_greedy_with_kv_cache():
    """
    Feature: greedy generation with KV cache
    Description: Decode uses single-token forward calls after prefill.
    Expectation: Output length and model call shapes match cache usage.
    """
    model = CacheLengthLM(vocab_size=32)
    input_ids = torch.tensor([[4, 5, 6]])

    out = generate(
        model,
        input_ids,
        GenerationConfig(max_new_tokens=3, do_sample=False, eos_token_id=None),
    )

    assert out.tolist() == [[4, 5, 6, 3, 4, 5]]
    assert [call["seq_len"] for call in model.calls] == [3, 1, 1]
    assert [call["past_len"] for call in model.calls] == [0, 3, 4]


def test_generate_no_cache_fallback_is_deterministic():
    """
    Feature: no-cache fallback
    Description: Models that do not return KV cache still generate by recomputing.
    Expectation: Greedy output is deterministic and preserves the prompt.
    """
    input_ids = torch.tensor([[7, 8]])
    config = GenerationConfig(max_new_tokens=2, do_sample=False, eos_token_id=None)

    out_a = generate(NoCacheLengthLM(vocab_size=32), input_ids, config)
    out_b = generate(NoCacheLengthLM(vocab_size=32), input_ids, config)

    assert torch.equal(out_a, out_b)
    assert out_a.tolist() == [[7, 8, 2, 3]]


def test_generate_strict_no_cache_model_omits_cache_kwargs():
    """
    Feature: no-cache signature fallback
    Description: Models without cache kwargs should be called without cache arguments.
    Expectation: Generation still falls back to full-sequence recompute.
    """
    out = generate(
        StrictNoCacheLM(vocab_size=32),
        torch.tensor([[7, 8]]),
        GenerationConfig(max_new_tokens=2, do_sample=False, eos_token_id=None),
    )

    assert out.tolist() == [[7, 8, 2, 3]]


def test_generate_preserves_internal_type_error():
    """
    Feature: model TypeError propagation
    Description: TypeError raised inside model.forward should not be treated as cache incompatibility.
    Expectation: The original TypeError is raised to the caller.
    """
    with pytest.raises(TypeError, match="internal model bug"):
        generate(
            InternalTypeErrorLM(vocab_size=32),
            torch.tensor([[1, 2]]),
            GenerationConfig(max_new_tokens=1, eos_token_id=None),
        )


def test_generate_stops_on_eos():
    """
    Feature: EOS stopping
    Description: Generation stops when every batch item emits eos_token_id.
    Expectation: Output includes the eos token and stops before max_new_tokens.
    """
    model = CacheLengthLM(vocab_size=32, eos_at_total_len=4)
    input_ids = torch.tensor([[4, 5, 6]])

    out = generate(
        model,
        input_ids,
        GenerationConfig(max_new_tokens=8, eos_token_id=2),
    )

    assert out.tolist() == [[4, 5, 6, 3, 2]]


def test_generate_batch_left_padding():
    """
    Feature: batch generation
    Description: Left-padded prompts are stripped in the returned sequences.
    Expectation: Batch output is right-padded to a common length.
    """
    model = NoCacheLengthLM(vocab_size=32)
    input_ids = torch.tensor([[0, 0, 4, 5], [7, 8, 9, 10]])
    attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

    out = generate(
        model,
        input_ids,
        GenerationConfig(max_new_tokens=2, eos_token_id=None, pad_token_id=0),
        attention_mask=attention_mask,
    )

    assert out.tolist() == [[4, 5, 4, 5, 0, 0], [7, 8, 9, 10, 4, 5]]


def test_generate_top_k_seeded_output_is_valid():
    """
    Feature: top-k generation
    Description: Sampling path returns valid token ids.
    Expectation: Generated tokens stay within vocabulary bounds.
    """
    torch.manual_seed(7)
    model = CacheLengthLM(vocab_size=16)
    out = generate(
        model,
        torch.tensor([[1, 2, 3]]),
        GenerationConfig(max_new_tokens=4, do_sample=True, top_k=4, eos_token_id=None),
    )

    assert out.shape == (1, 7)
    assert int(out.max()) < 16


def test_generate_logits_processor_changes_sampled_token():
    """
    Feature: logits processor extension
    Description: User processors can modify scores before sampling.
    Expectation: Greedy generation follows the processed logits.
    """
    def force_token_five(input_ids, scores):
        del input_ids
        processed = scores.new_full(scores.shape, -1000)
        processed[:, 5] = 1000
        return processed

    out = generate(
        CacheLengthLM(vocab_size=16),
        torch.tensor([[1, 2]]),
        GenerationConfig(
            max_new_tokens=1,
            eos_token_id=None,
            logits_processor=[force_token_five],
        ),
    )

    assert out.tolist() == [[1, 2, 5]]


def test_generate_stopping_criteria_stops_after_first_token():
    """
    Feature: stopping criteria extension
    Description: User criteria can stop generation after a decode step.
    Expectation: Generation stops before max_new_tokens.
    """
    def stop_after_prompt_plus_one(input_ids, scores):
        del scores
        return input_ids.shape[-1] >= 3

    out = generate(
        CacheLengthLM(vocab_size=16),
        torch.tensor([[1, 2]]),
        GenerationConfig(
            max_new_tokens=4,
            eos_token_id=None,
            stopping_criteria=[stop_after_prompt_plus_one],
        ),
    )

    assert out.tolist() == [[1, 2, 2]]


def test_tensor_parallel_logits_gather_falls_back_without_dist():
    """
    Feature: tensor-parallel logits gathering
    Description: Gathering is a no-op before torch.distributed is initialized.
    Expectation: Logits are returned unchanged.
    """
    logits = torch.randn(2, 8)
    config = GenerationConfig(gather_logits=True)

    gathered = gather_tensor_parallel_logits(logits, config)

    assert gathered is logits


def test_context_parallel_logits_gather_selects_owner_rank(monkeypatch):
    """
    Feature: context-parallel logits handoff
    Description: The final-token owner rank supplies logits to all ranks.
    Expectation: Gathered logits come from the configured CP owner rank.
    """
    import hyper_parallel.infer.utils as utils

    def fake_all_gather(output_tensors, tensor, group=None):
        del tensor, group
        output_tensors[0].copy_(torch.tensor([[0.0, 4.0, 1.0]]))
        output_tensors[1].copy_(torch.tensor([[0.0, 1.0, 7.0]]))

    logits = torch.zeros(1, 3)
    monkeypatch.setattr(utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(utils.dist, "all_gather", fake_all_gather)

    gathered = gather_context_parallel_logits(
        logits,
        GenerationConfig(context_logits_rank=1),
    )

    assert gathered.argmax(dim=-1).tolist() == [2]


def test_context_parallel_logits_gather_supports_batch_owner_ranks(monkeypatch):
    """
    Feature: context-parallel batch logits handoff
    Description: Different batch items can use different CP owner ranks.
    Expectation: Output rows are selected per batch item.
    """
    import hyper_parallel.infer.utils as utils

    def fake_all_gather(output_tensors, tensor, group=None):
        del tensor, group
        output_tensors[0].copy_(torch.tensor([[0.0, 5.0, 1.0], [0.0, 6.0, 1.0]]))
        output_tensors[1].copy_(torch.tensor([[0.0, 1.0, 7.0], [0.0, 1.0, 8.0]]))

    logits = torch.zeros(2, 3)
    monkeypatch.setattr(utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(utils.dist, "all_gather", fake_all_gather)

    gathered = gather_context_parallel_logits(
        logits,
        GenerationConfig(context_logits_rank=torch.tensor([0, 1])),
    )

    assert gathered.argmax(dim=-1).tolist() == [1, 2]


def test_generate_greedy_uses_gathered_tensor_parallel_logits(monkeypatch):
    """
    Feature: tensor-parallel greedy sampling
    Description: Vocab-sharded logits are gathered before selecting next tokens.
    Expectation: Greedy can select a token from a later gathered vocab shard.
    """
    import hyper_parallel.infer.utils as utils

    class ShardedLogitsLM(CacheLengthLM):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            logits = output["logits"].new_full(output["logits"].shape[:-1] + (4,), -1000)
            logits[:, -1, 1] = 1.0
            output["logits"] = logits
            return output

    def fake_all_gather(output_tensors, tensor, group=None):
        del group
        output_tensors[0].copy_(tensor)
        shard = tensor.new_full(tensor.shape, -1000)
        shard[:, 2] = 1000
        output_tensors[1].copy_(shard)

    monkeypatch.setattr(utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(utils.dist, "all_gather", fake_all_gather)

    out = generate(
        ShardedLogitsLM(vocab_size=8),
        torch.tensor([[1, 2]]),
        GenerationConfig(
            max_new_tokens=1,
            eos_token_id=None,
            gather_logits=True,
        ),
    )

    assert out.tolist() == [[1, 2, 6]]


def test_generate_with_project_qwen3_5_model():
    """
    Feature: project model compatibility
    Description: Generate can run against a project model through no-cache fallback.
    Expectation: Output appends valid token ids to the prompt.
    """
    config = Qwen3_5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        partial_rotary_factor=1.0,
        mrope_section=[2, 1, 1],
        layer_types=["full_attention"],
    )
    model = Qwen3_5ForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3]])

    out = generate(
        model,
        input_ids,
        GenerationConfig(max_new_tokens=2, eos_token_id=None),
    )

    assert out.shape == (1, 5)
    assert torch.equal(out[:, :3], input_ids)
    assert int(out.max()) < config.vocab_size


def test_generate_mixin_with_project_qwen3_5_model():
    """
    Feature: project model generate method
    Description: GenerateMixin exposes model.generate on a project Qwen3.5 model.
    Expectation: The project model method appends valid token ids to the prompt.
    """
    config = Qwen3_5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        partial_rotary_factor=1.0,
        mrope_section=[2, 1, 1],
        layer_types=["full_attention"],
    )
    model = MixinQwen3_5ForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3]])

    out = model.generate(
        input_ids,
        GenerationConfig(max_new_tokens=2, eos_token_id=None),
    )

    assert out.shape == (1, 5)
    assert torch.equal(out[:, :3], input_ids)
    assert int(out.max()) < config.vocab_size


def test_generate_with_project_qwen3_5_moe_model():
    """
    Feature: project MoE model compatibility
    Description: Generate can run against Qwen3.5-MoE through no-cache fallback.
    Expectation: Output appends valid token ids to the prompt.
    """
    config = Qwen3_5MoeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        partial_rotary_factor=0.5,
        mrope_section=[1, 1, 0],
        layer_types=["full_attention"],
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    model = Qwen3_5MoeForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3]])

    out = generate(
        model,
        input_ids,
        GenerationConfig(max_new_tokens=2, eos_token_id=None),
    )

    assert out.shape == (1, 5)
    assert torch.equal(out[:, :3], input_ids)
    assert int(out.max()) < config.vocab_size


def test_generate_mixin_with_project_qwen3_5_moe_model():
    """
    Feature: project MoE model generate method
    Description: GenerateMixin exposes model.generate on a project Qwen3.5-MoE model.
    Expectation: The project MoE model method appends valid token ids to the prompt.
    """
    config = Qwen3_5MoeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        partial_rotary_factor=0.5,
        mrope_section=[1, 1, 0],
        layer_types=["full_attention"],
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    model = MixinQwen3_5MoeForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3]])

    out = model.generate(
        input_ids,
        GenerationConfig(max_new_tokens=2, eos_token_id=None),
    )

    assert out.shape == (1, 5)
    assert torch.equal(out[:, :3], input_ids)
    assert int(out.max()) < config.vocab_size


def test_generate_mixin_method():
    """
    Feature: GenerateMixin
    Description: Models can expose a generate method through GenerateMixin.
    Expectation: The method returns the same style of generated token ids.
    """
    model = MixinLengthLM(vocab_size=32)
    out = model.generate(
        torch.tensor([[1, 2, 3]]),
        GenerationConfig(max_new_tokens=2, eos_token_id=None),
    )

    assert out.tolist() == [[1, 2, 3, 3, 4]]
