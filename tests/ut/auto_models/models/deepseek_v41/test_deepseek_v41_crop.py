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
"""Focused CPU tests for the cropped DeepSeek-V4.1 validation model."""
# pylint: disable=wrong-import-position

import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import ContextManagers
try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    from transformers.initialization import no_init_weights
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)

from hyper_parallel import init_empty_weights
from hyper_parallel.components.checkpoint.weight_conversion import (
    WeightConverter,
    WeightRenaming,
    get_model_conversion_mapping,
    rename_source_key,
)
from hyper_parallel.components.modules.engram import EngramModule, NgramHashMapping
from hyper_parallel.components.modules.mhc import PipelinedMhcModule
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedPackedSequence,
    SharedCompressedDSAAttention,
    SharedCompressedDSAIndexer,
    compressed_candidate_topk,
    compressed_causal_topk,
    select_candidate_block_indices,
    select_candidate_blocks,
    shared_compressed_indexer_kl_loss,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.data.batching import ParallelBatch
from hyper_parallel.data.batching.build_collate_fn import (
    DataBatchContext,
    TextPackingCollator,
)
from hyper_parallel.data.vlm.collator import VLMCollator
from hyper_parallel.data.vlm.get_batch import build_vlm_get_batch
from hyper_parallel.distributed._builder.forward_rewriter import (
    validate_local_compute_signature,
)
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.activation_checkpoint import (
    _apply_activation_checkpointing,
)
from hyper_parallel.distributed.expert_parallel.experts import (
    bind_local_expert_forward,
)
from hyper_parallel.distributed.recipe_spec import EP, TP, ModuleShardingSpec
from hyper_parallel.models._transformers.model_builder import (
    _materialize_and_load_model,
)
from hyper_parallel.models.deepseek_v41.adapter.checkpoint import (
    register_deepseek_v41_checkpoint_mapping,
)
from hyper_parallel.models.deepseek_v41.adapter.expert_parallel import (
    deepseek_v41_engram_compute_fn,
    deepseek_v41_ep_compute_fn,
)
from hyper_parallel.models.deepseek_v41.adapter.registration import (
    _get_fsdp_wrap_modules,
)
from hyper_parallel.models.deepseek_v41.adapter.packed_sequence import (
    DeepseekV41BatchAdapter,
)
from hyper_parallel.models.deepseek_v41.adapter.replacements import (
    replace_deepseek_v41_shared_attention,
)
from hyper_parallel.models.deepseek_v41.adapter.vlm_data import (
    DeepseekV41VLMBatchAdapter,
    build_deepseek_v41_vlm_data_transform,
)
from hyper_parallel.models.deepseek_v41.configuration import (
    build_scaled_engram_buckets,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41AttentionPlaceholder,
    DeepseekV41Compressor,
    DeepseekV41CroppedForCausalLM,
    DeepseekV41SharedCompressedAttention,
    SharedAttentionCPContext,
    SharedAttentionState,
    _window_indices,
)
from hyper_parallel.models.replacement import (
    apply_module_replacements,
    compile_module_replacements,
)
from hyper_parallel.trainer.config import (
    Target,
    entries_to_module_replacements,
    entries_to_plan_overrides,
)
from hyper_parallel.trainer.config.manager import parse_training_args
from tests.common.mark_utils import arg_mark


def _write_engram_assets(
        directory: str,
        num_hidden_layers: int = 4,
        layer_ids: tuple[int, ...] = (1,),
) -> Path:
    """Write a minimal but internally consistent scaled Engram asset."""
    available_primes = (
        [[17, 19], [23, 29]],
        [[31, 37], [41, 43]],
    )
    if len(layer_ids) > len(available_primes):
        raise ValueError("the test asset helper supports at most two Engram layers")
    primes = list(available_primes[:len(layer_ids)])
    assets = {
        "source_model_type": "deepseek_v41",
        "num_hidden_layers": num_hidden_layers,
        "layer_ids": list(layer_ids),
        "bucket_base": 16,
        "max_ngram_size": 3,
        "num_heads": 2,
        "head_dim": 4,
        "primes": primes,
        "num_embeddings": [
            sum(value for row in layer_primes for value in row)
            for layer_primes in primes
        ],
        "multipliers": [[101, 103, 107] for _ in layer_ids],
        "token_map": list(range(64)),
        "pad_token_id": 0,
    }
    path = Path(directory) / "engram.json"
    path.write_text(json.dumps(assets), encoding="utf-8")
    return path


def _tiny_config(
        assets_path: Path,
        num_hidden_layers: int = 4,
) -> DeepseekV4Config:
    """Build a small shape-compatible V4.1 validation configuration."""
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=16,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        max_position_embeddings=128,
        layer_types=["sliding_attention"] * num_hidden_layers,
        mlp_layer_types=["moe"] * num_hidden_layers,
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 2},
        compress_rope_theta=10000.0,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1.0e-6,
        swiglu_limit=10.0,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=2,
        rms_norm_eps=1.0e-6,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        partial_rotary_factor=0.5,
    )
    config.architectures = ["DeepseekV41ForCausalLM"]
    config.v41_compress_ratios = [
        0 if layer_index < 2 else 2 if layer_index < 20 else 1
        for layer_index in range(num_hidden_layers)
    ]
    config.v41_kv_source_layer_ids = [
        layer_index for layer_index in (2, 8, 14, 20)
        if layer_index < num_hidden_layers
    ]
    config.v41_index_source_layer_ids = [
        layer_index for layer_index in (2, 8, 14, 20, 24, 28, 32, 36)
        if layer_index < num_hidden_layers
    ]
    if num_hidden_layers < 25:
        config.v41_index_source_layer_ids.append(num_hidden_layers - 1)
        config.v41_index_source_layer_ids = sorted(set(config.v41_index_source_layer_ids))
        config.v41_candidate_source_layer_id = config.v41_kv_source_layer_ids[-1]
    else:
        config.v41_candidate_source_layer_id = 20
    config.v41_candidate_topk_blocks = 1
    config.v41_candidate_block_size = 2
    config.v41_indexer_loss_coeff = 0.01
    config.v41_engram_layer_ids = list(assets["layer_ids"])
    config.v41_engram_num_embeddings = list(assets["num_embeddings"])
    config.v41_engram_bucket_base = 16
    config.v41_engram_assets_path = str(assets_path)
    config.v41_source_model_type = "deepseek_v41"
    config.v41_validation_crop = True
    config.v41_vision_enabled = False
    config.v41_vision_num_hidden_layers = 1
    config.v41_vision_hidden_size = 32
    config.v41_vision_patch_size = 2
    config.v41_vision_num_attention_heads = 4
    config.v41_vision_intermediate_size = 48
    config.v41_vision_rope_theta = 10000.0
    config.v41_vision_downsample_ratio = 3
    config.v41_vision_max_image_tokens = 16
    config.v41_vision_min_pixels = 4
    config.v41_vision_max_wh_ratio = None
    config.v41_image_token_id = 3
    config._attn_implementation = "eager"  # pylint: disable=protected-access
    return config


def _replace_v41_modules(model: DeepseekV41CroppedForCausalLM) -> None:
    """Apply the recipe's attention, mHC, and Engram replacements."""
    for layer_index, layer in enumerate(model.model.layers):
        layer.self_attn = replace_deepseek_v41_shared_attention(
            module=layer.self_attn,
            module_fqn=f"model.layers.{layer_index}.self_attn",
            context={},
        )
        layer.attn_hc = PipelinedMhcModule(module=layer.attn_hc)
        layer.ffn_hc = PipelinedMhcModule(module=layer.ffn_hc)
    for layer_index in model.config.v41_engram_layer_ids:
        engram = model.model.layers[layer_index].engram
        model.model.layers[layer_index].engram = EngramModule(module=engram)


class TestDeepseekV41EngramScaling(unittest.TestCase):
    """Scaled Engram tables retain the source hash-layout invariants."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_default_validation_table_has_unique_synchronized_buckets(self):
        """The 4-layer crop shrinks the active layer-1 table to 100,776 rows."""
        primes, table_sizes = build_scaled_engram_buckets(
            [1],
            bucket_base=4096,
            max_ngram_size=4,
            num_heads=8,
        )
        flattened = [value for ngram in primes[0] for value in ngram]
        self.assertEqual(table_sizes, [100776])
        self.assertEqual(sum(flattened), table_sizes[0])
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(value >= 4096 for value in flattened))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dead_token_blocks_all_older_ngram_history(self):
        """A masked token prevents longer n-grams from crossing its boundary."""
        with tempfile.TemporaryDirectory() as directory:
            assets = json.loads(_write_engram_assets(directory).read_text(encoding="utf-8"))
            mapping = NgramHashMapping(assets, layer_id=1)
            token_mask = torch.tensor([[True, True, False, True, True]])
            first = mapping(torch.tensor([[1, 2, 3, 4, 5]]), token_mask=token_mask)
            second = mapping(torch.tensor([[9, 10, 3, 4, 5]]), token_mask=token_mask)

            self.assertTrue(torch.equal(first[:, 2], second[:, 2]))
            self.assertTrue(torch.equal(first[:, 3, mapping.num_heads:], second[:, 3, mapping.num_heads:]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_family_and_custom_model_are_discovered_lazily(self):
        """AutoModel path selection discovers the family without preheating."""
        code = (
            "import sys\n"
            "from types import SimpleNamespace\n"
            "from hyper_parallel.models._transformers.config_resolver import get_is_hf_model\n"
            "from hyper_parallel.models import registry\n"
            "registration = 'hyper_parallel.models.deepseek_v41.adapter.registration'\n"
            "model_module = 'hyper_parallel.models.deepseek_v41.modeling_deepseek_v41'\n"
            "assert registration not in sys.modules\n"
            "assert model_module not in sys.modules\n"
            "config = SimpleNamespace(model_type='deepseek_v41', "
            "architectures=['DeepseekV41ForCausalLM'])\n"
            "assert get_is_hf_model(config) is False\n"
            "assert registration in sys.modules\n"
            "model_cls = registry._resolve_custom_model_cls(config.architectures[0])\n"
            "assert model_cls.__name__ == 'DeepseekV41CroppedForCausalLM'\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


class TestDeepseekV41CroppedModel(unittest.TestCase):
    """Engram and shared compressed attention execute together."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_v41_query_projection_has_no_post_qb_norm(self):
        """V4.1 sends wq_b output directly into RoPE, unlike inherited V4."""
        with tempfile.TemporaryDirectory() as directory:
            config = _tiny_config(_write_engram_assets(directory))
            placeholder = DeepseekV41AttentionPlaceholder(config, 0)
            self.assertFalse(hasattr(placeholder, "q_b_norm"))
            attention = replace_deepseek_v41_shared_attention(
                module=placeholder,
                module_fqn="model.layers.0.self_attn",
                context={},
            )
            hidden_states = torch.randn(1, 4, config.hidden_size)
            position_ids = torch.zeros(1, 4, dtype=torch.long)
            model = DeepseekV41CroppedForCausalLM(config)
            cos, sin = model.model.rotary_emb(
                hidden_states,
                position_ids=position_ids,
                layer_type="main",
            )
            query_residual, query = attention._project_query(  # pylint: disable=protected-access
                hidden_states,
                cos,
                sin,
            )
            expected = attention.q_b_proj(query_residual).view(
                1,
                4,
                -1,
                attention.head_dim,
            ).transpose(1, 2)

        self.assertFalse(hasattr(attention, "q_b_norm"))
        torch.testing.assert_close(query, expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ratio_one_compressor_has_no_gate_and_keeps_every_token(self):
        """The released ratio-one path is norm(wkv(x)), with no gate weight."""
        with tempfile.TemporaryDirectory() as directory:
            config = _tiny_config(_write_engram_assets(directory))
            model = DeepseekV41CroppedForCausalLM(config)
            compressor = DeepseekV41Compressor(config, compress_ratio=1)
            hidden_states = torch.randn(1, 5, config.hidden_size, requires_grad=True)
            position_ids = torch.arange(5).unsqueeze(0)
            position_embeddings = model.model.rotary_emb(
                hidden_states,
                position_ids=position_ids,
                layer_type="compress",
            )
            latent, rotated = compressor(hidden_states, position_embeddings)
            expected = compressor.norm(compressor.wkv(hidden_states))
            (latent.sum() + rotated.sum()).backward()

        self.assertFalse(hasattr(compressor, "wgate"))
        self.assertNotIn("wgate.weight", compressor.state_dict())
        self.assertEqual(latent.shape[1], hidden_states.shape[1])
        torch.testing.assert_close(latent, expected)
        self.assertGreater(compressor.wkv.weight.grad.norm().item(), 0.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_official_checkpoint_names_route_to_v41_structure(self):
        """V4.1-specific leaves avoid the incompatible V4 Indexer mapping."""
        register_deepseek_v41_checkpoint_mapping()
        with tempfile.TemporaryDirectory() as directory:
            config = _tiny_config(_write_engram_assets(directory))
            config.v41_vision_enabled = True
            model = DeepseekV41CroppedForCausalLM(config)
            mapping = get_model_conversion_mapping(model)
            renamings = [item for item in mapping if isinstance(item, WeightRenaming)]
            converters = [item for item in mapping if isinstance(item, WeightConverter)]
            targets = model.state_dict()
            examples = {
                "layers.0.attn.wq_b.weight": "model.layers.0.self_attn.q_b_proj.weight",
                "layers.2.attn.compressor.wkv.weight": (
                    "model.layers.2.self_attn.compressor.wkv.weight"
                ),
                "layers.2.attn.compressor.wgate.weight": (
                    "model.layers.2.self_attn.compressor.wgate.weight"
                ),
                "layers.2.attn.indexer.wq_b.weight": (
                    "model.layers.2.self_attn.indexer.q_b_proj.weight"
                ),
                "layers.1.engram.embed.weight": "model.layers.1.engram.embed.weight",
                "vision.blocks.0.attn.wqkv.weight": "model.vision.blocks.0.attn.wqkv.weight",
                "aligner.w1.weight": "model.aligner.w1.weight",
                "image_start": "model.image_start",
            }
            routed = {
                source: rename_source_key(
                    source,
                    renamings,
                    converters,
                    base_model_prefix=model.base_model_prefix,
                    meta_state_dict=targets,
                )[0]
                for source in examples
            }

        self.assertEqual(routed, examples)
        self.assertNotIn(
            "model.layers.2.self_attn.compressor.kv_proj.weight",
            routed.values(),
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_training_state_keeps_multiple_source_groups_addressable(self):
        """A later Full layer cannot overwrite an earlier source dependency."""
        state = SharedAttentionState()
        source_two = torch.randn(1, 2, 8, requires_grad=True)
        source_eight = torch.randn(1, 2, 8, requires_grad=True)
        state.publish_compressed_kv(2, source_two)
        state.publish_compressed_kv(8, source_eight)

        self.assertIs(state.require_compressed_kv(2, 7), source_two)
        self.assertIs(state.require_compressed_kv(8, 13), source_eight)
        with self.assertRaisesRegex(RuntimeError, "layer 1.*source layer None"):
            state.require_compressed_kv(None, 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_full_checkpoint_keeps_shared_attention_autograd_graph(self):
        """KV-sharing fallback recomputes MLP but keeps source attention live."""
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(23)
            model = DeepseekV41CroppedForCausalLM(
                _tiny_config(_write_engram_assets(directory))
            )
            _replace_v41_modules(model)
            _apply_activation_checkpointing(model, "full")
            source_attention = model.model.layers[2].self_attn
            output = model(
                input_ids=torch.randint(3, 64, (1, 8)),
                labels=torch.randint(3, 64, (1, 8)),
            )
            output.loss.backward()

        self.assertGreater(model.config.num_kv_shared_layers, 0)
        self.assertIsInstance(source_attention, SharedCompressedDSAAttention)
        self.assertTrue(hasattr(model.model.layers[2].mlp, "_wrapped_module"))
        self.assertGreater(source_attention.compressor.wkv.weight.grad.norm().item(), 0.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_shared_source_consumer_and_engram_receive_gradients(self):
        """Layer 2 publishes compressed KV and layer 3 consumes it in backward."""
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(11)
            model = DeepseekV41CroppedForCausalLM(_tiny_config(_write_engram_assets(directory)))
            source_keys = [
                set(layer.self_attn.state_dict())
                for layer in model.model.layers
            ]
            _replace_v41_modules(model)
            replacement_keys = [
                set(layer.self_attn.state_dict())
                for layer in model.model.layers
            ]
            self.assertEqual(source_keys, replacement_keys)
            input_ids = torch.randint(3, 64, (1, 8))
            output = model(input_ids=input_ids, labels=input_ids)
            output.loss.backward()

            source = model.model.layers[2].self_attn
            consumer = model.model.layers[3].self_attn
            self.assertIsInstance(source, DeepseekV41SharedCompressedAttention)
            self.assertIsInstance(consumer, DeepseekV41SharedCompressedAttention)
            self.assertIsInstance(source, SharedCompressedDSAAttention)
            self.assertIsInstance(source.indexer, SharedCompressedDSAIndexer)
            self.assertTrue(hasattr(source, "compressor"))
            self.assertTrue(hasattr(source, "indexer"))
            self.assertFalse(hasattr(consumer, "compressor"))
            self.assertTrue(hasattr(consumer, "indexer"))
            self.assertFalse(hasattr(consumer.indexer, "wk"))
            self.assertFalse(hasattr(consumer.indexer, "k_norm"))
            self.assertIsNotNone(model.model.layers[1].engram.embed.weight.grad)
            self.assertGreater(model.model.layers[1].engram.embed.weight.grad.norm().item(), 0.0)
            self.assertIsNotNone(source.compressor.wkv.weight.grad)
            self.assertGreater(source.compressor.wkv.weight.grad.norm().item(), 0.0)
            self.assertIsNotNone(source.indexer.q_b_proj.weight.grad)
            self.assertGreater(source.indexer.q_b_proj.weight.grad.norm().item(), 0.0)
            self.assertIsNotNone(consumer.indexer.q_b_proj.weight.grad)
            self.assertGreater(consumer.indexer.q_b_proj.weight.grad.norm().item(), 0.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_native_vision_span_injects_trainable_image_features(self):
        """A V4.1 image span reaches vision, aligner, router, and LM loss."""
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(17)
            config = _tiny_config(_write_engram_assets(directory))
            config.v41_vision_enabled = True
            model = DeepseekV41CroppedForCausalLM(config)
            image_boundaries = torch.stack([
                model.model.image_start,
                model.model.image_newline,
                model.model.image_end,
            ])
            self.assertTrue(torch.isfinite(image_boundaries).all())
            self.assertGreater(torch.count_nonzero(image_boundaries).item(), 0)
            self.assertLess(image_boundaries.abs().max().item(), 0.5)
            _replace_v41_modules(model)
            input_ids = torch.tensor([[5, 6, 3, 3, 3, 3, 7, 8]])
            token_types = torch.tensor([[-1, -1, 0, 1, 2, 3, -1, -1]])
            labels = input_ids.clone()
            labels[:, :6] = -100
            output = model(
                input_ids=input_ids,
                labels=labels,
                token_types=token_types,
                pixel_values=torch.randn(9, 3, 2, 2),
                image_patch_offsets=torch.tensor([0, 9]),
                image_vit_grid_hw=torch.tensor([[3, 3]]),
                image_llm_grid_hw=torch.tensor([[1, 1]]),
                image_batch_indices=torch.tensor([0]),
                image_token_starts=torch.tensor([2]),
                packed_seq_params=SharedCompressedPackedSequence(
                    cu_seq_lens=torch.tensor([0, 8]),
                    local_query_start=0,
                    local_query_length=8,
                    global_sequence_length=8,
                ),
            )
            output.loss.backward()

        self.assertIsNotNone(model.model.vision.patch_embed.proj.weight.grad)
        self.assertIsNotNone(model.model.aligner.w1.weight.grad)
        self.assertIsNotNone(model.model.image_start.grad)
        self.assertIsNotNone(model.model.layers[0].mlp.gate.bias_vl)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_vlm_batch_builds_v41_packed_sequence_from_right_padding(self):
        """VLM right-padding maps to V4.1 compact DSA metadata, not a dense mask."""
        get_batch = build_vlm_get_batch(
            mesh_context=SimpleNamespace(tp_size=1, cp_size=1, pp_size=1),
            device=torch.device("cpu"),
            attention_mode="compressed",
            cp_algorithm="colossal",
            runtime_input_adapter=DeepseekV41BatchAdapter(
                SimpleNamespace(v41_compress_ratios=[2], pad_token_id=0)
            ),
        )
        model_inputs, _ = get_batch(
            iter(()),
            external_batch={
                "input_ids": torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0]]),
                "labels": torch.tensor([[-100, -100, 3, 4, -100, -100, -100, -100]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]]),
            },
        )

        self.assertNotIn("attention_mask", model_inputs)
        packed = model_inputs["packed_seq_params"]
        self.assertIsInstance(packed, SharedCompressedPackedSequence)
        self.assertEqual(packed.cu_seq_lens.tolist(), [0, 4, 8])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_vlm_adapter_owns_variable_image_field_collation(self):
        """The model adapter merges V4.1 patches, offsets, grids, and ownership."""
        adapter = DeepseekV41VLMBatchAdapter(
            SimpleNamespace(v41_compress_ratios=[2], pad_token_id=0)
        )
        collator = VLMCollator(
            context=DataBatchContext(source_type="online"),
            batch_adapter=adapter,
        )
        first = {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([-100, -100, 3, 4]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "token_types": torch.tensor([-1, 0, 1, 3]),
            "pixel_values": torch.ones(2, 3, 2, 2),
            "image_patch_offsets": torch.tensor([0, 2]),
            "image_vit_grid_hw": torch.tensor([[1, 2]]),
            "image_llm_grid_hw": torch.tensor([[1, 1]]),
            "image_token_starts": torch.tensor([1]),
        }
        second = {
            "input_ids": torch.tensor([5, 6, 7, 8]),
            "labels": torch.tensor([-100, -100, 7, 8]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "token_types": torch.tensor([0, 1, 2, 3]),
            "pixel_values": torch.full((3, 3, 2, 2), 2.0),
            "image_patch_offsets": torch.tensor([0, 1, 3]),
            "image_vit_grid_hw": torch.tensor([[1, 1], [1, 2]]),
            "image_llm_grid_hw": torch.tensor([[1, 1], [1, 1]]),
            "image_token_starts": torch.tensor([0, 2]),
        }

        batch = collator([first, second])

        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["pixel_values"].shape, (5, 3, 2, 2))
        torch.testing.assert_close(batch["image_patch_offsets"], torch.tensor([0, 2, 3, 5]))
        torch.testing.assert_close(batch["image_batch_indices"], torch.tensor([0, 1, 1]))
        torch.testing.assert_close(batch["image_token_starts"], torch.tensor([1, 0, 2]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_packed_samples_do_not_share_attention_or_engram_context(self):
        """Compact boundaries isolate CSA2, sliding windows, and Engram n-grams."""
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(13)
            model = DeepseekV41CroppedForCausalLM(_tiny_config(_write_engram_assets(directory)))
            _replace_v41_modules(model)
            model.eval()
            first = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
            second = first.clone()
            second[:, :4] = torch.tensor([11, 12, 13, 14])
            packed = SharedCompressedPackedSequence(
                cu_seq_lens=torch.tensor([0, 4, 8], dtype=torch.int32),
                local_query_start=0,
                local_query_length=8,
                global_sequence_length=8,
            )
            first_output = model.model(input_ids=first, packed_seq_params=packed).last_hidden_state
            second_output = model.model(input_ids=second, packed_seq_params=packed).last_hidden_state

        torch.testing.assert_close(first_output[:, 4:], second_output[:, 4:])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_online_packing_aligns_every_sample_to_compressor_groups(self):
        """Online packing pads each sample boundary to the CSA2 ratio."""
        adapter = DeepseekV41BatchAdapter(
            SimpleNamespace(v41_compress_ratios=[0, 2], pad_token_id=0)
        )
        collator = TextPackingCollator(
            context=DataBatchContext(source_type="online"),
            batch_adapter=adapter,
        )
        batch = collator([
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
            {"input_ids": torch.tensor([5, 6]), "labels": torch.tensor([6, 7])},
        ])

        torch.testing.assert_close(batch["input_ids"], torch.tensor([[1, 2, 3, 0, 5, 6]]))
        torch.testing.assert_close(batch["labels"], torch.tensor([[2, 3, 4, -100, 6, 7]]))
        torch.testing.assert_close(batch["cu_seq_lens"], torch.tensor([0, 4, 6], dtype=torch.int32))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_online_recipe_declares_one_model_owned_batch_adapter(self):
        """One configured adapter owns packing and forward runtime extensions."""
        examples_dir = Path(__file__).resolve().parents[5] / "examples/training_demo"
        for recipe_name in (
                "train_deepseek_v41_online.yaml",
                "train_deepseek_v41_megamoe.yaml",
        ):
            with self.subTest(recipe=recipe_name):
                recipe = parse_training_args([str(examples_dir / recipe_name)])
                get_batch_target = recipe.dataloader.get_batch
                batch_adapter_target = recipe.dataloader.batch_adapter

                self.assertIs(get_batch_target._target_, ParallelBatch)  # pylint: disable=protected-access
                self.assertIsNone(get_batch_target.runtime_input_adapter)
                self.assertIsInstance(batch_adapter_target, Target)
                self.assertIs(  # pylint: disable=protected-access
                    batch_adapter_target._target_,
                    DeepseekV41BatchAdapter,
                )

        vlm_recipe = parse_training_args([
            str(examples_dir / "train_deepseek_v41_vlm_online.yaml")
        ])
        self.assertIs(  # pylint: disable=protected-access
            vlm_recipe.dataset.data_transform._target_,
            build_deepseek_v41_vlm_data_transform,
        )
        self.assertIs(  # pylint: disable=protected-access
            vlm_recipe.dataloader.batch_adapter._target_,
            DeepseekV41VLMBatchAdapter,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_recipes_leave_fsdp_output_dtype_unset(self):
        """Model recipes must not downcast the FP32 objective at the FSDP boundary."""
        examples_dir = Path(__file__).resolve().parents[5] / "examples/training_demo"
        for recipe_name in (
                "train_deepseek_v41_online.yaml",
                "train_deepseek_v41_vlm_online.yaml",
        ):
            with self.subTest(recipe=recipe_name):
                recipe = parse_training_args([str(examples_dir / recipe_name)])
                self.assertIsNone(recipe.fsdp_config.mix_precision.output_dtype)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_meta_materialization_restores_hash_buffers(self):
        """Non-persistent hash metadata survives the trainer's meta build path."""
        with tempfile.TemporaryDirectory() as directory:
            assets_path = _write_engram_assets(directory)
            config = _tiny_config(assets_path)
            config.v41_vision_enabled = True
            with ContextManagers([no_init_weights(), init_empty_weights()]):
                model = DeepseekV41CroppedForCausalLM(config)
            self.assertTrue(next(model.parameters()).is_meta)
            source_engram = model.model.layers[1].engram
            model.model.layers[1].engram = EngramModule(module=source_engram)
            _materialize_and_load_model(
                model,
                is_meta_device=True,
                device=torch.device("cpu"),
                load_base_model=False,
                pretrained_path=None,
                weights_mapping=None,
            )
            expected = torch.arange(64)
            engram = model.model.layers[1].engram
            self.assertIsInstance(engram, EngramModule)
            self.assertTrue(torch.equal(engram.q_weight, torch.ones_like(engram.q_weight)))
            self.assertTrue(torch.equal(engram.k_weight, torch.ones_like(engram.k_weight)))
            hash_mapping = engram.hash_mapping
            self.assertTrue(torch.equal(hash_mapping.token_map, expected))
            self.assertEqual(hash_mapping.primes.tolist(), [[17, 19], [23, 29]])
            hash_buffer_names = {"token_map", "primes", "offsets", "multipliers"}
            self.assertFalse(
                any(key.rsplit(".", maxsplit=1)[-1] in hash_buffer_names for key in model.state_dict())
            )
            padding_row = model.model.embed_tokens.weight[model.config.pad_token_id]
            self.assertEqual(torch.count_nonzero(padding_row).item(), 0)
            for image_boundary in (
                    model.model.image_start,
                    model.model.image_end,
                    model.model.image_newline,
            ):
                self.assertTrue(torch.isfinite(image_boundary).all())
                self.assertGreater(torch.count_nonzero(image_boundary).item(), 0)
                self.assertLess(image_boundary.abs().max().item(), 1.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_recipe_replacements_preserve_parameter_identities(self):
        """The YAML replacement rules atomically install mHC and Engram modules."""
        recipe_path = Path(__file__).resolve().parents[5] / (
            "examples/training_demo/train_deepseek_v41_online.yaml"
        )
        recipe = parse_training_args([str(recipe_path)])
        rules = entries_to_module_replacements(recipe.plan_overrides)
        with tempfile.TemporaryDirectory() as directory:
            model = DeepseekV41CroppedForCausalLM(_tiny_config(_write_engram_assets(directory)))
            engram_weight = model.model.layers[1].engram.embed.weight
            replacement_plan = compile_module_replacements(model, rules)
            apply_module_replacements(model, replacement_plan, weights_mapping=[])
        self.assertIsInstance(model.model.layers[0].attn_hc, PipelinedMhcModule)
        self.assertIsInstance(model.model.layers[1].engram, EngramModule)
        self.assertIs(model.model.layers[1].engram.embed.weight, engram_weight)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_engram_fsdp_units_separate_expert_table_from_dense_gate_weights(self):
        """Engram table and projection are child units while q/k stay dense-owned."""
        with tempfile.TemporaryDirectory() as directory:
            model = DeepseekV41CroppedForCausalLM(_tiny_config(_write_engram_assets(directory)))
            _replace_v41_modules(model)

        units = _get_fsdp_wrap_modules(model)

        self.assertIn("model.layers.1.engram.embed", units)
        self.assertIn("model.layers.1.engram.wkv", units)
        self.assertNotIn("model.layers.1.engram", units)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_full_depth_recipe_preserves_released_shared_attention_roles(self):
        """Wildcard rules cover every Full, Reindex, Reuse, and Engram layer."""
        class _FakeMesh:
            mesh_dim_names = ("dp_shard", "tp")
            mesh_shape = (2, 2)

        recipe_path = Path(__file__).resolve().parents[5] / (
            "examples/training_demo/train_deepseek_v41_online.yaml"
        )
        recipe = parse_training_args([str(recipe_path)])
        rules = entries_to_module_replacements(recipe.plan_overrides)
        with tempfile.TemporaryDirectory() as directory:
            assets_path = _write_engram_assets(
                directory,
                num_hidden_layers=40,
                layer_ids=(1, 14),
            )
            model = DeepseekV41CroppedForCausalLM(
                _tiny_config(assets_path, num_hidden_layers=40)
            )
            replacement_plan = compile_module_replacements(model, rules)
            apply_module_replacements(model, replacement_plan, weights_mapping=[])

        overrides = entries_to_plan_overrides(
            recipe.plan_overrides,
            cp_size=1,
            ep_size=4,
            sequence_parallel=True,
        )
        plan = ShardingPlanner(plan_overrides=overrides).plan(
            model,
            _FakeMesh(),
            tp_size=2,
            cp_size=1,
            ep_size=4,
            sequence_parallel=True,
            loss_parallel=False,
        )

        self.assertEqual(len(model.model.layers), 40)
        self.assertTrue(all(
            isinstance(layer.self_attn, SharedCompressedDSAAttention)
            for layer in model.model.layers
        ))
        self.assertTrue(all(
            isinstance(layer.attn_hc, PipelinedMhcModule)
            and isinstance(layer.ffn_hc, PipelinedMhcModule)
            for layer in model.model.layers
        ))
        for layer_index in (1, 14):
            self.assertIsInstance(model.model.layers[layer_index].engram, EngramModule)

        expected_mhc_fqns = {
            f"model.layers.{layer_index}.{name}"
            for layer_index in range(40)
            for name in ("attn_hc", "ffn_hc")
        }
        actual_mhc_fqns = {
            module_fqn
            for module_fqn in plan.modules
            if module_fqn.endswith((".attn_hc", ".ffn_hc"))
        }
        self.assertEqual(actual_mhc_fqns, expected_mhc_fqns)
        self.assertEqual(
            {
                module_fqn
                for module_fqn in plan.modules
                if module_fqn.endswith(".engram")
            },
            {"model.layers.1.engram", "model.layers.14.engram"},
        )
        mhc_spec = plan.modules["model.layers.39.ffn_hc"]
        self.assertEqual(mhc_spec.in_src["hidden_streams"][TP], Shard(1))
        self.assertEqual(
            tuple(mhc_spec.out_names or mhc_spec.out_src),
            ("pre", "post", "residual"),
        )

        full_layer = model.model.layers[20].self_attn
        reindex_layer = model.model.layers[24].self_attn
        reuse_layer = model.model.layers[25].self_attn
        final_reuse_layer = model.model.layers[39].self_attn
        self.assertTrue(full_layer.is_kv_source)
        self.assertTrue(full_layer.is_index_source)
        self.assertTrue(full_layer.indexer.is_candidate_source)
        self.assertFalse(reindex_layer.is_kv_source)
        self.assertTrue(reindex_layer.is_index_source)
        self.assertTrue(reindex_layer.indexer.uses_candidates)
        self.assertFalse(hasattr(reindex_layer.indexer, "wk"))
        self.assertFalse(reuse_layer.is_kv_source)
        self.assertFalse(reuse_layer.is_index_source)
        self.assertEqual(reuse_layer.kv_source_layer_idx, 20)
        self.assertEqual(reuse_layer.index_source_layer_idx, 24)
        self.assertEqual(final_reuse_layer.kv_source_layer_idx, 20)
        self.assertEqual(final_reuse_layer.index_source_layer_idx, 36)

        _apply_activation_checkpointing(model, "full")
        self.assertIs(model.model.layers[20].self_attn, full_layer)
        self.assertTrue(all(
            hasattr(layer.mlp, "_wrapped_module") for layer in model.model.layers
        ))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_online_recipe_covers_v41_only_parameters_for_tp2_ep4(self):
        """The YAML recipe owns every mHC and Engram boundary for TP2/EP4."""
        class _FakeMesh:
            mesh_dim_names = ("dp_shard", "tp")
            mesh_shape = (2, 2)

        recipe_path = Path(__file__).resolve().parents[5] / (
            "examples/training_demo/train_deepseek_v41_online.yaml"
        )
        recipe = parse_training_args([str(recipe_path)])
        overrides = entries_to_plan_overrides(
            recipe.plan_overrides,
            cp_size=1,
            ep_size=4,
        )
        self.assertIn("model.layers.*.*_hc", overrides)
        self.assertIn("model.layers.*.engram", overrides)
        overrides["model.layers.*.attn_hc"] = ModuleShardingSpec(
            out_names=["pre", "post", "residual"]
        )
        with tempfile.TemporaryDirectory() as directory:
            config = _tiny_config(_write_engram_assets(directory))
            with ContextManagers([no_init_weights(), init_empty_weights()]):
                model = DeepseekV41CroppedForCausalLM(config)
            _replace_v41_modules(model)
            plan = ShardingPlanner(plan_overrides=overrides).plan(
                model,
                _FakeMesh(),
                tp_size=2,
                cp_size=1,
                ep_size=4,
                sequence_parallel=False,
                loss_parallel=False,
            )

        v41_boundaries = [
            *(f"model.layers.{layer_id}.{name}"
              for layer_id in range(4) for name in ("attn_hc", "ffn_hc")),
            "model.layers.1.engram",
        ]
        for boundary in v41_boundaries[:-1]:
            self.assertIn(boundary, plan.modules)
            self.assertTrue(plan.modules[boundary].params)
            for placement in plan.modules[boundary].params.values():
                self.assertEqual(placement[TP], Replicate())
        self.assertEqual(
            plan.modules["model.layers.3.attn_hc"].out_names,
            ["pre", "post", "residual"],
        )
        engram_spec = plan.modules["model.layers.1.engram"]
        self.assertEqual(engram_spec._ep_size, 4)  # pylint: disable=protected-access
        self.assertIsInstance(engram_spec.local_compute_fn, Target)
        self.assertIs(  # pylint: disable=protected-access
            engram_spec.local_compute_fn._target_,
            deepseek_v41_engram_compute_fn,
        )
        self.assertEqual(engram_spec.params["embed.weight"][EP], Shard(0))
        for parameter_name in ("q_weight", "k_weight", "wkv.weight"):
            self.assertEqual(engram_spec.params[parameter_name][TP], Replicate())
        attention_spec = plan.modules["model.layers.2.self_attn"]
        for parameter_name in (
                "indexer.q_b_proj.weight",
                "indexer.weights_proj.weight",
        ):
            self.assertEqual(attention_spec.params[parameter_name][TP], Shard(0))
        for parameter_name in ("indexer.wk.weight",):
            self.assertEqual(attention_spec.params[parameter_name][TP], Replicate())
        index_norm_spec = plan.modules["model.layers.2.self_attn.indexer.k_norm"]
        self.assertEqual(index_norm_spec.params["weight"][TP], Replicate())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_engram_tp_sequence_slice_keeps_left_ngram_context(self):
        """TP sequence rank one hashes the same rows as the full sequence."""
        with tempfile.TemporaryDirectory() as directory:
            model = DeepseekV41CroppedForCausalLM(_tiny_config(_write_engram_assets(directory)))
            _replace_v41_modules(model)
            engram = model.model.layers[1].engram
            input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
            full_hashes = engram.hash_mapping(input_ids)
            hidden = torch.zeros(1, 4, 2, 32)
            local_hashes = engram._aligned_hash_ids(  # pylint: disable=protected-access
                hidden,
                input_ids,
                None,
                tp_rank=1,
                tp_size=2,
            )
            torch.testing.assert_close(local_hashes, full_hashes[:, 4:])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_shared_attention_cp_uses_global_kv_and_offsets(self):
        """CP keeps local queries while gathering raw, compressed, and index KV."""
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(19)
            model = DeepseekV41CroppedForCausalLM(
                _tiny_config(_write_engram_assets(directory))
            )
            _replace_v41_modules(model)
            attention = model.model.layers[2].self_attn
            hidden_states = torch.randn(1, 4, model.config.hidden_size)
            position_ids = torch.arange(4, 8).unsqueeze(0)
            position_embeddings = {
                "main": model.model.rotary_emb(
                    hidden_states,
                    position_ids=position_ids,
                    layer_type="main",
                ),
                "compress": model.model.rotary_emb(
                    hidden_states,
                    position_ids=position_ids,
                    layer_type="compress",
                ),
            }
            gathered_dims = []

            def gather_sequence(tensor: torch.Tensor, sequence_dim: int) -> torch.Tensor:
                """Stand in for CP rank zero followed by the current rank-one shard."""
                gathered_dims.append(sequence_dim)
                return torch.cat([torch.zeros_like(tensor), tensor], dim=sequence_dim)

            cp_context = SharedAttentionCPContext(
                size=2,
                rank=1,
                gather_sequence=gather_sequence,
            )
            shared_state = SharedAttentionState()
            output, _ = attention(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=None,
                shared_attention_state=shared_state,
                shared_attention_cp_context=cp_context,
            )

        self.assertEqual(output.shape, hidden_states.shape)
        # Index K is consumed first. Raw and compressed KV were already
        # launched and wait only after local indexer work has completed.
        self.assertEqual(gathered_dims, [1, 2, 1])
        compressed_kv = shared_state.require_compressed_kv(2, 2)
        topk_indices = shared_state.require_topk_indices(2, 2)
        self.assertEqual(compressed_kv.shape[1], 4)
        self.assertEqual(topk_indices.shape[:2], (1, 4))
        thresholds = torch.arange(5, 9) // attention.compress_ratio
        for query_index, threshold in enumerate(thresholds):
            selected = topk_indices[0, query_index]
            selected = selected[selected >= 0]
            self.assertTrue(torch.all(selected < threshold))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_compressed_indexer_uses_ratio_aware_causal_boundary(self):
        """A compressed key becomes visible only after its source group closes."""
        query = torch.ones(1, 6, 2, 4)
        key = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0],
              [2.0, 0.0, 0.0, 0.0],
              [3.0, 0.0, 0.0, 0.0]]]
        )
        merge_weight = torch.ones(1, 6, 2)
        indices = compressed_causal_topk(
            query,
            key,
            merge_weight,
            compress_ratio=2,
            sparse_count=2,
            query_chunk_size=2,
        )

        expected = [set(), {0}, {0}, {0, 1}, {0, 1}, {1, 2}]
        for query_index, expected_indices in enumerate(expected):
            actual = indices[0, query_index]
            self.assertEqual(set(actual[actual >= 0].tolist()), expected_indices)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_hierarchical_candidate_blocks_pin_newest_reachable_block(self):
        """Candidate selection uses block maxima and pins the partial tail."""
        logits = torch.tensor(
            [[[
                10.0, 9.0,
                8.0, 7.0,
                1.0, 0.0,
                float("-inf"), float("-inf"),
            ]]]
        )
        candidates = select_candidate_blocks(
            logits,
            compress_lens=torch.tensor([[[6]]]),
            topk_blocks=2,
            block_size=2,
        )
        expected = torch.tensor([[[True, True, False, False, True, True, False, False]]])
        torch.testing.assert_close(candidates, expected)

        compact = select_candidate_block_indices(
            logits,
            compress_lens=torch.tensor([[[6]]]),
            topk_blocks=2,
            block_size=2,
        )
        self.assertEqual(set(compact.flatten().tolist()), {0, 2})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_reindex_scores_only_compact_candidate_blocks(self):
        """Reindex returns global ids from the compact candidate subset."""
        query = torch.ones(1, 12, 1, 1)
        key = torch.tensor([[[1.0], [10.0], [9.0], [8.0], [7.0], [6.0]]])
        merge_weight = torch.ones(1, 12, 1)
        candidate_blocks = torch.tensor([[[0, 2]] * 12], dtype=torch.int32)
        indices = compressed_candidate_topk(
            query,
            key,
            merge_weight,
            candidate_blocks,
            compress_ratio=2,
            sparse_count=2,
            block_size=2,
            query_chunk_size=3,
        )

        self.assertEqual(set(indices[0, -1].tolist()), {1, 4})
        for query_index, selected in enumerate(indices[0]):
            visible = (query_index + 1) // 2
            selected = selected[selected >= 0]
            self.assertTrue(torch.all(selected < visible))
            self.assertTrue(set(selected.tolist()).issubset({0, 1, 4, 5}))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_compressed_indexer_kl_updates_only_indexer_inputs(self):
        """PanGu-style sparse KL detaches the main-attention teacher."""
        torch.manual_seed(23)
        index_query = torch.randn(1, 4, 2, 3, requires_grad=True)
        index_key = torch.randn(1, 4, 3, requires_grad=True)
        merge_weight = torch.randn(1, 4, 2, requires_grad=True)
        attention_query = torch.randn(1, 2, 4, 5, requires_grad=True)
        compressed_key = torch.randn(1, 4, 5, requires_grad=True)
        topk_indices = torch.tensor([[[-1, -1], [0, -1], [0, 1], [1, 2]]])
        sinks = torch.zeros(2, requires_grad=True)
        loss = shared_compressed_indexer_kl_loss(
            index_query,
            index_key,
            merge_weight,
            attention_query,
            compressed_key,
            topk_indices,
            sinks,
            attention_scale=5**-0.5,
            loss_coeff=0.1,
            query_chunk_size=2,
        )
        loss.backward()

        self.assertGreater(loss.item(), 0.0)
        for tensor in (index_query, index_key, merge_weight):
            self.assertIsNotNone(tensor.grad)
            self.assertGreater(tensor.grad.norm().item(), 0.0)
        for tensor in (attention_query, compressed_key, sinks):
            self.assertIsNone(tensor.grad)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_compressed_indexer_kl_matches_direct_autograd(self):
        """Precomputed PanGu-style gradients match a direct sparse KL graph."""
        torch.manual_seed(29)
        source_tensors = (
            torch.randn(1, 4, 2, 3),
            torch.randn(1, 4, 3),
            torch.randn(1, 4, 2),
        )
        custom_inputs = [tensor.clone().requires_grad_() for tensor in source_tensors]
        reference_inputs = [tensor.clone().requires_grad_() for tensor in source_tensors]
        attention_query = torch.randn(1, 2, 4, 5)
        compressed_key = torch.randn(1, 4, 5)
        topk_indices = torch.tensor([[[-1, -1], [0, -1], [0, 1], [1, 2]]])
        sinks = torch.randn(2)
        scale = 5**-0.5
        coefficient = 0.13

        custom_loss = shared_compressed_indexer_kl_loss(
            *custom_inputs,
            attention_query,
            compressed_key,
            topk_indices,
            sinks,
            attention_scale=scale,
            loss_coeff=coefficient,
            query_chunk_size=2,
        )

        index_query, index_key, merge_weight = reference_inputs
        valid = topk_indices >= 0
        safe_indices = topk_indices.clamp_min(0).long()
        batch_indices = torch.arange(index_query.shape[0]).view(-1, 1, 1)
        selected_index_key = index_key[batch_indices, safe_indices]
        index_dots = torch.einsum("bsid,bskd->bsik", index_query, selected_index_key)
        index_scores = (index_dots.relu() * merge_weight.unsqueeze(-1)).sum(dim=2)
        index_scores = index_scores.masked_fill(~valid, -1.0e9)
        selected_attention_key = compressed_key[batch_indices, safe_indices]
        attention_scores = torch.einsum(
            "bhsd,bskd->bhsk",
            attention_query,
            selected_attention_key,
        ) * scale
        attention_scores = attention_scores.masked_fill(~valid.unsqueeze(1), -1.0e9)
        sink_logits = sinks.view(1, -1, 1, 1).expand(1, -1, index_query.shape[1], -1)
        target = torch.cat((attention_scores, sink_logits), dim=-1).softmax(dim=-1)
        target = target[..., :-1].masked_fill(~valid.unsqueeze(1), 0.0).sum(dim=1)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        row_loss = F.kl_div(
            index_scores.log_softmax(dim=-1),
            target,
            reduction="none",
        ).sum(dim=-1)
        reference_loss = row_loss[valid.any(dim=-1)].sum() * (
            coefficient / (index_query.shape[0] * index_query.shape[1])
        )

        custom_loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(custom_loss, reference_loss, rtol=1.0e-5, atol=1.0e-6)
        for custom, reference in zip(custom_inputs, reference_inputs):
            torch.testing.assert_close(custom.grad, reference.grad, rtol=1.0e-5, atol=1.0e-6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_window_indices_use_global_query_offset(self):
        """Rank-one local queries address the preceding global sliding window."""
        indices = _window_indices(
            batch_size=1,
            sequence_length=4,
            window_size=4,
            device=torch.device("cpu"),
            query_offset=4,
            key_length=8,
        )
        expected = torch.tensor(
            [[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]]
        )
        torch.testing.assert_close(indices, expected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_online_recipe_injects_v41_cp_wrapper(self):
        """An active CP axis selects the V4.1 wrapper on every attention boundary."""
        class _FakeMesh:
            mesh_dim_names = ("dp_shard", "cp")
            mesh_shape = (2, 2)

        recipe_path = Path(__file__).resolve().parents[5] / (
            "examples/training_demo/train_deepseek_v41_online.yaml"
        )
        recipe = parse_training_args([str(recipe_path)])
        overrides = entries_to_plan_overrides(
            recipe.plan_overrides,
            cp_size=2,
            ep_size=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            config = _tiny_config(_write_engram_assets(directory))
            with ContextManagers([no_init_weights(), init_empty_weights()]):
                model = DeepseekV41CroppedForCausalLM(config)
            _replace_v41_modules(model)
            plan = ShardingPlanner(plan_overrides=overrides).plan(
                model,
                _FakeMesh(),
                tp_size=1,
                cp_size=2,
                ep_size=4,
                sequence_parallel=False,
                loss_parallel=False,
            )

        for layer_index in range(4):
            spec = plan.modules[f"model.layers.{layer_index}.self_attn"]
            self.assertFalse(spec.region_dispatch)
            self.assertEqual(spec.inner_target, "self")
            self.assertIsNotNone(spec.inner_wrapper)


class TestDeepseekV41ExpertParallel(unittest.TestCase):
    """The EP local expert preserves DeepSeek's clamped SwiGLU."""

    @staticmethod
    def _build_signature_fixture(multimodal: bool) -> tuple[nn.Module, object]:
        """Create the smallest MoE and mesh accepted by the EP factory."""
        class _Experts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.num_experts = 1
                self.act_fn = F.silu

            @staticmethod
            def _apply_gate(gate_up: torch.Tensor) -> torch.Tensor:
                return gate_up

        class _TextMoe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = nn.Identity()
                self.experts = _Experts()
                self.shared_experts = nn.Identity()
                self.is_hash = False

            def forward(
                    self,
                    hidden_states: torch.Tensor,
                    input_ids: torch.Tensor | None = None,
            ) -> torch.Tensor:
                del input_ids
                return hidden_states

        class _MultimodalMoe(_TextMoe):
            def forward(
                    self,
                    hidden_states: torch.Tensor,
                    input_ids: torch.Tensor | None = None,
                    image_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                del input_ids, image_mask
                return hidden_states

        class _EpAxis:
            @staticmethod
            def size() -> int:
                return 1

        class _EpMesh:
            @staticmethod
            def get_group(name: str) -> None:
                del name
                return None

            def __getitem__(self, name: str) -> _EpAxis:
                del self
                del name
                return _EpAxis()

        return (_MultimodalMoe() if multimodal else _TextMoe()), _EpMesh()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ep_factory_matches_text_and_multimodal_forward_signatures(self):
        """The EP closure exposes only arguments accepted by its source MoE."""
        expected_parameters = {
            False: ["module", "hidden_states", "input_ids"],
            True: ["module", "hidden_states", "input_ids", "image_mask"],
        }
        for multimodal, expected in expected_parameters.items():
            with self.subTest(multimodal=multimodal):
                module, ep_mesh = self._build_signature_fixture(multimodal)
                compute_fn = deepseek_v41_ep_compute_fn(
                    module=module,
                    mesh=None,
                    tp_mesh=None,
                    cp_mesh=None,
                    ep_mesh=ep_mesh,
                )
                actual = list(inspect.signature(compute_fn).parameters)
                self.assertEqual(actual, expected)
                self.assertIsNone(
                    validate_local_compute_signature(
                        compute_fn,
                        module.forward,
                        owner="DeepSeek-V4.1 EP signature test",
                    )
                )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ep_factory_enables_grouped_gemm_with_model_specific_gate(self):
        """The V4.1 factory preserves its clamp hook in grouped-GEMM mode."""
        module, ep_mesh = self._build_signature_fixture(multimodal=False)
        deepseek_v41_ep_compute_fn(
            module=module,
            mesh=None,
            tp_mesh=None,
            cp_mesh=None,
            ep_mesh=ep_mesh,
            use_grouped_gemm=True,
        )
        self.assertTrue(module.experts._ep_use_grouped_gemm)  # pylint: disable=protected-access
        self.assertIs(module.experts._ep_apply_gate, module.experts._apply_gate)  # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    @patch("hyper_parallel.distributed.expert_parallel.experts.npu_grouped_swiglu")
    def test_grouped_expert_uses_model_specific_gate(self, grouped_swiglu):
        """The grouped local expert forwards the V4 clamp hook to its kernel helper."""
        class _Experts(nn.Module):
            def __init__(self) -> None:
                """Create a two-expert packed-weight holder."""
                super().__init__()
                self.num_experts = 2
                self.act_fn = F.silu
                self.gate_up_proj = nn.Parameter(torch.ones(2, 2, 1))
                self.down_proj = nn.Parameter(torch.ones(2, 1, 1))

        class _Moe(nn.Module):
            def __init__(self) -> None:
                """Wrap the expert holder for the EP binder."""
                super().__init__()
                self.experts = _Experts()

        def apply_clamped_gate(gate_up: torch.Tensor) -> torch.Tensor:
            """Stand in for DeepSeek-V4.1's model-specific clamp."""
            return gate_up[..., :1]

        module = _Moe()
        grouped_swiglu.return_value = torch.tensor([[11.0], [22.0]])
        bind_local_expert_forward(
            module,
            ep_size=1,
            use_grouped_gemm=True,
            apply_gate=apply_clamped_gate,
        )
        hidden_states = torch.tensor([[3.0], [4.0]])
        expert_indices = torch.tensor([1, 0])
        output = module.experts(hidden_states, expert_indices)

        grouped_swiglu.assert_called_once()
        call = grouped_swiglu.call_args
        self.assertIs(call.kwargs["apply_gate"], apply_clamped_gate)
        torch.testing.assert_close(output, torch.tensor([[22.0], [11.0]]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fused_expert_uses_model_specific_gate(self):
        """The generic local expert calls the V4 clamp hook when supplied."""
        class _Experts(nn.Module):
            def __init__(self) -> None:
                """Create one deterministic fused expert."""
                super().__init__()
                self.num_experts = 1
                self.act_fn = F.silu
                self.gate_up_proj = nn.Parameter(torch.tensor([[[2.0], [-2.0]]]))
                self.down_proj = nn.Parameter(torch.ones(1, 1, 1))

        class _Moe(nn.Module):
            def __init__(self) -> None:
                """Wrap the expert holder for the EP binder."""
                super().__init__()
                self.experts = _Experts()

        def apply_clamped_gate(gate_up: torch.Tensor) -> torch.Tensor:
            """Apply the model-specific clamp before SwiGLU multiplication."""
            gate, up = gate_up.chunk(2, dim=-1)
            return F.silu(gate.clamp(max=1.0)) * up.clamp(min=-1.0, max=1.0)

        module = _Moe()
        bind_local_expert_forward(module, ep_size=1, apply_gate=apply_clamped_gate)
        hidden_states = torch.tensor([[2.0], [-3.0]])
        expert_indices = torch.zeros(2, dtype=torch.long)
        output = module.experts(hidden_states, expert_indices)
        gate_up = F.linear(  # pylint: disable=not-callable
            hidden_states, module.experts.gate_up_proj[0]
        )
        expected = F.linear(  # pylint: disable=not-callable
            apply_clamped_gate(gate_up), module.experts.down_proj[0]
        )
        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
