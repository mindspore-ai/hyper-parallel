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
import tempfile
import unittest
from pathlib import Path
from typing import Optional

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
from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.components.modules.mhc import PipelinedMhcModule
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedDSAAttention,
    compressed_candidate_topk,
    compressed_causal_topk,
    select_candidate_block_indices,
    select_candidate_blocks,
    shared_compressed_indexer_kl_loss,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.data.batching import (
    OmniPackingLoader,
    OmniParallelBatch,
    TokenBatchLoader,
    build_online_text_collate_fn,
)
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
from hyper_parallel.models.deepseek_v41.adapter.data.runtime import DeepseekV41Runtime
from hyper_parallel.models.deepseek_v41.adapter.data.transform_fn import (
    build_deepseek_v41_omni_transform,
)
from hyper_parallel.models.deepseek_v41.adapter.distributed.moe_engram_expert_parallel import (
    deepseek_v41_engram_compute_fn,
    deepseek_v41_ep_compute_fn,
)
from hyper_parallel.models.deepseek_v41.adapter.policies.sharding import (
    get_fsdp_wrap_modules,
)
from hyper_parallel.models.deepseek_v41.adapter.registration import (
    DEEPSEEK_V41_ADAPTER_SPEC,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41Attention,
    DeepseekV41Engram,
    DeepseekV41ForCausalLM,
    DeepseekV41PipelinedHyperConnection,
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
from hyper_parallel.trainer.config.parser import parse_training_args
from tests.common.mark_utils import arg_mark


def _write_engram_assets(
        directory: str,
        num_hidden_layers: int = 4,
        layer_ids: tuple[int, ...] = (1,),
        head_dim: int = 4,
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
        "head_dim": head_dim,
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


def _write_released_config(directory: str) -> Path:
    """Write a release-shaped config whose dimensions have exact crop ratios."""
    text_config = {
        "vocab_size": 64,
        "hidden_size": 256,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "qk_rope_head_dim": 32,
        "q_lora_rank": 128,
        "o_lora_rank": 128,
        "o_groups": 8,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1.0e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "sliding_window": 128,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        "compress_rope_theta": 160000.0,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32,
        "index_head_dim": 64,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1.0e-6,
        "engram_layer_ids": [1, 14],
        "engram_head_dim": 32,
    }
    source = {
        "model_type": "deepseek_v41",
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "image_token_id": 3,
        "text_config": text_config,
        "vision_config": {
            "num_hidden_layers": 32,
            "hidden_size": 128,
            "num_attention_heads": 16,
            "intermediate_size": 192,
            "patch_size": 14,
            "rope_theta": 10000.0,
            "downsample_ratio": 3,
            "max_image_tokens": 1024,
            "min_pixels": 295936,
            "max_wh_ratio": None,
        },
    }
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(source), encoding="utf-8")
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
    config.v41_model_mode = "validation_crop"
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


class TestDeepseekV41EngramScaling(unittest.TestCase):
    """Scaled Engram tables retain the source hash-layout invariants."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_online_recipes_wire_omni_data_pipeline(self):
        """Validate the released DeepSeek text and VLM recipes.

        Feature: Omni data lifecycle recipe wiring.
        Description: Resolve both recipes through the typed Trainer config.
        Expectation: Generic loaders and model runtime metadata match each modality.
        """
        recipe_path = Path(__file__).resolve().parents[5] / (
            "examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml"
        )

        recipe = parse_training_args([str(recipe_path)])

        self.assertEqual(recipe.activation_checkpoint.selection.layer_count, 0)
        self.assertIsNone(recipe.activation_checkpoint.selection.layer_indices)
        self.assertIs(  # pylint: disable=protected-access
            recipe.dataloader._target_,
            TokenBatchLoader,
        )
        self.assertIs(  # pylint: disable=protected-access
            recipe.dataloader.collate_fn._target_,
            build_online_text_collate_fn,
        )
        text_runtime = recipe.dataloader.get_batch.runtime_input_adapter.build()
        self.assertEqual(text_runtime.runtime_input_fields(), ("packed_seq_params",))

        vlm_recipe = parse_training_args([
            str(recipe_path.with_name("train_deepseek_v41_vlm_online.yaml"))
        ])
        self.assertIs(  # pylint: disable=protected-access
            vlm_recipe.dataset.data_transform._target_,
            build_deepseek_v41_omni_transform,
        )
        self.assertIs(  # pylint: disable=protected-access
            vlm_recipe.dataloader._target_,
            OmniPackingLoader,
        )
        self.assertIs(  # pylint: disable=protected-access
            vlm_recipe.dataloader.get_batch._target_,
            OmniParallelBatch,
        )
        self.assertIs(  # pylint: disable=protected-access
            vlm_recipe.dataloader.get_batch.runtime_input_adapter._target_,
            DeepseekV41Runtime,
        )
        vlm_runtime = vlm_recipe.dataloader.get_batch.runtime_input_adapter.build()
        self.assertEqual(
            vlm_runtime.runtime_input_fields(),
            ("packed_seq_params", "position_ids", "image_sequence_start"),
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_recipes_leave_fsdp_output_dtype_unset(self):
        """Keep the FP32 objective outside FSDP output casting.

        Feature: DeepSeek recipe precision policy.
        Description: Resolve the text and VLM mixed-precision configurations.
        Expectation: Neither recipe declares an FSDP output dtype.
        """
        examples_dir = (
            Path(__file__).resolve().parents[5]
            / "examples/training_demo/deepseek_v41"
        )
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
        """Restore hash metadata through meta-device materialization.

        Feature: Engram meta initialization.
        Description: Materialize a model after wrapping its Engram child module.
        Expectation: Non-persistent hash buffers match the source assets.
        """
        with tempfile.TemporaryDirectory() as directory:
            assets_path = _write_engram_assets(directory)
            config = _tiny_config(assets_path)
            config.v41_vision_enabled = True
            with ContextManagers([no_init_weights(), init_empty_weights()]):
                model = DeepseekV41ForCausalLM(config)
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
    def test_model_directly_constructs_canonical_modules(self):
        """Construct the canonical architecture without replacement rules.

        Feature: Native DeepSeek V4.1 module construction.
        Description: Build the cropped model before applying optional optimizations.
        Expectation: Every specialized module uses its canonical model-owned type.
        """
        with tempfile.TemporaryDirectory() as directory:
            model = DeepseekV41ForCausalLM(_tiny_config(_write_engram_assets(directory)))
        self.assertIsInstance(
            model.model.layers[0].attn_hc,
            DeepseekV41PipelinedHyperConnection,
        )
        self.assertNotIsInstance(model.model.layers[0].attn_hc, PipelinedMhcModule)
        self.assertIsInstance(model.model.layers[0].self_attn, DeepseekV41Attention)
        self.assertNotIsInstance(model.model.layers[0].self_attn, SharedCompressedDSAAttention)
        self.assertIsInstance(model.model.layers[1].engram, DeepseekV41Engram)
        self.assertNotIsInstance(model.model.layers[1].engram, EngramModule)
        self.assertTrue(torch.equal(model.model.layers[0].self_attn.sinks, torch.zeros(4)))
        self.assertTrue(torch.equal(model.model.layers[0].attn_hc.base, torch.zeros(8)))
        self.assertTrue(torch.equal(model.model.layers[0].attn_hc.scale, torch.ones(3)))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_optional_replacements_preserve_state_keys_and_parameters(self):
        """Preserve model identity across optional module replacements.

        Feature: DeepSeek optimization replacements.
        Description: Compare state, outputs, and gradients before and after replacement.
        Expectation: Replacement modules preserve all observable model values.
        """
        recipe_path = Path(__file__).resolve().parents[5] / (
            "examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml"
        )
        recipe = parse_training_args([str(recipe_path)])
        with tempfile.TemporaryDirectory() as directory:
            model = DeepseekV41ForCausalLM(_tiny_config(_write_engram_assets(directory)))

        model.eval()
        input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
        reference_output = model(input_ids=input_ids, labels=input_ids)
        reference_output.loss.backward()
        gradient_names = (
            "model.layers.0.attn_hc.fn",
            "model.layers.1.engram.wkv.weight",
            "model.layers.2.self_attn.q_a_proj.weight",
        )
        reference_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if name in gradient_names
        }
        self.assertEqual(set(reference_gradients), set(gradient_names))
        model.zero_grad(set_to_none=True)
        state_keys = set(model.state_dict())
        parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
        attention_entries = [
            entry
            for entry in recipe.plan_overrides
            if entry.module_type
            == "hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.DeepseekV41Attention"
        ]
        self.assertEqual(len(attention_entries), 1)
        self.assertEqual(
            attention_entries[0].replace_module.to_dict()["_target_"],
            "hyper_parallel.components.modules.shared_compressed_dsa_attention."
            "SharedCompressedDSAAttention",
        )
        self.assertIsNone(DEEPSEEK_V41_ADAPTER_SPEC.replacements)
        rules = entries_to_module_replacements(recipe.plan_overrides)
        replacement_plan = compile_module_replacements(model, rules)
        apply_module_replacements(model, replacement_plan)

        candidate_output = model(input_ids=input_ids, labels=input_ids)
        candidate_output.loss.backward()

        self.assertEqual(set(model.state_dict()), state_keys)
        self.assertEqual(
            {name: id(parameter) for name, parameter in model.named_parameters()},
            parameter_ids,
        )
        self.assertIsInstance(model.model.layers[0].attn_hc, PipelinedMhcModule)
        self.assertIsInstance(model.model.layers[0].self_attn, SharedCompressedDSAAttention)
        self.assertIsInstance(model.model.layers[1].engram, EngramModule)
        torch.testing.assert_close(candidate_output.logits, reference_output.logits)
        torch.testing.assert_close(candidate_output.loss, reference_output.loss)
        candidate_parameters = dict(model.named_parameters())
        for name, reference_gradient in reference_gradients.items():
            torch.testing.assert_close(candidate_parameters[name].grad, reference_gradient)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_engram_fsdp_units_separate_expert_table_from_dense_gate_weights(self):
        """Separate Engram expert tables from dense gate weights.

        Feature: Engram FSDP unit selection.
        Description: Resolve wrap units for a model containing an Engram layer.
        Expectation: Table and projection children are units while q/k remain dense-owned.
        """
        with tempfile.TemporaryDirectory() as directory:
            model = DeepseekV41ForCausalLM(_tiny_config(_write_engram_assets(directory)))
        units = get_fsdp_wrap_modules(model)

        self.assertIn("model.layers.1.engram.embed", units)
        self.assertIn("model.layers.1.engram.wkv", units)
        self.assertNotIn("model.layers.1.engram", units)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_full_depth_recipe_preserves_released_shared_attention_roles(self):
        """Cover all released shared-attention layer roles.

        Feature: Full-depth recipe wildcard expansion.
        Description: Apply the recipe plan to the released decoder-depth layout.
        Expectation: Full, Reindex, Reuse, and Engram layers all receive valid rules.
        """
        class _FakeMesh:
            mesh_dim_names = ("dp_shard", "tp")
            mesh_shape = (2, 2)

        recipe_path = Path(__file__).resolve().parents[5] / (
            "examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml"
        )
        recipe = parse_training_args([str(recipe_path)])
        with tempfile.TemporaryDirectory() as directory:
            assets_path = _write_engram_assets(
                directory,
                num_hidden_layers=40,
                layer_ids=(1, 14),
            )
            model = DeepseekV41ForCausalLM(
                _tiny_config(assets_path, num_hidden_layers=40)
            )

        replacement_rules = entries_to_module_replacements(recipe.plan_overrides)
        replacement_plan = compile_module_replacements(model, replacement_rules)
        apply_module_replacements(model, replacement_plan)
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
        """Own every V4.1-only parameter under TP2 and EP4.

        Feature: Hybrid-parallel recipe coverage.
        Description: Resolve parameter placements for mHC and Engram boundaries.
        Expectation: Every model-specific parameter has an unambiguous owner.
        """
        class _FakeMesh:
            mesh_dim_names = ("dp_shard", "tp")
            mesh_shape = (2, 2)

        recipe_path = Path(__file__).resolve().parents[5] / (
            "examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml"
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
                model = DeepseekV41ForCausalLM(config)
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
        """Keep left n-gram context across TP sequence slices.

        Feature: Engram sequence-parallel hashing.
        Description: Compare rank-one local hashes with the matching full-sequence rows.
        Expectation: The local slice produces identical hash identifiers.
        """
        with tempfile.TemporaryDirectory() as directory:
            model = DeepseekV41ForCausalLM(_tiny_config(_write_engram_assets(directory)))
            engram = EngramModule(module=model.model.layers[1].engram)
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
        """Use global KV state and offsets under context parallelism.

        Feature: Shared-attention context parallelism.
        Description: Execute local queries against gathered raw and compressed KV state.
        Expectation: CP output matches the corresponding global-sequence result.
        """
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(19)
            model = DeepseekV41ForCausalLM(
                _tiny_config(_write_engram_assets(directory))
            )
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
        """Apply ratio-aware causal boundaries to compressed keys.

        Feature: Compressed indexer causal masking.
        Description: Select keys while successive source groups become complete.
        Expectation: A compressed key appears only after its source group closes.
        """
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
        """Pin the newest reachable hierarchical candidate block.

        Feature: Hierarchical compressed-key selection.
        Description: Select candidate blocks from maxima with a partial tail.
        Expectation: The newest causally reachable block is always retained.
        """
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
        """Score only the compact reindex candidate subset.

        Feature: Compressed candidate reindexing.
        Description: Select top keys from explicitly chosen compact blocks.
        Expectation: Returned identifiers map back to the correct global keys.
        """
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
        """Update only indexer inputs from sparse KL loss.

        Feature: PanGu-style compressed-indexer distillation.
        Description: Backpropagate sparse KL through student and teacher inputs.
        Expectation: Indexer inputs receive gradients while the teacher stays detached.
        """
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
        """Match direct autograd for compressed-indexer KL gradients.

        Feature: Precomputed sparse-KL gradient injection.
        Description: Compare custom backward values with a direct reference graph.
        Expectation: Every indexer input gradient is numerically identical.
        """
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
        """Apply the global query offset to CP sliding windows.

        Feature: Context-parallel window indexing.
        Description: Build rank-one local indices against an eight-token global key set.
        Expectation: Each local query addresses its preceding global window.
        """
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
        """Inject the V4.1 wrapper on every active CP boundary.

        Feature: Recipe-driven context-parallel wrapping.
        Description: Resolve plan overrides with a nontrivial CP mesh.
        Expectation: Every attention boundary selects the model-owned CP wrapper.
        """
        class _FakeMesh:
            mesh_dim_names = ("dp_shard", "cp")
            mesh_shape = (2, 2)

        recipe_path = Path(__file__).resolve().parents[5] / (
            "examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml"
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
                model = DeepseekV41ForCausalLM(config)
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
                """Create a one-expert signature fixture."""
                super().__init__()
                self.num_experts = 1
                self.act_fn = F.silu

            @staticmethod
            def _apply_gate(gate_up: torch.Tensor) -> torch.Tensor:
                return gate_up

        class _TextMoe(nn.Module):
            def __init__(self) -> None:
                """Create the text-only MoE signature fixture."""
                super().__init__()
                self.gate = nn.Identity()
                self.experts = _Experts()
                self.shared_experts = nn.Identity()
                self.is_hash = False

            def forward(
                    self,
                    hidden_states: torch.Tensor,
                    input_ids: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                """Expose the text MoE production-compatible signature."""
                del input_ids
                return hidden_states

        class _MultimodalMoe(_TextMoe):
            def forward(
                    self,
                    hidden_states: torch.Tensor,
                    input_ids: Optional[torch.Tensor] = None,
                    image_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                """Expose the multimodal MoE production-compatible signature."""
                del input_ids, image_mask
                return hidden_states

        class _EpAxis:
            @staticmethod
            def size() -> int:
                """Return the one-rank expert axis size."""
                return 1

        class _EpMesh:
            @staticmethod
            def get_group(name: str) -> None:
                """Return the local fixture's absent process group."""
                del name

            def __getitem__(self, name: str) -> _EpAxis:
                """Resolve the expert axis by name."""
                del self
                del name
                return _EpAxis()

        return (_MultimodalMoe() if multimodal else _TextMoe()), _EpMesh()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ep_factory_matches_text_and_multimodal_forward_signatures(self):
        """Match EP closures to text and multimodal MoE signatures.

        Feature: DeepSeek expert-parallel compute factories.
        Description: Inspect closures created for text-only and multimodal source modules.
        Expectation: Each closure exposes only arguments accepted by its source MoE.
        """
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
    def test_fused_expert_uses_model_specific_gate(self):
        """Use the model-specific gate in the fused local expert.

        Feature: DeepSeek fused expert activation.
        Description: Run a deterministic expert with the V4 clamp hook installed.
        Expectation: The generic expert applies the supplied model gate exactly once.
        """
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
