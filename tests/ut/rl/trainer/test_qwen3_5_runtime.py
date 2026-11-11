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
"""Contracts for the Qwen3.8 text policy using the public Qwen3.5 model."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from safetensors.torch import save_file

import rl
from rl.config.schema import build_model_registration
from rl.config.resolve import build_runtime_config
from rl.config.validate import validate_rollout_and_agentic
from rl.workers.engine.model import resolve_vllm_model
from rl.weight_sync.model_adapter import ModelWeightAdapter, alias_tied_embeddings
from rl.weight_sync.packed_weight import PackedWeight, unpack_packed_weights
from rl.weight_sync import vllm_worker


class TestQwen35Runtime(unittest.TestCase):
    """Keep model construction, native names and publication boundaries aligned."""

    def setUp(self) -> None:
        """Resolve a production GRPO recipe against a local hybrid identity."""
        # The fixture remains alive through the test; unittest owns its cleanup.
        self.directory = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        recipe = Path(rl.__file__).resolve().parent.parent / "examples/swe_smith/configs/qwen3_8_27b.yaml"
        self.config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        self.config["model"].update(name="qwen3_5", registry_name="qwen3_8_27b",
                                    weights_path=str(self.path), tokenizer_path=str(self.path))
        self.config["train"]["accelerator"].update(tp=1, ep=1, edp_shard=1)
        self.config["rollout"]["vllm"].update(enable_expert_parallel=False, max_num_seqs=2)
        self.metadata = {
            "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text", "tie_word_embeddings": False,
                            "layer_types": ["linear_attention", "full_attention"]},
        }
        self._write_config()

    def _write_config(self) -> None:
        self.path.joinpath("config.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def _registration(self):
        return resolve_vllm_model(build_model_registration(self.config), "native")

    def test_public_text_model_builder(self) -> None:
        """The hybrid policy uses the shared model loader and no dense-only patches."""
        registration = self._registration()
        self.assertEqual(registration.family, "qwen3_5")
        self.assertEqual(registration.architecture, "Qwen3_5ForConditionalGeneration")
        runtime = build_runtime_config(self.config)
        target = runtime.model.to_dict()
        self.assertEqual(target["_target_"], "hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained")
        self.assertNotIn("fused", target)
        self.assertEqual(runtime.accelerator.tp_size, 1)
        self.assertEqual(runtime.plan_overrides, [])
        self.assertTrue(runtime.fsdp_config.enable_offload)
        with self.assertRaisesRegex(ValueError, "Critic"):
            build_runtime_config(self.config, critic=True)

    def test_unsupported_combinations_fail_before_model_construction(self) -> None:
        """Hybrid policy support cannot expand through an unrelated dense default."""
        cases = [
            (("algorithm", "name"), "ppo", "GRPO only"),
            (("consistency", "enabled"), True, "consistency requires schema_version=1"),
            (("train", "accelerator", "tp"), 2, "tp=1"),
            (("train", "accelerator", "cp"), 2, "cp=1"),
            (("train", "accelerator", "ep"), 2, "ep=1"),
            (("rollout", "vllm", "model_implementation"), "hyper", "native vLLM"),
            (("rollout", "vllm", "deployment"), "disjoint", "colocated"),
            (("rollout", "vllm", "enforce_eager"), False, "enforce_eager"),
            (("rollout", "vllm", "max_num_seqs"), None, "explicit"),
            (("rollout", "vllm", "weight_sync", "strategy"), "direct_reshard", "full_gather"),
        ]
        for keys, value, error in cases:
            with self.subTest(keys=keys):
                config = deepcopy(self.config)
                section = config
                for key in keys[:-1]:
                    section = section.setdefault(key, {})
                section[keys[-1]] = value
                with self.assertRaisesRegex(ValueError, error):
                    build_runtime_config(config)

    def test_repository_parser_follows_registered_hybrid_family(self) -> None:
        """Reject a JSON parser for the hybrid model's XML calls before launching vLLM."""
        recipe = Path(rl.__file__).resolve().parent.parent / "examples/swe_smith/configs/qwen3_8_27b.yaml"
        config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        config["model"].update(weights_path=str(self.path), tokenizer_path=str(self.path))
        registration = build_model_registration(config)
        config["rollout"]["vllm"]["tool_call_parser"] = "qwen3_coder"
        validate_rollout_and_agentic(
            config["rollout"], config["agentic"], config["train"]["accelerator"], registration,
        )
        for parser in ("hermes", "qwen3_xml", "unreviewed"):
            with self.subTest(parser=parser):
                config["rollout"]["vllm"]["tool_call_parser"] = parser
                with self.assertRaisesRegex(ValueError, "tool_call_parser=qwen3_coder"):
                    validate_rollout_and_agentic(
                        config["rollout"], config["agentic"], config["train"]["accelerator"], registration,
                    )

    def test_checkpoint_identity_is_strict(self) -> None:
        """A marketing name cannot relabel a different architecture or MoE text."""
        self.metadata["text_config"]["model_type"] = "qwen3_5_moe_text"
        self._write_config()
        with self.assertRaisesRegex(ValueError, "Unsupported RL model identity"):
            build_model_registration(self.config)

    def test_canonical_text_tensors_survive_packing(self) -> None:
        """Only text namespace changes; numerical GDN and full-attention values survive."""
        adapter = ModelWeightAdapter(self._registration())
        source = {
            "model.layers.0.linear_attn.in_proj_qkv.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
            "model.layers.0.linear_attn.A_log": torch.tensor([-1., 2.]),
            "model.layers.1.self_attn.q_proj.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "lm_head.weight": torch.ones(3, 4),
        }
        mapped = adapter.map_local_state_dict(source)
        for name, tensor in source.items():
            destination = name if name == "lm_head.weight" else "model.language_model." + name[6:]
            self.assertIs(mapped[destination], tensor)
            metadata = adapter.packed_metadata([PackedWeight(destination, "float32", tuple(tensor.shape), 4)
                                                .worker_metadata()])
            unpacked = dict(unpack_packed_weights(tensor.contiguous().view(torch.uint8).flatten(), metadata))
            torch.testing.assert_close(unpacked[destination], tensor, rtol=0, atol=0)
        for invalid in ("model.visual.weight", "mtp.layers.0.weight", "model.language_model.norm.weight"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "text-only"):
                adapter.map_local_state_dict({invalid: torch.ones(1)})

    def test_tied_embedding_alias_uses_native_checkpoint_namespace(self) -> None:
        """A tied policy sends one storage object under both checkpoint aliases."""
        self.metadata["text_config"]["tie_word_embeddings"] = True
        self._write_config()
        registration = self._registration()
        embedding = torch.ones(3, 4)
        mapped = ModelWeightAdapter(registration).map_local_state_dict({
            "model.embed_tokens.weight": embedding, "lm_head.weight": embedding,
        })
        mapped = alias_tied_embeddings(mapped, registration)
        self.assertIs(mapped["model.language_model.embed_tokens.weight"], embedding)
        self.assertIs(mapped["lm_head.weight"], embedding)

    def test_single_file_and_tied_checkpoint_publication_coverage(self) -> None:
        """Single-file checkpoints and tied storage require exactly their text policy."""
        tensors = {
            "model.language_model.embed_tokens.weight": torch.ones(3, 4),
            "model.language_model.norm.weight": torch.ones(4),
            "model.visual.weight": torch.zeros(1), "mtp.weight": torch.zeros(1),
        }
        save_file(tensors, self.path / "model.safetensors")
        worker = SimpleNamespace(model_config=SimpleNamespace(
            model=str(self.path), hf_text_config=SimpleNamespace(tie_word_embeddings=True),
        ))
        self.assertEqual(vllm_worker._qwen3_5_text_checkpoint_names(worker), {
            "model.language_model.embed_tokens.weight", "model.language_model.norm.weight",
        })
        worker.model_config.hf_text_config.tie_word_embeddings = False
        with self.assertRaisesRegex(ValueError, "missing lm_head"):
            vllm_worker._qwen3_5_text_checkpoint_names(worker)

    def test_partial_publication_cannot_commit_a_policy_version(self) -> None:
        """Missing, duplicated or non-text parameters reject publication before commit."""
        names = {"model.language_model.norm.weight", "lm_head.weight"}
        self.path.joinpath("model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model-1.safetensors" for name in names | {"model.visual.weight", "mtp.weight"}},
        }), encoding="utf-8")
        worker = SimpleNamespace(
            model_config=SimpleNamespace(model=str(self.path),
                                         hf_text_config=SimpleNamespace(tie_word_embeddings=False),
                                         hf_config=SimpleNamespace(
                                             architectures=["Qwen3_5ForConditionalGeneration"])),
            _check_weight_transfer_engine=lambda: None, _weight_update_active=True,
            _rl_pending_policy_version=1, _rl_loaded_policy_version=0,
        )
        loaded = []
        model = SimpleNamespace(load_weights=lambda weights: loaded.extend(weights) or {weights[0][0]})
        source = torch.tensor([1., 2.])
        vllm_worker._load_qwen3_5_checkpoint_weights(worker, model, [("lm_head.weight", source)])
        source.zero_()
        torch.testing.assert_close(loaded[0][1], torch.tensor([1., 2.]))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            vllm_worker._finish_custom_weight_update(worker)
        self.assertEqual(worker._rl_loaded_policy_version, 0)
        self.assertTrue(worker._weight_update_active)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            vllm_worker._load_qwen3_5_checkpoint_weights(worker, model, [("lm_head.weight", source)])
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            vllm_worker._load_qwen3_5_checkpoint_weights(worker, model, [("model.visual.weight", source)])
        vllm_worker._load_qwen3_5_checkpoint_weights(worker, model, [("model.language_model.norm.weight", source)])
        vllm_worker._finish_custom_weight_update(worker)
        self.assertEqual(worker._rl_loaded_policy_version, 1)
        self.assertFalse(worker._weight_update_active)
