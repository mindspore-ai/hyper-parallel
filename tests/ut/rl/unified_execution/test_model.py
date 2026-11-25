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
"""HP model binding, projection ownership and rollout topology."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import rl
import torch
import yaml
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsConfig
from hyper_parallel.components.parallelism_invariant.row_parallel_linear import PikPlan, pik_linear_inference
from hyper_parallel.distributed.mesh import MeshContext
from hyper_parallel.distributed.module_binding import bind_parallel_modules
from rl.config.resolve import resolve_unified_execution_config
from rl.rollout.vllm import VLLMGenerationEngine
from rl.unified_execution.contract import build_unified_payload, fingerprint, parse_unified_payload
from rl.unified_execution.execution_plan import pik_row_options, resolve_pik_leaves
from rl.unified_execution.modules.grouped_query_attention import build_grouped_query_attention
from rl.unified_execution.modules.row_parallel_linear import PikLinear, build_pik_linear
from rl.unified_execution.vllm.model_builder import build_inference_model, normalized_compute_setup
from rl.unified_execution.models.structure import capture_training_structure, training_sequence_multiple
from rl.unified_execution.vllm.attention.paged_attention import paged_attention_forward
from rl.unified_execution.vllm.attention.grouped_query_attention import project_qkv_with_fused_rope
from rl.unified_execution.vllm.qwen3 import Qwen3RolloutModel
from rl.unified_execution.vllm.parallel_context import InferenceParallelContext, build_parallel_context
from rl.weight_sync.vllm_worker import _rollout_worker_topology
from rl.workers.actor import Actor
from rl.workers.engine.materialization import register_qwen3_materialization
from rl.workers.engine.model import ModelRegistration
from rl.workers.adapters.qwen3 import Qwen3ShardingPlanner
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh, FakeSubMesh


def consistency_recipe(name: str) -> dict:
    """Read the distributed example that ships with the installed RL package."""
    recipe = Path(rl.__file__).resolve().parent.parent / "examples/consistency/configs" / name
    return yaml.safe_load(recipe.read_text(encoding="utf-8"))


class _ReferencePagedCore(nn.Module):
    """CPU stand-in for request boundaries owned by vLLM forward metadata."""

    def __init__(self, attention: nn.Module, lengths: tuple[int, ...]) -> None:
        """Store the head layout and explicit request lengths for the CPU reference."""
        super().__init__()
        self.scale = attention.scaling
        self.head_dim = attention.config.head_dim
        self.heads = attention.config.num_attention_heads
        self.kv_heads = attention.config.num_key_value_heads
        self.lengths = lengths

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Independently compute each scheduled request with causal attention."""
        outputs, offset = [], 0
        for length in self.lengths:
            query_part = query[offset:offset + length].view(length, self.heads, self.head_dim).transpose(0, 1)
            key_part = key[offset:offset + length].view(length, self.kv_heads, self.head_dim).transpose(0, 1)
            value_part = value[offset:offset + length].view(length, self.kv_heads, self.head_dim).transpose(0, 1)
            # Pylint cannot infer this Torch C-extension callable; real CPU tests execute it.
            output = torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=not-callable
                query_part, key_part, value_part, is_causal=True, scale=self.scale, enable_gqa=True,
            )
            outputs.append(output.transpose(0, 1).flatten(1))
            offset += length
        return torch.cat(outputs)


class TestUnifiedModel(unittest.TestCase):
    """The wrapper must retain the original decoder and absolute-position RoPE."""

    def setUp(self) -> None:
        """Build a tiny actual Qwen3 model and a shared empty replacement plan."""
        device = patch(
            "hyper_parallel.models._transformers.auto_model._current_device", return_value=torch.device("cpu"),
        )
        device.start()
        self.addCleanup(device.stop)
        torch.manual_seed(13)
        self.config = Qwen3Config.from_dict({
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
            "max_position_embeddings": 64, "tie_word_embeddings": True,
        })
        self.reference = build_inference_model(self.config, [], attn_implementation="eager")
        self.payload = build_unified_payload({
            "consistency": {"schema_version": 1, "mode": "architecture"},
            "model": {"name": "qwen3"}, "train": {"plan_overrides": []},
            "rollout": {"vllm": {"enforce_eager": True}},
        })

    def _wrapper(self, lengths: tuple[int, ...]) -> Qwen3RolloutModel:
        """Replace the device-specific core while preserving the production bridge."""
        def install(model: nn.Module, unused_config: object, unused_prefix: str) -> None:
            """Keep the real interface conversion while supplying the CPU reference core."""
            del unused_config, unused_prefix
            ALL_ATTENTION_FUNCTIONS.register("hyper_parallel_vllm_test", paged_attention_forward)
            for layer in model.layers:
                layer.self_attn.vllm_attention = _ReferencePagedCore(layer.self_attn, lengths)
            model.config._attn_implementation = "hyper_parallel_vllm_test"  # pylint: disable=protected-access

        self.config.hyper_rl_unified_execution = self.payload
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=self.config, enforce_eager=True),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
            additional_config={}, quant_config=None,
        )
        with patch("rl.unified_execution.vllm.parallel_context.dist.is_initialized", return_value=False), \
                patch("rl.unified_execution.vllm.qwen3.install_paged_attention", side_effect=install):
            wrapper = Qwen3RolloutModel(vllm_config=config)
        wrapper.load_weights(self.reference.state_dict().items())
        return wrapper

    @torch.no_grad()
    def test_flattened_requests_remain_independent(self) -> None:
        """The shared decoder delegates packed request isolation to its core."""
        wrapper = self._wrapper((3, 2))
        tokens = torch.tensor([1, 2, 3, 17, 19])
        positions = torch.tensor([0, 1, 2, 12, 13])
        outputs = []
        for start, end in ((0, 3), (3, 5)):
            result = self.reference.model(
                tokens[start:end][None], position_ids=positions[start:end][None], use_cache=False,
            )
            outputs.append(result.last_hidden_state[0])
        torch.testing.assert_close(wrapper(tokens, positions), torch.cat(outputs), atol=1e-6, rtol=1e-5)

    def test_one_attention_graph_restores_dynamic_sp_padding(self) -> None:
        """Attention consumes scheduled rows and restores padded rows without shape branches."""
        for multiple in (2, 4):
            context = SimpleNamespace(
                parallel_context=InferenceParallelContext(tp_size=multiple, sequence_parallel=True),
            )
            module = SimpleNamespace(unified_sp_multiple=multiple,
                                     vllm_attention=lambda query, _key, _value: query)

            def make_forward(module, context):
                """Bind one parallel configuration without changing the traced signature."""
                def forward(value: torch.Tensor) -> torch.Tensor:
                    """Exercise the actual model entry and attention exit padding together."""
                    module.unified_token_count = value.shape[0]
                    padded = Qwen3RolloutModel._pad_rows(context, value)
                    query = padded.reshape(1, -1, 2, 4).transpose(1, 2)
                    return paged_attention_forward(module, query, query, query, None)[0]

                return forward

            graph = make_fx(make_forward(module, context), tracing_mode="symbolic")(torch.ones(128, 8))
            for rows in (1, 2, 3, 7, 8, 31, 127, 128):
                with self.subTest(tp=multiple, rows=rows):
                    value = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
                    actual = graph(value).flatten(2).squeeze(0)
                    self.assertEqual(actual.shape[0], ((rows + multiple - 1) // multiple) * multiple)
                    torch.testing.assert_close(actual[:rows], value, rtol=0, atol=0)
                    torch.testing.assert_close(actual[rows:], torch.zeros_like(actual[rows:]), rtol=0, atol=0)

    def test_actual_training_structure_is_required_to_match(self) -> None:
        """A valid payload digest cannot conceal a different realized Actor model."""
        report = capture_training_structure(self.reference, self.payload)
        self.payload["training_structure"] = report
        wrapper = self._wrapper((2,))
        self.assertEqual(wrapper.unified_structure_report, report)
        report["structure"]["modules"]["model.norm"] = "different.Norm"
        report["fingerprint"] = fingerprint(report["structure"])
        with self.assertRaisesRegex(ValueError, "differs from the Actor"):
            self._wrapper((2,))


class TestQwen3Materialization(unittest.TestCase):
    """Keep non-parameter RoPE state identical without replacing live storage."""

    def setUp(self) -> None:
        """Use the real Qwen3-4B head dimension and rotary base on CPU."""
        self.config = Qwen3Config.from_dict({
            "head_dim": 128, "rope_parameters": {"rope_type": "default", "rope_theta": 1000000},
        })
        with torch.device("cpu"):
            self.expected = Qwen3RotaryEmbedding(self.config).inv_freq.clone()

    def test_bf16_default_keeps_canonical_fp32_buffers(self) -> None:
        """The vLLM parameter dtype default must not reduce rotary-buffer precision."""
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cpu"):
                rotary = Qwen3RotaryEmbedding(self.config)
            register_qwen3_materialization(rotary)
        finally:
            torch.set_default_dtype(original_dtype)
        self.assertEqual(rotary.inv_freq.dtype, torch.float32)
        torch.testing.assert_close(rotary.inv_freq, self.expected, rtol=0, atol=0)


class TestPikModel(unittest.TestCase):
    """Test contract failure paths without initializing NPU or process groups."""

    def setUp(self) -> None:
        """Load the explicit shared recipe independently for each case."""
        self.config = consistency_recipe("qwen3_4b_rl.yaml")

    def test_worker_binds_projections_after_hp_build(self) -> None:
        """Exercise the real replacement-request path, stubbing only device-specific initialization."""
        self.config["train"]["accelerator"].update(tp=1, sequence_parallel=False)
        self.config["rollout"]["vllm"].update(tensor_parallel_size=1, data_parallel_size=1)
        payload = resolve_unified_execution_config(self.config)["rollout"]["vllm"]["unified_execution"]
        hf = Qwen3Config(vocab_size=32, hidden_size=256, intermediate_size=512, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=128)
        hf.hyper_rl_unified_execution = payload
        config = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf, enforce_eager=True),
                                 parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
                                 additional_config={}, quant_config=None)

        def local_pik(k, n, *, plan, **kwargs):
            return SimpleNamespace(plan=plan, global_k=k, output_features=n, world_size=1, sequence_parallel=False)

        worker = "rl.unified_execution.vllm.qwen3"
        # This CPU-only worker has no vLLM groups, even if earlier UTs initialized Gloo.
        with patch("hyper_parallel.models._transformers.auto_model._current_device",
                   return_value=torch.device("cpu")), \
                patch("rl.unified_execution.vllm.parallel_context.dist.is_initialized", return_value=False), \
                patch("rl.unified_execution.modules.row_parallel_linear.prepare_pik", side_effect=local_pik), \
                patch(worker + ".initialize_batch_invariant"), patch(worker + ".verify_batch_invariant"), \
                patch(worker + ".install_paged_attention"):
            model = Qwen3RolloutModel(vllm_config=config)

        layer = model.model.layers[0]
        self.assertIs(layer.self_attn.project_qkv.__func__, project_qkv_with_fused_rope)
        self.assertIs(layer.self_attn.forward.__func__, type(layer.self_attn).forward)
        self.assertIs(layer.self_attn.o_proj.compute_projection, pik_linear_inference)
        self.assertIs(layer.mlp.down_proj.compute_projection, pik_linear_inference)

    def test_hp_binds_cp_and_preserves_pik_output(self) -> None:
        """The shipped plan uses native CP ownership and consumes each PIK result once."""
        config = Qwen3Config(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                             num_attention_heads=8, num_key_value_heads=4, head_dim=8)
        path = "model.layers.0.self_attn"
        topology = FakeDeviceMesh((2, 2), ("cp", "tp"))
        entries = self.config["train"]["plan_overrides"]
        resolve_pik_leaves(entries, {"tp": 2}, {"tp": 2})
        for sp in (False, True):
            with self.subTest(sp=sp), patch.object(FakeSubMesh, "get_local_rank", return_value=0, create=True):
                model = Qwen3ForCausalLM(config)
                layer = model.model.layers[0]
                mesh = MeshContext(device_mesh=topology, tp_size=2, cp_size=2, sequence_parallel=sp)
                with patch("rl.unified_execution.modules.row_parallel_linear.prepare_pik") as prepare:
                    layer.self_attn = build_grouped_query_attention(
                        module=layer.self_attn, module_fqn=path, num_leaves=2,
                        context={"tp": True, "cp": True, "mesh": mesh, "sequence_parallel": sp})
                    self.assertEqual(prepare.call_args.kwargs["sequence_parallel"], sp)
                setup = normalized_compute_setup(entries, mesh)
                plan = Qwen3ShardingPlanner(plan_overrides=setup.plan_overrides).plan(
                    model, topology, tp_size=2, cp_size=2, sequence_parallel=sp)
                boundary = plan.modules[path]
                self.assertIsNone(boundary.inner_wrapper)
                self.assertEqual(boundary.out_src, boundary.out_dst)
                self.assertEqual(boundary.out_src["output"]["tp"].is_shard(), sp)
                self.assertNotIn(path + ".o_proj", plan.modules)
                self.assertIs(layer.self_attn.o_proj.forward.__func__, PikLinear.forward)
                bind_parallel_modules(model, plan, mesh)
                self.assertEqual(layer.self_attn.cp_context.size, 2)

    def test_shipped_recipes_roundtrip_worker_contract(self) -> None:
        for name in ("qwen3_4b_rl.yaml", "qwen3_moe_4l_rl.yaml", "qwen3_moe_4l_cp_rl.yaml"):
            with self.subTest(recipe=name):
                resolved = resolve_unified_execution_config(consistency_recipe(name))
                payload = resolved["rollout"]["vllm"]["unified_execution"]
                self.assertEqual(parse_unified_payload(payload), payload)
                self.assertEqual(payload["plan_overrides"], resolved["train"]["plan_overrides"])

    def test_automatic_leaves_are_shared_by_both_roles(self) -> None:
        """Use the larger TP degree even when the other role runs with TP1."""
        for train_tp, rollout_tp in ((1, 4), (4, 2)):
            with self.subTest(train_tp=train_tp, rollout_tp=rollout_tp):
                self.config["train"]["accelerator"].update(tp=train_tp, sequence_parallel=False)
                self.config["rollout"]["vllm"].update(tensor_parallel_size=rollout_tp, sequence_parallel=False)
                resolved = resolve_unified_execution_config(self.config)
                payload = resolved["rollout"]["vllm"]["unified_execution"]
                self.assertEqual(pik_row_options(resolved["train"]["plan_overrides"])["num_leaves"], 4)
                self.assertEqual(payload["plan_overrides"], resolved["train"]["plan_overrides"])
        self.assertEqual(pik_row_options(self.config["train"]["plan_overrides"])["num_leaves"], "auto")

    def test_factory_reads_current_parameters_and_keeps_gradients(self) -> None:
        """TP1 replacement preserves identity and reads later replacement weights."""
        original = torch.nn.Linear(8, 4)
        parameter = original.weight
        context = SimpleNamespace(world_size=1, sequence_parallel=False, group=None)
        with patch("rl.unified_execution.modules.row_parallel_linear.prepare_pik", return_value=context) as prepare:
            module = build_pik_linear(module=original, module_fqn="mlp.down_proj", context={})
        self.assertIs(prepare.call_args.kwargs["group"], None)
        self.assertEqual(prepare.call_args.kwargs["policy"], {"fill_uninitialized_memory": False})
        x = torch.randn(2, 3, 8, requires_grad=True)
        with patch.object(module, "compute_projection",
                   side_effect=lambda x, w, b, **_: x @ w.T + b):
            module(x).sum().backward()
            self.assertIs(module.weight, parameter)
            torch.testing.assert_close(parameter.grad, x.detach().flatten(0, 1).sum(0).expand(4, 8))
            module.weight = torch.nn.Parameter(torch.zeros_like(parameter))
            torch.testing.assert_close(module(x), module.bias.expand(2, 3, 4))


class _TokenModel(torch.nn.Module):
    """Expose input padding while retaining a real differentiable token projection."""

    def __init__(self) -> None:
        """Create an ordinary trainable projection and declare the SP input multiple."""
        super().__init__()
        self.projection = torch.nn.Embedding(11, 11)
        self.unified_sequence_multiple = 4
        self.observed = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = False,
                position_ids: Optional[torch.Tensor] = None) -> dict:
        """Record prepared model inputs without mutating caller token storage."""
        del use_cache
        if position_ids is None:
            # Model the HF default after SP embeddings, before attention gathers tokens.
            local_length = input_ids.shape[1] // self.unified_sequence_multiple
            position_ids = torch.arange(local_length, device=input_ids.device).unsqueeze(0)
        self.observed = (input_ids.detach().clone(), attention_mask.detach().clone(), position_ids.detach().clone())
        positional_bias = position_ids.unsqueeze(-1) * torch.arange(11, device=input_ids.device) / 13
        return {"logits": self.projection(input_ids) + positional_bias}


class TestPikActorPadding(unittest.TestCase):
    """Verify SP padding is invisible to original probabilities and gradients."""

    def test_native_expert_tp_boundary_without_sp(self) -> None:
        """EP's native TP row split needs padding even when dense SP is disabled."""
        dense = nn.Linear(8, 8)
        experts = EPGroupedExperts(config=GroupedExpertsConfig(
            num_experts=4, hidden_size=8, intermediate_size=8, num_layers=1,
        ))
        experts.ep_context = SimpleNamespace(mesh=SimpleNamespace(size=lambda: 4))
        for cp in (1, 2):
            with self.subTest(cp=cp):
                self.assertEqual(training_sequence_multiple(
                    dense, tp_size=2, cp_size=cp, sequence_parallel=False), cp)
                self.assertEqual(training_sequence_multiple(
                    experts, tp_size=2, cp_size=cp, sequence_parallel=False), 2 * cp)
                self.assertEqual(training_sequence_multiple(
                    experts, tp_size=2, cp_size=cp, sequence_parallel=True), 2 * cp)

    def test_odd_and_aligned_tokens(self) -> None:
        """Right padding must neither add loss positions nor change valid gradients."""
        for multiple, length in ((1, 7), (2, 7), (2, 8), (4, 7), (4, 8), (4, 9)):
            with self.subTest(multiple=multiple, length=length):
                model = _TokenModel()
                model.unified_sequence_multiple = multiple
                actor = Actor(model, algorithm=SimpleNamespace(), micro_batch_size=1,
                              optimizer=torch.optim.SGD(model.parameters(), lr=0.1), device=torch.device("cpu"))
                tokens = torch.arange(length).unsqueeze(0) % 11
                mask = torch.ones_like(tokens)
                positions = torch.arange(length).view(1, length, 1)
                logits = model.projection(tokens) + positions * torch.arange(11) / 13
                expected = logits[:, :-1].log_softmax(-1).gather(
                    -1, tokens[:, 1:, None]).squeeze(-1)
                expected.sum().backward()
                gradient = model.projection.weight.grad.clone()
                model.zero_grad(set_to_none=True)
                actual = actor.sequence_log_probs(tokens, mask)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.sum().backward()
                torch.testing.assert_close(model.projection.weight.grad, gradient, rtol=0, atol=0)
                seen_tokens, seen_mask, seen_positions = model.observed
                self.assertEqual(seen_tokens.shape[1] % multiple, 0)
                self.assertEqual(seen_tokens.shape[1], length + (-length) % multiple)
                self.assertEqual(seen_mask[:, length:].count_nonzero().item(), 0)
                torch.testing.assert_close(seen_positions, torch.arange(seen_tokens.shape[1]).unsqueeze(0))
                torch.testing.assert_close(seen_tokens[:, :length], tokens)
                self.assertEqual(mask.sum().item(), length)


class TestPrefillContextParallel(unittest.TestCase):
    """Validate independent training CP, inference PCP and publication domains."""

    def test_cross_pcp_mesh_extends_hp_planning_without_changing_native_dp(self) -> None:
        """Native DP1/PCP1/TP1 coordinates become HP DP3/TP1 in an eight-rank domain."""
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1,
                                            prefill_context_parallel_size=2, decode_context_parallel_size=1,
                                            data_parallel_size=2, data_parallel_rank=1),
            device_config=SimpleNamespace(device="cpu"),
            model_config=SimpleNamespace(enforce_eager=True, is_moe=True),
        )
        payload = {"parallelism": {"tp": 2, "cp": 2, "pp": 1}, "expert_parallelism": {"inference": 4},
                   "plan_overrides": consistency_recipe("qwen3_moe_4l_rl.yaml")["train"]["plan_overrides"]}
        native = ModuleType("vllm.distributed")
        tp_group = object()
        native.get_tp_group = lambda: SimpleNamespace(ranks=[6, 7], device_group=tp_group)
        get_native_dp = MagicMock()
        native.get_dp_group = get_native_dp
        native.get_ep_group = lambda: SimpleNamespace(ranks=list(range(8)))
        native.get_pcp_group = lambda: SimpleNamespace(world_size=2, rank_in_group=1)
        tp_mesh, dense_mesh, expert_mesh, root = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        tp_mesh.__getitem__.return_value.get_local_rank.return_value = 1
        dense_mesh.__getitem__.return_value.get_local_rank.return_value = 3
        expert_mesh.__getitem__.return_value.get_local_rank.return_value = 3
        module = "rl.unified_execution.vllm.parallel_context"
        with patch.dict("sys.modules", {"vllm.distributed": native}), \
                patch(module + ".dist.is_initialized", return_value=True), \
                patch(module + ".DeviceMesh.from_group", side_effect=[tp_mesh, dense_mesh]) as from_group, \
                patch(module + ".init_device_mesh", return_value=root) as build_mesh, \
                patch(module + ".build_expert_mesh", return_value=expert_mesh) as build_experts:
            context = build_parallel_context(config, payload)
        build_mesh.assert_called_once_with("cpu", (4, 2), mesh_dim_names=("dp", "tp"), rank_list=tuple(range(8)))
        build_experts.assert_called_once_with(root, 4)
        from_group.assert_called_with([root["dp"].get_group(), tp_group], "cpu",
                                      mesh=[[0, 1], [2, 3], [4, 5], [6, 7]], mesh_dim_names=("dp", "tp"))
        get_native_dp.assert_not_called()
        self.assertEqual((context.dp_size, context.dp_rank), (2, 1))
        self.assertEqual((context.mesh_context.dp_size, context.mesh_context.dp_rank), (4, 3))
        self.assertEqual(context.mesh_context.cp_size, 1)

    def test_cp_one_has_identical_payload_and_launch_selection(self) -> None:
        """Explicit PCP1 must not activate a new runner or change the old launch recipe."""
        config = consistency_recipe("qwen3_moe_4l_rl.yaml")
        old = build_unified_payload(config)
        config["rollout"]["vllm"]["context_parallel_size"] = 1
        self.assertEqual(build_unified_payload(config), old)
        engine = self.engine(config)
        command = engine._server_command("127.0.0.1", 8100)
        self.assertNotIn("--prefill-context-parallel-size", command)
        self.assertNotIn("enable_pcp_embedding_lmhead_weight_sharding",
                         json.loads(command[command.index("--additional-config") + 1]))
        with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0"}):
            self.assertEqual(engine._server_environment("0,1,2,3")["VLLM_USE_V2_MODEL_RUNNER"], "0")

    @staticmethod
    def engine(config: dict) -> VLLMGenerationEngine:
        """Construct a CPU-only controller without starting a server."""
        config = resolve_unified_execution_config(config)
        model = ModelRegistration("qwen", "qwen3_moe", "/model", "/tokenizer", "Qwen3MoeForCausalLM",
                                  "qwen3_moe", "qwen3_moe", False)
        return VLLMGenerationEngine(model, config["rollout"])

    def test_publication_addresses_every_pcp_replica_without_changing_tp(self) -> None:
        """Distinct PCP owners must never collapse onto one weight-update recipient."""
        native = ModuleType("vllm.distributed")
        ipc = ModuleType("vllm_ascend.distributed.weight_transfer.npu_ipc_engine")
        coordinates = set()
        for rank in range(8):
            native.get_tp_group = lambda current=rank: SimpleNamespace(rank_in_group=current % 2, world_size=2)
            native.get_pcp_group = lambda current=rank: SimpleNamespace(rank_in_group=(current // 2) % 2)
            ipc.npu_generate_uuid = lambda current=rank: f"host-{current}"
            worker = SimpleNamespace(parallel_config=SimpleNamespace(
                data_parallel_size=2, data_parallel_index=rank // 4, prefill_context_parallel_size=2,
            ))
            for visible in ("", "0,1,2,3,4,5,6,7"):
                with patch.dict("sys.modules", {"vllm.distributed": native,
                                                "vllm_ascend.distributed.weight_transfer.npu_ipc_engine": ipc}), \
                        patch.dict(os.environ, {"HYPER_RL_ROLLOUT_VISIBLE_DEVICES": visible}):
                    topology = _rollout_worker_topology(worker)
                self.assertEqual((topology["dp_rank"], topology["tp_rank"]), (rank // 2, rank % 2))
                self.assertEqual((topology["dp_size"], topology["tp_size"]), (4, 2))
                coordinates.add((topology["dp_rank"], topology["tp_rank"]))
        self.assertEqual(len(coordinates), 8)
