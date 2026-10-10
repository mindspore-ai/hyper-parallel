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
"""CPU-only precision checks for owner-local Host Engram rows."""

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file

from hyper_parallel.components.modules.engram import EngramModule, NgramHashMapping
from hyper_parallel.components.checkpoint.weight_conversion import WeightRenaming
from hyper_parallel.models.deepseek_v41.adapter.engram.host_replacement import HostEngramModule
from hyper_parallel.models.deepseek_v41.adapter.engram.host_state import DeepseekV41HostState
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.external_state import ExternalLoadContext
from tests.common.mark_utils import arg_mark


class TestHostEngramTable(unittest.TestCase):
    """Compare touched-row gradients with a full device-style embedding."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_ep1_repeated_rows_and_sparse_step(self):
        """Compare repeated Host rows to a dense embedding.

        Feature: EP1 sparse CPU update.
        Description: Gather repeated IDs and apply equal SparseAdam steps.
        Expectation: Coalesced gradients, updated weights, and optimizer moments equal the reference.
        """
        table = HostEngramTable(
            source_weight=torch.nn.Parameter(torch.empty((16, 2), device="meta")),
            logical_rows=15, physical_rows=16, width=2,
        )
        table.bind_planned_shard(ep_rank=0, ep_size=1)
        reference = torch.nn.Embedding(16, 2)
        with torch.no_grad():
            reference.weight.copy_(torch.arange(32, dtype=torch.float32).view(16, 2))
        original_id = id(table.weight)
        with torch.no_grad():
            reference_values = reference.weight.clone()
        torch.utils.swap_tensors(table.weight, torch.nn.Parameter(reference_values))
        self.assertEqual(id(table.weight), original_id)

        ids = torch.tensor([2, 10, 2])
        upstream = torch.tensor([[1.0, 1.0], [0.0, 5.0], [1.0, 3.0]])
        torch.testing.assert_close(table(ids), reference(ids))
        table(ids).backward(upstream)
        reference(ids).backward(upstream)
        sparse = table.coalesce_pending()
        self.assertIsNone(table.weight.grad)
        torch.testing.assert_close(sparse.to_dense(), reference.weight.grad)
        self.assertEqual(sparse.indices()[0].tolist(), [2, 10])

        table.pending = sparse
        table.install_grad()
        sparse_optimizer = torch.optim.SparseAdam([table.weight], lr=0.1)
        dense_optimizer = torch.optim.SparseAdam([reference.weight], lr=0.1)
        reference.weight.grad = reference.weight.grad.to_sparse().coalesce()
        sparse_optimizer.step()
        dense_optimizer.step()
        torch.testing.assert_close(table.weight, reference.weight)
        self.assertEqual(sparse_optimizer.state[table.weight]["step"], dense_optimizer.state[reference.weight]["step"])
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                sparse_optimizer.state[table.weight][key],
                dense_optimizer.state[reference.weight][key],
            )
        table.clear_step()
        self.assertIsNone(table.weight.grad)
        self.assertFalse(table.pending_chunks)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_empty_and_padding(self):
        """Check empty requests and logical padding.

        Feature: Host table ID bounds.
        Description: Query zero IDs and the first padding row.
        Expectation: Empty output succeeds and padding raises IndexError.
        """
        table = HostEngramTable(
            source_weight=torch.nn.Parameter(torch.empty((16, 2), device="meta")),
            logical_rows=15, physical_rows=16, width=2,
        )
        table.bind_planned_shard(ep_rank=0, ep_size=1)
        torch.utils.swap_tensors(table.weight, torch.nn.Parameter(torch.zeros(16, 2)))
        self.assertEqual(tuple(table(torch.empty(0, dtype=torch.long)).shape), (0, 2))
        self.assertEqual(table.coalesce_pending()._nnz(), 0)
        with self.assertRaises(IndexError):
            table(torch.tensor([15]))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_pretrained_slice_uses_mapped_key_and_rejects_duplicates(self):
        """Check mapped source slices and duplicate rejection.

        Feature: Owner-only pretrained materialization.
        Description: Load a mapped safetensors table for EP rank one.
        Expectation: Only its rows load and duplicate logical sources fail.
        """
        table = HostEngramTable(
            source_weight=torch.nn.Parameter(torch.empty((16, 2), device="meta")),
            logical_rows=15, physical_rows=16, width=2,
        )
        table.weight = torch.nn.Parameter(torch.empty((8, 2), device="meta"))
        table.bind_planned_shard(ep_rank=1, ep_size=2, ep_group=object())
        model = torch.nn.Module()
        model.embed = table
        state = DeepseekV41HostState(model, {"embed": table}, None)
        mapping = [WeightRenaming(source_patterns="source.weight", target_patterns="embed.weight")]
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            full = torch.arange(32, dtype=torch.float32).reshape(16, 2)
            save_file({"source.weight": full}, checkpoint)
            context = ExternalLoadContext(True, directory, mapping)
            result = state.materialize(context)
            self.assertEqual(result.source_keys, frozenset({"source.weight"}))
            torch.testing.assert_close(table.weight, full[8:])

            save_file({"source.weight": full, "embed.weight": full.clone()}, checkpoint)
            with self.assertRaisesRegex(ValueError, "Multiple Host Engram source keys"):
                state.materialize(context)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_full_engram_forward_and_backward_match_device_mode(self):
        """Compare all module outputs and gradients with device Engram.

        Feature: Host versus device Engram module precision.
        Description: Run hashing, segment mask, fusion, and backward identically.
        Expectation: Outputs and dense, input, and row gradients agree.
        """
        assets = {
            "layer_ids": [0], "max_ngram_size": 2, "num_heads": 2,
            "token_map": list(range(6)), "pad_token_id": 0,
            "primes": [[[5, 7]]], "multipliers": [[1, 3]],
        }
        source = torch.nn.Module()
        source.layer_id = 0
        source.hidden_size = 4
        source.hc_mult = 1
        source.eps = 1.0e-6
        source.clamp_value = 1.0e-6
        source.hash_mapping = NgramHashMapping(assets, 0)
        source.logical_num_embeddings = 12
        source.padded_num_embeddings = 16
        source.engram_max_pending_entries = 1000
        source.engram_max_sparse_rows_per_step = 1000
        source.embed = torch.nn.Embedding(16, 3)
        source.wkv = torch.nn.Linear(6, 8, bias=False)
        source.q_weight = torch.nn.Parameter(torch.randn(1, 4))
        source.k_weight = torch.nn.Parameter(torch.randn(1, 4))
        host_source = deepcopy(source)
        weight = host_source.embed.weight.detach().clone()
        host_source.embed.weight = torch.nn.Parameter(torch.empty((16, 3), device="meta"))
        device = EngramModule(module=source)
        host = HostEngramModule(module=host_source)
        host.embed.bind_planned_shard(ep_rank=0, ep_size=1)
        torch.utils.swap_tensors(host.embed.weight, torch.nn.Parameter(weight))

        input_ids = torch.tensor([[1, 2, 3, 4]])
        starts = torch.tensor([[1, 0, 1, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        hidden_device = torch.randn(1, 4, 1, 4, requires_grad=True)
        hidden_host = hidden_device.detach().clone().requires_grad_()
        device_output = device(hidden_device, input_ids, starts, mask)
        host_output = host(hidden_host, input_ids, starts, mask)
        torch.testing.assert_close(host_output, device_output)
        upstream = torch.randn_like(device_output)
        device_output.backward(upstream)
        host_output.backward(upstream)
        torch.testing.assert_close(hidden_host.grad, hidden_device.grad)
        for name in ("q_weight", "k_weight", "wkv.weight"):
            torch.testing.assert_close(host.get_parameter(name).grad,
                                       device.get_parameter(name).grad)
        torch.testing.assert_close(host.embed.coalesce_pending().to_dense(), device.embed.weight.grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_pretrained_manifest_reads_owner_and_rejects_row_gap(self):
        """Check physical row fragments for exact owner coverage.

        Feature: Pre-cut Host table source manifest.
        Description: Materialize EP rank one from two safetensors fragments.
        Expectation: Owner rows match and a physical row gap raises ValueError.
        """
        table = HostEngramTable(
            source_weight=torch.nn.Parameter(torch.empty((16, 2), device="meta")),
            logical_rows=15, physical_rows=16, width=2,
        )
        table.weight = torch.nn.Parameter(torch.empty((8, 2), device="meta"))
        table.bind_planned_shard(ep_rank=1, ep_size=2, ep_group=object())
        model = torch.nn.Module()
        model.embed = table
        state = DeepseekV41HostState(model, {"embed": table}, None)
        mapping = [WeightRenaming(source_patterns="source.weight", target_patterns="embed.weight")]
        full = torch.arange(32, dtype=torch.float32).reshape(16, 2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"dense.weight": torch.ones(1)}, root / "model.safetensors")
            fragments = []
            for start in (0, 8):
                filename = f"part_{start}.safetensors"
                save_file({"source.weight": full[start:start + 8].clone()}, root / filename)
                fragments.append({"start": start, "rows": 8,
                                  "file": filename, "key": "source.weight"})
            manifest = {"format_version": 1, "tables": {
                "source.weight": {"shape": [16, 2], "fragments": fragments},
            }}
            manifest_path = root / "engram_host_rows.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = state.materialize(ExternalLoadContext(True, directory, mapping))
            self.assertEqual(result.source_keys, frozenset({"source.weight"}))
            torch.testing.assert_close(table.weight, full[8:])

            fragments[1]["start"] = 9
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row gap or overlap"):
                state.materialize(ExternalLoadContext(True, directory, mapping))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_activation_checkpoint_adds_backward_rows_once(self):
        """Count sparse rows once across activation recomputation.

        Feature: Host autograd with activation checkpointing.
        Description: Recompute a repeated-row lookup in backward.
        Expectation: One backward chunk has the exact unrepeated gradient.
        """
        table = HostEngramTable(
            source_weight=torch.nn.Parameter(torch.empty((16, 2), device="meta")),
            logical_rows=15, physical_rows=16, width=2,
        )
        table.bind_planned_shard(ep_rank=0, ep_size=1)
        torch.utils.swap_tensors(table.weight, torch.nn.Parameter(torch.ones((16, 2))))
        ids = torch.tensor([2, 2])
        scale = torch.ones((), requires_grad=True)
        result = checkpoint(lambda value: table(ids).sum() * value, scale, use_reentrant=False)
        result.backward()
        self.assertEqual(len(table.pending_chunks), 1)
        torch.testing.assert_close(table.coalesce_pending().values(), torch.tensor([[2.0, 2.0]]))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_random_owner_initialization_is_layer_specific(self):
        """Keep deterministic owner replicas distinct across layers.

        Feature: Host table random materialization.
        Description: Materialize two layers with the same row interval twice.
        Expectation: Layers differ while repeated builds reproduce each layer.
        """
        def make_state():
            model = torch.nn.Module()
            tables = {}
            for layer in ("layer_0", "layer_1"):
                table = HostEngramTable(
                    source_weight=torch.nn.Parameter(torch.empty((16, 2), device="meta")),
                    logical_rows=15, physical_rows=16, width=2,
                )
                table.bind_planned_shard(ep_rank=0, ep_size=1)
                owner = torch.nn.Module()
                owner.embed = table
                setattr(model, layer, owner)
                tables[f"{layer}.embed"] = table
            return DeepseekV41HostState(model, tables, None)

        first = make_state()
        second = make_state()
        context = ExternalLoadContext(False, None, None)
        first.materialize(context)
        second.materialize(context)
        self.assertFalse(torch.equal(first.tables["layer_0.embed"].weight,
                                     first.tables["layer_1.embed"].weight))
        for name in first.tables:
            torch.testing.assert_close(first.tables[name].weight, second.tables[name].weight)
