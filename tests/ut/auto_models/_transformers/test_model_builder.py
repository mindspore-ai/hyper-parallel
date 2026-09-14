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
"""Unit tests for Transformers tied-weight model construction."""
# pylint: disable=protected-access,wrong-import-position

import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
from torch import nn

from hyper_parallel.models._transformers import model_builder
from hyper_parallel.models._transformers.checkpoint_loader import LoadReport
from tests.common.mark_utils import arg_mark


class _TiedCausalLM(nn.Module):
    """Minimal model exposing the Transformers tied-weight contract."""

    def __init__(self, *, device: str = "cpu", effective_tie: bool = True) -> None:
        """Create a split embedding/head pair on the requested device."""
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4, device=device)
        self.lm_head = nn.Linear(4, 8, bias=False, device=device)
        self.all_tied_weights_keys = {
            "lm_head.weight": "model.embed_tokens.weight",
        }
        self.effective_tie = effective_tie

    def tie_weights(self) -> None:
        """Install the input embedding Parameter at the output head."""
        if self.effective_tie:
            self.lm_head.weight = self.model.embed_tokens.weight


class _DeclaredWithoutTie(nn.Module):
    """Model declaring tied weights without implementing their binding API."""

    def __init__(self) -> None:
        """Create independent parameters and a declared tied mapping."""
        super().__init__()
        self.embed = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 2, bias=False)
        self.all_tied_weights_keys = {"head.weight": "embed.weight"}


class _RecordingCheckpointManager:
    """Record checkpoint discovery after materialization completes."""

    events: list[str] = []

    def __init__(self, model: nn.Module) -> None:
        """Record construction, which represents target discovery."""
        del model
        self.events.append("checkpoint-target-discovery")

    def load_checkpoint(self, *_args: object, **_kwargs: object) -> LoadReport:
        """Return a minimal report for the mocked finalization path."""
        self.events.append("checkpoint-load")
        return LoadReport((), (), ())


class TestTiedWeightPreparation(unittest.TestCase):
    """Validate tied identities before parameter sharding."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pre_shard_tie_restores_parameter_identity(self):
        """
        Feature: Pre-sharding tied-weight restoration.
        Description: Apply the model tie contract to an initially split pair.
        Expectation: The output head and embedding share one Parameter identity.
        """
        model = _TiedCausalLM()

        model_builder._tie_model_weights_before_sharding(model)

        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pre_shard_tie_requires_callable_contract(self):
        """
        Feature: Tied-weight contract validation.
        Description: Prepare a model that declares aliases without a tie API.
        Expectation: Preparation fails before distributed wrapping begins.
        """
        with self.assertRaisesRegex(ValueError, "no callable tie_weights"):
            model_builder._tie_model_weights_before_sharding(_DeclaredWithoutTie())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pre_shard_tie_rejects_ineffective_contract(self):
        """
        Feature: Tied-weight identity validation.
        Description: Run a tie API that leaves the declared pair independent.
        Expectation: Preparation rejects the ineffective model contract.
        """
        with self.assertRaisesRegex(ValueError, "do not share Parameter identity"):
            model_builder._tie_model_weights_before_sharding(
                _TiedCausalLM(effective_tie=False)
            )


class TestMaterializationHandshake(unittest.TestCase):
    """Validate tied restoration immediately after meta materialization."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_non_fsdp_materialization_reties_parameters(self):
        """
        Feature: Non-FSDP meta materialization.
        Description: Materialize a tied meta model directly onto CPU.
        Expectation: The materialized parameters remain non-meta aliases.
        """
        model = _TiedCausalLM(device="meta")
        model.tie_weights()

        result = model_builder._move_model_to_device(
            model,
            is_meta_device=True,
            device=torch.device("cpu"),
        )

        self.assertIs(result.lm_head.weight, result.model.embed_tokens.weight)
        self.assertFalse(result.lm_head.weight.is_meta)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_hsdp_refresh_runs_after_to_empty_and_restores_alias(self):
        """
        Feature: HSDP meta materialization handshake.
        Description: Materialize a model whose tied pair is managed by HSDP.
        Expectation: HSDP refresh sees real storage and restores the alias.
        """
        model = _TiedCausalLM(device="meta")
        model.tie_weights()
        hsdp_state = MagicMock()

        def lazy_init() -> None:
            """Emulate HSDP installing its materialized canonical Parameter."""
            self.assertFalse(model.model.embed_tokens.weight.is_meta)
            model.lm_head.weight = model.model.embed_tokens.weight

        hsdp_state.lazy_init.side_effect = lazy_init
        with patch.object(
            model_builder,
            "get_hsdp_state",
            side_effect=lambda module: hsdp_state if module is model else None,
        ):
            model_builder._move_model_to_device(
                model,
                is_meta_device=True,
                device=torch.device("cpu"),
            )

        hsdp_state.lazy_init.assert_called_once_with()
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_materialization_deduplicates_shared_hsdp_state(self):
        """
        Feature: HSDP materialization refresh deduplication.
        Description: Expose one HSDP state through every module in the model tree.
        Expectation: The shared state is refreshed exactly once.
        """
        model = _TiedCausalLM()
        model.tie_weights()
        hsdp_state = MagicMock()

        with patch.object(model_builder, "get_hsdp_state", return_value=hsdp_state):
            model_builder._restore_tied_weights_after_materialization(model)

        hsdp_state.lazy_init.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_hsdp_refresh_precedes_checkpoint_discovery(self):
        """
        Feature: HSDP checkpoint-loading order.
        Description: Record HSDP refresh, checkpoint discovery, and load events.
        Expectation: Alias restoration completes before checkpoint target discovery.
        """
        model = _TiedCausalLM(device="meta")
        model.tie_weights()
        events = []
        hsdp_state = MagicMock()

        def lazy_init() -> None:
            """Record HSDP restoration and install the canonical Parameter."""
            events.append("hsdp-refresh")
            model.lm_head.weight = model.model.embed_tokens.weight

        hsdp_state.lazy_init.side_effect = lazy_init
        _RecordingCheckpointManager.events = events
        with (
            patch.object(
                model_builder,
                "get_hsdp_state",
                side_effect=lambda module: hsdp_state if module is model else None,
            ),
            patch.object(model_builder, "CheckpointManager", _RecordingCheckpointManager),
            patch.object(model_builder, "_finalize_model_loading"),
        ):
            model_builder._materialize_and_load_model(
                model,
                is_meta_device=True,
                device=torch.device("cpu"),
                load_base_model=True,
                pretrained_path="unused",
                weights_mapping=None,
            )

        self.assertEqual(
            events,
            ["hsdp-refresh", "checkpoint-target-discovery", "checkpoint-load"],
        )


if __name__ == "__main__":
    unittest.main()
