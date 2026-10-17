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
"""Optimizer state swap for Muon and for ChainedOptimizer chains.

State swap started as an Adam/AdamW-only path: it materialized ``exp_avg``,
``exp_avg_sq`` and ``max_exp_avg_sq``, and it assumed the wrapped object was a
``torch.optim.Optimizer`` it could drive with the functional kernels. Muon keeps
a single ``momentum_buffer`` and mixes Newton-Schulz with HSDP shard all-gathers,
and a chain holds no state of its own -- the state lives in its leaves. Both are
covered here, together with the step bookkeeping and checkpoint behaviour that
only show up once state is actually swapped out.
"""
# pylint: disable=wrong-import-position

import copy
import os
import unittest
import warnings

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch

from hyper_parallel.core.optimizer import SwapOptimizerConfig, swap_optimizer
from hyper_parallel.core.optimizer.adamw import AdamW as NewAdamW
from hyper_parallel.core.optimizer.lr_scheduler import LRSchedulersContainer
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.core.optimizer.swap_optimizer_base import (
    ChainedOptimizerAdapter,
    PipelineSwapRuntime,
    TorchMuonAdapter,
    TorchNativeAdamWAdapter,
    TorchNewAdamWAdapter,
    UpdateUnit,
)


class _TwinProjections(torch.nn.Module):
    """Two independent projection stacks, one per leaf optimizer."""

    def __init__(self) -> None:
        """Build one stack for the Muon leaf and one for the AdamW leaf."""
        super().__init__()
        self.muon_stack = torch.nn.Sequential(
            torch.nn.Linear(8, 8, bias=False),
            torch.nn.Linear(8, 8, bias=False),
        )
        self.adamw_stack = torch.nn.Sequential(
            torch.nn.Linear(8, 8, bias=False),
            torch.nn.Linear(8, 8, bias=False),
        )


def _build_chain(flatten: bool = False):
    """Build a deterministic Muon + AdamW chained optimizer over a twin model."""
    torch.manual_seed(0)
    model = _TwinProjections()
    muon = Muon(model.muon_stack.parameters(), lr=0.02, momentum=0.95, nesterov=True, ns_steps=5)
    adamw = torch.optim.AdamW(model.adamw_stack.parameters(), lr=1e-3, betas=(0.9, 0.95))
    return model, ChainedOptimizer(model, {"muon": muon, "adamw": adamw}, flatten=flatten)


def _swap_config(**overrides) -> SwapOptimizerConfig:
    """Return a per-tensor swap config suitable for CPU-only runs."""
    values = {"swap_times": 2, "packed_swap": False, "min_numel": 0}
    values.update(overrides)
    return SwapOptimizerConfig(**values)


def _run_steps(model, optimizer, x, steps: int = 2) -> list:
    """Run deterministic forward/backward/optimizer steps and return the parameters."""
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model.muon_stack(x).float().pow(2).sum() + model.adamw_stack(x).float().pow(2).sum()
        loss.backward()
        optimizer.step()
    return [param.detach().clone() for param in model.parameters()]


class TestChainedAdapterWiring(unittest.TestCase):
    """Construction and delegation of the chained swap adapter."""

    def test_chained_optimizer_builds_one_leaf_adapter_per_sub_optimizer(self):
        """A Muon + AdamW chain gets one adapter per leaf and never packs."""
        _, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config(packed_swap=True))

        self.assertIsInstance(
            wrapped.adapter,
            ChainedOptimizerAdapter,
            f"a chain must take the chained adapter, got {type(wrapped.adapter)}",
        )
        self.assertEqual(
            [type(leaf) for leaf in wrapped.adapter.leaves],
            [TorchMuonAdapter, TorchNativeAdamWAdapter],
            f"unexpected leaf adapters: {[type(leaf) for leaf in wrapped.adapter.leaves]}",
        )

    def test_chained_swap_forces_the_per_tensor_path(self):
        """Packed host staging is disabled even when the config requests it."""
        _, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config(packed_swap=True))

        muon_leaf, adamw_leaf = wrapped.adapter.leaves
        self.assertTrue(wrapped.config.packed_swap, "the config itself keeps the packed request")
        self.assertFalse(
            wrapped.runtime.packed_enabled,
            "a composite must fall back to the per-tensor path",
        )
        self.assertTrue(wrapped.adapter.requires_per_tensor_swap, "the composite must ask for it")
        self.assertTrue(muon_leaf.requires_per_tensor_swap, "Muon must ask for it")
        self.assertFalse(adamw_leaf.requires_per_tensor_swap, "AdamW alone can still pack")

    def test_wrapper_delegates_the_chained_optimizer_surface(self):
        """Trainer-facing attributes keep resolving through the wrapper."""
        _, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config())

        self.assertTrue(getattr(wrapped, "_is_swap_optimizer", False), "wrapper must stay identifiable")
        self.assertEqual(
            list(wrapped.optimizers_dict),
            ["muon", "adamw"],
            f"leaf names must resolve through the wrapper, got {list(wrapped.optimizers_dict)}",
        )
        self.assertIs(
            wrapped.chained_optimizers[0],
            chain.chained_optimizers[0],
            "the wrapper must expose the chain's own leaves",
        )
        self.assertIs(wrapped.model, chain.model, "the wrapper must expose the chain's model")
        self.assertEqual(
            len(wrapped.param_groups), 2, f"both leaves' groups must be visible, got {len(wrapped.param_groups)}"
        )

    def test_empty_chain_is_rejected(self):
        """A chain without leaves cannot be swapped."""
        runtime = PipelineSwapRuntime(_swap_config())
        with self.assertRaisesRegex(ValueError, "at least one sub-optimizer"):
            ChainedOptimizerAdapter(torch.nn.Module(), _swap_config(), runtime)

    def test_unsupported_leaf_reports_the_supported_optimizers(self):
        """A leaf type outside the adapter registry fails the wrap loudly."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        chain = ChainedOptimizer(model, {"sgd": torch.optim.SGD(model.parameters(), lr=0.1)})

        with self.assertRaisesRegex(ValueError, "hyper_parallel.core.optimizer.muon.Muon"):
            swap_optimizer(chain, _swap_config())


class TestChainedSwapParity(unittest.TestCase):
    """The swapped chain must reproduce the unwrapped chain step for step."""

    def _assert_parity(self, swap_times: int) -> None:
        """Step a swapped and an unwrapped chain over identical data and compare."""
        model_ref, chain_ref = _build_chain()
        model_swap, chain_swap = _build_chain()
        wrapped = swap_optimizer(chain_swap, _swap_config(swap_times=swap_times))

        torch.manual_seed(7)
        x = torch.randn(4, 8)
        params_ref = _run_steps(model_ref, chain_ref, x)
        params_swap = _run_steps(model_swap, wrapped, x)

        for (name, expected), actual in zip(model_ref.named_parameters(), params_swap):
            self.assertTrue(
                torch.equal(expected, actual),
                f"{name} diverged at swap_times={swap_times}",
            )

        muon_group = chain_swap.chained_optimizers[0].param_groups[0]
        self.assertEqual(
            muon_group["step"], 2, f"Muon step must advance once per outer step, got {muon_group['step']}"
        )

    def test_multi_partition_step_matches_the_unwrapped_chain(self):
        """Several swap partitions keep Muon momentum, AdamW moments and step counters aligned."""
        self._assert_parity(swap_times=3)

    def test_single_partition_step_matches_the_unwrapped_chain(self):
        """A single swap partition is the degenerate pipeline case."""
        self._assert_parity(swap_times=1)

    def test_step_counters_advance_once_per_outer_step(self):
        """Muon's step counter must not follow the number of swap batches."""
        _, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config(swap_times=4))

        _run_steps(chain.model, wrapped, torch.randn(4, 8), steps=3)

        step = chain.chained_optimizers[0].param_groups[0]["step"]
        self.assertEqual(step, 3, f"three outer steps must count three, got {step}")

    def test_muon_metadata_covers_every_parameter_after_a_step(self):
        """The per-batch restriction is undone at the end of the outer step."""
        _, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config(swap_times=2))
        muon = chain.chained_optimizers[0]
        total_params = sum(len(group["params"]) for group in muon.param_groups)

        _run_steps(chain.model, wrapped, torch.randn(4, 8), steps=2)

        unshard = muon.unshard_params_by_group.get(0, [])
        self.assertEqual(
            len(unshard),
            total_params,
            f"every Muon parameter must be back in the metadata, got {len(unshard)} of {total_params}",
        )
        self.assertEqual(
            len(muon.unshard_params_by_group),
            1,
            f"one group must be registered, got {len(muon.unshard_params_by_group)}",
        )

    def test_swapped_state_keys_are_tracked_per_leaf(self):
        """Muon tracks momentum buffers while AdamW tracks both Adam moments."""
        _, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config())
        _run_steps(chain.model, wrapped, torch.randn(4, 8), steps=1)

        names = {slot.name for slot in wrapped.adapter.all_slots()}
        self.assertEqual(
            names,
            {"momentum_buffer", "exp_avg", "exp_avg_sq"},
            f"unexpected swapped state keys: {names}",
        )


class TestPerLeafMuonSwap(unittest.TestCase):
    """The Trainer wraps each leaf, so a lone Muon must swap on its own."""

    def test_muon_only_wrap_matches_the_unwrapped_optimizer(self):
        """Wrapping one Muon leaf reproduces the unwrapped update."""
        torch.manual_seed(0)
        model_ref = _TwinProjections()
        model_swap = copy.deepcopy(model_ref)
        reference = Muon(model_ref.muon_stack.parameters(), lr=0.02, momentum=0.95, ns_steps=5)
        wrapped = swap_optimizer(
            Muon(model_swap.muon_stack.parameters(), lr=0.02, momentum=0.95, ns_steps=5),
            _swap_config(),
        )

        torch.manual_seed(7)
        x = torch.randn(4, 8)
        for _ in range(2):
            for model, optimizer in ((model_ref, reference), (model_swap, wrapped)):
                optimizer.zero_grad()
                model.muon_stack(x).float().pow(2).sum().backward()
                optimizer.step()

        for (name, expected), actual in zip(model_ref.named_parameters(), model_swap.parameters()):
            self.assertTrue(
                torch.equal(expected, actual),
                f"{name} diverged under Muon swap",
            )
        self.assertTrue(
            isinstance(wrapped.adapter, TorchMuonAdapter),
            f"a Muon leaf must take the Muon adapter, got {type(wrapped.adapter)}",
        )
        self.assertFalse(
            wrapped.runtime.packed_enabled,
            "Muon state must swap per tensor, never through a shared packed region",
        )


class TestMuonAdapterValidation(unittest.TestCase):
    """Muon-specific configuration guards."""

    def _adapter(self, optimizer) -> TorchMuonAdapter:
        """Build a Muon adapter over ``optimizer`` with a per-tensor runtime."""
        config = _swap_config()
        return TorchMuonAdapter(optimizer, config, PipelineSwapRuntime(config))

    def test_replicated_hsdp_assignment_is_rejected(self):
        """Replica broadcasts cover a whole assignment and cannot be split."""

        class _Replicated:
            """An HSDP assignment that broadcasts across ranks."""

            is_replicated = True

        class _ReplicatedMuon:
            """Muon-shaped stand-in whose only assignment is replicated."""

            param_groups = [{"params": []}]
            _hsdp_assignment_batches = {0: {"batch_groups": [{"sub_batches": [_Replicated()]}]}}

        with self.assertRaisesRegex(ValueError, "replicate"):
            self._adapter(_ReplicatedMuon()).validate()

    def test_non_replicated_assignments_validate(self):
        """A schedule without replica broadcasts is accepted."""

        class _Owned:
            """An HSDP assignment that stays on the owning rank."""

            is_replicated = False

        class _OwnedMuon:
            """Muon-shaped stand-in whose assignment is not replicated."""

            param_groups = [{"params": []}]
            _hsdp_assignment_batches = {0: {"batch_groups": [{"sub_batches": [_Owned()]}]}}

        self._adapter(_OwnedMuon()).validate()


class TestChainedSwapIntegration(unittest.TestCase):
    """Scheduler, checkpoint and unit-routing behaviour of the chained wrapper."""

    def test_lr_scheduler_container_steps_through_the_wrapper(self):
        """The trainer's scheduler container resolves the chain and mutates its leaf groups."""
        _, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config())
        # The trainer builds the scheduler before the first optimizer step, which
        # is where torch checks that the optimizer reported itself as stepped.
        container = LRSchedulersContainer(
            wrapped,
            scheduler=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step),
        )
        before = [group["lr"] for leaf in chain.chained_optimizers for group in leaf.param_groups]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _run_steps(chain.model, wrapped, torch.randn(4, 8), steps=1)
            container.step()

        after = [group["lr"] for leaf in chain.chained_optimizers for group in leaf.param_groups]
        scheduler_warnings = [w for w in caught if "lr_scheduler.step()" in str(w.message)]

        self.assertEqual(
            after, [lr * 0.5 for lr in before], f"learning rates must halve: before={before}, after={after}"
        )
        self.assertEqual(
            len(container.get_last_lr()),
            2,
            f"the container must expose one lr per leaf, got {container.get_last_lr()}",
        )
        self.assertEqual(
            scheduler_warnings, [], f"no scheduler warning expected, got {[str(w.message) for w in scheduler_warnings]}"
        )

    def test_every_leaf_reports_itself_as_stepped(self):
        """Both the functional AdamW path and the Muon path flag their optimizer."""
        _, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config())
        # Torch only tracks the flag when a scheduler patched the optimizer step.
        LRSchedulersContainer(
            wrapped,
            scheduler=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step),
        )
        _run_steps(chain.model, wrapped, torch.randn(4, 8), steps=1)

        flags = {type(leaf).__name__: getattr(leaf, "_opt_called", False) for leaf in chain.chained_optimizers}
        self.assertTrue(
            all(flags.values()), f"every leaf must report itself as stepped, got {flags}"
        )

    def test_update_units_route_to_the_owning_leaf(self):
        """Every unit of one outer step belongs to exactly one leaf adapter."""
        _, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config(swap_times=2))
        model = chain.model
        wrapped.zero_grad()
        loss = model.muon_stack(torch.randn(4, 8)).float().pow(2).sum()
        loss += model.adamw_stack(torch.randn(4, 8)).float().pow(2).sum()
        loss.backward()

        context = wrapped.adapter.prepare_step()
        units = wrapped.adapter.iter_update_units(context)
        owners = sorted({wrapped.adapter._owner_by_unit[id(unit)] for unit in units})  # pylint: disable=protected-access

        self.assertEqual(owners, [0, 1], f"both leaves must own units, got {owners}")
        self.assertTrue(
            all(isinstance(unit, UpdateUnit) for unit in units),
            f"unexpected unit types: {[type(unit) for unit in units]}",
        )

    def test_checkpoint_state_dict_uses_host_mirrors(self):
        """Swapped buffers are exported from the host mirror that owns the values."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
        chain = ChainedOptimizer(model, {"adamw": torch.optim.AdamW(model.parameters(), lr=1e-2)})
        wrapped = swap_optimizer(chain, _swap_config(min_numel=0))
        wrapped.zero_grad()
        model(torch.ones(2, 4)).pow(2).sum().backward()
        wrapped.step()

        param = next(model.parameters())
        slot = wrapped.adapter.leaves[0]._slots[(id(param), "exp_avg")]  # pylint: disable=protected-access
        slot.cpu_tensor = torch.full_like(slot.tensor, 3.0)
        slot.state = "host"

        exported = next(iter(wrapped.state_dict()["state"].values()))
        self.assertTrue(
            torch.equal(exported["exp_avg"], torch.full_like(exported["exp_avg"], 3.0)),
            f"the host mirror must be exported, got {exported['exp_avg'].flatten()[:4].tolist()}",
        )

    def test_checkpoint_state_dict_uses_host_mirrors_when_the_chain_flattens(self):
        """The flat ``state.<fqn>.<key>`` layout is substituted too.

        A chain holding both an AdamW leaf and a Muon leaf sets ``flatten``, so
        its state dict has *no* ``state`` mapping -- only flat
        ``state.<fqn>.<key>`` entries. Walking only ``state_dict["state"]``
        therefore exported the released placeholders untouched, i.e. a
        checkpoint that saved cleanly and restored garbage.
        """
        model, chain = _build_chain(flatten=True)
        wrapped = swap_optimizer(chain, _swap_config())
        wrapped.zero_grad()
        loss = model.muon_stack(torch.ones(2, 8)).float().pow(2).sum()
        loss += model.adamw_stack(torch.ones(2, 8)).float().pow(2).sum()
        loss.backward()
        wrapped.step()

        param = chain.chained_optimizers[1].param_groups[0]["params"][0]
        slot = wrapped.adapter.leaves[1]._slots[(id(param), "exp_avg")]  # pylint: disable=protected-access
        slot.cpu_tensor = torch.full_like(slot.tensor, 3.0)
        slot.state = "host"

        state_dict = wrapped.state_dict()
        self.assertNotIn(
            "state", state_dict, f"a flattened chain has no state mapping, got {sorted(state_dict)[:4]}"
        )
        flat_key = f"state.{param.model_name}.exp_avg"
        self.assertIn(
            flat_key, state_dict, f"expected {flat_key} in {[k for k in state_dict if k.endswith('exp_avg')]}"
        )
        self.assertTrue(
            torch.equal(state_dict[flat_key], torch.full_like(slot.tensor, 3.0)),
            f"{flat_key} should carry the host mirror, got {state_dict[flat_key].flatten()[:4].tolist()}",
        )

    def test_checkpoint_state_dict_refuses_an_unknown_layout(self):
        """Mirrors that land nowhere must fail loudly, not export placeholders.

        With no mirror on the host there is nothing to lose, so the guard only
        fires once a slot really is swapped out and still cannot be placed.
        """
        model, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config())
        wrapped.zero_grad()
        loss = model.muon_stack(torch.ones(2, 8)).float().pow(2).sum()
        loss += model.adamw_stack(torch.ones(2, 8)).float().pow(2).sum()
        loss.backward()
        wrapped.step()

        param = chain.chained_optimizers[1].param_groups[0]["params"][0]
        slot = wrapped.adapter.leaves[1]._slots[(id(param), "exp_avg")]  # pylint: disable=protected-access
        slot.cpu_tensor = torch.full_like(slot.tensor, 3.0)
        slot.state = "host"

        original = wrapped.optimizer.state_dict
        wrapped.optimizer.state_dict = lambda *args, **kwargs: {"unexpected.layout": torch.zeros(1)}
        try:
            with self.assertRaisesRegex(RuntimeError, "released placeholders"):
                wrapped.state_dict()
        finally:
            wrapped.optimizer.state_dict = original

    def test_checkpoint_restore_is_rejected_with_a_clear_error(self):
        """Chained restore cannot re-offload FQN-keyed state and must fail loudly."""
        _, chain = _build_chain()
        wrapped = swap_optimizer(chain, _swap_config())

        with self.assertRaisesRegex(NotImplementedError, "Resume with optimizer.swap disabled"):
            wrapped.load_state_dict({})

    def test_new_adamw_leaf_keeps_the_adamw_adapter(self):
        """A hyper-parallel AdamW leaf keeps its own adapter instead of the torch one."""
        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
        chain = ChainedOptimizer(model, {"adamw": NewAdamW(model.parameters(), lr=1e-2)})
        wrapped = swap_optimizer(chain, _swap_config())

        self.assertIsInstance(
            wrapped.adapter.leaves[0],
            TorchNewAdamWAdapter,
            f"unexpected adapter for the fused AdamW leaf: {type(wrapped.adapter.leaves[0])}",
        )


class TestChainedSwapReferenceParameters(unittest.TestCase):
    """Guard the parity harness itself: the reference chain must move the model."""

    def test_reference_chain_changes_the_parameters(self):
        """A parity test is meaningless if the reference optimizer is a no-op."""
        model, chain = _build_chain()
        before = copy.deepcopy([param.detach().clone() for param in model.parameters()])
        after = _run_steps(model, chain, torch.randn(4, 8), steps=1)

        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, after)),
            "the unwrapped chain must move at least one parameter",
        )


if __name__ == "__main__":
    unittest.main()
