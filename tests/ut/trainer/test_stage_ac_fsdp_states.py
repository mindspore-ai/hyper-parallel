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
"""``_iter_hsdp_states`` must see FSDP states created in the PP stage tree.

Under PP+FSDP the stage builder's activation-checkpoint pass re-points
``stage.layers[i]`` at a fresh ``CheckpointWrapper`` which is then
``fully_shard``-ed. That HSDP module exists only in the stage tree (the full
model still references the bare layer), so a materialize walk over
``self.model`` alone misses its state: ``lazy_init`` and the flat-buffer
rebase never run and the first all-gather copies out of meta local shards
("Cannot copy out of meta tensor"). These tests pin the reachability contract
with the exact production mechanics (wrapper class-swapped to an HSDP
subclass), no process group needed.
"""
import types

from torch import nn

# The dispatched ``checkpoint_wrapper`` binds at import time, so it can be the
# MindSpore one when another module imported the package first; these tests
# build torch modules, so take the torch implementation directly.
from hyper_parallel.platform.torch.activation_checkpoint.checkpoint_wrapper import (
    ckpt_wrapper as checkpoint_wrapper,
)
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.trainer.base import BaseTrainer


class _Block(nn.Module):
    """Tiny stand-in decoder layer."""

    def __init__(self) -> None:
        """One linear so the wrapper has parameters to share."""
        super().__init__()
        self.lin = nn.Linear(4, 4)


def _attach_hsdp_state(module: nn.Module) -> object:
    """Mimic ``fully_shard``: class-swap onto an HSDPModule subclass + state."""
    swapped = type(f"HSDP{type(module).__name__}", (type(module), HSDPModule), {})
    module.__class__ = swapped
    state = types.SimpleNamespace(tag=id(module))
    module.hsdp_scheduler = types.SimpleNamespace(hsdp_state=state)
    return state


def _trainer_with(model: nn.Module) -> BaseTrainer:
    """Bare trainer exposing only what ``_iter_hsdp_states`` reads."""
    trainer = object.__new__(BaseTrainer)
    trainer.model = model
    return trainer


def test_stage_only_wrapper_state_missed_without_stage_roots():
    """Without stage roots the stage-tree-only state is not enumerated."""
    model = nn.Module()
    layer = _Block()
    model.layers = nn.ModuleList([layer])

    # The stage builder wraps the SHARED layer in a stage-local container.
    stage_module = nn.Module()
    stage_module.layers = nn.ModuleList([checkpoint_wrapper(layer)])
    stage_state = _attach_hsdp_state(stage_module.layers[0])

    trainer = _trainer_with(model)
    assert not list(trainer._iter_hsdp_states())  # pylint: disable=protected-access

    trainer._pp_stage_modules = [stage_module]  # pylint: disable=protected-access
    assert list(trainer._iter_hsdp_states()) == [stage_state]  # pylint: disable=protected-access


def test_states_deduped_when_reachable_from_both_trees():
    """A state reachable from both trees is yielded once."""
    model = nn.Module()
    layer = _Block()
    model.layers = nn.ModuleList([layer])
    shared_state = _attach_hsdp_state(layer)

    stage_module = nn.Module()
    stage_module.layers = nn.ModuleList([layer])  # same object in both trees

    trainer = _trainer_with(model)
    trainer._pp_stage_modules = [stage_module]  # pylint: disable=protected-access
    assert list(trainer._iter_hsdp_states()) == [shared_state]  # pylint: disable=protected-access


def test_absent_stage_roots_attribute_is_tolerated():
    """A trainer without the stage-roots attribute still enumerates."""
    model = nn.Module()
    layer = _Block()
    model.layers = nn.ModuleList([layer])
    state = _attach_hsdp_state(layer)

    trainer = _trainer_with(model)  # no _pp_stage_modules set (non-PP path)
    assert list(trainer._iter_hsdp_states()) == [state]  # pylint: disable=protected-access
