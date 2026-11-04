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
"""Local model/optimizer state and rank-owned training state codecs."""

import copy
import hashlib
import random
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Access persistent local storage without redistributing a DTensor."""
    return getattr(tensor, "_local_tensor", tensor)


def freeze(value: Any) -> Any:
    """Copy rank-owned training metadata; never use this for model or optimizer tensors."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("training metadata cannot contain device tensors; register them as model buffers")
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    if isinstance(value, list):
        return [freeze(item) for item in value]
    if isinstance(value, tuple):
        return tuple(freeze(item) for item in value)
    return copy.deepcopy(value)


class ModelOptimizerState:
    """Bind live device shards without optimizer ``state_dict`` collectives.

    A collection of optimizers is one transaction: every leaf is included.
    Main parameters are discovered from the model's persistent ``main_param``
    identities. Restore updates storage in place and copies main values back
    to computation parameters explicitly.
    """

    def __init__(self, model: torch.nn.Module, optimizer: Any) -> None:
        """Bind persistent local parameters and every supported optimizer leaf."""
        self.model = model
        self.parameters = dict(model.named_parameters())
        self.buffers = dict(model.named_buffers())
        self.extra_modules = {name: module for name, module in model.named_modules()
                              if type(module).get_extra_state is not torch.nn.Module.get_extra_state}
        self.leaves = []
        roots = optimizer if isinstance(optimizer, (tuple, list)) else [optimizer]
        for root in roots:
            self.leaves.extend(getattr(root, "chained_optimizers", [root]))
        if not self.leaves or any(not isinstance(leaf, (AdamW, Muon)) for leaf in self.leaves):
            raise ValueError("process recovery supports HyperParallel AdamW and Muon leaves only")
        self.names = {}
        for name, parameter in self.parameters.items():
            self.names[id(parameter)] = name
            main = getattr(parameter, "main_param", None)
            if main is not None:
                self.names[id(main)] = name
        owned = []
        for leaf in self.leaves:
            for group in leaf.param_groups:
                if group.get("amsgrad", False):
                    raise ValueError("AMSGrad process recovery is not supported")
                for parameter in group["params"]:
                    if id(parameter) not in self.names:
                        raise ValueError("optimizer parameter has no stable model FQN")
                    owned.append(id(parameter))
        if len(owned) != len(set(owned)):
            raise ValueError("a recovery parameter must belong to exactly one optimizer leaf")
        self.schema = self._schema()
        self.optimizer_ownership = self._optimizer_ownership()

    def _optimizer_ownership(self) -> str:
        """Fingerprint the state slots this rank owns after Muon's second partition."""
        ownership = []
        for leaf in self.leaves:
            if isinstance(leaf, Muon):
                owned = set()
                for info in leaf._hsdp_assignment_batches.values():
                    owned.update(id(param) for param in info["no_comm"])
                    for batch in info["batch_groups"]:
                        for assignment in batch["sub_batches"]:
                            owned.update(id(param) for param in assignment.owned_params)
            else:
                owned = {id(param) for group in leaf.param_groups for param in group["params"]}
            ownership.append(tuple(tuple(self.names[id(param)] for param in group["params"] if id(param) in owned)
                                   for group in leaf.param_groups))
        return hashlib.sha256(repr(ownership).encode()).hexdigest()

    def _schema(self) -> str:
        """Fingerprint local shard shape, dtype, aliases and optimizer ownership."""
        items = [(name, tuple(local_tensor(param).shape), str(param.dtype),
                  str(getattr(param, "placements", ()))) for name, param in self.parameters.items()]
        items += [("buffer:" + name, tuple(buffer.shape), str(buffer.dtype))
                  for name, buffer in self.buffers.items()]
        items += [("aliases", tuple((name, self.names[id(param)])
                                    for name, param in self.model.named_parameters(remove_duplicate=False)))]
        items += [("optimizer", tuple((type(leaf).__module__, type(leaf).__qualname__,
                                       getattr(leaf, "hsdp_replica_count", None),
                                       tuple(tuple(self.names[id(param)] for param in group["params"])
                                             for group in leaf.param_groups)) for leaf in self.leaves))]
        items += [("extra_modules", tuple(self.extra_modules))]
        return hashlib.sha256(repr(items).encode()).hexdigest()

    def metadata(self) -> dict[str, Any]:
        """Describe live device slots without exporting parameter or moment values."""
        self.validate_identities()
        optimizers = []
        for leaf in self.leaves:
            groups = []
            for group in leaf.param_groups:
                states = {}
                for parameter in group["params"]:
                    values = {}
                    for key, value in leaf.state.get(parameter, {}).items():
                        if isinstance(value, torch.Tensor):
                            values[key] = {"tensor": True, "shape": tuple(local_tensor(value).shape),
                                           "dtype": value.dtype}
                        else:
                            values[key] = {"tensor": False, "value": copy.deepcopy(value)}
                    states[self.names[id(parameter)]] = values
                options = copy.deepcopy({key: value for key, value in group.items() if key != "params"})
                groups.append({"options": options,
                               "states": states})
            optimizers.append(groups)
        return {"schema": self.schema, "optimizer_ownership": self.optimizer_ownership, "optimizers": optimizers}

    def validate_identities(self) -> None:
        """Reject objects that no longer refer to the persistent local shards."""
        for expected, actual in ((self.parameters, dict(self.model.named_parameters())),
                                 (self.buffers, dict(self.model.named_buffers()))):
            if expected.keys() != actual.keys() or any(actual[name] is not value for name, value in expected.items()):
                raise RuntimeError("recovery requires stable persistent parameter and buffer identities")

    def tensor_items(self) -> list[tuple[str, torch.Tensor]]:
        """Return source device tensors themselves, never model/optimizer snapshots."""
        self.validate_identities()
        result = [("parameter:" + name, local_tensor(parameter)) for name, parameter in self.parameters.items()]
        for index, leaf in enumerate(self.leaves):
            for group in leaf.param_groups:
                for parameter in group["params"]:
                    name = self.names[id(parameter)]
                    if parameter is not self.parameters[name]:
                        result.append((f"main:{index}:{name}", local_tensor(parameter)))
                    for key, value in sorted(leaf.state.get(parameter, {}).items()):
                        if isinstance(value, torch.Tensor):
                            result.append((f"state:{index}:{name}:{key}", local_tensor(value)))
        return result

    def prepare_receive(self, metadata: Mapping[str, Any]) -> None:
        """Allocate matching moment slots on the receiver's device and preserve layouts."""
        if metadata["schema"] != self.schema:
            raise RuntimeError("recovery state schema/layout mismatch")
        if metadata["optimizer_ownership"] != self.optimizer_ownership:
            raise RuntimeError("recovery optimizer-state ownership mismatch")
        for leaf, groups in zip(self.leaves, metadata["optimizers"]):
            for group, saved in zip(leaf.param_groups, groups):
                for key in list(group):
                    if key != "params":
                        del group[key]
                group.update(copy.deepcopy(saved["options"]))
                for parameter in group["params"]:
                    values = {}
                    for key, entry in saved["states"][self.names[id(parameter)]].items():
                        if entry["tensor"]:
                            if tuple(local_tensor(parameter).shape) != tuple(entry["shape"]):
                                raise RuntimeError("optimizer moment does not match its local parameter shard")
                            values[key] = torch.zeros_like(parameter, dtype=entry["dtype"])
                        else:
                            values[key] = copy.deepcopy(entry["value"])
                    leaf.state[parameter] = values

    def rank_state(self) -> dict[str, Any]:
        """Retain rank-local buffers on their original device and small extra metadata."""
        extra = {name: freeze(module.get_extra_state()) for name, module in self.extra_modules.items()}
        return {"buffers": {name: local_tensor(buffer).detach().clone() for name, buffer in self.buffers.items()},
                "extra": extra}

    @torch.no_grad()
    def restore_rank_state(self, buffers: Mapping[str, torch.Tensor], extra: Mapping[str, Any]) -> None:
        """Restore rank-owned device buffers, then align computation and main parameters."""
        for name, buffer in self.buffers.items():
            local_tensor(buffer).copy_(buffers[name])
        for name, module in self.extra_modules.items():
            module.set_extra_state(copy.deepcopy(extra[name]))
        for name, parameter in self.parameters.items():
            main = getattr(parameter, "main_param", None)
            if main is not None and main is not parameter:
                local_tensor(parameter).copy_(local_tensor(main))
            parameter.grad = None
            if hasattr(parameter, "main_grad"):
                parameter.main_grad = None
        for leaf in self.leaves:
            for group in leaf.param_groups:
                for parameter in group["params"]:
                    parameter.grad = None

    def rebuild_runtime_state(self) -> None:
        """Collect Muon shard metadata only after every worker's groups are usable."""
        for leaf in self.leaves:
            if isinstance(leaf, Muon):
                leaf.rebuild_recovery_state()
        if self._optimizer_ownership() != self.optimizer_ownership:
            raise RuntimeError("optimizer-state ownership changed after communication rebuild")


class TrainState:
    """Capture rank-owned RNG, consumed data cursor and named state providers.

    Providers and the loader must implement state_dict/load_state_dict with
    collective-free operations. A loader must describe consumed data, not
    prefetched positions. Iterator recreation belongs to the training body,
    after rollback has restored this provider. Only providers explicitly
    listed in ``replica_providers`` may use the donor's same-step snapshot;
    RNG, loader and unlisted providers always retain target-rank state.
    """

    def __init__(self, loader: Any, providers: Mapping[str, Any], device: Any = None,
                 generators: Mapping[str, torch.Generator] = None,
                 replica_providers: tuple[str, ...] = ()) -> None:
        """Validate providers and optional donor-reconstructible provider names.

        ``replica_providers`` are stored locally and transferred only during
        repair. Their states must be interchangeable between full replicas,
        or observational values whose donor provenance is recorded.
        """
        self.loader = loader
        self.providers = dict(providers)
        self.device = device
        self.generators = dict(generators or {})
        self.replica_providers = frozenset(replica_providers)
        if not self.replica_providers.issubset(self.providers):
            raise ValueError("replica providers must refer to registered training state providers")
        for value in [loader, *self.providers.values()]:
            if (not callable(getattr(value, "state_dict", None))
                    or not callable(getattr(value, "load_state_dict", None))):
                raise ValueError("recovery requires state_dict/load_state_dict for every training state provider")

    def capture(self) -> dict[str, Any]:
        """Snapshot the current logical rank, including forward-mutated buffers via providers."""
        return freeze({"python": random.getstate(), "numpy": np.random.get_state(),
                       "torch": torch.get_rng_state(),
                       "device": self.device.get_rng_state() if self.device is not None else None,
                       "generators": {name: generator.get_state() for name, generator in self.generators.items()},
                       "loader": self.loader.state_dict(),
                       "providers": {name: provider.state_dict() for name, provider in self.providers.items()}})

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore this rank's state, never the equivalent shard donor's RNG/cursor."""
        if set(state["providers"]) != set(self.providers):
            raise RuntimeError("training state provider schema mismatch")
        if set(state["generators"]) != set(self.generators):
            raise RuntimeError("custom RNG generator schema mismatch")
        if self.loader.load_state_dict(copy.deepcopy(state["loader"])) is False:
            raise RuntimeError("data loader refused the consumed-data cursor rollback")
        for name, provider in self.providers.items():
            provider.load_state_dict(copy.deepcopy(state["providers"][name]))
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if self.device is not None:
            self.device.set_rng_state(state["device"])
        for name, generator in self.generators.items():
            generator.set_state(state["generators"][name])

    def make_iterator(self, preserve_rng: bool = False) -> Iterator[Any]:
        """Keep DataLoader iterator construction from consuming committed RNG twice."""
        if not preserve_rng:
            return iter(self.loader)
        state = self.capture()
        iterator = iter(self.loader)
        # The cursor/provider state may legitimately change while restoring
        # an iterator. Only undo RNG draws introduced by that construction.
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if self.device is not None:
            self.device.set_rng_state(state["device"])
        for name, generator in self.generators.items():
            generator.set_state(state["generators"][name])
        return iterator
