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
from typing import Any, Mapping

import numpy as np
import torch

from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.swap_optimizer_base import SwapOptimizer
from hyper_parallel.high_availability.validation import validate_recovery_state_providers
from hyper_parallel.models.external_state import get_model_external_state


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Access persistent local storage without redistributing a DTensor."""
    return getattr(tensor, "_local_tensor", tensor)


def clone_metadata(value: Any) -> Any:
    """Copy rank-owned training metadata; never use this for model or optimizer tensors."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("training metadata cannot contain device tensors; register them as model buffers")
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_metadata(item) for item in value)
    return copy.deepcopy(value)


class ModelOptimizerState:
    """Bind live local shards without optimizer ``state_dict`` collectives.

    A collection of optimizers is one transaction: every leaf is included.
    Main parameters are discovered from the model's persistent ``main_param``
    identities. Restore updates storage in place and copies main values back
    to computation parameters explicitly.
    """

    def __init__(self, model: torch.nn.Module, optimizer: Any) -> None:
        """Bind persistent local parameters and every supported optimizer leaf."""
        self.model = model
        self.external_state = get_model_external_state(model)
        self.parameters = dict(model.named_parameters())
        self.buffers = dict(model.named_buffers())
        self.extra_modules = {name: module for name, module in model.named_modules()
                              if type(module).get_extra_state is not torch.nn.Module.get_extra_state}
        self.leaves = []
        roots = optimizer if isinstance(optimizer, (tuple, list)) else [optimizer]
        for root in roots:
            self.leaves.extend(getattr(root, "chained_optimizers", [root]))
        self._swap_leaves = {index: leaf for index, leaf in enumerate(self.leaves) if isinstance(leaf, SwapOptimizer)}
        self.leaves = [self._swap_leaves[index].optimizer if index in self._swap_leaves else leaf
                       for index, leaf in enumerate(self.leaves)]
        sparse = self.external_state.sparse_optimizer if self.external_state is not None else None
        if not self.leaves or any(not isinstance(leaf, (AdamW, Muon))
                                  and not (leaf is sparse and isinstance(leaf, torch.optim.SparseAdam))
                                  for leaf in self.leaves):
            raise ValueError("process recovery supports HyperParallel AdamW/Muon and Host Engram SparseAdam only")
        self.names = {}
        for name, parameter in self.parameters.items():
            self.names[id(parameter)] = name
            main = getattr(parameter, "main_param", None)
            if main is not None:
                self.names[id(main)] = name
        self._validate_optimizer_parameters()
        self.schema = self._schema()
        self.optimizer_ownership = self._optimizer_ownership()

    def _validate_optimizer_parameters(self) -> None:
        """Require supported moments and one optimizer owner per model parameter."""
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

    def _optimizer_ownership(self, rank: int = None) -> str:
        """Fingerprint the state slots this rank owns after Muon's second partition."""
        ownership = []
        for leaf in self.leaves:
            if isinstance(leaf, Muon):
                owned = self._muon_owned_parameter_ids(leaf, rank)
            else:
                owned = {id(param) for group in leaf.param_groups for param in group["params"]}
            ownership.append(tuple(tuple(self.names[id(param)] for param in group["params"] if id(param) in owned)
                                   for group in leaf.param_groups))
        return hashlib.sha256(repr(ownership).encode()).hexdigest()

    @staticmethod
    def _muon_owned_parameter_ids(optimizer: Muon, rank: int = None) -> set[int]:
        """Resolve owners from Muon's existing local and distributed assignments."""
        owned = set()
        assignments = vars(optimizer)["_hsdp_assignment_batches"]
        for info in assignments.values():
            owned.update(id(param) for param in info["no_comm"])
            for batch in info["batch_groups"]:
                for assignment in batch["sub_batches"]:
                    if rank is None:
                        owned.update(id(param) for param in assignment.owned_params)
                        continue
                    domain = batch["hsdp_group"].comm_key
                    coordinate = np.unravel_index(domain.mesh_rank_list.index(rank), domain.mesh_shape)
                    logical = tuple(coordinate[axis] % size for axis, size in
                                    zip(domain.replicate_mesh_dims, assignment.replicate_sizes))
                    owned.update(id(record.param) for record in assignment.all_records
                                 if assignment.owner_by_index.get(record.index) == logical)
        return owned

    def replica_ownership(self, rank: int) -> str:
        """Derive a model replica's optimizer owners from the existing Muon assignment map.

        Args:
            rank: Global rank with the same local model shard layout.

        Returns:
            Optimizer-state ownership fingerprint without remote communication.
        """
        return self._optimizer_ownership(rank)

    def _schema(self) -> str:
        """Fingerprint local shard shape, dtype, storage device, aliases and optimizer ownership."""
        items = [(name, tuple(local_tensor(param).shape), str(param.dtype), local_tensor(param).device.type,
                  str(getattr(param, "placements", ()))) for name, param in self.parameters.items()]
        items += [("buffer:" + name, tuple(buffer.shape), str(buffer.dtype), local_tensor(buffer).device.type)
                  for name, buffer in self.buffers.items()]
        items += [("aliases", tuple((name, self.names[id(param)])
                                    for name, param in self.model.named_parameters(remove_duplicate=False)))]
        items += [("optimizer", tuple((type(leaf).__module__, type(leaf).__qualname__,
                                       getattr(leaf, "hsdp_replica_count", None),
                                       tuple(tuple(self.names[id(param)] for param in group["params"])
                                             for group in leaf.param_groups)) for leaf in self.leaves))]
        items += [("optimizer_swap", tuple((index, swap.config.packed_swap, swap.config.min_numel,
                                              swap.config.state_keys) for index, swap in self._swap_leaves.items()))]
        items += [("extra_modules", tuple(self.extra_modules))]
        if self.external_state is not None:
            items += [("external_state", self.external_state.recovery_schema())]
        return hashlib.sha256(repr(items).encode()).hexdigest()

    def _optimizer_state(self, index: int, parameter: torch.nn.Parameter) -> Mapping[str, Any]:
        swap = self._swap_leaves.get(index)
        if swap is not None:
            return swap.adapter.recovery_state(parameter)
        return self.leaves[index].state.get(parameter, {})

    def metadata(self) -> dict[str, Any]:
        """Describe live device slots without exporting parameter or moment values."""
        self.validate_identities()
        optimizers = []
        for index, leaf in enumerate(self.leaves):
            groups = []
            for group in leaf.param_groups:
                states = {}
                for parameter in group["params"]:
                    values = {}
                    for key, value in self._optimizer_state(index, parameter).items():
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

    def validate_layout(self) -> None:
        """Check that training initialization preserved the validated recovery layout."""
        self.validate_identities()
        if self._schema() != self.schema or self._optimizer_ownership() != self.optimizer_ownership:
            raise RuntimeError("training initialization changed the validated recovery layout")

    def validate_identities(self) -> None:
        """Reject objects that no longer refer to the persistent local shards."""
        for expected, actual in ((self.parameters, dict(self.model.named_parameters())),
                                 (self.buffers, dict(self.model.named_buffers()))):
            if expected.keys() != actual.keys() or any(actual[name] is not value for name, value in expected.items()):
                raise RuntimeError("recovery requires stable persistent parameter and buffer identities")

    def tensor_items(self) -> list[tuple[str, torch.Tensor]]:
        """Return live local tensors, including CPU offload, never model/optimizer snapshots."""
        self.validate_identities()
        result = [("parameter:" + name, local_tensor(parameter)) for name, parameter in self.parameters.items()]
        for index, leaf in enumerate(self.leaves):
            for group in leaf.param_groups:
                for parameter in group["params"]:
                    name = self.names[id(parameter)]
                    if parameter is not self.parameters[name]:
                        result.append((f"main:{index}:{name}", local_tensor(parameter)))
                    for key, value in sorted(self._optimizer_state(index, parameter).items()):
                        if isinstance(value, torch.Tensor):
                            result.append((f"state:{index}:{name}:{key}", local_tensor(value)))
        return result

    def prepare_receive(self, metadata: Mapping[str, Any]) -> None:
        """Allocate matching moment slots on the parameter storage device and preserve layouts.

        Args:
            metadata: Donor schema, optimizer ownership and per-parameter state descriptions.
        """
        if metadata["schema"] != self.schema:
            raise RuntimeError("recovery state schema/layout mismatch")
        if metadata["optimizer_ownership"] != self.optimizer_ownership:
            raise RuntimeError("recovery optimizer-state ownership mismatch")
        for index, (leaf, groups) in enumerate(zip(self.leaves, metadata["optimizers"])):
            swap = self._swap_leaves.get(index)
            if swap is not None:
                swap.adapter.clear_recovery_slots()
            for group, saved in zip(leaf.param_groups, groups):
                for key in list(group):
                    if key != "params":
                        del group[key]
                group.update(copy.deepcopy(saved["options"]))
                for parameter in group["params"]:
                    values = {}
                    for key, entry in saved["states"][self.names[id(parameter)]].items():
                        values[key] = self._receive_optimizer_value(swap, parameter, key, entry, group=group)
                    leaf.state[parameter] = values
            if swap is not None:
                swap.adapter.publish_packed_state()

    @staticmethod
    def _receive_optimizer_value(
            swap: Any,
            parameter: torch.nn.Parameter,
            key: str,
            entry: Mapping,
            *,
            group: Mapping[str, Any],
    ) -> Any:
        """Allocate the actual receive destination for one local optimizer value."""
        if not entry["tensor"]:
            return copy.deepcopy(entry["value"])
        shape = tuple(entry["shape"])
        parameter_shape = tuple(local_tensor(parameter).shape)
        if swap is not None and key == "step" and not shape:
            device = swap.adapter.step_state_device(parameter, group)
            return torch.empty((), dtype=entry["dtype"], device=device)
        if shape != parameter_shape:
            raise RuntimeError("optimizer moment does not match its local parameter shard")
        if swap is not None:
            return swap.adapter.allocate_recovery_state(parameter, key, entry["dtype"])
        return torch.zeros_like(parameter, dtype=entry["dtype"])

    def rank_state(self) -> dict[str, Any]:
        """Retain rank-local buffers on their original device and small extra metadata."""
        extra = {name: clone_metadata(module.get_extra_state()) for name, module in self.extra_modules.items()}
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
    """Capture rank-owned RNG/cursor and replica-equivalent training providers.

    The loader and replica providers must implement collective-free
    state_dict/load_state_dict. Loader/RNG/Generators retain target-rank
    state; every replica provider may use the donor's same-step snapshot.
    Iterator recreation follows rollback in the original training body.
    """

    def __init__(self, loader: Any, replica_providers: Mapping[str, Any], device: Any = None,
                 generators: Mapping[str, torch.Generator] = None) -> None:
        """Bind rank-specific state and replica-equivalent provider objects.

        Args:
            loader: Stateful loader describing the consumed cursor and pending data.
            replica_providers: Named stateful objects interchangeable between full replicas.
            device: Optional device API exposing RNG get/set operations.
            generators: Explicit rank-specific Torch generators keyed by name.
        """
        validate_recovery_state_providers(loader, replica_providers)
        self.loader = loader
        self.replica_providers = dict(replica_providers)
        self.device = device
        self.generators = dict(generators or {})
        self._pending_rng_state = None

    def capture(self) -> dict[str, Any]:
        """Snapshot separate rank-owned and replica-equivalent state groups."""
        rank_state = self._capture_rng()
        rank_state["loader"] = self.loader.state_dict()
        return clone_metadata({
            "rank_state": rank_state,
            "replica_state": {name: provider.state_dict() for name, provider in self.replica_providers.items()},
        })

    def _capture_rng(self) -> dict[str, Any]:
        """Read random state without exporting loader or replica-provider state."""
        return {
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "device": self.device.get_rng_state() if self.device is not None else None},
            "generators": {name: generator.get_state() for name, generator in self.generators.items()},
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        """Restore loader/providers and stage RNG until iterator reconstruction finishes.

        Args:
            state: Rank-specific loader/RNG state and replica-equivalent provider snapshot.

        Note:
            The training main thread must call restore_pending_rng after startup/epoch callbacks
            and iterator construction, before consuming training data.
        """
        if set(state["replica_state"]) != set(self.replica_providers):
            raise RuntimeError("replica training state provider schema mismatch")
        rank_state = state["rank_state"]
        if set(rank_state["generators"]) != set(self.generators):
            raise RuntimeError("custom RNG generator schema mismatch")
        if self.loader.load_state_dict(copy.deepcopy(rank_state["loader"])) is False:
            raise RuntimeError("data loader refused the consumed-data cursor rollback")
        for name, provider in self.replica_providers.items():
            provider.load_state_dict(copy.deepcopy(state["replica_state"][name]))
        self._pending_rng_state = clone_metadata({"rng": rank_state["rng"], "generators": rank_state["generators"]})

    def restore_pending_rng(self) -> None:
        """Apply pending RNG once after callbacks and iterator reconstruction.

        Failed application retains the pending state so that the caller can retry.
        Ordinary epoch initialization without a recovery does not change RNG.
        """
        if self._pending_rng_state is not None:
            self._restore_rng(self._pending_rng_state)
            self._pending_rng_state = None

    def _restore_rng(self, rng_state: Mapping[str, Any]) -> None:
        """Apply the committed random streams before training consumes restored data."""
        rng = rng_state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if self.device is not None:
            self.device.set_rng_state(rng["device"])
        for name, generator in self.generators.items():
            generator.set_state(rng_state["generators"][name])
