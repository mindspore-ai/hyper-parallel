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
"""ShardBackend protocol + registry. No platform deps here."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

from tests.shard_ops.framework.case_spec import CompareSpec, InputSpec


class ShardBackend(ABC):
    """Adapter that hides framework/device differences from the harness."""

    # Identity used by registry keys.
    framework: str = "<override>"
    device_type: str = "<override>"

    @abstractmethod
    def maybe_init_dist(self) -> None:
        """Idempotent: initialise the distributed runtime once per process."""

    @abstractmethod
    def get_or_init_mesh(self, shape: Tuple[int, ...],
                         names: Tuple[str, ...]) -> Any:
        """Return a cached device mesh; build it on first call."""

    @abstractmethod
    def make_tensor(self, spec: InputSpec) -> Any:
        """Materialise a full (unsharded) tensor on the local device."""

    @abstractmethod
    def distribute(self, full_tensor: Any, mesh: Any,
                   placements: Tuple[Any, ...]) -> Any:
        """Wrap ``full_tensor`` as a distributed tensor."""

    @abstractmethod
    def local_to_global(self, dist_tensor: Any) -> Any:
        """Gather all shards back to a full tensor."""

    @abstractmethod
    def assert_close(self, expected: Any, actual: Any,
                     spec: CompareSpec) -> None:
        """Compare two full tensors per ``spec``."""

    def recover_after_failure(self) -> bool:
        """Best-effort recovery after a case raised.

        Returns True if the process is still usable for subsequent cases.
        Default: True (most local AssertionErrors leave the comm group OK).
        """
        return True


# (framework, device_type) -> ShardBackend subclass
_BACKEND_REGISTRY: Dict[Tuple[str, str], Callable[[], ShardBackend]] = {}
_INSTANCES: Dict[Tuple[str, str], ShardBackend] = {}


def register_backend(framework: str, device_type: str,
                     factory: Callable[[], ShardBackend]) -> None:
    """Platform-specific ``framework/__init__.py`` calls this at import time."""
    key = (framework, device_type)
    _BACKEND_REGISTRY[key] = factory


def resolve_backend(framework: str, device_type: str) -> ShardBackend:
    """Look up and instantiate (singleton) the backend for the given pair."""
    key = (framework, device_type)
    if key not in _BACKEND_REGISTRY:
        raise RuntimeError(
            f"no backend registered for framework={framework!r}, "
            f"device_type={device_type!r}. Did you import the platform "
            f"framework package (e.g. tests.torch.shard.ops.framework)?"
        )
    if key not in _INSTANCES:
        _INSTANCES[key] = _BACKEND_REGISTRY[key]()
    return _INSTANCES[key]


def list_backends() -> List[Tuple[str, str]]:
    """Return all currently registered ``(framework, device_type)`` keys."""
    return list(_BACKEND_REGISTRY.keys())
