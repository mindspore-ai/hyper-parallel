# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Checkpointer interface --- owns the on-disk checkpoint format.

The split against ``CheckpointerCallback`` is deliberate:

* the **callback** decides *when* to save, *what* goes into the payload, and how
  a restored payload maps back onto trainer objects (progress counters, LR
  scheduler, dataloader position, RNG);
* the **checkpointer** knows only how to write that payload to a directory and
  read it back --- storage layout, marker files, async plumbing, and the
  backend-specific load semantics.

Swapping the persistence backend is therefore a matter of registering another
:class:`CheckpointerBase`, with no change to the callback.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from hyper_models.components.utils.registry import Registry


CHECKPOINTER_REGISTRY = Registry("checkpointer")


def build_checkpointer(
    ckpt_manager: str = "dcp",
    **kwargs: Any,
) -> "CheckpointerBase":
    """Instantiate the checkpointer backend named by ``ckpt_manager``.

    Args:
        ckpt_manager: Registered backend name.
        **kwargs: Forwarded to the backend constructor.

    Returns:
        The constructed checkpointer.
    """
    return CHECKPOINTER_REGISTRY[ckpt_manager](**kwargs)


class CheckpointerBase(ABC):
    """Persist and restore a training payload for one checkpoint directory.

    The payload is a plain dict whose ``model`` / ``optimizer`` entries are
    state dicts, plus an optional ``extra_state`` bundle of non-tensor training
    progress. Implementations own where each part physically lands.
    """

    @abstractmethod
    def save(
        self,
        path: str,
        state: Dict[str, Any],
        *,
        global_step: int,
        save_async: bool = False,
    ) -> None:
        """Write ``state`` into the checkpoint directory ``path``.

        Args:
            path: Directory for this checkpoint; created if absent.
            state: Payload to persist (``model`` / ``optimizer`` /
                optionally ``extra_state``).
            global_step: Step this checkpoint represents, recorded so a later
                run can tell a complete checkpoint from a half-written one.
            save_async: Return before persistence finishes. The caller must
                reach :meth:`maybe_wait_for_async_save` before relying on the
                checkpoint or exiting the process.
        """

    @abstractmethod
    def load(
        self,
        path: str,
        state: Dict[str, Any],
        *,
        strict_model: bool = True,
        extra_state_skeleton: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fill ``state`` in place from the checkpoint at ``path``.

        Args:
            path: Directory holding the checkpoint.
            state: Payload to fill; its existing keys define what is read.
            strict_model: Require every requested ``model`` key to exist in the
                checkpoint. Disable for adapter-only (PEFT) checkpoints, which
                legitimately omit frozen base weights.
            extra_state_skeleton: A dict shaped like the ``extra_state`` that
                was saved. Required to restore a bundle that was embedded in
                the payload, because a backend fills only the keys the caller
                already declares. Pass ``None`` to skip ``extra_state`` restore.

        Returns:
            The same ``state`` object, filled.
        """

    def maybe_wait_for_async_save(self) -> None:
        """Block until an in-flight async save is fully persisted.

        Safe to call when nothing is pending. Backends without async support
        inherit this no-op.
        """
        return

    def find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """Return the newest resumable checkpoint under ``checkpoint_dir``.

        An instance method like the rest of this interface, so a backend whose
        discovery needs configuration (an object-store client, credentials, a
        bucket) has somewhere to read it from. Returns ``None`` when the
        directory holds no usable checkpoint.
        """
        raise NotImplementedError


@CHECKPOINTER_REGISTRY.register("dcp")
def _dcp_checkpointer(**kwargs: Any) -> CheckpointerBase:
    """Construct the distributed-checkpoint (DCP) backend."""
    # Imported lazily: dcp_checkpointer imports CheckpointerBase from this
    # module, so a top-level import here would be circular.
    from hyper_models.components.checkpoint.dcp_checkpointer import (  # pylint: disable=import-outside-toplevel
        DistributedCheckpointer,
    )

    return DistributedCheckpointer(**kwargs)


__all__ = [
    "CHECKPOINTER_REGISTRY",
    "CheckpointerBase",
    "build_checkpointer",
]
