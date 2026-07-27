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
"""Checkpoint metadata version migration.

Provides a sequential migration chain for :class:`Metadata` objects loaded from
disk.  Each rule advances the version by exactly one step; unknown or future
versions fail fast with a descriptive :class:`ValueError`.
"""

from typing import Callable

from hyper_parallel.core.distributed_checkpoint.metadata import Metadata


CURRENT_CHECKPOINT_VERSION = "2.0"


def _migrate_1_0_to_2_0(metadata: Metadata) -> Metadata:
    """Migrate metadata from format version 1.0 to 2.0.

    The 1.0 format has no explicit ``version`` field on deserialized pickled
    objects.  This rule sets the version to ``"2.0"`` without altering any
    tensor/chunk/storage structure.

    Args:
        metadata: A Metadata object whose effective version is ``"1.0"``.

    Returns:
        Metadata: The same object with ``version`` set to ``"2.0"``.
    """
    metadata.version = "2.0"
    return metadata


_MIGRATION_RULES: dict[str, Callable[[Metadata], Metadata]] = {
    "1.0": _migrate_1_0_to_2_0,
}


def _get_effective_version(metadata: Metadata) -> str:
    """Return the effective version string of *metadata*.

    For pickled objects that predate the ``version`` field the instance
    attribute may be missing entirely; in that case we treat the version as
    ``"1.0"``.  We check ``metadata.__dict__`` directly so that a missing
    instance-level attribute is not masked by the dataclass default.

    Args:
        metadata: A Metadata object, possibly deserialized from an old pickle.

    Returns:
        str: The effective version string.
    """
    if "version" in metadata.__dict__:
        return metadata.__dict__["version"]
    return "1.0"


def migrate_metadata(metadata: Metadata) -> Metadata:
    """Migrate *metadata* to the current checkpoint format version.

    The migration follows a directed, per-version rule chain.  On each
    iteration the effective version is looked up in :data:`_MIGRATION_RULES`.
    If the version equals :data:`CURRENT_CHECKPOINT_VERSION` the metadata is
    returned as-is.  If a rule exists it is applied and the loop continues.
    The loop terminates with a :class:`ValueError` if a rule does not advance
    the version, if the version is not registered, or if the final version
    would exceed the current format version.

    Args:
        metadata: A Metadata object, possibly loaded from an old checkpoint.

    Returns:
        Metadata: The migrated metadata at the current format version.

    Raises:
        ValueError: If the version cannot be migrated (unknown, future,
            no-progress rule, or migration result type error).
    """
    max_steps = len(_MIGRATION_RULES) + 1
    for _ in range(max_steps):
        version = _get_effective_version(metadata)
        if version == CURRENT_CHECKPOINT_VERSION:
            return metadata
        if version not in _MIGRATION_RULES:
            raise ValueError(
                f"Cannot migrate metadata from version {version!r} to "
                f"{CURRENT_CHECKPOINT_VERSION!r}: no migration rule registered."
            )
        prev_version = version
        metadata = _MIGRATION_RULES[version](metadata)
        if not isinstance(metadata, Metadata):
            raise ValueError(
                f"Migration rule for version {prev_version!r} returned "
                f"{type(metadata).__name__} instead of Metadata."
            )
        new_version = _get_effective_version(metadata)
        if new_version == prev_version:
            raise ValueError(
                f"Migration rule for version {prev_version!r} did not advance "
                f"the version (still {new_version!r})."
            )
        if new_version > CURRENT_CHECKPOINT_VERSION:
            raise ValueError(
                f"Migration rule for version {prev_version!r} produced a "
                f"future version {new_version!r} which exceeds the current "
                f"format version {CURRENT_CHECKPOINT_VERSION!r}."
            )
    raise ValueError(
        "Migration did not converge within the expected number of steps."
    )
