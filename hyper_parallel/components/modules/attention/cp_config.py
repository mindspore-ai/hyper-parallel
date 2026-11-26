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
"""Shared public context-parallel configuration for attention families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CP_SIZE_DEFAULTS = {
    "ulysses_cp_size": 1,
    "kv_cp_size": 1,
    "ring_cp_size": 1,
}
CP_STRATEGY_DEFAULTS = {
    "ulysses_cp_strategy": "alltoall",
    "kv_cp_strategy": "allgather",
    "ring_cp_strategy": "p2p",
}
CP_LOAD_BALANCE_DEFAULTS = {
    "kv_cp_load_balance": False,
    "ring_cp_load_balance": False,
}
CP_OVERLAP_DEFAULT = {"cp_comm_overlap": False}
CP_ARGUMENT_DEFAULTS = {
    **CP_SIZE_DEFAULTS,
    **CP_STRATEGY_DEFAULTS,
    **CP_LOAD_BALANCE_DEFAULTS,
    **CP_OVERLAP_DEFAULT,
}


def resolve_attention_cp_config(
    member: Any,
) -> dict[str, Any]:
    """Validate common CP arguments from a resolved family member.

    Args:
        member: Resolved family selection with public arguments and CP topology.
    Returns:
        A fresh mapping containing all public CP arguments with defaults applied.
    """
    return resolve_attention_cp_arguments(member.arguments, member.axis_sizes["cp"])


def resolve_attention_cp_arguments(
    arguments: Mapping[str, Any],
    cp_size: int,
) -> dict[str, Any]:
    """Resolve common CP arguments when only constructor arguments are available."""
    config = dict(CP_ARGUMENT_DEFAULTS)
    config.update({name: arguments[name] for name in CP_ARGUMENT_DEFAULTS if name in arguments})

    for name in CP_SIZE_DEFAULTS:
        value = config[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in CP_STRATEGY_DEFAULTS:
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError(f"{name} must be a non-empty string")
    for name in CP_LOAD_BALANCE_DEFAULTS:
        value = config[name]
        if value is not False and value != "zigzag":
            raise ValueError(f"{name} must be false or 'zigzag'")
    if not isinstance(config["cp_comm_overlap"], bool):
        raise TypeError("cp_comm_overlap must be bool")

    configured_size = config["ulysses_cp_size"] * config["kv_cp_size"] * config["ring_cp_size"]
    if cp_size > 1 and configured_size != cp_size:
        raise ValueError(
            "cp_size must equal ulysses_cp_size * kv_cp_size * ring_cp_size; "
            f"got {cp_size} != {config['ulysses_cp_size']} * {config['kv_cp_size']} * "
            f"{config['ring_cp_size']}"
        )
    return config


def validate_supported_cp_options(
    config: Mapping[str, Any],
    family: str,
    *,
    ulysses_strategies: tuple[str, ...] = ("alltoall",),
    kv_strategies: tuple[str, ...] = ("allgather",),
    ring_strategies: tuple[str, ...] = ("p2p",),
    allow_ulysses: bool = True,
    allow_kv: bool = True,
    allow_ring: bool = False,
    allow_kv_load_balance: bool = False,
    allow_ring_load_balance: bool = False,
    allow_overlap: bool = False,
) -> None:
    """Reject CP choices that a concrete attention family cannot implement."""
    strategies = (
        ("ulysses_cp_strategy", ulysses_strategies),
        ("kv_cp_strategy", kv_strategies),
        ("ring_cp_strategy", ring_strategies),
    )
    for name, supported in strategies:
        if config[name] not in supported:
            raise ValueError(f"{family} does not support {name}={config[name]!r}; supported: {supported}")

    unsupported_domains = []
    if not allow_ulysses and config["ulysses_cp_size"] > 1:
        unsupported_domains.append("ulysses_cp_size")
    if not allow_kv and config["kv_cp_size"] > 1:
        unsupported_domains.append("kv_cp_size")
    if not allow_ring and config["ring_cp_size"] > 1:
        unsupported_domains.append("ring_cp_size")
    if unsupported_domains:
        raise NotImplementedError(f"{family} does not support CP domains: {', '.join(unsupported_domains)}")

    if config["kv_cp_load_balance"] and not allow_kv_load_balance:
        raise NotImplementedError(f"{family} does not support kv_cp_load_balance")
    if config["ring_cp_load_balance"] and not allow_ring_load_balance:
        raise NotImplementedError(f"{family} does not support ring_cp_load_balance")
    if config["cp_comm_overlap"] and not allow_overlap:
        raise NotImplementedError(f"{family} does not expose cp_comm_overlap")
