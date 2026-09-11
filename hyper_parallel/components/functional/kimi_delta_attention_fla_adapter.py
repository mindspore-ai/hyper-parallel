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
"""Lazy adapter for the external FLA Triton-Ascend KDA backend."""
from __future__ import annotations

import importlib
import importlib.util
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional

import torch


_MIN_FLA_VERSION = (0, 6, 0)


@dataclass(frozen=True)
class FLAKDAStagedOps:
    """FLA operators required around Hyper's affine state summary."""

    l2norm_fwd: Callable[..., Any]
    l2norm_bwd: Callable[..., Any]
    fused_beta_sigmoid: Callable[..., Any]
    fused_beta_sigmoid_bwd: Callable[..., Any]
    kda_gate_chunk_cumsum: Callable[..., Any]
    kda_gate_bwd: Callable[..., Any]
    chunk_kda_fwd_intra: Callable[..., Any]
    recompute_w_u_fwd: Callable[..., Any]
    chunk_gated_delta_rule_fwd_h: Callable[..., Any]
    chunk_gla_fwd_o_gk: Callable[..., Any]
    chunk_kda_bwd_dav: Callable[..., Any]
    chunk_gated_delta_rule_bwd_dhu: Callable[..., Any]
    chunk_kda_bwd_wy_dqkg_fused: Callable[..., Any]
    chunk_kda_bwd_intra: Callable[..., Any]
    chunk_local_cumsum: Callable[..., Any]


@dataclass(frozen=True)
class _FLAKDARuntime:
    """Resolved external FLA runtime and its required KDA entry points."""

    version: str
    chunk_kda: Callable[..., Any]
    staged: FLAKDAStagedOps


def _get_attribute(module_name: str, attribute: str) -> Any:
    """Import one required FLA symbol with a useful compatibility error."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None)
        is_fla_module = (
            missing_name is not None and str(missing_name).startswith("fla.")
        )
        if missing_name == module_name or is_fla_module:
            raise RuntimeError(
                "The installed FLA package is incompatible with Hyper KDA: "
                f"missing module {module_name}."
            ) from exc
        raise RuntimeError(
            f"FLA module {module_name} could not load runtime dependency "
            f"{missing_name!r}."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            f"FLA module {module_name} failed to import; inspect the chained "
            "exception for the incompatible runtime library."
        ) from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise RuntimeError(
            "The installed FLA package is incompatible with Hyper KDA: "
            f"missing {module_name}.{attribute}. Install an FLA revision with "
            "the Triton-Ascend KDA backend."
        ) from exc


def _parse_version(version: str) -> tuple[int, int, int]:
    """Parse the numeric prefix of an FLA semantic version."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"Unable to parse the installed FLA version {version!r}.")
    return tuple(int(part) for part in match.groups())


def _require_triton_ascend() -> None:
    """Require the external FLA runtime to have a Triton Ascend backend."""
    try:
        importlib.import_module("triton")
        ascend_backend = importlib.util.find_spec("triton.backends.ascend")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "KDA backend='triton' requires Triton-Ascend in the runtime environment."
        ) from exc
    if ascend_backend is None:
        raise RuntimeError(
            "KDA backend='triton' found Triton, but its Ascend backend is missing."
        )


def _require_npu_backends() -> None:
    """Verify that FLA registered every backend used by the staged P2P path."""
    backend_specs = (
        (
            "fla.ops.kda.backends.triton_ascend",
            "TritonAscendKDABackend",
        ),
        (
            "fla.ops.common.backends.triton_ascend",
            "TritonAscendCommonBackend",
        ),
        (
            "fla.ops.gla.backends.triton_ascend",
            "TritonAscendGLABackend",
        ),
    )
    for module_name, class_name in backend_specs:
        backend = _get_attribute(module_name, class_name)
        if not backend.is_available():
            raise RuntimeError(
                "KDA backend='triton' requires FLA's Triton-Ascend backend "
                f"{class_name}, but it is unavailable in this process."
            )


@lru_cache(maxsize=1)
def _require_fla_kda_runtime() -> _FLAKDARuntime:
    """Resolve and validate the optional external FLA dependency once."""
    try:
        fla = importlib.import_module("fla")
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None)
        if missing_name != "fla":
            raise RuntimeError(
                "The external FLA package is present but cannot load runtime "
                f"dependency {missing_name!r}."
            ) from exc
        raise RuntimeError(
            "KDA backend='triton' requires the optional flash-linear-attention "
            "package with its Triton-Ascend KDA backend."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "The external FLA package is present but failed to import; inspect "
            "the chained exception for the incompatible runtime library."
        ) from exc

    version = getattr(fla, "__version__", None)
    if not isinstance(version, str):
        raise RuntimeError("The installed FLA package does not expose __version__.")
    if _parse_version(version) < _MIN_FLA_VERSION:
        minimum = ".".join(str(part) for part in _MIN_FLA_VERSION)
        raise RuntimeError(
            f"KDA backend='triton' requires FLA >= {minimum}, got {version}."
        )

    _require_triton_ascend()
    _require_npu_backends()
    staged = FLAKDAStagedOps(
        l2norm_fwd=_get_attribute("fla.modules.l2norm", "l2norm_fwd"),
        l2norm_bwd=_get_attribute("fla.modules.l2norm", "l2norm_bwd"),
        fused_beta_sigmoid=_get_attribute(
            "fla.ops.common.gate", "fused_beta_sigmoid"
        ),
        fused_beta_sigmoid_bwd=_get_attribute(
            "fla.ops.common.gate", "fused_beta_sigmoid_bwd"
        ),
        kda_gate_chunk_cumsum=_get_attribute(
            "fla.ops.kda.gate", "kda_gate_chunk_cumsum"
        ),
        kda_gate_bwd=_get_attribute("fla.ops.kda.gate", "kda_gate_bwd"),
        chunk_kda_fwd_intra=_get_attribute(
            "fla.ops.kda.chunk_intra", "chunk_kda_fwd_intra"
        ),
        recompute_w_u_fwd=_get_attribute(
            "fla.ops.kda.wy_fast", "recompute_w_u_fwd"
        ),
        chunk_gated_delta_rule_fwd_h=_get_attribute(
            "fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_h"
        ),
        chunk_gla_fwd_o_gk=_get_attribute(
            "fla.ops.gla.chunk", "chunk_gla_fwd_o_gk"
        ),
        chunk_kda_bwd_dav=_get_attribute(
            "fla.ops.kda.chunk_bwd", "chunk_kda_bwd_dAv"
        ),
        chunk_gated_delta_rule_bwd_dhu=_get_attribute(
            "fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_bwd_dhu"
        ),
        chunk_kda_bwd_wy_dqkg_fused=_get_attribute(
            "fla.ops.kda.chunk_bwd", "chunk_kda_bwd_wy_dqkg_fused"
        ),
        chunk_kda_bwd_intra=_get_attribute(
            "fla.ops.kda.chunk_intra", "chunk_kda_bwd_intra"
        ),
        chunk_local_cumsum=_get_attribute(
            "fla.ops.utils.cumsum", "chunk_local_cumsum"
        ),
    )
    return _FLAKDARuntime(
        version=version,
        chunk_kda=_get_attribute("fla.ops.kda", "chunk_kda"),
        staged=staged,
    )


def is_fla_triton_kda_available() -> bool:
    """Return whether the optional external FLA KDA backend is usable."""
    try:
        _require_fla_kda_runtime()
    except RuntimeError:
        return False
    return True


def get_fla_kda_staged_ops() -> FLAKDAStagedOps:
    """Return FLA staged operators for Hyper's state-P2P autograd function."""
    return _require_fla_kda_runtime().staged


def run_fla_chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    lower_bound: float = -5.0,
    chunk_size: int = 64,
    safe_gate: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run FLA's public KDA API with the validated Kimi K3 fused contract."""
    runtime = _require_fla_kda_runtime()
    return runtime.chunk_kda(
        q=query,
        k=key,
        v=value,
        g=gate,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
    )


__all__ = [
    "FLAKDAStagedOps",
    "get_fla_kda_staged_ops",
    "is_fla_triton_kda_available",
    "run_fla_chunk_kda",
]
