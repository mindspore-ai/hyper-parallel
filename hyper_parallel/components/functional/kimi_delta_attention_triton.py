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
"""FLA Triton-Ascend KDA stages, state summaries and ordered affine scans."""
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
    backend = _get_attribute("fla.ops.kda.backends.triton_ascend", "TritonAscendKDABackend")
    if not backend.is_available():
        raise RuntimeError("FLA Triton-Ascend KDA backend is unavailable in this process.")
    return _FLAKDARuntime(version=version, chunk_kda=_get_attribute("fla.ops.kda", "chunk_kda"))


def is_fla_triton_kda_available() -> bool:
    """Return whether the optional external FLA KDA backend is usable."""
    try:
        _require_fla_kda_runtime()
    except RuntimeError:
        return False
    return True


@lru_cache(maxsize=1)
def get_fla_kda_staged_ops() -> FLAKDAStagedOps:
    """Return FLA staged operators for Hyper's state-P2P autograd function."""
    _require_fla_kda_runtime()
    _require_npu_backends()
    return FLAKDAStagedOps(
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
    cu_seqlens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run FLA's public KDA API with the validated Kimi K3 fused contract.

    Args:
        query: Raw query tensor before fused normalization.
        key: Raw key tensor before fused normalization.
        value: Value tensor in the input token layout.
        gate: Raw gate projection before the fused decay transform.
        beta: Raw beta projection before the fused sigmoid.
        a_log: Learned log-scale of the gate transform.
        dt_bias: Learned gate bias.
        scale: Optional attention scale; FLA supplies the default when omitted.
        initial_state: Optional incoming recurrent state.
        output_final_state: Whether to return the outgoing recurrent state.
        lower_bound: Lower bound of the safe gate transform.
        chunk_size: Token chunk size accepted by the selected fused implementation.
        safe_gate: Whether to use the bounded gate transform.
        cu_seqlens: Optional cumulative document boundaries for packed input.

    Returns:
        Attention output and the optional outgoing state.
    """
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
        cu_seqlens=cu_seqlens,
    )


def _validate_local_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Validate local KDA tensor shapes and return dimensions used by later checks."""
    if (query.ndim, key.ndim, value.ndim, gate.ndim) != (4, 4, 4, 4):
        raise ValueError("Fused KDA expects rank-4 query/key/value/gate tensors.")
    if beta.ndim != 3:
        raise ValueError("Fused KDA expects a rank-3 beta tensor.")
    if query.shape != key.shape:
        raise ValueError("Fused KDA query and key must have identical shapes.")
    batch, sequence_length, num_query_heads, key_dim = query.shape
    num_value_heads, value_dim = value.shape[2:]
    if value.shape[:2] != (batch, sequence_length):
        raise ValueError("Fused KDA value must match query batch and sequence dimensions.")
    if gate.shape != (batch, sequence_length, num_value_heads, key_dim):
        raise ValueError("Fused KDA gate has an incompatible shape.")
    if beta.shape != (batch, sequence_length, num_value_heads):
        raise ValueError("Fused KDA beta has an incompatible shape.")
    if num_value_heads % num_query_heads:
        raise ValueError("Fused KDA value heads must be divisible by query heads.")
    return sequence_length, key_dim, value_dim, num_value_heads


def validate_triton_kda_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    chunk_size: int,
    lower_bound: float,
) -> None:
    """Validate paired Triton KDA inputs before local kernels or state communication.

    Args:
        query: BF16 queries [B,T,H,K].
        key: BF16 keys matching query.
        value: BF16 values with compatible token and head axes.
        gate: Raw BF16 channel gate logits.
        beta: Raw BF16 per-head update logits.
        a_log: FP32 per-head decay parameter.
        dt_bias: FP32 per-channel gate bias.
        chunk_size: Kernel chunk size, currently 64.
        lower_bound: Safe gate lower bound in [-5,0).
    """
    _, key_dim, value_dim, num_value_heads = _validate_local_shapes(
        query, key, value, gate, beta
    )
    if not (
        query.dtype == key.dtype == value.dtype == gate.dtype == beta.dtype
        == torch.bfloat16
    ):
        raise TypeError("Fused KDA requires q/k/v/gate/beta tensors in bfloat16.")
    if a_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise TypeError("Fused KDA requires a_log and dt_bias tensors in float32.")
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "Fused KDA currently requires key_dim=value_dim=128 and chunk_size=64."
        )
    if a_log.numel() != num_value_heads:
        raise ValueError("Fused KDA a_log must contain one value per value head.")
    if dt_bias.numel() != num_value_heads * key_dim:
        raise ValueError("Fused KDA dt_bias must contain one value per gate channel.")
    if not -5.0 <= lower_bound < 0:
        raise ValueError("Fused KDA lower_bound must lie in [-5, 0).")
    if not query.is_npu:
        raise RuntimeError("The fused KDA backend requires Ascend NPU tensors.")
    if any(
        tensor.device != query.device
        for tensor in (key, value, gate, beta, a_log, dt_bias)
    ):
        raise ValueError("Fused KDA inputs must reside on the same NPU device.")
    _require_fla_kda_runtime()


def fused_chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float] = None,
    lower_bound: float = -5.0,
    chunk_size: int = 64,
    safe_gate: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the local Triton-Ascend KDA backend without CP communication.

    Args:
        query: Local query tensor [B,T,H,K].
        key: Matching local key tensor.
        value: Local value tensor [B,T,HV,V].
        gate: Per-token gate logits [B,T,HV,K].
        beta: Per-token beta logits [B,T,HV].
        a_log: Decay parameter with one value per value head.
        dt_bias: Learned bias with one value per gate channel.
        scale: Optional attention scaling factor.
        lower_bound: Lower bound for the activated gate.
        chunk_size: Tokens per local recurrence chunk.
        safe_gate: Use the staged lower-bounded gate implementation.
    """
    validate_triton_kda_inputs(
        query,
        key,
        value,
        gate,
        beta,
        a_log,
        dt_bias,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    output, _ = run_fla_chunk_kda(
        query,
        key,
        value,
        gate,
        beta,
        a_log=a_log,
        dt_bias=dt_bias.reshape(-1),
        scale=scale,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        cu_seqlens=cu_seqlens,
    )
    return output


def validate_kda_state_summary(state: torch.Tensor, matrix: torch.Tensor) -> None:
    """Validate the fixed KDA affine contract before communication.

    Args:
        state: Nonempty FP32 S or G, shaped [B,H,128,128].
        matrix: Matching FP32 transition on the same device.
    """
    if (state.ndim != 4 or state.shape[-2:] != (128, 128) or matrix.shape != state.shape
            or state.shape[0] == 0 or state.shape[1] == 0):
        raise ValueError("KDA state CP requires nonempty matching [B,H,128,128] summaries.")
    if state.dtype != torch.float32 or matrix.dtype != torch.float32 or state.device != matrix.device:
        raise ValueError("KDA state CP requires FP32 summaries on the same device.")
    if state.device.type not in ("cpu", "npu"):
        raise ValueError("KDA state CP supports CPU protocol checks and Ascend NPU execution only.")


def _validate_prepared_summary_inputs(
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gate: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int]:
    """Validate the fixed Kimi K3 summary contract and return dimensions."""
    if (key.ndim, w.ndim, u.ndim, gate.ndim) != (4, 4, 4, 4):
        raise ValueError("KDA prepared summary expects rank-4 key/w/u/gate tensors.")
    batch, sequence_length, heads, key_dim = key.shape
    value_dim = u.shape[-1]
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "Triton KDA state summary requires key_dim=value_dim=128 and "
            "chunk_size=64."
        )
    if sequence_length <= 0:
        raise ValueError("State summary requires a nonempty sample fragment.")
    if w.shape != key.shape or gate.shape != key.shape:
        raise ValueError(
            "KDA prepared key, w, and gate must have identical shapes, got "
            f"key={tuple(key.shape)}, w={tuple(w.shape)}, gate={tuple(gate.shape)}."
        )
    if u.shape[:3] != key.shape[:3]:
        raise ValueError(
            f"KDA prepared u prefix {tuple(u.shape[:3])} must match "
            f"key prefix {tuple(key.shape[:3])}."
        )
    if not key.is_npu:
        raise RuntimeError("Triton KDA state summary requires Ascend NPU tensors.")
    return batch, sequence_length, heads, key_dim, value_dim


def _validate_gradient_summary_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    w: torch.Tensor,
    gate: torch.Tensor,
    grad_output: torch.Tensor,
    grad_value: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int]:
    """Validate prepared tensors used by the reverse state wavefront."""
    tensors = (query, key, w, gate, grad_output, grad_value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError(
            "KDA gradient summary expects rank-4 query/key/w/gate/do/dv tensors."
        )
    batch, sequence_length, heads, key_dim = query.shape
    value_dim = grad_output.shape[-1]
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "Triton KDA state-gradient summary requires key_dim=value_dim=128 "
            "and chunk_size=64."
        )
    if sequence_length <= 0:
        raise ValueError("State summary requires a nonempty sample fragment.")
    prepared_shape = (batch, sequence_length, heads, key_dim)
    value_shape = (batch, sequence_length, heads, value_dim)
    if key.shape != prepared_shape or w.shape != prepared_shape:
        raise ValueError(
            "KDA prepared query, key, and w must have identical shapes, got "
            f"query={tuple(query.shape)}, key={tuple(key.shape)}, "
            f"w={tuple(w.shape)}."
        )
    if gate.shape != prepared_shape:
        raise ValueError(
            f"KDA prepared gate shape {tuple(gate.shape)} must match "
            f"query shape {prepared_shape}."
        )
    if grad_output.shape != value_shape or grad_value.shape != value_shape:
        raise ValueError(
            "KDA grad_output and grad_value must have shape "
            f"{value_shape}, got do={tuple(grad_output.shape)}, "
            f"dv={tuple(grad_value.shape)}."
        )
    if not query.is_npu:
        raise RuntimeError(
            "Triton KDA state-gradient summary requires Ascend NPU tensors."
        )
    return batch, sequence_length, heads, key_dim, value_dim


def _launch_kda_state_summary(
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gate: torch.Tensor,
    *,
    chunk_size: int,
    output_transition: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch the shared S/M summary used by the state CP protocols."""
    return _launch_kda_paired_state_summary(
        key,
        w,
        u,
        gate,
        chunk_size=chunk_size,
        output_transition=output_transition,
    )


def _launch_kda_state_gradient_summary(
    query: torch.Tensor,
    key: torch.Tensor,
    w: torch.Tensor,
    gate: torch.Tensor,
    grad_output: torch.Tensor,
    grad_value: torch.Tensor,
    scale: float,
    *,
    chunk_size: int,
    block_size: int = 64,
) -> torch.Tensor:
    """Launch the reverse-wavefront Triton-Ascend summary kernel."""
    # Triton is optional until a summary kernel is actually launched.
    import triton  # pylint: disable=import-outside-toplevel

    from ._triton.kimi_delta_attention.state_summary import (  # pylint: disable=import-outside-toplevel
        kda_state_grad_ext_kernel,
    )

    batch, sequence_length, heads, key_dim, value_dim = (
        _validate_gradient_summary_inputs(
            query,
            key,
            w,
            gate,
            grad_output,
            grad_value,
            chunk_size,
        )
    )
    query, key, w, gate, grad_output, grad_value = (
        tensor.contiguous()
        for tensor in (query, key, w, gate, grad_output, grad_value)
    )
    grad_state_ext = torch.empty(
        batch,
        heads,
        key_dim,
        value_dim,
        dtype=torch.float32,
        device=query.device,
    )
    kda_state_grad_ext_kernel[
        (triton.cdiv(value_dim, block_size), batch * heads)
    ](
        query,
        key,
        w,
        gate,
        grad_output,
        grad_value,
        grad_state_ext,
        float(scale),
        sequence_length,
        H=heads,
        H_TOTAL=heads,
        HEAD_OFFSET=0,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=block_size,
        num_warps=2,
    )
    return grad_state_ext


@torch.compiler.disable
def _launch_kda_paired_state_summary(
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gate: torch.Tensor,
    *,
    chunk_size: int = 64,
    output_transition: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build Ascend FLA-paired ``S_ext`` and an explicit FP32 transfer ``M``."""
    # Triton is optional until a summary kernel is actually launched.
    import triton  # pylint: disable=import-outside-toplevel

    from ._triton.kimi_delta_attention.state_summary import (  # pylint: disable=import-outside-toplevel
        kda_split_state_summary_kernel,
        kda_state_transition_kernel,
    )

    batch, sequence_length, heads, key_dim, value_dim = (
        _validate_prepared_summary_inputs(key, w, u, gate, chunk_size)
    )
    key, w, u, gate = (
        tensor.contiguous() for tensor in (key, w, u, gate)
    )
    state_ext = torch.empty(
        batch,
        heads,
        key_dim,
        value_dim,
        dtype=torch.float32,
        device=key.device,
    )
    transition = None
    if output_transition:
        transition = torch.empty(
            batch,
            heads,
            key_dim,
            key_dim,
            dtype=torch.float32,
            device=key.device,
        )
    kda_split_state_summary_kernel[(1, batch * heads)](
        key,
        w,
        u,
        gate,
        state_ext,
        transition if transition is not None else state_ext,
        sequence_length,
        H=heads,
        H_TOTAL=heads,
        HEAD_OFFSET=0,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=128,
        OUTPUT_MODE=1,
        num_warps=2,
    )
    if output_transition:
        kda_state_transition_kernel[(triton.cdiv(key_dim, 64), batch * heads)](
            key,
            w,
            gate,
            transition,
            sequence_length,
            H=heads,
            K=key_dim,
            BT=chunk_size,
            BV=64,
            num_warps=4,
            num_stages=2,
        )
    return state_ext, transition


@torch.compiler.disable
def kda_state_summary_forward_from_prepared(
    key: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gate: torch.Tensor,
    *,
    chunk_size: int = 64,
    output_transition: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build the local affine summary for state context parallelism.

    ``S_ext`` follows Ascend FLA h's FP32 feedback. ``M`` composes explicit
    chunk transfers with FP32 feedback, following FLA CP.
    ``output_transition=False`` returns only S and skips M allocation/computation.
    It is valid for a zero-entry P2P endpoint whose backward never applies M.

    Args:
        key: Prepared keys, shaped [batch, tokens, heads, key_dim].
        w: Prepared feedback weights with the same shape as key.
        u: Prepared values, shaped [batch, tokens, heads, value_dim].
        gate: Per-chunk cumulative log2 gates matching key.
        chunk_size: Number of tokens per local chunk; the kernel requires 64.
        output_transition: Whether the caller needs the local transition M.

    Returns:
        The FP32 state S and the FP32 transition M, or None when M is omitted.
    """
    return _launch_kda_state_summary(
        key,
        w,
        u,
        gate,
        chunk_size=chunk_size,
        output_transition=output_transition,
    )


@torch.compiler.disable
def kda_state_gradient_summary_from_prepared(
    query: torch.Tensor,
    key: torch.Tensor,
    w: torch.Tensor,
    gate: torch.Tensor,
    grad_output: torch.Tensor,
    grad_value: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Return the local-loss contribution to the incoming KDA state gradient.

    Match Ascend FLA ``dhu`` at zero terminal state: scale ``grad_output``
    before its dot-input cast and preserve the two K64 slabs and subtraction
    grouping. This summary omits the full ``dh``/``dv`` outputs. All inputs
    are read-only; the affine application to a nonzero state is a separate step.

    Args:
        query: Prepared gated queries [B,T,H,K].
        key: Matching prepared gated keys.
        w: Prepared feedback weights matching key.
        gate: Per-chunk cumulative log2 gates matching key.
        grad_output: Original token-output gradients [B,T,H,V].
        grad_value: Local value gradients before recurrent feedback.
        scale: Attention scaling factor.
        chunk_size: Tokens per local chunk; the kernel requires 64.

    Returns:
        FP32 local contribution G, shaped [B,H,K,V].
    """
    return _launch_kda_state_gradient_summary(
        query,
        key,
        w,
        gate,
        grad_output,
        grad_value,
        scale,
        chunk_size=chunk_size,
        block_size=128,
    )


def merge_affine_summaries(
    states: torch.Tensor, matrices: torch.Tensor, rank: int, forward: bool,
) -> torch.Tensor:
    """Merge the exclusive prefix or transposed suffix in chronological order.

    Args:
        states: Affine offsets shaped [P,B,H,128,128].
        matrices: Matching transfer matrices, possibly views of packed storage.
        rank: Chronological rank whose input state or final-state gradient is requested.
        forward: Select forward prefix instead of backward suffix.

    Returns:
        Contiguous FP32 recurrent state or state gradient shaped [B,H,128,128].
    """
    if states.ndim != 5 or states.shape != matrices.shape or states.shape[0] == 0:
        raise ValueError("KDA merge requires matching rank-stacked summaries.")
    size, batch, heads = states.shape[:3]
    validate_kda_state_summary(states[0], matrices[0])
    if not 0 <= rank < size:
        raise ValueError("KDA merge rank is outside the summary group.")
    if any(t.stride(1) != heads * t.stride(2) or t.stride(-1) != 1 for t in (states, matrices)):
        raise ValueError("KDA merge requires consecutive batch/head slabs and columns.")
    if states.device.type != "npu":
        raise ValueError("KDA fused state merge requires Ascend NPU tensors.")
    count = rank if forward else size - rank - 1
    output = states.new_empty((batch, heads, 128, 128))
    if count == 0:
        return output.zero_()
    # Triton is optional for imports; production merging requires Ascend.
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.components.functional._triton.kimi_delta_attention.cp_merge import (
        kda_cached_affine_merge_kernel,
    )
    # pylint: enable=import-outside-toplevel

    kda_cached_affine_merge_kernel[(2, batch * heads)](
        output, states, matrices, count, rank,
        state_rank_stride=states.stride(0), state_head_stride=states.stride(2), state_row_stride=states.stride(3),
        matrix_rank_stride=matrices.stride(0), matrix_head_stride=matrices.stride(2),
        matrix_row_stride=matrices.stride(3),
        forward=forward, block_v=64, num_warps=2, num_stages=2)
    return output


def _validate_scan_range(indices: range, size: int, forward: bool, prefix_count: int | None) -> None:
    """Check the requested chronological interval and optional saved prefix."""
    count = len(indices)
    if prefix_count is not None and not 0 <= prefix_count <= count:
        raise ValueError("The saved prefix must lie within the scan interval.")
    if count:
        endpoints_valid = all(0 <= index < size for index in (indices[0], indices[-1]))
        if indices.step != (1 if forward else -1) or not endpoints_valid:
            raise ValueError("KDA scans require an ordered interval inside the group.")


def scan_kda_affine_maps(maps: torch.Tensor, incoming: torch.Tensor,
               indices: range, forward: bool, prefix_count: int | None = None,
               ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scan ordered group maps from a nonzero entry state.

    Args:
        maps: Rank-stacked S/M summaries in packed [P,B,H,128,256] layout.
        incoming: Contiguous FP32 entry state.
        indices: Contiguous chronological prefix or reverse suffix.
        forward: Use forward maps or their transposed reverse application.
        prefix_count: Optionally return the state after this many applications.

    Returns:
        The outgoing state and the optional saved intermediate prefix.
    """
    count = len(indices)
    _validate_scan_range(indices, maps.shape[0], forward, prefix_count)
    if not count:
        return incoming, incoming if prefix_count == 0 else None
    if incoming.device.type != "npu" or not incoming.is_contiguous():
        raise ValueError("KDA fused scans require a contiguous Ascend input state.")
    output = torch.empty_like(incoming)
    prefix = None
    if prefix_count is not None:
        prefix = incoming if prefix_count == 0 else output if prefix_count == count else torch.empty_like(incoming)
    save_prefix = prefix is not None and prefix_count not in (0, count)
    # Keep the optional Triton dependency out of CPU-only model imports.
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.components.functional._triton.kimi_delta_attention.cp_merge import kda_affine_scan_kernel
    # pylint: enable=import-outside-toplevel

    batch, heads = incoming.shape[:2]
    kda_affine_scan_kernel[(2, batch * heads)](
        output, prefix, maps, incoming, count, indices.start, prefix_count or 0,
        rank_stride=maps.stride(0), head_stride=maps.stride(2), row_stride=maps.stride(3),
        forward=forward, save_prefix=save_prefix, block_v=64, num_warps=2, num_stages=2)
    return output, prefix
