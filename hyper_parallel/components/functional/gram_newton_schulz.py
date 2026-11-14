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
"""Optional Gram Newton-Schulz adapter for Muon's zeropower callback."""

import copy
import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import torch


def _load_gram_newton_schulz() -> Callable[..., Any]:
    """Load the optional package only when its backend is explicitly requested."""
    try:
        # CUDA packages must stay off the default CPU/NPU import path.
        package = importlib.import_module("gram_newton_schulz")
    except (ImportError, OSError) as error:
        raise ImportError(
            "Could not load gram-newton-schulz. Install a compatible version following "
            "hyper_parallel/components/functional/README.md. "
            f"Original error: {error}"
        ) from error
    implementation = getattr(package, "GramNewtonSchulz", None)
    if not callable(implementation):
        raise ImportError("gram-newton-schulz must export a callable GramNewtonSchulz")
    return implementation


def _normalize_coefficients(coefficients: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    """Own an immutable, finite coefficient schedule independently of the caller."""
    normalized = []
    for index, row in enumerate(coefficients):
        if len(row) != 3:
            raise ValueError(f"ns_coefficients[{index}] must contain exactly three values")
        values = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"ns_coefficients[{index}] must contain only finite values")
        normalized.append(values)
    if not normalized:
        raise ValueError("ns_coefficients must not be empty")
    return tuple(normalized)


class _GramNewtonSchulzAdapter:
    """Bind an NS schedule and cache upstream instances per device and step count."""

    def __init__(
        self,
        implementation: Callable[..., Any],
        coefficients: tuple[tuple[float, float, float], ...],
        epsilon: float,
        reset_iterations: tuple[int, ...],
        use_kernels: Optional[bool],
        compile_kwargs: Optional[Mapping[str, Any]],
    ) -> None:
        self._implementation = implementation
        self._coefficients = coefficients
        self._epsilon = epsilon
        self._reset_iterations = reset_iterations
        self._use_kernels = use_kernels
        self._compile_kwargs = copy.deepcopy(dict(compile_kwargs)) if compile_kwargs is not None else None
        self._instances: dict[tuple[torch.device, int], Any] = {}

    def _resolve_use_kernels(self, device: torch.device) -> bool:
        """Choose kernels from the input device, preserving explicit overrides."""
        if device.type != "cuda":
            if self._use_kernels is True:
                raise ValueError("Gram Newton-Schulz custom kernels require a Hopper or Blackwell CUDA device")
            if device.type != "cpu":
                raise ValueError("Gram Newton-Schulz's Torch path supports CPU/CUDA; NPU is not qualified")
            return False
        if self._use_kernels is False:
            return False
        major, _ = torch.cuda.get_device_capability(device)
        supported = major in (9, 10)
        if self._use_kernels is True and not supported:
            raise ValueError(
                f"Gram Newton-Schulz custom kernels require Hopper/Blackwell (SM 9.x/10.x), got SM {major}.x"
            )
        return supported

    @torch.no_grad()
    def __call__(self, ns_inputs: torch.Tensor, steps: int) -> torch.Tensor:
        """Orthogonalize a local 2D matrix or 3D batch without retaining inputs."""
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= len(self._coefficients):
            raise ValueError(f"steps must be an integer in [1, {len(self._coefficients)}], got {steps!r}")
        if ns_inputs.ndim not in (2, 3):
            raise ValueError(f"Gram Newton-Schulz expects a 2D matrix or 3D batch, got {ns_inputs.ndim}D")
        if ns_inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Gram Newton-Schulz expects float16, bfloat16 or float32, got {ns_inputs.dtype}")
        if ns_inputs.numel() == 0:
            self._resolve_use_kernels(ns_inputs.device)
            return torch.empty_like(ns_inputs)
        key = (ns_inputs.device, steps)
        if key not in self._instances:
            self._instances[key] = self._implementation(
                ns_epsilon=self._epsilon,
                ns_use_kernels=self._resolve_use_kernels(ns_inputs.device),
                ns_coefficients=[list(row) for row in self._coefficients[:steps]],
                gram_newton_schulz_reset_iterations=[index for index in self._reset_iterations if index < steps],
                compile_kwargs=copy.deepcopy(self._compile_kwargs),
            )
        return self._instances[key](ns_inputs.contiguous())


def make_gram_newton_schulz(
    ns_coefficients: Sequence[Sequence[float]],
    *,
    epsilon: float = 1e-10,
    reset_iterations: Sequence[int] = (2,),
    use_kernels: Optional[bool] = None,
    compile_kwargs: Optional[Mapping[str, Any]] = None,
) -> Callable[..., torch.Tensor]:
    """Create a Gram Newton-Schulz callback accepted by ``Muon(zeropower_fn=...)``.

    Args:
        ns_coefficients: Explicit coefficient schedule. The callback uses its first
            ``steps`` entries. Pass the same schedule to Muon with ``ns_variant='custom'``.
        epsilon: Finite positive normalization epsilon; share it with Muon's ``ns_epsilon``.
        reset_iterations: Positive iteration counts after which to restart. Counts
            greater than or equal to the requested step count are ignored.
        use_kernels: None (default) selects kernels for Hopper/Blackwell CUDA inputs
            and the upstream Torch path for CPU/other CUDA inputs. True requires
            supported CUDA hardware; False forces the Torch path on CPU/CUDA.
        compile_kwargs: Options forwarded to upstream ``torch.compile``. None disables
            compilation; pass a mapping to enable it (including an empty mapping).

    Returns:
        Callable with signature ``callback(ns_inputs, steps)`` that preserves input
        shape, dtype and device. Instances are cached per device and step count.

    Raises:
        ImportError: The optional package or its public entry point is unavailable.
        ValueError: Configuration, input, step count or device is unsupported.

    Note:
        Coefficients and epsilon are bound here: Muon's existing callback only
        forwards ``steps``. All groups must share the bound schedule and epsilon.
        Recreate the callback with matching configuration when restoring a checkpoint.
        Upstream uses FP32 normalization and FP16 iteration, including for BF16 input.
        Automatic selection uses device capability, not exception recovery. Import,
        compilation and execution errors propagate. NPU is not qualified.
    """
    coefficients = _normalize_coefficients(ns_coefficients)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon!r}")
    resets = tuple(reset_iterations)
    if any(not isinstance(index, int) or isinstance(index, bool) or index <= 0 for index in resets):
        raise ValueError("reset_iterations must contain positive integer iteration counts")
    if use_kernels is not None and not isinstance(use_kernels, bool):
        raise ValueError("use_kernels must be a bool or None")
    if compile_kwargs is not None and not isinstance(compile_kwargs, Mapping):
        raise ValueError("compile_kwargs must be a mapping or None")
    return _GramNewtonSchulzAdapter(
        _load_gram_newton_schulz(), coefficients, float(epsilon), tuple(sorted(set(resets))),
        use_kernels, compile_kwargs,
    )
