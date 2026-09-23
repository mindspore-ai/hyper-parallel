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
"""Typed low-precision policies and their FQN-independent dtype-scheme catalog."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Optional


# One registry row per accepted dtype: (family, bit width, MX block check).
# weight_format and act_format must share one family (operators never mix
# families). The weight/activation bit widths are read directly from these
# rows to select the native strategy (e.g. mxfp4 weight + mxfp8 act => W4A8).
# Future dtypes (int/fp/bf, e.g. bf16) are one new row here; hif4 is
# pre-listed as part of the hifloat family (reserved for its flow).
_FORMAT_SPECS = {
    "mxfp4": ("mxfp", 4, True),
    "mxfp8": ("mxfp", 8, True),
    "hif4": ("hifloat", 4, False),
    "hif8": ("hifloat", 8, False),
}
_FORMATS = tuple(_FORMAT_SPECS)


def _format_spec(format_name: str) -> tuple[str, int, bool]:
    """Look up one dtype format in ``_FORMAT_SPECS``.

    Drives ``LowPrecisionDtypeScheme.__post_init__`` validation (family /
    bit-width / MX-block rules). Raises ``ValueError`` on an unregistered
    format so a bad ``weight_format``/``act_format`` fails at config time.
    """

    try:
        return _FORMAT_SPECS[format_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unsupported low-precision format {format_name!r}; "
            f"expected {_FORMATS}."
        ) from exc


def _format_family(format_name: str) -> str:
    """Resolve one dtype format to its family name for the factory family gate.

    Exported for the strategy factory, which must reject cross-family schemes
    without coupling to the raw ``_FORMAT_SPECS`` tuple layout.
    """

    return _format_spec(format_name)[0]


@dataclass(frozen=True)
class LowPrecisionDtypeScheme:
    """One complete low-precision dtype scheme referenced by a plan entry.
    Defines the quantization dtypes for weights and activations together with
    the quantization granularity and whether the scheme runs the native
    operator flow or a future fake-quantize (fake QAT) seam. The native
    strategy (W4A8/W8A8) is derived downstream from ``weight_format`` /
    ``act_format`` bit widths, so it is not a separate field here.
    """

    # Optional: True = future fake-QAT seam; False (default) = native flow.
    is_fake_quantize: bool = False
    weight_format: Literal["mxfp4", "mxfp8", "hif4", "hif8"] = "mxfp8"
    act_format: Literal["mxfp4", "mxfp8", "hif4", "hif8"] = "mxfp8"
    block_size: int = 32

    def __post_init__(self) -> None:
        """Validate that the scheme names a consistent native/QAT policy.
        ``weight_format``/``act_format`` must share one format family (the
        underlying operator never mixes families); their bit widths drive
        strategy selection downstream. MX formats require a 32/128 block size.
        """

        if not isinstance(self.is_fake_quantize, bool):
            raise ValueError(
                "LowPrecisionDtypeScheme.is_fake_quantize must be a bool, "
                f"but got {self.is_fake_quantize!r}."
            )
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise ValueError("LowPrecisionDtypeScheme.block_size must be an integer.")

        weight_family, weight_bits, weight_is_mx = _format_spec(self.weight_format)
        act_family, act_bits, act_is_mx = _format_spec(self.act_format)
        if weight_family != act_family:
            raise ValueError(
                "weight_format and act_format must belong to the same format "
                "family (the underlying operator never mixes families); got "
                f"weight={self.weight_format!r}/act={self.act_format!r}."
            )
        if (weight_is_mx or act_is_mx) and self.block_size not in (32, 128):
            raise ValueError(
                "MX low-precision block_size must be 32 or 128, "
                f"but got {self.block_size}."
            )
        if (
            not self.is_fake_quantize
            and weight_family == "mxfp"
            and weight_bits == 8
            and self.block_size != 32
        ):
            raise ValueError(
                "native w8a8 mxfp8 scheme requires block_size=32 because the "
                "current MXFP8 operator contract uses 32-element blocks."
            )


@dataclass
class LowPrecisionConfig:
    """Global switch plus a catalog of named dtype schemes.

    A dtype scheme is selected by the surrounding ``PlanOverride`` entry.
    Matching module FQNs and selecting a dtype scheme are intentionally
    separate concerns: this object is only the low-precision catalog.

    ``format``/``scaling`` retain the pre-catalog configuration surface. They
    are converted to one implicit scheme when no named catalog is supplied,
    so existing callers remain valid while mixed W8A8/W4A8 plans use
    ``dtype_schemes`` explicitly.
    """

    enabled: bool = False
    format: Literal["mxfp8_e4m3", "mxfp4_e2m1", "hif8"] = "mxfp8_e4m3"
    scaling: Literal["mx_block", "current"] = "mx_block"
    dtype_schemes: dict[str, LowPrecisionDtypeScheme] = field(default_factory=dict)
    default_dtype_scheme: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate catalog keys and the optional default dtype scheme."""

        if not isinstance(self.enabled, bool):
            raise ValueError("LowPrecisionConfig.enabled must be a bool.")
        legacy_pairs = {
            ("mxfp8_e4m3", "mx_block"),
            ("mxfp4_e2m1", "mx_block"),
            ("hif8", "current"),
        }
        if (self.format, self.scaling) not in legacy_pairs:
            raise ValueError(
                "Unsupported low-precision format/scaling combination "
                f"{self.format!r}/{self.scaling!r}."
            )

        if not isinstance(self.dtype_schemes, Mapping):
            raise ValueError(
                "LowPrecisionConfig.dtype_schemes must be a mapping of "
                "non-empty names to dtype schemes."
            )
        typed_schemes = {}
        for name, scheme in self.dtype_schemes.items():
            if isinstance(scheme, Mapping):
                try:
                    scheme = LowPrecisionDtypeScheme(**dict(scheme))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid low-precision dtype scheme {name!r}: {exc}"
                    ) from exc
            typed_schemes[name] = scheme
        self.dtype_schemes = typed_schemes

        if any(
            not isinstance(name, str) or not name
            or not isinstance(scheme, LowPrecisionDtypeScheme)
            for name, scheme in self.dtype_schemes.items()
        ):
            raise ValueError(
                "LowPrecisionConfig.dtype_schemes must map non-empty names to "
                "LowPrecisionDtypeScheme objects."
            )
        if self.default_dtype_scheme is not None:
            if not isinstance(self.default_dtype_scheme, str) or not self.default_dtype_scheme:
                raise ValueError(
                    "LowPrecisionConfig.default_dtype_scheme must be a dtype scheme name."
                )
            if self.default_dtype_scheme not in self.dtype_schemes:
                raise ValueError(
                    f"LowPrecisionConfig.default_dtype_scheme "
                    f"{self.default_dtype_scheme!r} is not present in dtype_schemes."
                )

    def resolve_dtype_scheme(
        self, dtype_scheme_name: Optional[str] = None,
    ) -> LowPrecisionDtypeScheme:
        """Map a dtype-scheme *name* to the scheme object used downstream.

        The input is either ``plan_overrides[].low_precision_dtype_scheme`` or
        ``default_dtype_scheme``; the returned ``LowPrecisionDtypeScheme`` is
        what the strategy factory and adapters actually consume. Only resolves
        catalog names -- which modules are matched is decided by the plan.
        """

        selected = dtype_scheme_name or self.default_dtype_scheme
        if selected is not None:
            if not isinstance(selected, str) or not selected:
                raise ValueError(
                    "low-precision dtype scheme name must be a non-empty string."
                )
            try:
                return self.dtype_schemes[selected]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown low-precision dtype scheme {selected!r}; "
                    f"available dtype schemes: {sorted(self.dtype_schemes)}."
                ) from exc
        if self.dtype_schemes:
            # A catalog exists but nothing was selected: fail loudly instead of
            # silently running the model unquantized.
            raise ValueError(
                "LowPrecisionConfig.dtype_schemes is configured, but the replacement "
                "plan did not select low_precision_dtype_scheme and no "
                "default_dtype_scheme was declared."
            )
        legacy_scheme = {
            ("mxfp8_e4m3", "mx_block"): LowPrecisionDtypeScheme(),
            ("mxfp4_e2m1", "mx_block"): LowPrecisionDtypeScheme(
                weight_format="mxfp4",
                act_format="mxfp8",
            ),
            ("hif8", "current"): LowPrecisionDtypeScheme(
                weight_format="hif8",
                act_format="hif8",
            ),
        }
        return legacy_scheme[(self.format, self.scaling)]

    def to_dict(self) -> dict[str, object]:
        """Serialize legacy fields and the optional named scheme catalog."""

        result: dict[str, object] = {
            "enabled": self.enabled,
            "format": self.format,
            "scaling": self.scaling,
        }
        if self.dtype_schemes:
            result["dtype_schemes"] = {
                name: {
                    "is_fake_quantize": scheme.is_fake_quantize,
                    "weight_format": scheme.weight_format,
                    "act_format": scheme.act_format,
                    "block_size": scheme.block_size,
                }
                for name, scheme in self.dtype_schemes.items()
            }
        if self.default_dtype_scheme is not None:
            result["default_dtype_scheme"] = self.default_dtype_scheme
        return result


__all__ = [
    "LowPrecisionConfig",
    "LowPrecisionDtypeScheme",
]
