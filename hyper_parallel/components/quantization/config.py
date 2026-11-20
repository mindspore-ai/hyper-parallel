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
from dataclasses import dataclass, field, fields
from typing import Literal, Optional, TypeAlias, Union, cast


MXBlockSizeValue: TypeAlias = Literal[32]
FP4_BITS = 4
FP8_BITS = 8
MX_BLOCK_SIZE_32 = 32
SUPPORTED_MX_BLOCK_SIZES = (MX_BLOCK_SIZE_32,)

# MX tensor and scale layout.
MX_SCALE_SLOTS = 2
MX_SCALE_TILE = MX_BLOCK_SIZE_32 * MX_SCALE_SLOTS
MXFP4_VALUES_PER_BYTE = 2
MXFP4_NIBBLE_BITS = 4
MXFP4_NIBBLE_MASK = (1 << MXFP4_NIBBLE_BITS) - 1

# E8M0/FP8 conversion limits used by the fake-QAT path.
E8M0_BIAS = 127.0
E8M0_MAX_EXPONENT = 255.0
MXFP4_MAX_VALUE = 6.0
FP8_E4M3_MAX = 448.0
MAIN_KV_GROUP_SIZE = 16
SWA_KV_GROUP_SIZE = 32

# GroupedMatmul and DynamicMxQuant protocol values.
GROUP_LIST_OFFSETS = 0
GROUP_LIST_COUNTS = 1
GMM_GROUP_M = 0
GMM_GROUP_K = 2
GMM_SPLIT_ANTIQ = 2
GMM_SPLIT_MXFP8 = 2
GMM_SPLIT_SCALE = 3
MX_SCALE_ALG_OCP = 0
MX_SCALE_ALG_CUBALS = 1
NPU_FORMAT_FRACTAL_NZ = 29

# Shape ranks used by grouped quantization contracts.
MATRIX_NDIM = 2
GROUPED_NDIM = 3
SCALE_NDIM = 4
_SUPPORTED_BLOCK_SIZE_TEXT = " or ".join(
    str(block_size) for block_size in SUPPORTED_MX_BLOCK_SIZES
)


def validate_mx_block_size(block_size: int) -> MXBlockSizeValue:
    """Validate a configured integer and preserve its numeric representation."""

    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError(
            f"MX block_size must be {_SUPPORTED_BLOCK_SIZE_TEXT}, "
            f"got {block_size!r}."
        )
    if block_size not in SUPPORTED_MX_BLOCK_SIZES:
        raise ValueError(
            f"MX block_size must be {_SUPPORTED_BLOCK_SIZE_TEXT}, "
            f"got {block_size!r}."
        )
    return cast(MXBlockSizeValue, block_size)


# One registry row per accepted dtype: (format family, bit width).
# weight_format and act_format must share one family (operators never mix
# families). The weight/activation bit widths are read directly from these
# rows to select the native strategy (e.g. mxfp4 weight + mxfp8 act => W4A8).
# Future dtypes (int/fp/bf, e.g. bf16) are one new row here; hif4 is
# pre-listed as part of the hifloat family (reserved for its flow).
_FORMAT_SPECS = {
    "mxfp4": ("mxfp", FP4_BITS),
    "mxfp8": ("mxfp", FP8_BITS),
    "hif4": ("hifloat", FP4_BITS),
    "hif8": ("hifloat", FP8_BITS),
}
_FORMATS = tuple(_FORMAT_SPECS)


def _format_spec(format_name: str) -> tuple[str, int]:
    """Look up one dtype format in ``_FORMAT_SPECS``.

    Drives ``LowPrecisionDtypeScheme.__post_init__`` validation (family and
    bit width). Raises ``ValueError`` on an unregistered
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
    block_size: int = MX_BLOCK_SIZE_32

    def __post_init__(self) -> None:
        """Validate that the scheme names a consistent native/QAT policy.
        ``weight_format``/``act_format`` must share one format family (the
        underlying operator never mixes families); their bit widths drive
        strategy selection downstream. ``block_size`` must be an integer;
        only MX formats restrict it to the currently supported block32 value.
        """

        if not isinstance(self.is_fake_quantize, bool):
            raise ValueError(
                "LowPrecisionDtypeScheme.is_fake_quantize must be a bool, "
                f"but got {self.is_fake_quantize!r}."
            )
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise ValueError("LowPrecisionDtypeScheme.block_size must be an integer.")

        weight_family, _ = _format_spec(self.weight_format)
        act_family, _ = _format_spec(self.act_format)
        if weight_family != act_family:
            raise ValueError(
                "weight_format and act_format must belong to the same format "
                "family (the underlying operator never mixes families); got "
                f"weight={self.weight_format!r}/act={self.act_format!r}."
            )
        if weight_family == "mxfp":
            try:
                block_size = validate_mx_block_size(self.block_size)
            except ValueError as exc:
                raise ValueError(f"LowPrecisionDtypeScheme.{exc}") from exc
            object.__setattr__(self, "block_size", block_size)


@dataclass(frozen=True)
class CSA2IndexerQKFormat:
    """OCP MXFP4 activation QDQ for CSA2 Indexer queries and keys."""

    act_format: Literal["mxfp4"]

    def __post_init__(self) -> None:
        if self.act_format != "mxfp4":
            raise ValueError("indexer_qk.act_format must be 'mxfp4'.")


@dataclass(frozen=True)
class CSA2MainKVFormat:
    """Main KV E2M1 with one E4M3 or E8M7 scale per 16 channels."""

    act_format: Literal["fp4_e2m1"]
    scale_format: Literal["e4m3", "e8m7"]
    group: Literal[16] = MAIN_KV_GROUP_SIZE

    def __post_init__(self) -> None:
        if self.act_format != "fp4_e2m1":
            raise ValueError("main_kv.act_format must be 'fp4_e2m1'.")
        if self.scale_format not in ("e4m3", "e8m7"):
            raise ValueError("main_kv.scale_format must be 'e4m3' or 'e8m7' (BF16).")
        if isinstance(self.group, bool) or not isinstance(self.group, int) or self.group != MAIN_KV_GROUP_SIZE:
            raise ValueError("main_kv.group must be 16.")


@dataclass(frozen=True)
class CSA2SWAKVFormat:
    """FP8 E4M3 sliding-window KV with E8M0 or E8M7 scales."""

    act_format: Literal["fp8_e4m3"]
    scale_format: Literal["e8m0", "e8m7"] = "e8m0"
    group: Literal[32] = SWA_KV_GROUP_SIZE

    def __post_init__(self) -> None:
        if self.act_format != "fp8_e4m3":
            raise ValueError("swa_kv.act_format must be 'fp8_e4m3'.")
        if self.scale_format not in ("e8m0", "e8m7"):
            raise ValueError("swa_kv.scale_format must be 'e8m0' or 'e8m7' (BF16).")
        if isinstance(self.group, bool) or not isinstance(self.group, int) or self.group != SWA_KV_GROUP_SIZE:
            raise ValueError("swa_kv.group must be 32.")


_CSA2_FORMAT_TYPES = {
    "indexer_qk": CSA2IndexerQKFormat,
    "main_kv": CSA2MainKVFormat,
    "swa_kv": CSA2SWAKVFormat,
}

CSA2DtypeScheme: TypeAlias = Union[
    LowPrecisionDtypeScheme, CSA2IndexerQKFormat, CSA2MainKVFormat, CSA2SWAKVFormat,
]


@dataclass
class LowPrecisionConfig:
    """Global switch plus a catalog of named dtype schemes.

    A W/A dtype scheme is selected by the surrounding ``PlanOverride`` entry.
    CSA2 formats are read directly by a matching attention replacement from
    the shared build context; unused formats remain in this catalog.

    ``format``/``scaling`` retain the pre-catalog configuration surface. They
    are converted to one implicit scheme when no named catalog is supplied,
    so existing callers remain valid while mixed W8A8/W4A8 plans use
    ``dtype_schemes`` explicitly.
    """

    enabled: bool = False
    format: Literal["mxfp8_e4m3", "mxfp4_e2m1", "hif8"] = "mxfp8_e4m3"
    scaling: Literal["mx_block", "current"] = "mx_block"
    dtype_schemes: dict[str, CSA2DtypeScheme] = field(default_factory=dict)
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
            if not isinstance(name, str) or not name:
                raise ValueError("LowPrecisionConfig.dtype_schemes keys must be non-empty strings.")
            scheme_type = _CSA2_FORMAT_TYPES.get(name, LowPrecisionDtypeScheme)
            if isinstance(scheme, Mapping):
                try:
                    scheme = scheme_type(**dict(scheme))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid LowPrecisionConfig.dtype_schemes.{name}: {exc}"
                    ) from exc
            if not isinstance(scheme, scheme_type):
                raise ValueError(
                    f"LowPrecisionConfig.dtype_schemes.{name} must be a {scheme_type.__name__}."
                )
            typed_schemes[name] = scheme
        self.dtype_schemes = typed_schemes
        if self.default_dtype_scheme is not None:
            if not isinstance(self.default_dtype_scheme, str) or not self.default_dtype_scheme:
                raise ValueError(
                    "LowPrecisionConfig.default_dtype_scheme must be a dtype scheme name."
                )
            if not isinstance(self.dtype_schemes.get(self.default_dtype_scheme), LowPrecisionDtypeScheme):
                raise ValueError(
                    f"LowPrecisionConfig.default_dtype_scheme "
                    f"{self.default_dtype_scheme!r} is not a W/A scheme in dtype_schemes."
                )

    @property
    def indexer_qk(self) -> Optional[CSA2IndexerQKFormat]:
        """Return the configured Indexer activation format, if present."""
        value = self.dtype_schemes.get("indexer_qk")
        return value if isinstance(value, CSA2IndexerQKFormat) else None

    @property
    def main_kv(self) -> Optional[CSA2MainKVFormat]:
        """Return the configured Main KV cache format, if present."""
        value = self.dtype_schemes.get("main_kv")
        return value if isinstance(value, CSA2MainKVFormat) else None

    @property
    def swa_kv(self) -> Optional[CSA2SWAKVFormat]:
        """Return the configured SWA KV cache format, if present."""
        value = self.dtype_schemes.get("swa_kv")
        return value if isinstance(value, CSA2SWAKVFormat) else None

    def active_direct_formats(self) -> frozenset[str]:
        """Return explicitly configured CSA2 formats."""
        return frozenset(self.dtype_schemes.keys() & _CSA2_FORMAT_TYPES.keys())

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
            scheme = self.dtype_schemes.get(selected)
            if not isinstance(scheme, LowPrecisionDtypeScheme):
                available = sorted(
                    name for name, value in self.dtype_schemes.items()
                    if isinstance(value, LowPrecisionDtypeScheme)
                )
                raise ValueError(
                    f"Unknown W/A low-precision dtype scheme {selected!r}; "
                    f"available W/A schemes: {available}."
                )
            return scheme
        if any(isinstance(value, LowPrecisionDtypeScheme) for value in self.dtype_schemes.values()):
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
                name: {scheme_field.name: getattr(scheme, scheme_field.name) for scheme_field in fields(scheme)}
                for name, scheme in self.dtype_schemes.items()
            }
        if self.default_dtype_scheme is not None:
            result["default_dtype_scheme"] = self.default_dtype_scheme
        return result


__all__ = [
    "MXBlockSizeValue",
    "LowPrecisionConfig",
    "LowPrecisionDtypeScheme",
    "CSA2IndexerQKFormat",
    "CSA2MainKVFormat",
    "CSA2SWAKVFormat",
    "MAIN_KV_GROUP_SIZE",
    "SWA_KV_GROUP_SIZE",
    "MX_BLOCK_SIZE_32",
    "SUPPORTED_MX_BLOCK_SIZES",
    "validate_mx_block_size",
]
