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
"""Reserved VeOmni-style field collation interfaces."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

CollateMode = Literal["pack", "concat", "stack", "list"]


@dataclass(frozen=True)
class FieldCollateSpec:
    """Describe how one sample field is combined across a micro-batch.

    Args:
        mode: Combination operation for this field.
        dim: Tensor dimension used by ``pack`` and ``concat``.
        pad_value: Value used when a packed field is padded to a target length.
    """

    mode: CollateMode = "stack"
    dim: int = 0
    pad_value: int = 0


class ModelSampleCollator:
    """Declare the deferred VeOmni-style per-field collation stage."""

    def __init__(
            self,
            field_specs: Mapping[str, FieldCollateSpec],
            *,
            pad_to_length: int | None = None,
    ) -> None:
        """Initialize the field collator.

        Args:
            field_specs: Collation policy keyed by sample field.
            pad_to_length: Optional target length for fields using ``pack``.

        Raises:
            ValueError: If ``pad_to_length`` is not a positive integer.
        """
        if pad_to_length is not None and (
                isinstance(pad_to_length, bool)
                or not isinstance(pad_to_length, int)
                or pad_to_length <= 0
        ):
            raise ValueError("pad_to_length must be a positive integer or None")
        self.field_specs = dict(field_specs)
        self.pad_to_length = pad_to_length

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Reserve field-specific collation for a later implementation.

        Args:
            samples: Model samples selected for one micro-batch.

        Raises:
            NotImplementedError: Always, until VeOmni-style field collation is
                implemented.
        """
        del samples
        raise NotImplementedError("VeOmni-style field collation is not implemented yet")


__all__ = ["FieldCollateSpec", "ModelSampleCollator"]
