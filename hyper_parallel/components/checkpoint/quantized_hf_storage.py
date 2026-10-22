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
"""Read FineGrained FP8/FP4 HF storage through shared DCP region reads.

The planner sees logical, unpacked shapes. Each physical read decodes only the
requested region, expanded to quantization block boundaries. BF16 rounding is
performed before any model/replacement conversion. Decoded values, including
the sign of zero, follow the installed Transformers implementation.
"""
# Native Torch is the only supported backend; the legacy lint checker still flags its import.
# pylint: disable=forbidden-backend-import

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Union

import torch
from safetensors import safe_open
from torch import nn

from hyper_parallel.components.checkpoint.hf_storage import HuggingFaceStorageReader
from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata, Metadata, MetadataIndex, TensorProperties, TensorStorageMetadata,
)

_SCRATCH_BYTES = 32 * 1024 * 1024


def supports_hf_dequantization(config: Any) -> bool:
    """Return whether shared loading supports the declared method.

    Args:
        config: Quantization dictionary or Transformers quantization configuration.

    Returns:
        Whether the shared reader supports the quantization method.
    """
    method = config.get("quant_method") if isinstance(config, dict) else getattr(config, "quant_method", None)
    return method == "fp8"


def validate_quantization_config(config: dict[str, Any]) -> None:
    """Validate supported storage formats without depending on a model family.

    Args:
        config: Quantization configuration from the checkpoint.

    Raises:
        ValueError: The method, block dimensions or scale format is unsupported.
    """
    if not supports_hf_dequantization(config):
        raise ValueError("Unsupported HF quantization method for online DCP decoding; expected fp8")
    block = config.get("weight_block_size") or [128, 128]
    if (not isinstance(block, (list, tuple)) or len(block) != 2
            or any(not isinstance(size, int) or size <= 0 for size in block)):
        raise ValueError("weight_block_size must contain two positive integers")
    if config.get("scale_fmt", "float") not in ("float", "ue8m0"):
        raise ValueError("Unsupported FP8 scale format")


def _validate_scales(scale: torch.Tensor) -> None:
    if scale.dtype in (torch.float8_e8m0fnu, torch.uint8):
        codes = scale.contiguous().view(torch.uint8)
        if torch.any(codes == 255):
            raise ValueError("UE8M0 scale contains reserved NaN code 255")
    elif scale.dtype != torch.float32:
        raise ValueError(f"Unsupported scale dtype: {scale.dtype}")
    result = scale.view(torch.float8_e8m0fnu).float() if scale.dtype == torch.uint8 else scale.float()
    if not torch.all(torch.isfinite(result) & (result > 0)):
        raise ValueError("Quantization scales must be positive and finite")


@dataclass(frozen=True)
class _EncodedWeight:
    scale: str
    shape: tuple[int, int]
    block_shape: tuple[int, int]
    packed_elements: int


class QuantizedHuggingFaceStorageReader(HuggingFaceStorageReader):
    """Decode two-dimensional FineGrained FP8/FP4 weights using Transformers.

    Args:
        checkpoint_dir: Local HF directory containing a supported quantization_config, weights and scales.
    """

    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        """Initialize metadata caches without reading weight payloads."""
        super().__init__(checkpoint_dir)
        config = json.loads((self.checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        self.quantization_config = config.get("quantization_config", {})
        validate_quantization_config(self.quantization_config)
        # Keep optional/version-dependent HF quantization imports off ordinary HF loading.
        try:
            from transformers.integrations.finegrained_fp8 import Fp8Dequantize  # pylint: disable=C0415
            from transformers.quantizers.quantizer_finegrained_fp8 import (  # pylint: disable=C0415
                FineGrainedFP8HfQuantizer,
            )
            from transformers.utils.quantization_config import FineGrainedFP8Config  # pylint: disable=C0415
        except ImportError as exc:
            raise ValueError("Online HF decoding requires Transformers FineGrainedFP8/Fp8Dequantize support") from exc
        quantizer = FineGrainedFP8HfQuantizer(FineGrainedFP8Config(
            dequantize=True, weight_block_size=tuple(self.quantization_config.get("weight_block_size") or [128, 128]),
            scale_fmt=self.quantization_config.get("scale_fmt", "float"),
        ))
        self._dequantizer = Fp8Dequantize(quantizer)
        self._dtype_target = nn.Module()
        self._dtype_target.register_parameter(
            "weight", nn.Parameter(torch.empty(0, device="meta", dtype=torch.bfloat16)))
        self._physical: Optional[Metadata] = None
        self._logical: Optional[Metadata] = None
        self._encoded: dict[str, _EncodedWeight] = {}

    def load_metadata(self, **kwargs: Any) -> Metadata:
        """Read headers and expose unpacked shapes, hiding consumed scales."""
        if self._logical is not None:
            return self._logical
        physical = super().load_metadata(**kwargs)
        entries = physical.state_dict_metadata
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        if index_path.is_file():
            selected = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            missing = set(selected) - entries.keys()
            if missing:
                raise ValueError(f"HF index references missing tensors: {sorted(missing)[:10]}")
            entries = {name: entries[name] for name in selected}
        logical = {}
        used_scales = set()
        for name, entry in entries.items():
            encoded = self._describe_weight(name, entry, entries)
            if encoded is None:
                logical[name] = entry
                continue
            self._encoded[name] = encoded
            used_scales.add(encoded.scale)
            logical[name] = TensorStorageMetadata(
                properties=TensorProperties(dtype="torch.bfloat16"), size=encoded.shape,
                chunks=[ChunkStorageMetadata(offsets=(0, 0), sizes=encoded.shape)],
            )
        for name in used_scales:
            logical.pop(name, None)
        orphan_scales = [name for name, entry in logical.items()
                         if str(entry.properties.dtype) == "torch.float8_e8m0fnu"]
        if orphan_scales:
            raise ValueError(f"Unconsumed quantization scales: {orphan_scales[:10]}")
        self._physical = physical
        self._logical = replace(physical, state_dict_metadata=logical)
        return self._logical

    def _describe_weight(self, name: str, entry: TensorStorageMetadata,
                         entries: dict[str, Any]) -> Optional[_EncodedWeight]:
        """Resolve a quantized weight's scale, logical shape, and block layout."""
        dtype = str(entry.properties.dtype)
        if not name.endswith("weight"):
            return None
        candidates = [name[:-6] + suffix for suffix in ("scale", "weight_scale_inv")
                      if name[:-6] + suffix in entries]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous quantization scales: {name}")
        packed = 2 if dtype in ("torch.int8", "torch.uint8") else 1
        if dtype not in ("torch.float8_e4m3fn", "torch.int8", "torch.uint8"):
            if candidates or dtype.startswith("torch.float8"):
                raise ValueError(f"Unsupported encoded weight: {name} ({dtype})")
            return None
        if not candidates or len(entry.size) != 2:
            raise ValueError(f"Missing scale or unsupported encoded shape: {name}")
        rows, columns = entry.size
        columns *= packed
        scale = entries[candidates[0]]
        block_shape = self._scale_block_shape(name, (rows, columns), scale, packed)
        return _EncodedWeight(candidates[0], (rows, columns), block_shape, packed)

    def _scale_block_shape(self, name: str, shape: tuple[int, int],
                           scale: TensorStorageMetadata, packed: int) -> tuple[int, int]:
        """Validate a scale grid and derive its logical block dimensions."""
        if (len(scale.size) != 2 or min(scale.size) <= 0
                or str(scale.properties.dtype) not in ("torch.float8_e8m0fnu", "torch.float32", "torch.uint8")):
            raise ValueError(f"Invalid quantization scale shape or dtype: {name}")
        rows, columns = shape
        block_rows, block_columns = self.quantization_config.get("weight_block_size") or [128, 128]
        expected = ((rows + block_rows - 1) // block_rows, (columns + block_columns - 1) // block_columns)
        if packed == 2:
            # FP4 packing and scale-grid granularity are properties of storage,
            # not of expert/module names. The same layout can belong to any model.
            if rows % scale.size[0] or columns % scale.size[1]:
                raise ValueError(f"FP4 weight is not divisible by its scale grid: {name}")
            block_rows, block_columns = rows // scale.size[0], columns // scale.size[1]
            if block_columns % packed:
                raise ValueError(f"FP4 scale blocks must align with packed byte pairs: {name}")
        elif tuple(scale.size) != expected:
            # Row-wise MX-style scaling can coexist with block-wise FP8 scaling.
            if tuple(scale.size) != (rows, (columns + 31) // 32):
                raise ValueError(f"Invalid quantization scale shape: {name}")
            block_rows, block_columns = 1, 32
        return block_rows, block_columns

    def _physical_slice(self, name: str, region: tuple[slice, slice], open_files: Any = None) -> torch.Tensor:
        info = self._physical.storage_data[MetadataIndex(name, (0, 0))]
        path = str(self.checkpoint_dir / info.relative_path)
        if open_files is not None:
            return open_files.reader(path).get_slice(name)[region]
        with safe_open(path, framework="pt", device="cpu") as source:
            return source.get_slice(name)[region]

    def read_region(self, name: str, offsets: tuple[int, int], lengths: tuple[int, int],
                    open_files: Any = None) -> torch.Tensor:
        """Read a logical rectangle, with bounded decoding scratch on CPU.

        Args:
            name: Original HF weight name.
            offsets: Logical row and column starts, before FP4 packing.
            lengths: Logical rectangle shape.
            open_files: Optional file cache supplied by the DCP reader.

        Returns:
            BF16 rectangle with the same rounding as offline conversion; ordinary
            floating-point tensors retain their stored dtype.
        """
        metadata = self.load_metadata()
        shape = metadata.state_dict_metadata[name].size
        if (len(offsets) != 2 or len(lengths) != 2
                or len(shape) != 2
                or any(start < 0 or length < 0 or start + length > size
                       for start, length, size in zip(offsets, lengths, shape))):
            raise ValueError(f"Invalid logical read region for {name}: {offsets}, {lengths}")
        encoded = self._encoded.get(name)
        if encoded is None:
            region = tuple(slice(start, start + length) for start, length in zip(offsets, lengths))
            return self._physical_slice(name, region, open_files)
        first_row, first_column = offsets
        row_count, column_count = lengths
        result = torch.empty(lengths, dtype=torch.bfloat16)
        if not row_count or not column_count:
            return result
        row_block, column_block = encoded.block_shape
        left = first_column // column_block * column_block
        right = min((first_column + column_count + column_block - 1) // column_block * column_block,
                    encoded.shape[1])
        rows_per_chunk = max(row_block, _SCRATCH_BYTES // max(1, (right - left) * 32))
        rows_per_chunk = rows_per_chunk // row_block * row_block
        cursor = first_row
        while cursor < first_row + row_count:
            start = cursor // row_block * row_block
            stop = min(start + rows_per_chunk, encoded.shape[0], first_row + row_count)
            # Read complete quantization blocks even for unaligned end slices.
            read_stop = min((stop + row_block - 1) // row_block * row_block, encoded.shape[0])
            value = self._decode_chunk(name, encoded, start, read_stop, left, right, open_files)
            result[cursor - first_row:stop - first_row].copy_(
                value[cursor - start:stop - start, first_column - left:first_column - left + column_count])
            cursor = stop
        return result

    def _decode_chunk(self, name: str, encoded: _EncodedWeight, start: int, stop: int,
                      left: int, right: int, open_files: Any) -> torch.Tensor:
        """Read aligned weight and scale blocks and decode the requested BF16 rectangle."""
        packed = encoded.packed_elements
        weight = self._physical_slice(name, (slice(start, stop), slice(left // packed, (right + packed - 1) // packed)),
                                      open_files)
        row_block, column_block = encoded.block_shape
        scale = self._physical_slice(encoded.scale,
                                     (slice(start // row_block, (stop + row_block - 1) // row_block),
                                      slice(left // column_block, (right + column_block - 1) // column_block)),
                                     open_files)
        _validate_scales(scale)
        if packed == 2 and weight.dtype == torch.uint8:
            weight = weight.view(torch.int8)
        padded_shape = (scale.shape[0] * row_block, scale.shape[1] * column_block // packed)
        if tuple(weight.shape) != padded_shape:
            # HF infers block sizes from the scale grid. Pad only the global edge
            # so a partial last block is not mistaken for a different block size.
            padded = torch.zeros(padded_shape, dtype=weight.dtype)
            padded[:weight.shape[0], :weight.shape[1]].copy_(weight)
            weight = padded
        value = self._dequantizer.convert(
            {"weight$": weight, "weight_scale_inv": scale},
            full_layer_name="weight", model=self._dtype_target,
        )["weight"][:stop - start, :right - left]
        if tuple(value.shape) != (stop - start, right - left) or value.dtype != torch.bfloat16:
            raise ValueError("Transformers Fp8Dequantize did not produce the requested BF16 rectangle")
        if not torch.all(torch.isfinite(value)):
            raise ValueError(f"Nonfinite decoded value or BF16 overflow: {name}")
        return value

    def _fetch_from_storage(self, items: list, open_files: Any) -> list:
        ordinary, encoded = [], []
        for item in items:
            (encoded if item.storage_index.fqn in self._encoded else ordinary).append(item)
        fetched = super()._fetch_from_storage(ordinary, open_files)
        for item in encoded:
            fetched.append((item, self.read_region(item.storage_index.fqn, tuple(item.storage_offsets),
                                                   tuple(item.lengths), open_files)))
        return fetched


def build_quantized_hf_reader(location: Union[Path, str]) -> Optional[QuantizedHuggingFaceStorageReader]:
    """Select shared decoding only for a directory declaring supported quantization.

    Args:
        location: Resolved local checkpoint directory or file.

    Returns:
        Quantized reader, or None for an ordinary floating-point checkpoint.
    """
    root = Path(location)
    config_path = root / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    if not quantization:
        return None
    validate_quantization_config(quantization)
    return QuantizedHuggingFaceStorageReader(root)
