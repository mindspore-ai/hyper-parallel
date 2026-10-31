# Copyright 2026
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
"""Read a pinned GPT sample index and prepare batches for Xing40 examples."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import default_collate

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.indexed.indexed_data_reader import IndexedDataReader

_INDEX_NAMES = ("document_index", "sample_index", "shuffle_index")
_FIELDS = ("input_ids", "labels", "loss_mask", "position_ids")
_TOPOLOGIES = {(1, 1, 1, 1, 1), (4, 1, 1, 1, 1), (2, 1, 2, 1, 1), (4, 1, 1, 1, 2)}


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _verify_file(path: Path, expected: str) -> tuple[int, int, int, int]:
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("SHA256 must contain 64 lowercase hexadecimal characters")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = _identity(os.fstat(stream.fileno()))
        while True:
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if _identity(os.fstat(stream.fileno())) != before:
            raise ValueError(f"file changed while hashing: {path}")
    if digest.hexdigest() != expected or _identity(path.stat()) != before:
        raise ValueError(f"file hash or identity differs: {path}")
    return before


def _load_index(path: str, expected: str, dimensions: int) -> np.ndarray:
    file_path = Path(path)
    identity = _verify_file(file_path, expected)
    array = np.load(file_path, mmap_mode="r", allow_pickle=False)
    if _identity(file_path.stat()) != identity:
        raise ValueError(f"index changed while opening: {file_path}")
    if array.ndim != dimensions or array.dtype.kind not in "iu":
        raise ValueError(f"index must be an integer array of rank {dimensions}: {file_path}")
    return array


def _validate_mesh(mesh_context: Any) -> tuple[int, int, int, int, int]:
    topology = tuple(getattr(mesh_context, name, None)
                     for name in ("dp_size", "tp_size", "cp_size", "pp_size", "ep_size"))
    if topology not in _TOPOLOGIES:
        raise ValueError(f"unsupported alignment topology: {topology}")
    expected_world_size = 1 if topology[0] == 1 else 4
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != expected_world_size:
        raise ValueError(f"expected world size {expected_world_size} for topology {topology}")
    return topology


def _validate_data_options(data_config: Mapping[str, Any], sequence_length: int) -> None:
    expected = {
        "seq_length": sequence_length,
        "eod": -1,
        "pad": -1,
        "add_extra_token_to_sequence": True,
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "eod_mask_loss": False,
        "create_attention_mask": False,
        "create_compressed_eod_mask": False,
    }
    if not isinstance(data_config, Mapping):
        raise TypeError("data_config must be a mapping")
    for key, value in expected.items():
        if type(data_config.get(key)) is not type(value) or data_config.get(key) != value:
            raise ValueError(f"data_config requires {key}={value!r}")


class AlignmentDataset:
    """Map logical sample IDs through externally supplied GPT indices.

    Args:
        data_prefix: Prefix of the paired indexed token files.
        index_paths: Document, sample, and shuffle NPY paths.
        index_sha256: Expected SHA256 for each index file.
        data_sha256: Expected SHA256 for the ``bin`` and ``idx`` files.
        sequence_length: Number of input tokens in one sample.

    Note:
        Sampler state owns the resume cursor; this dataset adds no offset.
    """

    def __init__(
            self,
            data_prefix: str,
            index_paths: Mapping[str, str],
            index_sha256: Mapping[str, str],
            data_sha256: Mapping[str, str],
            sequence_length: int,
    ) -> None:
        """Open the token reader and three explicitly pinned sample indices."""
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer")
        if not isinstance(index_paths, Mapping) or set(index_paths) != set(_INDEX_NAMES):
            raise ValueError("index_paths must contain document_index, sample_index and shuffle_index")
        if not isinstance(index_sha256, Mapping) or set(index_sha256) != set(_INDEX_NAMES):
            raise ValueError("index_sha256 must contain the same three index names")
        if not isinstance(data_sha256, Mapping) or set(data_sha256) != {"bin", "idx"}:
            raise ValueError("data_sha256 must contain bin and idx")
        self.sequence_length = sequence_length
        for suffix in ("bin", "idx"):
            _verify_file(Path(f"{data_prefix}.{suffix}"), data_sha256[suffix])
        self.reader = IndexedDataReader(str(data_prefix), mmap=True)
        self.document_index = _load_index(index_paths["document_index"], index_sha256["document_index"], 1)
        self.sample_index = _load_index(index_paths["sample_index"], index_sha256["sample_index"], 2)
        self.shuffle_index = _load_index(index_paths["shuffle_index"], index_sha256["shuffle_index"], 1)
        if self.sample_index.shape[1] != 2 or self.sample_index.shape[0] < 2:
            raise ValueError("sample_index must contain at least two [document, offset] rows")
        if len(self.shuffle_index) != len(self.sample_index) - 1 or not len(self.document_index):
            raise ValueError("sample and shuffle index lengths are inconsistent")
        if np.any(self.document_index < 0) or np.any(self.document_index >= len(self.reader.sequence_lengths)):
            raise ValueError("document_index contains out-of-range sequence IDs")
        if np.any(self.sample_index < 0) or np.any(self.sample_index[:, 0] >= len(self.document_index)):
            raise ValueError("sample_index contains out-of-range document positions or offsets")
        if np.any(self.shuffle_index < 0) or np.any(self.shuffle_index >= len(self)):
            raise ValueError("shuffle_index contains out-of-range sample IDs")

    def __len__(self) -> int:
        """Return the number of logical samples represented by the indices."""
        return len(self.sample_index) - 1

    def __getitem__(self, index: int | None) -> dict[str, np.ndarray]:
        """Return four int32 fields using the pinned sample order."""
        is_padding = index is None
        if is_padding:
            index = 0
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("sample index must be an integer or None")
        if index < 0 or index >= len(self):
            raise IndexError(f"sample index out of range: {index}")
        selected = int(self.shuffle_index[index])
        first_document, first_offset = (int(value) for value in self.sample_index[selected])
        last_document, last_offset = (int(value) for value in self.sample_index[selected + 1])
        if first_document > last_document:
            raise ValueError("sample document range is reversed")
        pieces = []
        for position in range(first_document, last_document + 1):
            offset = first_offset if position == first_document else 0
            length = None if position < last_document else last_offset - offset + 1
            pieces.append(self.reader.get(int(self.document_index[position]), offset=offset, length=length))
        wanted = self.sequence_length + 1
        length = sum(piece.size for piece in pieces)
        if length > wanted:
            raise ValueError("sample is longer than sequence_length + 1")
        if length < wanted:
            pieces.append(np.full(wanted - length, -1, dtype=np.int64))
        tokens_with_target = np.concatenate(pieces, dtype=np.int64)
        tokens = tokens_with_target[:-1].copy()
        labels = tokens_with_target[1:].copy()
        loss_mask = (labels != -1).astype(np.int32)
        if is_padding:
            loss_mask.fill(0)
        tokens[tokens == -1] = 0
        labels[labels == -1] = 0
        bounds = np.iinfo(np.int32)
        if np.any(tokens < bounds.min) or np.any(tokens > bounds.max) or np.any(labels < bounds.min) or np.any(
                labels > bounds.max):
            raise ValueError("sample tokens exceed int32 range")
        return {
            "input_ids": tokens.astype(np.int32),
            "labels": labels.astype(np.int32),
            "loss_mask": loss_mask,
            "position_ids": np.arange(self.sequence_length, dtype=np.int32),
        }


def build_alignment_dataset(
        *,
        data_prefix: str,
        index_paths: Mapping[str, str],
        index_sha256: Mapping[str, str],
        data_sha256: Mapping[str, str],
        sequence_length: int,
        eod_token_id: int,
        pad_token_id: int,
        add_extra_token_to_sequence: bool,
        reset_position_ids: bool,
        reset_attention_mask: bool,
        eod_mask_loss: bool,
        create_attention_mask: bool,
        create_compressed_eod_mask: bool,
        data_config: Mapping[str, Any],
        mesh_context: Any,
        transform: Any = None,
        tokenizer: Any = None,
        training_config: Any = None,
) -> AlignmentDataset:
    """Build a pinned training dataset without constructing new indices.

    Args:
        data_prefix: Prefix of the paired indexed token files.
        index_paths: Paths to document, sample, and shuffle index arrays.
        index_sha256: Expected index-file hashes.
        data_sha256: Expected token and low-level index hashes.
        sequence_length: Number of input tokens per sample.
        eod_token_id: EOD ID, fixed to -1 for this example.
        pad_token_id: Padding ID, fixed to -1 for this example.
        add_extra_token_to_sequence: Whether the source contains one target token.
        reset_position_ids: Whether positions reset at EOD.
        reset_attention_mask: Whether attention resets at EOD.
        eod_mask_loss: Whether the EOD token is masked in the loss.
        create_attention_mask: Whether the dataset creates an attention mask.
        create_compressed_eod_mask: Whether the dataset creates a compressed mask.
        data_config: Explicit matching data contract.
        mesh_context: Trainer topology.
        transform: Unused; this example rejects a second transformation.
        tokenizer: Optional tokenizer with matching EOD and padding IDs.
        training_config: Unused Trainer runtime context.

    Returns:
        The map dataset consumed by the Trainer's native batch sampler.
    """
    del training_config
    _validate_mesh(mesh_context)
    _validate_data_options(data_config, sequence_length)
    options = (eod_token_id, pad_token_id, add_extra_token_to_sequence, reset_position_ids,
               reset_attention_mask, eod_mask_loss, create_attention_mask, create_compressed_eod_mask)
    if options != (-1, -1, True, False, False, False, False, False):
        raise ValueError("unsupported GPT index and mask options")
    if transform is not None:
        raise ValueError("alignment samples must not be transformed again")
    if tokenizer is not None and (getattr(tokenizer, "eod", None) != -1 or getattr(tokenizer, "pad", None) != -1):
        raise ValueError("tokenizer EOD and pad IDs must match the pinned data contract")
    return AlignmentDataset(data_prefix, index_paths, index_sha256, data_sha256, sequence_length)


def build_alignment_collate_fn(*, mesh_context: Any) -> Any:
    """Return ordinary PyTorch collation for the supported topology.

    Args:
        mesh_context: Trainer topology.

    Returns:
        PyTorch's default collation function.
    """
    _validate_mesh(mesh_context)
    return default_collate


class AlignmentBatch:
    """Preserve the explicit mask while preparing single, DP, CP, or EP inputs."""

    def __init__(
            self,
            mesh_context: Any,
            device: Any,
            tokenizer: Any,
            data_config: Mapping[str, Any],
            pp_shared_data: bool,
            *,
            expected_batch_size: int = 1,
            sequence_length: int = 4096,
    ) -> None:
        """Validate the supported topology and fixed-size sample contract."""
        topology = _validate_mesh(mesh_context)
        _validate_data_options(data_config, sequence_length)
        if pp_shared_data:
            raise ValueError("pipeline shared data is unsupported")
        if expected_batch_size != 1 or isinstance(expected_batch_size, bool):
            raise ValueError("the alignment example requires micro batch size 1")
        if sequence_length <= 0 or sequence_length % topology[2]:
            raise ValueError("sequence length must be divisible by the CP size")
        if tokenizer is not None and (getattr(tokenizer, "eod", None) != -1 or getattr(tokenizer, "pad", None) != -1):
            raise ValueError("tokenizer EOD and pad IDs must match the pinned data contract")
        self.cp_size = topology[2]
        self.cp_rank = getattr(mesh_context, "cp_rank", 0)
        if type(self.cp_rank) is not int or not 0 <= self.cp_rank < self.cp_size:
            raise ValueError("invalid CP rank")
        self.device = torch.device(device)
        self.expected_shape = (expected_batch_size, sequence_length)

    def prepare_raw(self, source: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Validate and slice one batch without shifting labels again.

        Args:
            source: Collated CPU int32 input, labels, mask, and position IDs.

        Returns:
            Model inputs and loss inputs for the local CP interval.
        """
        if not isinstance(source, Mapping) or set(source) != set(_FIELDS):
            raise ValueError("source must contain the four alignment fields")
        raw = {name: torch.as_tensor(source[name]) for name in _FIELDS}
        for name, value in raw.items():
            if value.device.type != "cpu" or value.dtype != torch.int32 or tuple(value.shape) != self.expected_shape:
                raise ValueError(f"{name} must be a CPU int32 tensor of shape {self.expected_shape}")
        if bool(torch.any((raw["loss_mask"] != 0) & (raw["loss_mask"] != 1))):
            raise ValueError("loss_mask must contain only zero or one")
        if bool(torch.any(raw["input_ids"] < 0)) or bool(torch.any(raw["labels"] < 0)):
            raise ValueError("input IDs and labels must have nonnegative padding replacements")
        expected_positions = torch.arange(self.expected_shape[1], dtype=torch.int32).expand(self.expected_shape)
        if not torch.equal(raw["position_ids"], expected_positions):
            raise ValueError("position IDs must follow the no-reset contract")
        width = self.expected_shape[1] // self.cp_size
        part = slice(self.cp_rank * width, (self.cp_rank + 1) * width)
        inputs = raw["input_ids"][..., part].to(device=self.device, dtype=torch.long)
        labels = raw["labels"][..., part].to(device=self.device, dtype=torch.long)
        positions = raw["position_ids"][..., part].to(device=self.device, dtype=torch.long)
        mask = raw["loss_mask"][..., part].to(device=self.device, dtype=torch.int32)
        targets = labels.masked_fill(mask == 0, IGNORE_INDEX)
        model_inputs = {"input_ids": inputs, "labels": labels, "shift_labels": targets, "position_ids": positions}
        loss_inputs = {"labels": labels, "shift_labels": targets, "loss_mask": mask}
        return model_inputs, loss_inputs

    def __call__(self, data_iterator: Any) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Consume exactly one loader batch."""
        return self.prepare_raw(next(data_iterator))
