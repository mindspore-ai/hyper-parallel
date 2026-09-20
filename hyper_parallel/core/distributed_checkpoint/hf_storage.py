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
"""Safetensors checkpoints written outside DCP, loaded through the DCP read path.

Two kinds of directory are read, each by a reader of its own, since what makes a directory of one
kind valid makes one of the other kind wrong:

* :class:`HuggingFaceStorageReader` reads a Hugging Face checkpoint, as ``save_pretrained`` writes
  one: whole tensors, each in exactly one file, which is ``model.safetensors`` or one of the files a
  ``model.safetensors.index.json`` names.
* :class:`TorchShardedSafetensorsReader` reads what torch's ``HuggingFaceStorageWriter`` writes with
  ``save_distributed=True`` before consolidating it: every rank's shard of a tensor in a file of its
  own, where the shard sits in the whole tensor recorded in the file's header, and no index.

A safetensors file opens with a JSON header giving the dtype and shape of every tensor in it, and
that is all a load plans against. Both readers build the checkpoint :class:`Metadata` out of those
headers, where :class:`FileSystemReader` unpickles a ``.metadata`` file, and inherit the rest of the
load: planning, resharding, the replicated-shard broadcast and the reads themselves.

Tensors are known by the names the files give them. Mapping those onto the names of a model, and
merging or splitting tensors laid out differently from it, is left to the layer above.
"""
import json
import os
import struct
from collections.abc import Iterator
from math import prod
from pathlib import Path
from typing import Any, Optional

from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemReader
from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from hyper_parallel.core.distributed_checkpoint.storage import StorageInfo
from hyper_parallel.core.distributed_checkpoint.utils import dcp_timer_decorator

SAFETENSORS_SUFFIX = ".safetensors"
SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json"
# torch's HuggingFaceStorageWriter records where in the whole tensor each tensor of a file starts: a
# JSON string under this key of the header's ``__metadata__``, mapping every tensor name in the file
# to ``{"saved_offsets": [...]}``. The format allows nothing but strings there, hence JSON in a string.
DCP_SHARDING_INFO_KEY = "DCP_SHARDING_INFO"
SAVED_OFFSETS_KEY = "saved_offsets"

_SAFE_WEIGHTS_NAME = "model.safetensors"
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
_HEADER_METADATA_KEY = "__metadata__"
_HEADER_LENGTH_BYTES = 8

# Every dtype code safetensors (0.8) reads into torch, with the dtype it is read as, spelled the
# way a DCP checkpoint records one: ``str(tensor.dtype)``.
_SAFETENSORS_DTYPES = {
    "BOOL": "torch.bool",
    "U8": "torch.uint8",
    "I8": "torch.int8",
    "U16": "torch.uint16",
    "I16": "torch.int16",
    "U32": "torch.uint32",
    "I32": "torch.int32",
    "U64": "torch.uint64",
    "I64": "torch.int64",
    "F8_E4M3": "torch.float8_e4m3fn",
    "F8_E4M3FNUZ": "torch.float8_e4m3fnuz",
    "F8_E5M2": "torch.float8_e5m2",
    "F8_E5M2FNUZ": "torch.float8_e5m2fnuz",
    "F16": "torch.float16",
    "BF16": "torch.bfloat16",
    "F32": "torch.float32",
    "F64": "torch.float64",
    "C64": "torch.complex64",
}


def _loads(document: Any, source: str) -> Any:
    """
    Parse a JSON document, saying which one it was when it does not parse.

    Args:
        document (Any): The text or bytes to parse.
        source (str): What the document is, for the error.

    Returns:
        Any: The parsed document.

    Raises:
        ValueError: If the document is not JSON.
    """
    try:
        return json.loads(document)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{source} is not valid JSON: {e}") from e


def _read_header(path: Path) -> dict[str, Any]:
    """
    Read the JSON header a safetensors file opens with, and nothing past it.

    The header is an eight-byte little-endian length followed by that many bytes of JSON.

    Args:
        path (Path): The safetensors file.

    Returns:
        dict[str, Any]: One entry per tensor, plus ``__metadata__`` when the writer left one.

    Raises:
        ValueError: If the file is too short for the header it declares, or the header is not a
            JSON object.
    """
    with open(path, "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        prefix = f.read(_HEADER_LENGTH_BYTES)
        header_length = struct.unpack("<Q", prefix)[0] if len(prefix) == _HEADER_LENGTH_BYTES else None
        if header_length is None or header_length > file_size - _HEADER_LENGTH_BYTES:
            raise ValueError(f"{path} is not a safetensors file: it is too short for its header.")
        header = _loads(f.read(header_length), f"The header of {path}")
    if not isinstance(header, dict):
        raise ValueError(f"The header of {path} is not a JSON object.")
    return header


def _sharding_info(header: dict[str, Any], path: Path) -> Optional[dict[str, Any]]:
    """
    The offsets torch's Hugging Face writer recorded for the tensors of one file, if it did.

    Args:
        header (dict[str, Any]): The header of the file.
        path (Path): The file, for the error.

    Returns:
        Optional[dict[str, Any]]: ``{tensor name: {"saved_offsets": [...]}}``, or None for a file
        that records none, which is every file Hugging Face writes itself.

    Raises:
        ValueError: If the recorded offsets are not a JSON object.
    """
    extra = header.get(_HEADER_METADATA_KEY)
    if not isinstance(extra, dict) or DCP_SHARDING_INFO_KEY not in extra:
        return None
    info = _loads(extra[DCP_SHARDING_INFO_KEY], f"{DCP_SHARDING_INFO_KEY} of {path}")
    if not isinstance(info, dict):
        raise ValueError(f"{DCP_SHARDING_INFO_KEY} of {path} is not a JSON object.")
    return info


def _dtype_and_shape(entry: Any, key: str, path: Path) -> tuple[str, tuple[int, ...]]:
    """
    The dtype, spelled as DCP spells it, and the shape of one tensor entry of a header.

    Args:
        entry (Any): The header entry of the tensor.
        key (str): Name of the tensor, for the error.
        path (Path): The file, for the error.

    Returns:
        tuple[str, tuple[int, ...]]: The dtype and the shape.

    Raises:
        ValueError: If the entry has no valid shape, or a dtype safetensors does not read into torch.
    """
    shape = entry.get("shape") if isinstance(entry, dict) else None
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise ValueError(f"Tensor {key!r} in {path} has no valid shape: {entry!r}.")
    code = entry.get("dtype")
    if not isinstance(code, str) or code not in _SAFETENSORS_DTYPES:
        raise ValueError(
            f"Tensor {key!r} in {path} has dtype {code!r}, which is none of {', '.join(_SAFETENSORS_DTYPES)}."
        )
    return _SAFETENSORS_DTYPES[code], tuple(shape)


def _saved_offsets(
        sharding_info: Optional[dict[str, Any]], key: str, ndim: int, path: Path
) -> tuple[int, ...]:
    """
    Where in the whole tensor the part of it stored under ``key`` in one file starts.

    Args:
        sharding_info (Optional[dict[str, Any]]): The offsets the file records, as
            :func:`_sharding_info` returns them.
        key (str): Name of the tensor.
        ndim (int): Number of dimensions of the stored part.
        path (Path): The file, for the error.

    Returns:
        tuple[int, ...]: The offsets, all zero for a file that records none.

    Raises:
        ValueError: If the file records offsets but not a valid set for this tensor. Taking it to
            start at the origin instead could lay it over another shard of the same tensor.
    """
    if sharding_info is None:
        return (0,) * ndim
    entry = sharding_info.get(key)
    offsets = entry.get(SAVED_OFFSETS_KEY) if isinstance(entry, dict) else None
    if (not isinstance(offsets, list) or len(offsets) != ndim
            or not all(isinstance(offset, int) and offset >= 0 for offset in offsets)):
        raise ValueError(
            f"{DCP_SHARDING_INFO_KEY} of {path} has no valid {SAVED_OFFSETS_KEY} for the "
            f"{ndim}-dimensional tensor {key!r}: {entry!r}."
        )
    return tuple(offsets)


def _read_chunks(path: Path, sharded: bool) -> Iterator[tuple[str, str, ChunkStorageMetadata]]:
    """
    Every tensor one safetensors file holds, as a chunk of the tensor its name stands for.

    Args:
        path (Path): The safetensors file.
        sharded (bool): Whether the file is a shard written by torch, which has to record where
            each of its tensors starts. Otherwise every tensor in it has to be whole.

    Yields:
        tuple[str, str, ChunkStorageMetadata]: Name, dtype, and where in the whole tensor it sits.

    Raises:
        ValueError: If a shard records no offsets, or a file of whole tensors records a tensor as
            starting anywhere but the origin.
    """
    header = _read_header(path)
    sharding_info = _sharding_info(header, path)
    if sharded and sharding_info is None:
        raise ValueError(
            f"{path} records no {DCP_SHARDING_INFO_KEY}, so it is not a shard written by torch's "
            f"HuggingFaceStorageWriter; read a Hugging Face checkpoint with HuggingFaceStorageReader."
        )
    for key, entry in header.items():
        if key == _HEADER_METADATA_KEY:
            continue
        dtype, shape = _dtype_and_shape(entry, key, path)
        offsets = _saved_offsets(sharding_info, key, len(shape), path)
        if not sharded and any(offsets):
            raise ValueError(
                f"Tensor {key!r} in {path} is the shard at offsets {offsets} of a larger tensor; read a "
                f"directory of shards written by torch with TorchShardedSafetensorsReader."
            )
        yield key, dtype, ChunkStorageMetadata(offsets=offsets, sizes=shape)


def _tensor_size(key: str, chunks: list[ChunkStorageMetadata]) -> tuple[int, ...]:
    """
    The size of the tensor a set of chunks makes up, checking that they make up exactly that.

    Args:
        key (str): Name of the tensor, for the error.
        chunks (list[ChunkStorageMetadata]): Every chunk of it, no two at the same offsets.

    Returns:
        tuple[int, ...]: The smallest size that holds every chunk.

    Raises:
        ValueError: If the chunks differ in number of dimensions, or hold more or fewer elements
            than that size does: a file of the checkpoint is missing, or two chunks overlap.
    """
    ndim = len(chunks[0].sizes)
    if any(len(chunk.sizes) != ndim for chunk in chunks):
        raise ValueError(f"The chunks of tensor {key!r} differ in number of dimensions.")
    size = tuple(max(chunk.offsets[dim] + chunk.sizes[dim] for chunk in chunks) for dim in range(ndim))
    held = sum(prod(chunk.sizes) for chunk in chunks)
    if held != prod(size):
        raise ValueError(
            f"The chunks of tensor {key!r} hold {held} elements between them, where a tensor of size "
            f"{size} has {prod(size)}: a file of the checkpoint is missing, or two chunks overlap."
        )
    return size


def _describe_checkpoint(checkpoint_dir: Path, file_names: list[str], sharded: bool) -> Metadata:
    """
    Build the metadata of a checkpoint out of the headers of its safetensors files.

    Args:
        checkpoint_dir (Path): The checkpoint directory.
        file_names (list[str]): The files of the checkpoint, relative to ``checkpoint_dir``.
        sharded (bool): Whether the files are shards written by torch; see :func:`_read_chunks`.

    Returns:
        Metadata: Size, dtype and chunks of every tensor, and the file each chunk is in.

    Raises:
        ValueError: If a file is not of the kind ``sharded`` says, a header is malformed, or the
            chunks of one tensor do not fit together: the same chunk twice, chunks in different
            dtypes, or chunks that do not make up the whole tensor.
    """
    chunks: dict[str, list[ChunkStorageMetadata]] = {}
    dtypes: dict[str, str] = {}
    storage_data: dict[MetadataIndex, StorageInfo] = {}
    for file_name in file_names:
        for key, dtype, chunk in _read_chunks(checkpoint_dir / file_name, sharded):
            index = MetadataIndex(fqn=key, offset=chunk.offsets)
            if index in storage_data:
                raise ValueError(
                    f"Tensor {key!r} at offsets {chunk.offsets} is in both "
                    f"{storage_data[index].relative_path} and {file_name}."
                )
            if dtypes.setdefault(key, dtype) != dtype:
                raise ValueError(f"Tensor {key!r} is {dtype} in {file_name} but {dtypes[key]} in another file.")
            chunks.setdefault(key, []).append(chunk)
            storage_data[index] = StorageInfo(relative_path=file_name, offset=0, length=-1, tensor_key=key)
    state_dict_metadata = {
        key: TensorStorageMetadata(
            properties=TensorProperties(dtype=dtypes[key]),
            size=_tensor_size(key, key_chunks),
            chunks=key_chunks,
        )
        for key, key_chunks in chunks.items()
    }
    return Metadata(state_dict_metadata=state_dict_metadata, storage_data=storage_data)


class HuggingFaceStorageReader(FileSystemReader):
    """
    Storage reader for a Hugging Face checkpoint: whole tensors, each in exactly one file.

    Pass it to :func:`load` in place of a checkpoint path::

        load(model.state_dict(), storage_reader=HuggingFaceStorageReader("/models/Qwen3-8B"))

    The files read are the ones ``model.safetensors.index.json`` names, or ``model.safetensors``
    alone when there is no index. Nothing else in the directory is read, such as a second copy of
    the weights in a file of another name. A file whose header records offsets the way torch's
    HuggingFaceStorageWriter does is read as long as every offset is zero: anything else is a shard
    of a larger tensor, and a directory of those is for :class:`TorchShardedSafetensorsReader`.

    The state dict is filled by name, as from any DCP checkpoint: each of its keys has to name a
    tensor in the files, at the same global shape. Beyond that the target is free. A DTensor may
    be laid out any way, and each rank reads only its own slice of the file. The dtype may differ
    from the file's, and the copy into the target converts it. Ranks holding the same slice read
    it once and broadcast it, as :func:`load` does for any checkpoint.

    A key the files do not have fails the load, unless the planner is
    ``StandardLoadPlanner(allow_partial_load=True)``. A model with tied weights needs that, since
    Hugging Face stores a tied tensor once. Quantized tensors are read as they are stored: an FP8
    weight arrives as FP8 values, and its scales are tensors of their own.
    """

    @dcp_timer_decorator
    def load_metadata(self, **kwargs: Any) -> Metadata:
        """
        Describe the checkpoint from the headers of its safetensors files, and nothing past them.

        Args:
            **kwargs (Any): Ignored. :func:`load` asks again with ``rank`` when a first ask finds
                nothing, and a Hugging Face checkpoint has no rank-local metadata to find instead.

        Returns:
            Metadata: Size, dtype and file of every tensor.

        Raises:
            FileNotFoundError: If the directory holds neither the index nor the single weights
                file, or a file the index names is not there.
            ValueError: If the index or a header is malformed, a dtype has no torch counterpart, a
                tensor is a shard of a larger one, or two files hold the same tensor.
        """
        return _describe_checkpoint(self.checkpoint_dir, self._weight_files(), sharded=False)

    def _weight_files(self) -> list[str]:
        """
        Name the files that make up the checkpoint.

        Returns:
            list[str]: File names relative to the checkpoint directory, sorted.

        Raises:
            FileNotFoundError: If the directory holds neither the index nor the single weights file.
            ValueError: If the index does not map tensor names to file names.
        """
        index_path = self.checkpoint_dir / _SAFE_WEIGHTS_INDEX_NAME
        if index_path.is_file():
            index = _loads(index_path.read_bytes(), str(index_path))
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if (not isinstance(weight_map, dict) or not weight_map
                    or not all(isinstance(file_name, str) for file_name in weight_map.values())):
                raise ValueError(f"{index_path} has no weight_map mapping tensor names to file names.")
            return sorted(set(weight_map.values()))
        if not (self.checkpoint_dir / _SAFE_WEIGHTS_NAME).is_file():
            raise FileNotFoundError(
                f"Neither {_SAFE_WEIGHTS_INDEX_NAME} nor {_SAFE_WEIGHTS_NAME} found in {self.checkpoint_dir}. "
                f"A directory of shards written by torch is read with TorchShardedSafetensorsReader."
            )
        return [_SAFE_WEIGHTS_NAME]


class TorchShardedSafetensorsReader(FileSystemReader):
    """
    Storage reader for the shards torch's ``HuggingFaceStorageWriter`` writes with ``save_distributed``.

    Saving that way without consolidating, every rank writes its shard of each tensor to a file of
    its own, ``shard-00001-model-00001-of-00001.safetensors`` and so on, under the tensor's own
    name, and records in the file's header where in the whole tensor the shard sits. No index is
    written, since a tensor is spread over several files. torchtitan writes such a directory as the
    ``sharded`` directory of a checkpoint it saves in Hugging Face format, before consolidating it.

    Every safetensors file in the directory is read, and every one has to record those offsets.
    The shards of a tensor are its chunks, and have to make up the whole tensor between them,
    leaving no gap and overlapping nowhere. A directory with an index holds whole tensors, and is
    for :class:`HuggingFaceStorageReader`.

    Loading from it is as from any DCP checkpoint; see :class:`HuggingFaceStorageReader` for what
    the state dict may be.
    """

    @dcp_timer_decorator
    def load_metadata(self, **kwargs: Any) -> Metadata:
        """
        Describe the checkpoint from the headers of its shard files, and nothing past them.

        Args:
            **kwargs (Any): Ignored, as :meth:`HuggingFaceStorageReader.load_metadata` ignores them.

        Returns:
            Metadata: Size, dtype and chunks of every tensor, and the file each chunk is in.

        Raises:
            FileNotFoundError: If the directory holds no safetensors file.
            ValueError: If the directory has an index, a file records no offsets, a header is
                malformed, a dtype has no torch counterpart, or the shards of one tensor do not fit
                together: the same shard twice, shards in different dtypes, or shards that do not
                make up the whole tensor.
        """
        indexes = sorted(path.name for path in self.checkpoint_dir.glob(f"*{SAFETENSORS_INDEX_SUFFIX}"))
        if indexes:
            raise ValueError(
                f"{self.checkpoint_dir} has {indexes[0]}, which a checkpoint of whole tensors has and one "
                f"of shards does not; read it with HuggingFaceStorageReader."
            )
        files = sorted(path.name for path in self.checkpoint_dir.glob(f"*{SAFETENSORS_SUFFIX}"))
        if not files:
            raise FileNotFoundError(f"No *{SAFETENSORS_SUFFIX} file found in {self.checkpoint_dir}.")
        return _describe_checkpoint(self.checkpoint_dir, files, sharded=True)
