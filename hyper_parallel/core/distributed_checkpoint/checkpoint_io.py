# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
"""Reading and writing one checkpoint file.

The whole-file end of checkpoint storage: a flat tensor dict in, a file out, and back.
Which shards go in which file is decided elsewhere -- see ``filesystem_storage`` for the
save and load that call these.
"""
import torch
from safetensors.torch import load_file, save_file


def unshare_storage(state_dict: dict) -> dict:
    """
    Return the same mapping with no two tensors sharing one storage.

    ``safetensors`` refuses to write a mapping whose entries alias each other,
    so a model with tied weights -- ``lm_head.weight`` pointing at the
    embedding under ``tie_word_embeddings``, the common case -- fails at save
    time after training has already succeeded. Entries after the first in an
    alias group are copied, which keeps every key, every value and the
    round trip intact at the cost of writing the tied tensor once per name.

    Tensors with no storage (``data_ptr() == 0``: empty, meta) are left alone;
    they alias each other trivially and copying them would mean nothing.

    Args:
        state_dict (dict): Tensors to write, keyed by their physical name.

    Returns:
        dict: The same keys, with aliases replaced by independent copies.
    """
    owners = {}
    unshared = {}
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            unshared[name] = value
            continue
        pointer = value.untyped_storage().data_ptr()
        key = (value.device, pointer)
        if pointer and key in owners:
            unshared[name] = value.detach().clone().contiguous()
        else:
            owners[key] = name
            unshared[name] = value
    return unshared


def save_checkpoint_file(state_dict: dict, file_path: str, ckpt_format: str = "safetensors") -> None:
    """
    Write a flat tensor dict to one checkpoint file.

    Args:
        state_dict (dict): Tensors to write, keyed by their physical name.
        file_path (str): Where to write them.
        ckpt_format (str): ``safetensors`` for the safetensors format, anything else for
            a pickle written by :func:`torch.save`. Default ``safetensors``.
    """
    if ckpt_format == "safetensors":
        save_file(tensors=unshare_storage(state_dict), filename=file_path)
    else:
        torch.save(obj=state_dict, f=file_path)


def load_checkpoint_file(file_path: str, ckpt_format: str = "safetensors") -> dict:
    """
    Read one checkpoint file back into a flat tensor dict.

    Args:
        file_path (str): The file to read.
        ckpt_format (str): ``safetensors`` for the safetensors format, anything else for
            a pickle read by :func:`torch.load`. Default ``safetensors``.

    Returns:
        dict: What the file holds, keyed by physical name.
    """
    if ckpt_format == "safetensors":
        return load_file(filename=file_path)
    return torch.load(f=file_path, weights_only=True)
