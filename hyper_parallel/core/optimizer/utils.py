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

"""Install ``info_rank0`` / ``warning_rank0`` on ``logging.Logger``."""

import logging
import os

import torch


def _get_rank() -> int:
    try:
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except (ImportError, RuntimeError):
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def info_rank0(self, msg, *args, **kwargs) -> None:
    if _get_rank() == 0:
        kwargs.setdefault("stacklevel", 2)
        self.info(msg, *args, **kwargs)


def warning_rank0(self, msg, *args, **kwargs) -> None:
    if _get_rank() == 0:
        kwargs.setdefault("stacklevel", 2)
        self.warning(msg, *args, **kwargs)


def debug_rank0(self, msg, *args, **kwargs) -> None:
    if _get_rank() == 0:
        kwargs.setdefault("stacklevel", 2)
        self.debug(msg, *args, **kwargs)


def get_device_count() -> int:
    """Return the active accelerator count, defaulting to 1."""
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return npu.device_count()
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def get_current_device() -> torch.device:
    """Return the current accelerator device, or CPU when none is available."""
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return torch.device("npu", npu.current_device())
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def empty_accelerator_cache() -> None:
    """Clear the active accelerator cache when supported."""
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


_INSTALLED = False


def _install_logger_methods() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    logging.Logger.info_rank0 = info_rank0
    logging.Logger.warning_rank0 = warning_rank0
    logging.Logger.debug_rank0 = debug_rank0
    _INSTALLED = True


_install_logger_methods()
