# Copyright 2025-2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# FIXME: VeOmni version -> HyperModels version
"""Helper utils"""

import datetime
import gc
import logging as builtin_logging
import os
import random
import subprocess
import sys
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from transformers import set_seed as set_seed_func

from .device import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_device_type,
    get_torch_device,
)

if IS_NPU_AVAILABLE:
    import torch_npu


# internal use
VALID_CONFIG_TYPE = None
FlopsCounter = None


def convert_hdfs_fuse_path(*args, **kwargs):
    if len(args) > 0:
        return args[0]
    return kwargs.get("path", None)


logger = builtin_logging.getLogger(__name__)

CACHE_DIR = os.path.expanduser(os.getenv("CACHE_DIR", os.path.join("~/.cache", "veomni")))


def _info_rank0(message: str) -> None:
    """Log an informational message on global rank zero."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(message)


def enable_high_precision_for_bf16():
    """
    Set high accumulation dtype for matmul and reduction.
    """
    if IS_CUDA_AVAILABLE:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    if IS_NPU_AVAILABLE:
        torch.npu.matmul.allow_tf32 = False
        torch.npu.matmul.allow_bf16_reduced_precision_reduction = False


def enable_full_determinism(seed: int):
    """
    Helper function for reproducibility in distributed training.
    See https://pytorch.org/docs/stable/notes/randomness.html for details.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["NCCL_DETERMINISTIC"] = "1"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    if IS_NPU_AVAILABLE:
        # The environment variable required to enable deterministic mode on Ascend NPUs.
        os.environ["NCCL_DETERMINISTIC"] = "true"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if IS_NPU_AVAILABLE:
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def set_seed(seed: int | None, full_determinism: bool = False) -> None:
    """
    Sets a manual seed on all devices.
    """
    if seed is None:
        return
    if full_determinism:
        enable_full_determinism(seed)
    else:
        set_seed_func(seed)


def create_logger(name: Optional[str] = None) -> "logging._Logger":
    """
    Creates a pretty logger for the third-party program.
    """
    logger = builtin_logging.getLogger(name)
    formatter = builtin_logging.Formatter(
        fmt="[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = builtin_logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(builtin_logging.INFO)
    logger.propagate = False
    return logger


def enable_third_party_logging() -> None:
    """
    Enables explicit logger of the third-party libraries.
    """
    transformers.logging.set_verbosity_info()
    transformers.logging.enable_default_handler()
    transformers.logging.enable_explicit_format()


def disable_warning() -> None:
    """
    Enables warning filter.
    """
    from pyiceberg.metrics import LoggingMetricsReporter

    builtin_logging.basicConfig(level=builtin_logging.ERROR)
    warnings.simplefilter("ignore")
    LoggingMetricsReporter()
    LoggingMetricsReporter._logger = builtin_logging.getLogger(LoggingMetricsReporter.__name__)
    LoggingMetricsReporter._logger.setLevel(builtin_logging.WARNING)
    LoggingMetricsReporter._logger.propagate = False


def print_device_mem_info(prompt: str = "VRAM usage") -> None:
    """
    Logs VRAM info.
    """
    if get_device_type() == "cpu":
        print_cpu_memory_info()
    else:
        memory_allocated = get_torch_device().memory_allocated() / (1024**3)
        max_memory_allocated = get_torch_device().max_memory_allocated() / (1024**3)
        _info_rank0(f"{prompt}: cur {memory_allocated:.2f}GB, max {max_memory_allocated:.2f}GB.")


def print_cpu_memory_info():
    cpu_usage = psutil.cpu_percent(interval=1)  # sampling for 1 sec
    _info_rank0(f"CPU Usage: {cpu_usage}%")

    memory_info = psutil.virtual_memory()
    _info_rank0(f"Total Memory: {memory_info.total / (1024**3):.2f} GB")
    _info_rank0(f"Available Memory: {memory_info.available / (1024**3):.2f} GB")
    _info_rank0(f"Used Memory: {memory_info.used / (1024**3):.2f} GB")
    _info_rank0(f"Memory Usage: {memory_info.percent}%")


def empty_cache() -> None:
    """
    Collects system memory.
    """
    gc.collect()

    if IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE:
        from ..utils.device import empty_cache

        empty_cache()


def get_cache_dir(path: Optional[str] = None) -> str:
    """
    Returns the cache directory for the given path.
    """
    if path is None:
        return CACHE_DIR

    path = os.path.normpath(path)
    if not os.path.splitext(path)[-1]:  # is a dir
        path = os.path.join(path, "")

    path = os.path.split(os.path.dirname(path))[-1]
    return os.path.join(CACHE_DIR, path, "")  # must endswith os.path.sep


@lru_cache
def get_dtype_size(dtype: "torch.dtype") -> int:
    """
    Taken from https://github.com/huggingface/safetensors/blob/v0.4.5/bindings/python/py_src/safetensors/torch.py#L350
    """
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)
    _SIZE = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
        _float8_e4m3fn: 1,
        _float8_e5m2: 1,
    }
    return _SIZE[dtype]


def unwrap_model(model: "nn.Module") -> "nn.Module":
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Taken from: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_utils.py#L4808
    """
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def print_example(example: Dict[str, "torch.Tensor"], rank: int, print_tensor: bool = True) -> None:
    """
    Logs a single example to screen.

    Nested dicts (e.g. ``multimodal_metadata`` from ``PackingCollator``)
    are expanded one level so inner tensor shapes/devices stay visible
    instead of being collapsed into a single dict-repr line.
    """

    def _log(key: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if print_tensor:
                logger.info(f"[rank {rank}]: {key}'s shape: {value.shape}, device: {value.device}, {value}")
            else:
                logger.info(f"[rank {rank}]: {key}'s shape: {value.shape}, device: {value.device}")
        else:
            logger.info(f"[rank {rank}]: {key}'s value: {value}")

    for key, value in example.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                _log(f"{key}[{inner_key!r}]", inner_value)
        else:
            _log(key, value)


def dict2device(input_dict: dict):
    """
    Move a dict of Tensor to GPUs.
    """
    output_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            output_dict[k] = v.to(get_device_type())
        elif isinstance(v, dict):
            output_dict[k] = dict2device(v)
        else:
            output_dict[k] = v
    return output_dict


def make_list(item):
    if isinstance(item, List) or isinstance(item, np.ndarray):
        return item
    return [item]


class ProfilerWithMem:
    """Thin wrapper that toggles CUDA-allocator tracing around profiler.step()"""

    def __init__(self, inner):
        self._p = inner

    # delegate ctx-manager behaviour
    def __enter__(self):
        return self._p.__enter__()

    def __exit__(self, *a):
        return self._p.__exit__(*a)

    def start(self):
        out = self._p.start()
        get_torch_device().memory._record_memory_history()
        return out

    def stop(self):
        out = self._p.stop()
        get_torch_device().memory._record_memory_history(enabled=None)  # step recording memory snapshot
        return out

    def step(self, *a, **kw):
        return self._p.step(*a, **kw)


def create_profiler(
    start_step: int,
    end_step: int,
    trace_dir: str,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    with_modules: bool,
    global_rank: int,
):
    """
    Creates a profiler to record the CPU and CUDA activities. Default export to trace.json.
    Profile steps in [start_step, end_step).

    When is_npu_available = True, the profiler will be created as torch_npu.profiler.

    Args:
        start_step (int): The step to start recording.
        end_step (int): The step to end recording.
        trace_dir (str): The path to save the profiling result.
        record_shapes (bool): Whether to record the shapes of the tensors.
        profile_memory (bool): Whether to profile the memory usage.
        with_stack (bool): Whether to include the stack trace.
    """
    import hdfs_io
    from hdfs_io import copy

    def handler_fn(p):
        time = int(datetime.datetime.now().timestamp())

        trace_file_extention = "pt.trace.json.gz"
        gpu_memory_file_extension = "pkl"

        if trace_dir.startswith("hdfs://"):
            hdfs_io.makedirs(trace_dir, exist_ok=True)
            os.makedirs(CACHE_DIR, exist_ok=True)
            trace_file = os.path.join(CACHE_DIR, f"veomni_rank{global_rank}_{time}.{trace_file_extention}")
            gpu_memory_file = os.path.join(CACHE_DIR, f"veomni_rank{global_rank}_{time}.{gpu_memory_file_extension}")
        else:
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = os.path.join(trace_dir, f"veomni_rank{global_rank}_{time}.{trace_file_extention}")
            gpu_memory_file = os.path.join(trace_dir, f"veomni_rank{global_rank}_{time}.{gpu_memory_file_extension}")

        if IS_NPU_AVAILABLE:
            nonlocal npu_trace_handler
            npu_trace_handler(p)
            trace_file = p.prof_if.prof_path
        elif IS_CUDA_AVAILABLE:
            p.export_chrome_trace(trace_file)
        logger.info(f"Profiling result saved at {trace_file}.")

        get_torch_device().memory._dump_snapshot(gpu_memory_file)
        logger.info(f"Profiling memory visualization saved at {gpu_memory_file}.")

        if trace_dir.startswith("hdfs://"):
            copy(trace_file, trace_dir)
            logger.info(f"Profiling result uploaded to {trace_dir}.")

    if IS_NPU_AVAILABLE:
        profiler_module = torch_npu.profiler
        activities = [profiler_module.ProfilerActivity.CPU, profiler_module.ProfilerActivity.NPU]
        npu_trace_handler = torch_npu.profiler.tensorboard_trace_handler(
            CACHE_DIR if trace_dir.startswith("hdfs://") else trace_dir
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            data_simplification=False,
        )
    else:
        profiler_module = torch.profiler
        activities = [profiler_module.ProfilerActivity.CPU, profiler_module.ProfilerActivity.CUDA]
        experimental_config = None

    warmup = 0 if start_step == 1 else 1
    wait = start_step - warmup - 1
    active = end_step - start_step
    logger.info(f"build profiler schedule - wait: {wait}, warmup: {warmup}, active: {active}.")

    schedule = profiler_module.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=1,
    )
    base_profiler = profiler_module.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=handler_fn,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_modules=with_modules,
        with_stack=with_stack,
        experimental_config=experimental_config,
    )
    if (IS_CUDA_AVAILABLE or IS_NPU_AVAILABLE) and profile_memory:
        return ProfilerWithMem(base_profiler)
    else:
        return base_profiler


if os.getenv("DISABLE_WARNINGS", "0").lower() in ["true", "1"]:
    disable_warning()
