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

"""Trainer-side profiler construction.

Split out of the former ``auto_models/components/utils/helper.py`` in stage 7
(05 §10.4); names, signatures and the trace-export behaviour are unchanged.
"""

import datetime
import logging
import os
from typing import Any, Optional

import torch

from hyper_parallel.models.build_options import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_torch_device,
)

if IS_NPU_AVAILABLE:
    import torch_npu


logger = logging.getLogger(__name__)

CACHE_DIR = os.path.expanduser(os.getenv("CACHE_DIR", os.path.join("~/.cache", "veomni")))


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


class ProfilerWithMem:
    """Thin wrapper that toggles CUDA-allocator tracing around profiler.step()"""

    def __init__(self, inner: Any) -> None:
        """Initialize the wrapper around an underlying profiler instance.

        Args:
            inner: The wrapped torch/torch_npu profiler.
        """
        self._p = inner

    # delegate ctx-manager behaviour
    def __enter__(self) -> Any:
        """Enter the wrapped profiler's context manager."""
        return self._p.__enter__()

    def __exit__(self, *a: Any) -> Any:
        """Exit the wrapped profiler's context manager."""
        return self._p.__exit__(*a)

    def start(self) -> Any:
        """Start profiling and begin recording the allocator history."""
        out = self._p.start()
        get_torch_device().memory._record_memory_history()
        return out

    def stop(self) -> Any:
        """Stop profiling and stop recording the allocator history."""
        out = self._p.stop()
        get_torch_device().memory._record_memory_history(enabled=None)  # step recording memory snapshot
        return out

    def step(self, *a: Any, **kw: Any) -> Any:
        """Advance the wrapped profiler by one step."""
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
) -> Any:
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
    copy = None
    if trace_dir.startswith("hdfs://"):
        # hdfs_io is an optional dependency needed only for remote trace output.
        import hdfs_io  # pylint: disable=import-outside-toplevel
        from hdfs_io import copy  # pylint: disable=import-outside-toplevel

    def handler_fn(p: Any) -> None:
        """Export the trace (and memory snapshot) when a profiling trace is ready.

        Args:
            p: The profiler instance that produced the trace.
        """
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

        if profile_memory:
            get_torch_device().memory._dump_snapshot(gpu_memory_file)
            logger.info(f"Profiling memory visualization saved at {gpu_memory_file}.")

        if trace_dir.startswith("hdfs://"):
            if copy is None:
                raise ValueError("hdfs_io.copy is required for an HDFS profiling trace directory")
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
    return base_profiler


__all__ = [
    "CACHE_DIR",
    "ProfilerWithMem",
    "create_profiler",
    "get_cache_dir",
]
