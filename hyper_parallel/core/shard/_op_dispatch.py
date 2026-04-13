# Copyright 2025 Huawei Technologies Co., Ltd
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
"""_op_dispatch"""
import os
import sys
import atexit
import glob
import importlib
from typing import Any, List, Dict, Optional, Set
from itertools import chain

import yaml

from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.random import OffsetBasedRNGTracker, is_rng_supported_mesh
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType

platform = get_platform()
Tensor = platform.Tensor


def _apply_shard_offset_to_rng_args(args, offset_incr):
    """Apply per-shard offset increment to seed/offset tensors in MindSpore random op args.

    MindSpore random ops (e.g. ``randn_like_``) receive ``(seed, offset)`` as
    explicit int64 scalar tensors from ``default_generator._step()`` in the
    Python wrapper *before* the C++ dispatch triggers ``__fallback__``.  By the
    time ``_dispatch_random_op`` is called, the kernel will use whatever
    ``(seed, offset)`` values are in the args—it does **not** read the
    generator again. This function finds the offset tensor and adds the
    per-rank offset increment so each shard gets a unique random stream.

    The (seed, offset) pair is identified as the last two consecutive int64
    0-dim tensors in *args* (scanning from the end to skip trailing dtype /
    device arguments).

    Args:
        args: The list of local args for the random op.
        offset_incr (int): Per-shard offset increment.

    Returns:
        list: Modified args with the offset tensor adjusted.
    """
    int64_dtype = platform.tensor_dtype.int64
    last_int64_idx = -1
    for i in range(len(args) - 1, -1, -1):
        arg = args[i]
        if isinstance(arg, Tensor) and arg.dtype == int64_dtype and arg.ndim == 0:
            if last_int64_idx == i + 1:
                offset_idx = i + 1
                new_args = list(args)
                new_offset = int(new_args[offset_idx].item()) + offset_incr
                new_args[offset_idx] = platform.tensor([new_offset], dtype=int64_dtype).reshape(())
                return new_args
            last_int64_idx = i
    return args

_dtensor_dispatch = True
_no_skip_ops: Set[str] = set()


def get_no_skip_ops() -> Set[str]:
    """Return the set of op names that are exempt from SkipDTensorDispatch."""
    return _no_skip_ops


def add_no_skip_ops(op_names: Set[str]) -> None:
    """Add op names to the no-skip set so they are always dispatched through DTensor.

    Args:
        op_names: Set of canonical op name strings to register as no-skip.
    """
    global _no_skip_ops
    _no_skip_ops = _no_skip_ops | op_names


def remove_no_skip_ops(op_names: Set[str]) -> None:
    """Remove op names from the no-skip set.

    Args:
        op_names: Set of canonical op name strings to remove.
    """
    global _no_skip_ops
    _no_skip_ops = _no_skip_ops - op_names


def enable_dtensor_dispatch():
    """
    Enable DTensor dispatch for distributed tensor operations.

    When enabled, tensor operations will be dispatched through the
    distributed operator dispatcher for layout inference and redistribution.
    """
    global _dtensor_dispatch
    _dtensor_dispatch = True


def disable_dtensor_dispatch():
    """
    Disable DTensor dispatch for distributed tensor operations.

    When disabled, tensor operations will bypass the distributed operator
    dispatcher and use native implementations directly.
    """
    global _dtensor_dispatch
    _dtensor_dispatch = False


def get_dtensor_dispatch():
    """
    Get the current DTensor dispatch status.

    Returns:
        bool: True if DTensor dispatch is enabled, False otherwise.
    """
    return _dtensor_dispatch


class LayoutCacheKey:
    """Immutable layout cache key."""
    __slots__ = ('_tuple', '_hash')

    def __init__(self, layout_ids: List[str]):
        self._tuple = tuple(layout_ids)
        self._hash = hash(self._tuple)

    @classmethod
    def from_cache_values(cls, cache_values):
        key_values = []
        for v in cache_values:
            if hasattr(v, 'compact_str'):
                key_values.append(str(v.compact_str))
            else:
                key_values.append(str(v))
        return cls(key_values)

    def __eq__(self, other):
        if not isinstance(other, LayoutCacheKey):
            return False
        return self._tuple == other._tuple

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"LayoutCacheKey({self._tuple})"

class LayoutCacheManager:
    """
    Cache layout in infer layout.

    A singleton class that manages layout caches for distributed operations.
    It caches the inferred layouts and operation implementations to avoid
    redundant computation during repeated calls with the same input layouts.
    """
    _instance = None

    def __init__(self):
        self.layout_cache: Dict[str, Dict[LayoutCacheKey, Any]] = {}
        atexit.register(self.clear_cache)

    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of LayoutCacheManager.

        Returns:
            LayoutCacheManager: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = LayoutCacheManager()
        return cls._instance

    def get_layout_cache(self) -> Dict[str, Dict[LayoutCacheKey, Any]]:
        """
        Get the layout cache dictionary.

        Returns:
            Dict[str, Dict[LayoutCacheKey, Any]]: The nested dictionary mapping
                operation names to their layout caches.
        """
        return self.layout_cache

    def distributed_op(self, op_name: str) -> Any:
        """
        Get the distributed operation implementation by name.

        Args:
            op_name (str): The name of the distributed operation.

        Returns:
            Any: The distributed operation class or implementation.
        """
        op = get_distributed_op(op_name)
        return op

    def clear_cache(self):
        """
        Clear all cached layouts.

        This method is automatically registered with atexit to ensure
        cache is cleared when the program exits.
        """
        self.layout_cache.clear()


class OpDispatcher:
    """
    OpDispatcher
    """
    def __init__(self):
        self._env_yaml_dir: Optional[str] = os.environ.get("HYPER_PARALLEL_OPS_YAML_DIR")
        self._env_python_path: Optional[str] = os.environ.get("HYPER_PARALLEL_OPS_PYTHON_PATH")
        # The following attributes are initialized in _setup_yaml_dir()
        self.work_dir = ""  # Initialized in _setup_yaml_dir()
        self.yaml_dir = ""  # Initialized in _setup_yaml_dir()

        self._setup_paths_from_env()

        self.layout_infer_ops = self.safe_load_yaml_from_dir()
        self.whitelist = ["InplaceAddExt", "InplaceSubExt", "InplaceMul", "InplaceDiv", "typeof", "DistCommIsend",
                          "DistCommIrecv", "DistCommBroadcast", "DistCommAllReduce", "DistCommAllGather",
                          "DistCommReduceScatter", "requires_grad_", "item", "__get__", "__set__", "register_hook",
                          "is_complex", "chunk", "__bool__", "__len__", "__format__", "dim",
                          "_has_compatible_shallow_copy_type", "is_floating_point", "is_contiguous"]

        # Ops requiring args unpacking for layout inference (packed as prim, name, real_args).
        self.unpack_ops = ["ScatterUpdate", "Mod", "GatherNd"]

        self._random_ops = {
            "normal_", "uniform_", "bernoulli", "bernoulli_",
            "native_dropout", "rand", "rand_like", "randn",
            "randn_like", "randint_like", "kaiming_uniform_",
        }
        # Only mint random op support
        # MindSpore use the actual kernel name.
        self._random_ms_ops = {
            "BernoulliExt", "MultinomialExt", "RandpermExt",
            "NormalTensorTensor", "NormalTensorFloat", "NormalFloatTensor", "NormalFloatFloat",
            "Randn", "RandLikeExt", "RandnLike", "RandInt", "RandIntLike", "RandExt",
            "FuncDropoutExt"
        }
        self._rng_tracker: Optional[OffsetBasedRNGTracker] = None

        self._suffix_dispatch: Dict[str, str] = {
            "WithShape": "_with_layout_infer_with_shape",
            "Reshape": "_with_layout_infer_reshape",
            "WithTupleExpand": "_with_layout_infer_with_tuple_expand",
            "Slice": "_with_layout_infer_slice",
        }

        self._register_distributed_ops()

    def _setup_paths_from_env(self):
        """
        Setup YAML directory and Python path from environment variables.

        This method initializes the YAML directory and extends sys.path based on
        environment variables HYPER_PARALLEL_OPS_YAML_DIR and HYPER_PARALLEL_OPS_PYTHON_PATH.
        """
        self._setup_yaml_dir(self._env_yaml_dir)
        self._extend_sys_path(self._env_python_path)

    def _setup_yaml_dir(self, env_yaml_dir: Optional[str]):
        """
        Feature: Configure yaml_dir/work_dir for OpDispatcher
        Description: Resolve the YAML directory used to load distributed op definitions.
                     If env_yaml_dir is an absolute path, use it directly; otherwise treat it
                     as a path relative to the project work_dir. If env_yaml_dir is not set,
                     fall back to the default 'shard/ops/yaml' under work_dir.
        Expectation: self.yaml_dir and self.work_dir are set to valid values used later by
                     safe_load_yaml_from_dir(); no functional behavior is changed.
        """
        if env_yaml_dir:
            if os.path.isabs(env_yaml_dir):
                self.yaml_dir = env_yaml_dir
                self.work_dir = ""
            else:
                self.work_dir = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
                )
                self.yaml_dir = env_yaml_dir
        else:
            self.yaml_dir = "shard/ops/yaml"
            self.work_dir = os.path.normpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
            )

    def _extend_sys_path(self, env_python_path: Optional[str]):
        if not env_python_path:
            return
        python_paths = env_python_path.split(":")
        for path in python_paths:
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

    def _register_distributed_ops(self):
        for op_name, config in self.layout_infer_ops.items():
            self._register_single_distributed_op(op_name, config)

    def _register_single_distributed_op(self, op_name: str, config: dict):
        """
        Feature: Register a single distributed op implementation
        Description: Import the distributed op class specified by config and instantiate it
                     with op_name to trigger registration in the distributed op registry.
                     Prefer 'distributed_op_module' when provided; otherwise import from
                     built-in module prefix 'hyper_parallel.core.shard.ops.' plus
                     'distributed_op_file'. If import fails and an external python path is
                     provided via env, fall back to importing 'distributed_op_file' directly.
        Expectation: The distributed op class is imported and instantiated successfully,
                     or the original import error is raised; no functional behavior is changed.
        """
        class_name = config["distributed_op_class"]

        if "distributed_op_module" in config:
            module_name = config["distributed_op_module"]
            module = importlib.import_module(module_name)
            op_class = getattr(module, class_name)
            _ = op_class(op_name)
            return

        module_file = config["distributed_op_file"]
        try:
            module_name = "hyper_parallel.core.shard.ops." + module_file
            module = importlib.import_module(module_name)
            op_class = getattr(module, class_name)
            _ = op_class(op_name)
        except (ModuleNotFoundError, ImportError):
            if self._env_python_path:
                module = importlib.import_module(module_file)
                op_class = getattr(module, class_name)
                _ = op_class(op_name)
            else:
                raise

    @staticmethod
    def _process_args_and_kwargs(
        args, kwargs
    ) -> tuple[list, list, list, dict, list]:
        """_process_args_and_kwargs"""
        input_layouts = []
        extra_args = []
        input_args = []
        input_kwargs = kwargs.copy()
        cache_key_values = []

        for arg in args:
            if arg is None:
                input_layouts.append(None)
                input_args.append(arg)
                continue

            if not hasattr(arg, "_layout"):
                id_str = "scalar"
                if not isinstance(arg, Tensor):
                    id_str = str(arg)
                cache_key_values.append(id_str)
                extra_args.append(arg)
                input_layouts.append(None)
                input_args.append(arg)
            else:
                layout = arg.layout
                layout_id = layout.compact_str
                cache_key_values.append(str(layout_id))
                input_layouts.append(layout)
                if isinstance(arg, DTensor):
                    input_args.append(arg.to_local())
                else:
                    input_args.append(arg)

        for k, val in kwargs.items():
            if val is None:
                input_layouts.append(None)
                continue
            if not hasattr(val, "_layout"):
                id_str = "scalar"
                if not isinstance(val, Tensor):
                    id_str = str(val)
                cache_key_values.append(id_str)
                extra_args.append(val)
                input_layouts.append(None)
            else:
                layout = val.layout
                layout_id = layout.compact_str
                cache_key_values.append(str(layout_id))
                input_layouts.append(layout)
                if isinstance(val, DTensor):
                    input_kwargs[k] = val.to_local()

        return input_layouts, extra_args, input_args, input_kwargs, cache_key_values

    def _with_layout_infer(self, func: callable, *args, **kwargs) -> Tensor:
        """_with_layout_infer"""
        func_name = platform.get_op_name(func)
        packed_call = None
        if(func_name in self.unpack_ops and len(args) == 3 and
            isinstance(args[1], str) and isinstance(args[2],(tuple,list))):
            packed_call = (args[0], args[1])
            args = tuple(args[2])

        input_layouts, extra_args, input_args, input_kwargs, cache_key_values = \
            OpDispatcher._process_args_and_kwargs(args, kwargs)
        cache_key = LayoutCacheKey(cache_key_values)
        cache_manager = LayoutCacheManager.get_instance()
        layout_cache = cache_manager.get_layout_cache()
        if func_name not in layout_cache:
            layout_cache[func_name] = {}

        op_layout_cache = layout_cache[func_name]

        distribute_op = cache_manager.distributed_op(func_name)
        if cache_key in op_layout_cache:
            output_layout, op_impl = op_layout_cache[cache_key]
        else:
            all_args = (input_layouts, extra_args)
            output_layout = distribute_op.infer_layout(*all_args)
            op_impl = distribute_op.get_expand_impl(func, output_layout, input_layouts, extra_args)
            op_layout_cache[cache_key] = (output_layout, op_impl)

        if op_impl is None:
            op_impl = func

        if packed_call is not None:
            py_output = op_impl(packed_call[0], packed_call[1], tuple(input_args), **input_kwargs)
        else:
            py_output = op_impl(*input_args, **input_kwargs)

        if isinstance(py_output, (tuple, list)):
            output = ()
            if isinstance(output_layout, (tuple, list)):
                if len(py_output) == len(output_layout):
                    for i, output_item in enumerate(py_output):
                        output += (DTensor.from_local(
                            output_item, output_layout[i].mesh,
                            output_layout[i].alias_placements),)
                else:
                    raise RuntimeError(f"Output tuple size ({len(py_output)}) "
                                       f"does not match layout tuple size ({len(output_layout)})")
            else:
                raise RuntimeError("Output is a tuple but layout is not")
            return output

        return DTensor.from_local(
            py_output, output_layout.mesh, output_layout.alias_placements)

    def _extract_single_arg_layout(self, expanded_args, kwargs_value):
        """Helper to extract layout and cache info for a single argument."""
        cache_key_values = []
        input_layouts = []
        extra_args = []

        for arg in chain(expanded_args, kwargs_value):
            if arg is None:
                input_layouts.append(None)
                continue

            if not hasattr(arg, "_layout"):
                id_str = "scalar" if isinstance(arg, Tensor) else str(arg)
                cache_key_values.append(id_str)
                extra_args.append(arg)
                input_layouts.append(None)
            else:
                layout = arg.layout
                cache_key_values.append(str(layout.compact_str))
                input_layouts.append(layout)
        return cache_key_values, input_layouts, extra_args

    def _pack_infer_output(self, py_output, output_layout):
        """Helper to pack py_output into DTensors using output_layout."""
        if isinstance(py_output, (tuple, list)):
            if not isinstance(output_layout, (tuple, list)):
                raise RuntimeError("Output is a tuple but layout is not")
            if len(py_output) != len(output_layout):
                raise RuntimeError(f"Output tuple size ({len(py_output)}) "
                                   f"does not match layout tuple size ({len(output_layout)})")

            return tuple(
                DTensor.from_local(item, layout.mesh, layout.alias_placements)
                for item, layout in zip(py_output, output_layout)
            )

        return DTensor.from_local(py_output, output_layout.mesh, output_layout.alias_placements)

    def _with_layout_infer_with_tuple_expand(self, func: callable, *args, **kwargs) -> Tensor:
        """_with_layout_infer_with_tuple_expand"""
        expanded_args = []
        input_args = []
        for arg in args:
            if isinstance(arg, (tuple, list)):
                expanded_args.extend(arg)
                # pylint: disable=R1728
                input_args.append(tuple(item.to_local() if hasattr(item, "_layout") else item for item in arg))
            else:
                expanded_args.append(arg)
                input_args.append(arg.to_local() if isinstance(arg, DTensor) else arg)

        # Process kwargs into local tensors
        input_kwargs = {k: (v.to_local() if isinstance(v, DTensor) else v) for k, v in kwargs.items()}

        # Extract layouts for positional args
        cache_key_values, input_layouts, extra_args = self._extract_single_arg_layout(expanded_args, kwargs.values())

        cache_key = LayoutCacheKey(cache_key_values)

        cache_manager = LayoutCacheManager.get_instance()
        layout_cache = cache_manager.get_layout_cache()
        func_name = platform.get_op_name(func)
        if func_name not in layout_cache:
            layout_cache[func_name] = {}

        op_layout_cache = layout_cache[func_name]
        distribute_op = cache_manager.distributed_op(func_name)

        if cache_key in op_layout_cache:
            output_layout, op_impl = op_layout_cache[cache_key]
        else:
            all_args = (input_layouts, extra_args)
            output_layout = distribute_op.infer_layout(*all_args)
            op_impl = distribute_op.get_expand_impl(func, output_layout, input_layouts, extra_args)
            op_layout_cache[cache_key] = (output_layout, op_impl)

        if op_impl is None:
            op_impl = func

        py_output = op_impl(*input_args, **input_kwargs)

        return self._pack_infer_output(py_output, output_layout)

    @staticmethod
    def _with_layout_infer_reshape(func: callable, *args) -> Tensor:
        """_with_layout_infer_reshape"""
        input_tensor = args[0]
        shape = args[1]

        layout = input_tensor.layout
        input_layouts = [layout]

        extra_args = [shape, input_tensor.shape]

        cache_key_values = [str(layout.compact_str), str(shape), str(input_tensor.shape)]
        cache_key = LayoutCacheKey(cache_key_values)

        cache_manager = LayoutCacheManager.get_instance()
        layout_cache = cache_manager.get_layout_cache()
        func_name = platform.get_op_name(func)
        if func_name not in layout_cache:
            layout_cache[func_name] = {}

        op_layout_cache = layout_cache[func_name]

        distribute_op = cache_manager.distributed_op(func_name)
        if cache_key in op_layout_cache:
            infer_output, op_impl = op_layout_cache[cache_key]
        else:
            all_args = (input_layouts, extra_args)
            infer_output = distribute_op.infer_layout(*all_args)
            op_impl = distribute_op.get_expand_impl(func, infer_output, input_layouts, extra_args)
            op_layout_cache[cache_key] = (infer_output, op_impl)

        infer_output_tuple = infer_output
        local_shape = infer_output_tuple[1]

        if op_impl is None:
            op_impl = func

        py_output = op_impl(input_tensor.to_local(), local_shape)

        return DTensor.from_local(py_output, infer_output_tuple[0].mesh, infer_output_tuple[0].alias_placements)

    @staticmethod
    def _process_args_and_kwargs_with_shape(args, kwargs):
        """Process args and kwargs with input shapes for WithShape suffix operators.

        Args:
            args: Positional arguments from dispatch.
            kwargs: Keyword arguments from dispatch.

        Returns:
            tuple: (input_layouts, input_shapes, extra_args, input_args, input_kwargs, cache_key_values)
        """
        input_layouts = []
        extra_args = []
        input_shapes = []
        input_args = []
        input_kwargs = kwargs.copy()
        cache_key_values = []

        for arg in args:
            if arg is None:
                input_layouts.append(None)
                input_shapes.append(None)
                input_args.append(arg)
                continue

            if not hasattr(arg, "_layout"):
                id_str = "scalar"
                if not isinstance(arg, Tensor):
                    id_str = str(arg)
                cache_key_values.append(id_str)
                extra_args.append(arg)
                input_layouts.append(None)
                input_args.append(arg)
            else:
                layout = arg.layout
                layout_id = layout.compact_str
                cache_key_values.append(str(layout_id))
                input_layouts.append(layout)
                if isinstance(arg, DTensor):
                    input_args.append(arg.to_local())
                else:
                    input_args.append(arg)

            if not hasattr(arg, "shape"):
                input_shapes.append(None)
            else:
                input_shape = arg.shape
                input_shapes.append(input_shape)
                cache_key_values.append(str(input_shape))

        for k, val in kwargs.items():
            if val is None:
                input_layouts.append(None)
                continue
            if not hasattr(val, "_layout"):
                id_str = "scalar"
                if not isinstance(val, Tensor):
                    id_str = str(val)
                cache_key_values.append(id_str)
                extra_args.append(val)
                input_layouts.append(None)
            else:
                layout = val.layout
                layout_id = layout.compact_str
                cache_key_values.append(str(layout_id))
                input_layouts.append(layout)
                if isinstance(val, DTensor):
                    input_kwargs[k] = val.to_local()

            if not hasattr(val, "shape"):
                input_shapes.append(None)
            else:
                input_shape = val.shape
                input_shapes.append(input_shape)
                cache_key_values.append(str(input_shape))

        return input_layouts, input_shapes, extra_args, input_args, input_kwargs, cache_key_values

    def _with_layout_infer_with_shape(self, func: callable, *args, **kwargs) -> Tensor:
        """_with_layout_infer_with_shape"""
        func_name = platform.get_op_name(func)
        packed_call = None
        # Packed fallback args for some ops (e.g. Mod: (prim_obj, "Mod", (x, y))).
        if (func_name in self.unpack_ops and len(args) == 3 and
            isinstance(args[1], str) and isinstance(args[2], (tuple, list))):
            packed_call = (args[0], args[1])
            args = tuple(args[2])

        (input_layouts, input_shapes, extra_args, input_args,
        input_kwargs, cache_key_values) = OpDispatcher._process_args_and_kwargs_with_shape(args, kwargs)
        cache_key = LayoutCacheKey(cache_key_values)

        cache_manager = LayoutCacheManager.get_instance()
        layout_cache = cache_manager.get_layout_cache()
        if func_name not in layout_cache:
            layout_cache[func_name] = {}

        op_layout_cache = layout_cache[func_name]

        distribute_op = cache_manager.distributed_op(func_name)
        if cache_key in op_layout_cache:
            output_layout, op_impl = op_layout_cache[cache_key]
        else:
            extra_args.append(input_shapes)
            all_args = (input_layouts, extra_args)
            output_layout = distribute_op.infer_layout(*all_args)
            op_impl = distribute_op.get_expand_impl(func, output_layout, input_layouts, extra_args)
            op_layout_cache[cache_key] = (output_layout, op_impl)

        if op_impl is None:
            op_impl = func

        if packed_call is not None:
            py_output = op_impl(packed_call[0], packed_call[1], tuple(input_args), **input_kwargs)
        else:
            py_output = op_impl(*input_args, **input_kwargs)

        # set output layout
        if isinstance(py_output, (tuple, list)):
            output = ()
            if isinstance(output_layout, (tuple, list)):
                if len(py_output) == len(output_layout):
                    for i, output_item in enumerate(py_output):
                        output += (DTensor.from_local(
                            output_item, output_layout[i].mesh,
                            output_layout[i].alias_placements),)
                else:
                    raise RuntimeError(f"Output tuple size ({len(py_output)}) "
                                       f"does not match layout tuple size ({len(output_layout)})")
            else:
                raise RuntimeError("Output is a tuple but layout is not")
            return output

        return DTensor.from_local(
            py_output, output_layout.mesh, output_layout.alias_placements)

    def _with_layout_infer_slice(self, func: callable, *args) -> Tensor:
        """_with_layout_infer_slice"""
        input_tensor = args[0]
        begin = args[1]
        end = args[2]

        # input layout
        input_layouts = []

        layout = input_tensor.layout
        global_shape = input_tensor.shape
        input_layouts.append(layout)
        layout_id = layout.compact_str

        extra_args = []
        extra_args.append(begin)
        extra_args.append(end)
        extra_args.append(global_shape)
        cache_key_values = [str(layout_id), str(begin), str(end), str(global_shape)]
        cache_key = LayoutCacheKey(cache_key_values)

        cache_manager = LayoutCacheManager.get_instance()
        layout_cache = cache_manager.get_layout_cache()
        func_name = platform.get_op_name(func)
        if func_name not in layout_cache:
            layout_cache[func_name] = {}

        op_layout_cache = layout_cache[func_name]

        distribute_op = cache_manager.distributed_op(func_name)
        if cache_key in op_layout_cache:
            infer_output, op_impl = op_layout_cache[cache_key]
        else:
            all_args = (input_layouts, extra_args)
            infer_output = distribute_op.infer_layout(*all_args)
            op_impl = distribute_op.get_expand_impl(func, infer_output, input_layouts, extra_args)
            op_layout_cache[cache_key] = (infer_output, op_impl)

        infer_output_tuple = infer_output
        new_begin = infer_output_tuple[1]
        new_end = infer_output_tuple[2]

        if op_impl is None:
            op_impl = func

        py_output = op_impl(input_tensor.to_local(), new_begin, new_end)

        return DTensor.from_local(py_output, infer_output_tuple[0].mesh, infer_output_tuple[0].alias_placements)

    @staticmethod
    def _merge_default(config: dict):
        """Apply __default__ values to all ops in this YAML file."""
        if "__default__" not in config:
            return config

        default_cfg = config["__default__"]
        merged = {}

        for op_name, op_cfg in config.items():
            if op_name == "__default__":
                continue

            new_cfg = default_cfg.copy()
            new_cfg.update(op_cfg)
            merged[op_name] = new_cfg

        return merged

    def safe_load_yaml_from_dir(self):
        """
        Load yaml dictionary from directory.
        """
        yaml_dict = {}
        yaml_path = os.path.join(self.work_dir, self.yaml_dir) if self.work_dir else self.yaml_dir
        if not os.path.isdir(yaml_path):
            raise ValueError(f"Invalid yaml directory path: {yaml_path}")

        for yaml_file_path in glob.glob(os.path.join(yaml_path, '*.yaml')):
            with open(yaml_file_path, 'r', encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            yaml_data = OpDispatcher._merge_default(yaml_data)
            for name, data in yaml_data.items():
                if name in yaml_dict:
                    raise ValueError(f"Duplicate yaml object with name '{name}'.")
                yaml_dict[name] = data

        return yaml_dict

    def _dispatch_random_op(self, op_name: str, op_call: callable, args, kwargs):
        """Handle dispatch for random ops that operate on DTensors."""
        first_arg = next(
            (x for x in chain(args, kwargs.values()) if isinstance(x, DTensor)),
            None,
        )
        # Fall back to the default op if no DTensor is found.
        if first_arg is None:
            return op_call(*args, **kwargs)

        local_args = [arg.to_local() if isinstance(arg, DTensor) else arg for arg in args]
        local_kwargs = {k: v.to_local() if isinstance(v, DTensor) else v for k, v in kwargs.items()}
        first_local_arg = first_arg.to_local()

        if self._rng_tracker is None and is_rng_supported_mesh():
            self._rng_tracker = OffsetBasedRNGTracker()

        maybe_user_generator = local_kwargs.pop("generator", None)
        if (
            self._rng_tracker is not None
            and not first_local_arg.is_meta
            and self._rng_tracker.distribute_region_enabled
        ):
            # pylint: disable=W0212
            with self._rng_tracker._distribute_region(
                device_mesh=first_arg.device_mesh,
                placements=first_arg.placements,
                global_shape=first_arg.shape,
                generator=maybe_user_generator,
            ):
                # MindSpore random ops (e.g. mint.randn_like) extract (seed, offset)
                # from default_generator._step() in the Python wrapper *before* the
                # C++ dispatch triggers __fallback__. The callback reuses these
                # pre-fetched tensor args, so set_rng_state inside _distribute_region
                # has no effect on the kernel. Fix: apply the per-shard offset
                # increment directly to the offset tensor in the args.
                if platform.platform_type == PlatformType.MINDSPORE:
                    offset_incr = self._rng_tracker.compute_offset_incr(
                        first_arg.device_mesh, first_arg.placements, first_arg.shape,
                    )
                    local_args = _apply_shard_offset_to_rng_args(local_args, offset_incr)
                local_results = op_call(*local_args, **local_kwargs)
        else:
            local_results = op_call(*local_args, **local_kwargs)

        # in-place ops
        if op_name.endswith('_'):
            return first_arg
        # non-in-place ops
        # Some ops return tuple/list, e.g. native_dropout returns (output, mask).
        if isinstance(local_results, (tuple, list)):
            return tuple(
                DTensor.from_local(r, first_arg.device_mesh, first_arg.layout.alias_placements)
                if isinstance(r, Tensor) else r
                for r in local_results
            )
        if isinstance(local_results, Tensor):
            return DTensor.from_local(local_results, first_arg.device_mesh, first_arg.layout.alias_placements)
        # Fallback: return as-is for non-Tensor results (currently unreachable with existing _random_ops).
        return local_results

    @staticmethod
    def _unwrap_args(args: tuple) -> list:
        """Strip DTensor wrappers from args, preserving tuple/list container structure.

        Args:
            args: Op call positional arguments, may contain DTensor instances.

        Returns:
            List of args with DTensor replaced by their local tensors.
        """
        def unwrap(arg):
            if isinstance(arg, DTensor):
                return arg.to_local()
            if isinstance(arg, tuple):
                return tuple(e.to_local() if isinstance(e, DTensor) else e for e in arg)
            if isinstance(arg, list):
                return [e.to_local() if isinstance(e, DTensor) else e for e in arg]
            return arg
        return [unwrap(arg) for arg in args]

    def _should_bypass_dispatch(self, op_name: str) -> bool:
        """Return True if the op should bypass DTensor dispatch and run locally.

        Args:
            op_name: Canonical operator name from platform.get_op_name().

        Returns:
            True when the op is whitelisted or DTensor dispatch is globally disabled.
        """
        skip_dispatch = get_dtensor_dispatch() is False and op_name not in get_no_skip_ops()
        return op_name in self.whitelist or skip_dispatch

    def _dispatch_layout_infer(
        self, op_name: str, op_call: callable, args: tuple, kwargs: dict
    ):
        """Dispatch an op through the layout-inference path.

        Args:
            op_name: Canonical operator name.
            op_call: The raw operator callable.
            args: Positional arguments for op_call.
            kwargs: Keyword arguments for op_call.

        Returns:
            Result of the layout-infer dispatch.

        Raises:
            RuntimeError: If op_name is not registered or has an unknown suffix.
        """
        if op_name not in self.layout_infer_ops:
            raise RuntimeError(f"Operator {op_name} dose not contain parallel layout infer func.")

        cache_manager = LayoutCacheManager.get_instance()
        distribute_op = cache_manager.distributed_op(op_name)

        result = distribute_op.preprocess(args, kwargs)
        if result is not None:
            return self._dispatch_new(op_call, distribute_op, result)

        suffix = self.layout_infer_ops[op_name].get('infer_layout_suffix', '')
        if not suffix:
            return self._with_layout_infer(op_call, *args, **kwargs)

        handler_name = self._suffix_dispatch.get(suffix)
        if handler_name is None:
            raise RuntimeError(f"Operator {op_name} specified wrong suffix in parallel yaml.")
        return getattr(self, handler_name)(op_call, *args, **kwargs)

    def _wrap_output(self, py_output, output_layouts) -> Tensor:
        if isinstance(py_output, (tuple, list)):
            if len(py_output) != len(output_layouts):
                raise RuntimeError(
                    f"Output tuple size ({len(py_output)}) "
                    f"does not match layout tuple size ({len(output_layouts)})")
            return tuple(
                DTensor.from_local(item, layout.mesh, layout.alias_placements)
                for item, layout in zip(py_output, output_layouts))
        return DTensor.from_local(
            py_output, output_layouts[0].mesh, output_layouts[0].alias_placements)

    def _dispatch_new(self, func, distribute_op, result) -> Tensor:
        """New dispatch flow using preprocess result.

        Args:
            func: Original function.
            distribute_op: Distributed operation instance.
            result: Preprocessed result (local_args, local_kwargs, cache_values).

        Returns:
            Tensor: Dispatched result as DTensor.
        """
        local_args, local_kwargs, cache_values = result
        cache_key = LayoutCacheKey.from_cache_values(cache_values)
        func_name = platform.get_op_name(func)
        cache_manager = LayoutCacheManager.get_instance()
        layout_cache = cache_manager.get_layout_cache()
        if func_name not in layout_cache:
            layout_cache[func_name] = {}
        op_layout_cache = layout_cache[func_name]
        if cache_key in op_layout_cache:
            infer_result, op_impl = op_layout_cache[cache_key]
        else:
            infer_result = distribute_op.infer_layout(cache_values)
            op_impl = distribute_op.get_expand_impl(func, infer_result, cache_values)
            op_layout_cache[cache_key] = (infer_result, op_impl)
        output_layouts, extra_info = infer_result
        if op_impl is None:
            op_impl = func
        if extra_info is not None:
            py_output = op_impl(*local_args, *extra_info, **local_kwargs)
        else:
            py_output = op_impl(*local_args, **local_kwargs)
        return self._wrap_output(py_output, output_layouts)

    def dispatch(self, op_call: callable, args: tuple[object, ...], kwargs: dict[str, object]):
        """Route an op call through the appropriate DTensor dispatch path.

        Args:
            op_call: The raw operator callable.
            args: Positional arguments for op_call.
            kwargs: Keyword arguments for op_call.

        Returns:
            Result of the dispatched op call.
        """
        op_name = platform.get_op_name(op_call)

        if self._should_bypass_dispatch(op_name):
            return op_call(*self._unwrap_args(args), **kwargs)

        if op_name in self._random_ops or op_name in self._random_ms_ops:
            return self._dispatch_random_op(op_name, op_call, args, kwargs)

        return self._dispatch_layout_infer(op_name, op_call, args, kwargs)

_OP_DISPATCHER = OpDispatcher()
