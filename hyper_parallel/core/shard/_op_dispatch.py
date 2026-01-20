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
from typing import Any, List, Dict, Optional
import yaml

from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.platform import get_platform

platform = get_platform()
Tensor = platform.Tensor

_dtensor_dispatch = True

def enable_dtensor_dispatch():
    global _dtensor_dispatch
    _dtensor_dispatch = True

def disable_dtensor_dispatch():
    global _dtensor_dispatch
    _dtensor_dispatch = False

def get_dtensor_dispatch():
    return _dtensor_dispatch


class LayoutCacheKey:
    """
    Layout cache key
    """
    def __init__(self, layout_ids: List[str]):
        self.layout_ids = layout_ids

    def __eq__(self, other):
        if not isinstance(other, LayoutCacheKey):
            return False
        return self.layout_ids == other.layout_ids

    def __hash__(self):
        seed = 0
        for id_str in self.layout_ids:
            h = hash(id_str)
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2)
        return seed

class LayoutCacheManager:
    """
    Cache layout in infer layout.
    """
    _instance = None

    def __init__(self):
        self.layout_cache: Dict[str, Dict[LayoutCacheKey, Any]] = {}
        atexit.register(self.clear_cache)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LayoutCacheManager()
        return cls._instance

    def get_layout_cache(self) -> Dict[str, Dict[LayoutCacheKey, Any]]:
        return self.layout_cache

    def distributed_op(self, op_name: str) -> Any:
        op = get_distributed_op(op_name)
        return op

    def clear_cache(self):
        self.layout_cache.clear()


class OpDispatcher:
    """
    OpDispatcher
    """
    def __init__(self):
        self._env_yaml_dir: Optional[str] = os.environ.get("HYPER_PARALLEL_OPS_YAML_DIR")
        self._env_python_path: Optional[str] = os.environ.get("HYPER_PARALLEL_OPS_PYTHON_PATH")

        self._setup_paths_from_env()

        self.layout_infer_ops = self.safe_load_yaml_from_dir()
        self.whitelist = ["InplaceAddExt", "InplaceSubExt", "InplaceMul", "InplaceDiv", "typeof", "DistCommIsend",
                          "DistCommIrecv", "DistCommBroadcast", "DistCommAllReduce", "DistCommAllGather",
                          "DistCommReduceScatter", "requires_grad_", "item", "__get__", "__set__", "register_hook",
                          "is_complex", "chunk", "__bool__", "__len__", "__format__"]

        # Ops requiring args unpacking for layout inference (packed as prim, name, real_args).
        self.unpack_ops = ["ScatterUpdate", "Mod"]

        self._register_distributed_ops()

    def _setup_paths_from_env(self):
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

    def _with_layout_infer(self, func: callable, *args, **kwargs) -> Tensor:
        """_with_layout_infer"""
        func_name = platform.get_op_name(func)
        packed_call = None
        # Ops in unpack_ops use packed fallback args (e.g. ScatterUpdate: (prim_obj, op_name: str, (input_x, indices, updates))).
        if(func_name in self.unpack_ops and len(args) == 3 and
            isinstance(args[1], str) and isinstance(args[2],(tuple,list))):
            packed_call = (args[0], args[1])
            args = tuple(args[2])

        cache_key = LayoutCacheKey([])
        input_layouts = []
        extra_args = []
        input_args = []

        # Normal ops pass real inputs directly (e.g. SumExt: args = (dtensor, axis: list, keep_dims: bool, dtype: None)).
        for arg in args:
            if arg is None:
                input_layouts.append(None)
                input_args.append(arg)
                continue

            if not hasattr(arg, "_layout"):
                id_str = "scalar"
                if not isinstance(arg, Tensor):
                    id_str = str(arg)
                cache_key.layout_ids.append(id_str)
                extra_args.append(arg)
                input_layouts.append(None)
                input_args.append(arg)
            else:
                layout = arg.layout
                layout_id = layout.compact_str
                cache_key.layout_ids.append(str(layout_id))
                input_layouts.append(layout)
                if isinstance(arg, DTensor):
                    input_args.append(arg.to_local())
                else:
                    input_args.append(arg)

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
            py_output = op_impl(packed_call[0], packed_call[1], tuple(input_args), **kwargs)
        else:
            py_output = op_impl(*input_args, **kwargs)

        if isinstance(py_output, (tuple, list)):
            output = ()
            if isinstance(output_layout, (tuple, list)):
                if len(py_output) == len(output_layout):
                    for i, output_item in enumerate(py_output):
                        output += (DTensor.from_local(output_item, output_layout[i]),)
                else:
                    raise RuntimeError(f"Output tuple size ({len(py_output)}) "
                                       f"does not match layout tuple size ({len(output_layout)})")
            else:
                raise RuntimeError("Output is a tuple but layout is not")
            return output

        return DTensor.from_local(py_output, output_layout)

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

        cache_key = LayoutCacheKey([])
        input_layouts = []
        extra_args = []

        for arg in expanded_args:
            if arg is None:
                input_layouts.append(None)
                continue

            if not hasattr(arg, "_layout"):
                id_str = "scalar"
                if not isinstance(arg, Tensor):
                    id_str = str(arg)
                cache_key.layout_ids.append(id_str)
                extra_args.append(arg)
                input_layouts.append(None)
            else:
                layout = arg.layout
                layout_id = layout.compact_str
                cache_key.layout_ids.append(str(layout_id))
                input_layouts.append(layout)

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

        py_output = op_impl(*input_args, **kwargs)

        if isinstance(py_output, (tuple, list)):
            output = ()
            if isinstance(output_layout, (tuple, list)):
                if len(py_output) == len(output_layout):
                    for i, output_item in enumerate(py_output):
                        output += (DTensor.from_local(output_item, output_layout[i]),)
                else:
                    raise RuntimeError(f"Output tuple size ({len(py_output)}) "
                                       f"does not match layout tuple size ({len(output_layout)})")
            else:
                raise RuntimeError("Output is a tuple but layout is not")
            return output

        return DTensor.from_local(py_output, output_layout)

    def _with_layout_infer_reshape(self, func: callable, *args) -> Tensor:
        """_with_layout_infer_reshape"""
        input_tensor = args[0]
        shape = args[1]
        cache_key = LayoutCacheKey([])
        input_layouts = []

        layout = input_tensor.layout
        input_layouts.append(layout)
        layout_id = layout.compact_str
        cache_key.layout_ids.append(str(layout_id))

        extra_args = []
        extra_args.append(shape)
        cache_key.layout_ids.append(str(shape))

        input_shape = input_tensor.shape
        extra_args.append(input_shape)
        cache_key.layout_ids.append(str(input_shape))

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

        return DTensor.from_local(py_output, infer_output_tuple[0])

    def _with_layout_infer_with_shape(self, func: callable, *args, **kwargs) -> Tensor:
        """_with_layout_infer_with_shape"""
        func_name = platform.get_op_name(func)
        packed_call = None
        # Packed fallback args for some ops (e.g. Mod: (prim_obj, "Mod", (x, y))).
        if (func_name in self.unpack_ops and len(args) == 3 and
            isinstance(args[1], str) and isinstance(args[2], (tuple, list))):
            packed_call = (args[0], args[1])
            args = tuple(args[2])

        cache_key = LayoutCacheKey([])
        input_layouts = []
        extra_args = []
        input_shapes = []
        input_args = []

        # Normal ops pass real inputs directly (e.g. GreaterEqual: (dtensor_x, dtensor_y)).
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
                cache_key.layout_ids.append(id_str)
                extra_args.append(arg)
                input_layouts.append(None)
                input_args.append(arg)
            else:
                layout = arg.layout
                layout_id = layout.compact_str
                cache_key.layout_ids.append(str(layout_id))
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
                cache_key.layout_ids.append(str(input_shape))

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
            py_output = op_impl(packed_call[0], packed_call[1], tuple(input_args), **kwargs)
        else:
            py_output = op_impl(*input_args, **kwargs)

        # 设置输出布局
        if isinstance(py_output, (tuple, list)):
            output = ()
            if isinstance(output_layout, (tuple, list)):
                if len(py_output) == len(output_layout):
                    for i, output_item in enumerate(py_output):
                        output += (DTensor.from_local(output_item, output_layout[i]),)
                else:
                    raise RuntimeError(f"Output tuple size ({len(py_output)}) "
                                       f"does not match layout tuple size ({len(output_layout)})")
            else:
                raise RuntimeError("Output is a tuple but layout is not")
            return output

        return DTensor.from_local(py_output, output_layout)

    def _with_layout_infer_slice(self, func: callable, *args) -> Tensor:
        """_with_layout_infer_slice"""
        input_tensor = args[0]
        begin = args[1]
        end = args[2]

        # 输入布局
        cache_key = LayoutCacheKey([])
        input_layouts = []

        layout = input_tensor.layout
        global_shape = input_tensor.shape
        input_layouts.append(layout)
        layout_id = layout.compact_str
        cache_key.layout_ids.append(str(layout_id))

        extra_args = []
        extra_args.append(begin)
        extra_args.append(end)
        extra_args.append(global_shape)
        cache_key.layout_ids.append(str(begin))
        cache_key.layout_ids.append(str(end))
        cache_key.layout_ids.append(str(global_shape))

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

        return DTensor.from_local(py_output, infer_output_tuple[0])

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
            for name, data in yaml_data.items():
                if name in yaml_dict:
                    raise ValueError(f"Duplicate yaml object with name '{name}'.")
                yaml_dict[name] = data

        return yaml_dict

    def dispatch(self, op_call: callable, args: tuple[object, ...], kwargs: dict[str, object]):
        """
        dispatch
        :param op_call:
        :param args:
        :param kwargs:
        :return:
        """
        op_name = platform.get_op_name(op_call)
        if op_name in self.whitelist or get_dtensor_dispatch() is False:
            input_args = [arg.to_local() if isinstance(arg, DTensor) else arg for arg in args]
            return op_call(*input_args, **kwargs)
        if op_name not in self.layout_infer_ops:
            raise RuntimeError(f"Operator {op_name} dose not contain parallel layout infer func.")

        layout_infer_info = self.layout_infer_ops[op_name]
        suffix = layout_infer_info.get('infer_layout_suffix', '')
        if not suffix:
            return self._with_layout_infer(op_call, *args, **kwargs)

        if suffix == "WithShape":
            return self._with_layout_infer_with_shape(op_call, *args, **kwargs)
        if suffix == "Reshape":
            return self._with_layout_infer_reshape(op_call, *args)
        if suffix == "WithTupleExpand":
            return self._with_layout_infer_with_tuple_expand(op_call, *args, **kwargs)
        if suffix == "Slice":
            return self._with_layout_infer_slice(op_call, *args)
        raise RuntimeError(f"Operator {op_name} specified wrong suffix in parallel yaml.")

_OP_DISPATCHER = OpDispatcher()
