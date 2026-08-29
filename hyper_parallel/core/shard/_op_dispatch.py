# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
import atexit
import copy
import glob
import importlib
import logging
import os
import sys
import warnings
from contextvars import ContextVar
from itertools import chain
from typing import Any, Dict, FrozenSet, List, Optional

import yaml

from hyper_parallel.core.shard.ops.parallel_ops_register import get_distributed_op
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import RaggedShardInfo
from hyper_parallel.core.dtensor.random import OffsetBasedRNGTracker, is_rng_supported_mesh
from hyper_parallel.core.dtensor.debug._dispatch_logger import log_dispatch_enter, log_dispatch_exit
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType

from hyper_parallel.core.tensor_parallel._ce_op_registry import is_loss_parallel_op, is_decomposed_ce_op
from hyper_parallel.core.tensor_parallel.loss_parallel import is_loss_parallel_active
from hyper_parallel.core.tensor_parallel.loss_parallel_ops_common import _is_shard_on_last_dim

platform = get_platform()
Tensor = platform.Tensor

logger = logging.getLogger(__name__)


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

_dtensor_dispatch_disabled: ContextVar[bool] = ContextVar('_dtensor_dispatch_disabled', default=False)
_no_skip_ops: ContextVar[FrozenSet[str]] = ContextVar('_no_skip_ops', default=frozenset())
_debug_mode_observer: ContextVar = ContextVar('_debug_mode_observer', default=None)

_RAGGED_ELEMENTWISE_OPS = {
    "abs": "unary", "absolute": "unary", "clone": "unary", "cos": "unary",
    "conj": "unary", "empty_like": "unary", "exp": "unary", "gelu": "unary",
    "isinf": "unary", "isnan": "unary",
    "log": "unary", "neg": "unary", "negative": "unary", "relu": "unary",
    "rsqrt": "unary", "sigmoid": "unary", "silu": "unary", "sin": "unary",
    "sqrt": "unary", "square": "unary", "zeros_like": "unary",
    "add": "binary", "add_": "binary", "addcdiv_": "binary",
    "addcmul_": "binary", "div": "binary", "lerp_": "binary",
    "mul": "binary", "mul_": "binary", "pow": "binary", "real_div": "binary",
    "sub": "binary", "__rsub__": "binary", "__rpow__": "binary",
    "true_divide": "binary",
}

_RAGGED_INPLACE_ELEMENTWISE_OPS = frozenset({
    "add_",
    "addcdiv_",
    "addcmul_",
    "lerp_",
    "mul_",
})

# Tensor subclass bookkeeping must stay available when a Ragged DTensor is
# wrapped as a Parameter. These operations only inspect or update the local
# autograd wrapper and do not reinterpret the logical distributed shape.
_RAGGED_METADATA_BYPASS_OPS = frozenset({
    "requires_grad_",
    "__get__",
    "__set__",
    "register_hook",
    "_has_compatible_shallow_copy_type",
    "is_complex",
    "is_floating_point",
    "is_contiguous",
})


def get_no_skip_ops() -> FrozenSet[str]:
    """Return the set of op names that are exempt from SkipDTensorDispatch."""
    return _no_skip_ops.get()


def get_dtensor_dispatch() -> bool:
    """
    Get the current DTensor dispatch status.

    Returns:
        bool: True if DTensor dispatch is enabled, False otherwise.
    """
    return not _dtensor_dispatch_disabled.get()


class LayoutCacheKey:
    """Immutable layout cache key."""
    __slots__ = ('_tuple', '_hash')

    def __init__(self, layout_ids: List[str]):
        self._tuple = tuple(layout_ids)
        self._hash = hash(self._tuple)

    @classmethod
    def from_cache_values(cls, cache_values: list) -> "LayoutCacheKey":
        """Build a LayoutCacheKey from a cache_values list.

        Args:
            cache_values (list): Mixed list of Layout objects (with compact_str) and raw scalars.

        Returns:
            LayoutCacheKey: Immutable key derived from the string representation of each value.
        """
        # Read the cached ``_compact_str`` attribute directly instead of going through
        # the ``compact_str`` property getter (one fewer Python frame per Layout), and
        # build the key in one comprehension. The resulting tuple is byte-for-byte the
        # legacy string key, so eq/hash semantics are unchanged.
        # NOTE: the key stays a string tuple by design (cross-checked against manually
        # built legacy keys in the UTs); CPython caches each compact_str's hash on the
        # string object, so this is not the bottleneck a full integer key would target.
        return cls([cs if (cs := getattr(v, '_compact_str', None)) is not None else str(v)
                    for v in cache_values])

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
    def get_instance(cls) -> "LayoutCacheManager":
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

    @staticmethod
    def distributed_op(op_name: str) -> Any:
        """
        Get the distributed operation implementation by name.

        Args:
            op_name (str): The name of the distributed operation.

        Returns:
            Any: The distributed operation class or implementation.
        """
        op = get_distributed_op(op_name)
        return op

    def clear_cache(self) -> None:
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

    # Whitelisted ops that mutate args[0]'s storage in place. The dispatch bypass
    # must return the original DTensor self for these, not the unwrapped local
    # result, or it demotes a DTensor accumulator to a plain Tensor and breaks the
    # next op that adds a DTensor to it (e.g. grad-accumulation `loss += micro_loss`).
    # Class-level so it stays available on instances built via __new__ (e.g. tests).
    _INPLACE_BYPASS_OPS = frozenset(
        {"InplaceAddExt", "InplaceSubExt", "InplaceMul", "InplaceDiv"})

    # MindSpore random kernels that always mutate an existing tensor in place.
    # Out-of-place random kernels belong in _random_ms_ops only, not here.
    _RANDOM_INPLACE_MS_OPS = frozenset({
        "InplaceBernoulliScalar",
        "InplaceBernoulliTensor",
        "InplaceNormal",
        "InplaceRandom",
        "InplaceUniform",
    })

    def __init__(self):
        self._env_yaml_dir: Optional[str] = os.environ.get("HYPER_PARALLEL_OPS_YAML_DIR")
        self._env_python_path: Optional[str] = os.environ.get("HYPER_PARALLEL_OPS_PYTHON_PATH")
        # The following attributes are initialized in _setup_yaml_dir()
        self.work_dir = ""  # Initialized in _setup_yaml_dir()
        self.yaml_dir = ""  # Initialized in _setup_yaml_dir()

        self._setup_paths_from_env()

        self.layout_infer_ops = self.safe_load_yaml_from_dir()
        # frozenset for O(1) membership (checked on every dispatch's bypass test).
        self.whitelist = frozenset({"typeof", "DistCommIsend",
                                    "DistCommIrecv", "DistCommBroadcast", "DistCommAllReduce", "DistCommAllGather",
                                    "DistCommBatchIsendIrecv",
                                    "DistCommReduceScatter", "requires_grad_", "item", "__get__", "__set__",
                                    "register_hook",
                                    "is_complex", "chunk", "__bool__", "__len__", "__format__", "dim",
                                    "_has_compatible_shallow_copy_type", "is_floating_point", "is_contiguous",
                                    "get_device"})

        # Ops requiring args unpacking for layout inference (packed as prim, name, real_args).
        # frozenset so the aclop-normalization gate in _dispatch_layout_infer is O(1).
        self.unpack_ops = frozenset({"ScatterUpdate", "Mod", "GatherNd", "StopGradient"})

        self._random_ops = {
            "normal_", "uniform_", "bernoulli", "bernoulli_",
            "native_dropout", "rand", "rand_like", "randn",
            "randn_like", "randint_like", "kaiming_uniform_",
            "multinomial",
        }
        # Only mint random op support
        # MindSpore use the actual kernel name.
        self._random_ms_ops = {
            "BernoulliExt", "MultinomialExt",
            "InplaceBernoulliScalar", "InplaceBernoulliTensor",
            "InplaceNormal", "InplaceRandom", "InplaceUniform",
            "NormalFloatFloat", "NormalFloatTensor", "NormalTensorFloat", "NormalTensorTensor",
            "RandpermExt", "Randn", "RandLikeExt", "RandnLike", "RandInt", "RandIntLike", "RandExt",
            "FuncDropoutExt", "UniformExt",
        }
        self._rng_tracker: Optional[OffsetBasedRNGTracker] = None
        # Op names proven to be loss/CE-irrelevant (both is_loss_parallel_op and
        # is_decomposed_ce_op are False). For these the loss_parallel / decomposed-CE
        # guards in dispatch() are always no-ops regardless of context, so we skip
        # them (and their is_loss_parallel_active() contextvar reads) on later calls.
        self._non_loss_ops: set = set()

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

    @staticmethod
    def _extend_sys_path(env_python_path: Optional[str]):
        if not env_python_path:
            return
        python_paths = env_python_path.split(":")
        for path in python_paths:
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.append(path)

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

    def safe_load_yaml_from_dir(self) -> dict:
        """
        Load yaml dictionary from directory.

        Returns:
            dict: Merged dictionary of all operator configurations loaded from YAML files.
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

        if self._rng_tracker is None and is_rng_supported_mesh(first_arg.device_mesh):
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
            if maybe_user_generator is not None:
                local_kwargs["generator"] = maybe_user_generator
            local_results = op_call(*local_args, **local_kwargs)

        return self._wrap_random_result(op_name, local_results, first_arg, args, kwargs)

    @staticmethod
    def _func_dropout_ext_inplace(args, kwargs) -> bool:
        """Return True when FuncDropoutExt is invoked with inplace=True."""
        # Kernel signature: (input, p, training, inplace, seed, offset).
        if len(args) >= 4:
            return bool(args[3])
        return bool(kwargs.get("inplace", False))

    @staticmethod
    def _random_op_returns_self(op_name: str, args, kwargs) -> bool:
        """Return True when a random op mutates an existing DTensor in place."""
        if op_name in OpDispatcher._RANDOM_INPLACE_MS_OPS:
            return True
        if op_name == "FuncDropoutExt":
            return OpDispatcher._func_dropout_ext_inplace(args, kwargs)
        # Torch random inplace ops follow the ATen '_' suffix convention.
        return op_name.endswith('_')

    @staticmethod
    def _wrap_random_result(op_name, local_results, first_arg, args, kwargs):
        """Wrap a random op's local result(s) back into DTensor(s).

        In-place ops return the input DTensor itself. Torch random inplace ops use
        the ATen '_' suffix; MindSpore inplace random kernels are listed in
        ``_RANDOM_INPLACE_MS_OPS``. ``FuncDropoutExt`` is handled separately
        because the same kernel serves both modes via its ``inplace`` argument.
        """
        if OpDispatcher._random_op_returns_self(op_name, args, kwargs):
            return first_arg
        mesh = first_arg.device_mesh
        placements = first_arg.layout.alias_placements
        # Some ops return tuple/list, e.g. native_dropout returns (output, mask).
        if isinstance(local_results, (tuple, list)):
            return tuple(
                DTensor.from_local(r, mesh, placements) if isinstance(r, Tensor) else r
                for r in local_results
            )
        if isinstance(local_results, Tensor):
            return DTensor.from_local(local_results, mesh, placements)
        # Fallback: return as-is for non-Tensor results (currently unreachable with existing _random_ops).
        return local_results

    @staticmethod
    def _unwrap_value(value: object) -> object:
        """Replace DTensor with its local tensor; pass scalars and plain tensors through.

        Args:
            value (object): A single argument value from an op call.

        Returns:
            object: The local tensor if value is a DTensor, otherwise value unchanged.
        """
        if isinstance(value, DTensor):
            return value.to_local()
        if isinstance(value, tuple):
            return tuple(OpDispatcher._unwrap_value(e) for e in value)
        if isinstance(value, list):
            return [OpDispatcher._unwrap_value(e) for e in value]
        return value

    @staticmethod
    def _unwrap_args(args: tuple) -> list:
        """Strip DTensor wrappers from args, preserving tuple/list container structure.

        Args:
            args: Op call positional arguments, may contain DTensor instances.

        Returns:
            List of args with DTensor replaced by their local tensors.
        """
        return [OpDispatcher._unwrap_value(arg) for arg in args]

    @staticmethod
    def _unwrap_kwargs(kwargs: dict) -> dict:
        """Strip DTensor wrappers from kwargs values, preserving tuple/list container structure.

        Args:
            kwargs: Op call keyword arguments, values may contain DTensor instances.

        Returns:
            Dict of kwargs with DTensor values replaced by their local tensors.
        """
        return {k: OpDispatcher._unwrap_value(v) for k, v in kwargs.items()}

    @staticmethod
    def _collect_dtensors(value: object) -> List[DTensor]:
        """Return all DTensors nested in one dispatch argument."""
        if isinstance(value, DTensor):
            return [value]
        if isinstance(value, (tuple, list)):
            return list(chain.from_iterable(
                OpDispatcher._collect_dtensors(item) for item in value
            ))
        if isinstance(value, dict):
            return list(chain.from_iterable(
                OpDispatcher._collect_dtensors(item) for item in value.values()
            ))
        return []

    def _validate_ragged_dispatch(
        self, op_name: str, args: tuple, kwargs: dict
    ) -> Optional[DTensor]:
        """Return the first Ragged input for a whitelisted elementwise op."""
        reference = next(
            (
                dtensor for dtensor in self._collect_dtensors((args, kwargs))
                if isinstance(getattr(dtensor.layout, "ragged_shard", None), RaggedShardInfo)
            ),
            None,
        )
        if reference is None:
            return None
        if op_name not in _RAGGED_ELEMENTWISE_OPS:
            raise RuntimeError(
                f"Operator {op_name!r} does not support RaggedShard in phase one"
            )
        return reference

    def _dispatch_ragged_elementwise(
        self, op_call: callable, args: tuple, kwargs: dict,
        reference: DTensor,
    ) -> DTensor:
        """Execute a whitelisted op locally and inherit its Ragged Layout."""
        local_args = tuple(self._unwrap_args(args))
        local_kwargs = self._unwrap_kwargs(kwargs)
        py_output = op_call(*local_args, **local_kwargs)
        op_name = platform.get_op_name(op_call)
        if op_name in _RAGGED_INPLACE_ELEMENTWISE_OPS:
            if not args or not isinstance(args[0], DTensor):
                raise ValueError(
                    f"Ragged in-place operator {op_name!r} requires a DTensor first argument"
                )
            return args[0]
        return DTensor.from_local_with_layout(
            py_output,
            copy.deepcopy(reference.layout),
            shape=tuple(reference.shape),
        )

    @staticmethod
    def _gather_dtensors_to_full(args: tuple, kwargs: dict) -> tuple:
        """Gather all DTensor arguments to full tensors for fallback execution.

        Used when an operator has no parallel layout implementation. All DTensor
        arguments are gathered to full tensors before calling the standard operator.

        Args:
            args: Op call positional arguments, may contain DTensor instances.
            kwargs: Op call keyword arguments, may contain DTensor instances.

        Returns:
            Tuple of (unwrapped_args, unwrapped_kwargs) with DTensor values
            replaced by their full tensor representations.

        Warning:
            This fallback performs all-gather which may consume significant memory.
            Operators without layout implementations should be registered properly.
        """
        def gather(value: object) -> object:
            if isinstance(value, DTensor):
                return value.full_tensor()
            if isinstance(value, tuple):
                return tuple(gather(e) for e in value)
            if isinstance(value, list):
                return [gather(e) for e in value]
            return value

        gathered_args = [gather(arg) for arg in args]
        gathered_kwargs = {k: gather(v) for k, v in kwargs.items()}

        warnings.warn(
            "Operator has no distributed layout implementation. "
            "Falling back to all-gather which may consume significant memory. "
            "Consider registering a proper distributed operator.",
            UserWarning,
            stacklevel=4
        )

        return gathered_args, gathered_kwargs

    def _should_bypass_dispatch(self, op_name: str) -> bool:
        """Return True if the op should bypass DTensor dispatch and run locally.

        Args:
            op_name: Canonical operator name from platform.get_op_name().

        Returns:
            True when the op is whitelisted or DTensor dispatch is globally disabled.
        """
        # Cheap O(1) frozenset checks first, short-circuit before the ContextVar
        # read (get_dtensor_dispatch) which is the priciest part of this guard.
        if op_name in self.whitelist or op_name in self._INPLACE_BYPASS_OPS:
            return True
        return get_dtensor_dispatch() is False and op_name not in get_no_skip_ops()

    @staticmethod
    def _validate_inplace_partial_inputs(op_name: str, args: tuple, kwargs: dict) -> None:
        """Reject local in-place add/sub when Partial contributions need gating."""
        if op_name not in {"InplaceAddExt", "InplaceSubExt"} or not args:
            return
        first = args[0]
        if len(args) >= 2:
            second = args[1]
        elif "other" in kwargs:
            second = kwargs["other"]
        else:
            return
        if not isinstance(first, DTensor):
            return
        mesh_ndim = len(first.layout.partial)
        first_partial = tuple(first.layout.partial)
        if isinstance(second, DTensor):
            second_partial = tuple(second.layout.partial)
            if len(second_partial) != mesh_ndim:
                raise ValueError(
                    f"For {op_name}, in-place input mesh dimensions must match, "
                    f"but got {mesh_ndim} and {len(second_partial)}."
                )
        else:
            second_partial = (None,) * mesh_ndim
        if first_partial != second_partial:
            raise ValueError(
                f"For {op_name}, input Partial placements must be identical for "
                f"local in-place execution, but got {first_partial} and {second_partial}."
            )

    def _should_dispatch_loss_parallel(self, op_name: str) -> bool:
        """Check if should dispatch through loss_parallel path.

        Args:
            op_name: Canonical operator name from platform.get_op_name().

        Returns:
            True when in loss_parallel context and op is a CE entry point.
        """
        return is_loss_parallel_active() and is_loss_parallel_op(op_name)

    def _check_decomposed_ce_op_in_loss_parallel(self, op_name: str, args: tuple, kwargs: dict):
        """Check if decomposed CE ops are called in loss_parallel context.

        Args:
            op_name: Canonical operator name.
            args: Positional arguments for op_call.
            kwargs: Keyword arguments for op_call.

        Raises:
            ValueError: If decomposed CE op is called in loss_parallel context
                       with vocab-sharded DTensor input.
        """
        if not is_loss_parallel_active() or not is_decomposed_ce_op(op_name):
            return

        has_vocab_sharded_dtensor = False
        for arg in args:
            if isinstance(arg, DTensor) and _is_shard_on_last_dim(arg):
                has_vocab_sharded_dtensor = True
                break
        if not has_vocab_sharded_dtensor:
            for val in kwargs.values():
                if isinstance(val, DTensor) and _is_shard_on_last_dim(val):
                    has_vocab_sharded_dtensor = True
                    break

        if has_vocab_sharded_dtensor:
            raise ValueError(
                f"Operator '{op_name}' is a decomposed component of cross_entropy and should not be called "
                f"directly within loss_parallel() context. Use F.cross_entropy(logits, targets) instead. "
                f"For example, replace:\n"
                f"  with loss_parallel():\n"
                f"      log_probs = F.log_softmax(logits, dim=-1)\n"
                f"      loss = F.nll_loss(log_probs, targets)\n"
                f"with:\n"
                f"  with loss_parallel():\n"
                f"      loss = F.cross_entropy(logits, targets)"
            )

    def _dispatch_loss_parallel(self, op_call: callable, args: tuple, kwargs: dict):
        """Dispatch cross_entropy through the loss_parallel distributed kernel.

        Args:
            op_call: The raw operator callable.
            args: Positional arguments for op_call.
            kwargs: Keyword arguments for op_call.

        Returns:
            Result of the distributed cross_entropy computation.
        """
        if platform.platform_type == PlatformType.PYTORCH:
            # pylint: disable=C0415
            from hyper_parallel.platform.torch.loss_parallel_ops import distributed_cross_entropy_from_op_call
        elif platform.platform_type == PlatformType.MINDSPORE:
            # pylint: disable=C0415
            from hyper_parallel.platform.mindspore.loss_parallel_ops import distributed_cross_entropy_from_op_call
        else:
            raise RuntimeError(f"Unsupported platform for loss_parallel: {platform.platform_type}")
        return distributed_cross_entropy_from_op_call(op_call, args, kwargs)

    def _check_ce_op_without_loss_parallel_context(self, op_name: str, args: tuple):
        """Check if CE op is called with Shard(-1) DTensor outside loss_parallel context.

        Args:
            op_name: Canonical operator name.
            args: Positional arguments for op_call.

        Raises:
            ValueError: If CE op is called with Shard(-1) logits outside loss_parallel context.
        """
        if is_loss_parallel_active() or not is_loss_parallel_op(op_name):
            return

        if len(args) == 0 or not isinstance(args[0], DTensor):
            return

        logits = args[0]
        if _is_shard_on_last_dim(logits):
            raise ValueError(
                f"Operator '{op_name}' requires loss_parallel context when input logits are "
                f"sharded on the vocabulary dimension (Shard(-1)). Please wrap your forward "
                f"and backward pass with loss_parallel():\n"
                f"  with loss_parallel():\n"
                f"      loss = F.cross_entropy(logits, targets)\n"
                f"      loss.backward()\n"
                f"If you intentionally want to gather all shards to compute cross_entropy "
                f"(not recommended for large vocabulary), use logits.full_tensor() explicitly."
            )

    @staticmethod
    def _normalize_aclop_args(op_name: str, unpack_ops: list, args: tuple) -> tuple:
        """
        Normalize aclop-packed arguments for MindSpore backend operators.

        NOTE: This handles MindSpore aclop operators whose kernel signature packs
        arguments as ``(prim, op_name_str, (real_arg0, real_arg1, ...))``. The
        ``prim`` and ``op_name_str`` are preserved as ``packed_call`` for the
        final kernel invocation, while the real tensor arguments are extracted
        for layout inference and preprocessing.

        **aclop is planned for deprecation.** Once aclop is fully removed, this
        normalization and the associated ``unpack_ops`` list can be deleted.

        Args:
            op_name (str): Canonical operator name.
            unpack_ops (list): List of op names that may use aclop packed format.
            args (tuple): Raw positional arguments from the op call.

        Returns:
            tuple: ``(packed_call, normalized_args)``
                - **packed_call**: ``(prim, op_name_str)`` tuple for kernel
                  invocation, or ``None`` if no unpacking was performed.
                - **normalized_args**: The real tensor arguments (unpacked if
                  the packed format was detected, otherwise the original args).
        """
        if OpDispatcher._is_aclop_packed(op_name, unpack_ops, args):
            return (args[0], args[1]), tuple(args[2])
        return None, args

    @staticmethod
    def _is_aclop_packed(op_name: str, unpack_ops: list, args: tuple) -> bool:
        """Check if arguments use aclop packed format."""
        return (
            op_name in unpack_ops
            and len(args) == 3
            and isinstance(args[1], str)
            and isinstance(args[2], (tuple, list))
        )

    @staticmethod
    def _call_op_impl(op_impl: callable, packed_call, args, kwargs: dict):
        """Invoke *op_impl* with optional aclop packed-call wrapping.

        When *packed_call* is not ``None`` the MindSpore aclop kernel expects
        ``(prim, op_name, (arg0, arg1, ...))``.  Otherwise *args* are spread
        as positional arguments in the usual way.

        Args:
            op_impl: The op implementation callable.
            packed_call: ``(prim, op_name)`` tuple or ``None``.
            args: Local tensor arguments (list or tuple).
            kwargs: Keyword arguments dict.

        Returns:
            Result of the *op_impl* invocation.
        """
        if packed_call is not None:
            return op_impl(packed_call[0], packed_call[1], tuple(args), **kwargs)
        return op_impl(*args, **kwargs)

    def _handle_unregistered_op(
        self, op_name: str, op_call: callable, args: tuple, kwargs: dict
    ):
        """Handle ops that have no registered layout-inference entry.

        This is a fallback path for ops that are not registered in
        ``layout_infer_ops``. When arguments contain DTensors it either raises
        (with a hint to register a distributed op) or, for loss-parallel ops,
        gathers tensors to full and dispatches through the raw callable.

        Args:
            op_name: Canonical operator name.
            op_call: The raw operator callable.
            args: Positional arguments for op_call.
            kwargs: Keyword arguments for op_call.

        Returns:
            Raw dispatch result (plain Tensor, not wrapped as DTensor).

        Raises:
            RuntimeError: If op_name is not registered for layout inference.
        """
        has_dtensor = any(isinstance(arg, DTensor) for arg in args)
        has_dtensor = has_dtensor or any(isinstance(v, DTensor) for v in kwargs.values())
        if has_dtensor:
            self._check_ce_op_without_loss_parallel_context(op_name, args)

            if not is_loss_parallel_op(op_name):
                raise RuntimeError(
                    f"Operator {op_name} does not contain parallel layout infer func. "
                    f"DTensor dispatch requires explicit layout inference registration. "
                    f"Please register a distributed operator for '{op_name}' or use local tensors."
                )

            gathered_args, gathered_kwargs = self._gather_dtensors_to_full(args, kwargs)

            # Special handling for cross_entropy with 3D logits (only when NOT in loss_parallel context)
            # PyTorch expects: logits [N, C], targets [N]
            # But LLM forward returns: logits [batch, seq, vocab], targets [batch, seq]
            # Note: nll_loss input is log_probs, typically already 2D, so we only reshape for cross_entropy
            if op_name == "cross_entropy" and len(gathered_args) >= 2:
                logits = gathered_args[0]
                targets = gathered_args[1]
                if isinstance(logits, Tensor) and isinstance(targets, Tensor):
                    if logits.ndim > 2 and targets.ndim > 1 and targets.ndim == logits.ndim - 1:
                        vocab_size = logits.shape[-1]
                        gathered_args[0] = logits.reshape(-1, vocab_size)
                        gathered_args[1] = targets.reshape(-1)

            return op_call(*gathered_args, **gathered_kwargs)
        raise RuntimeError(f"Operator {op_name} does not contain parallel layout infer func.")

    def _dispatch_layout_infer(
        self, op_name: str, op_call: callable, args: tuple, kwargs: dict
    ):
        """Standard dispatch through layout-inference: preprocess → infer → execute → wrap.

        Args:
            op_name: Canonical operator name (already resolved by the caller).
            op_call: The raw operator callable.
            args: Positional arguments for op_call.
            kwargs: Keyword arguments for op_call.

        Returns:
            DTensor: Dispatched result wrapped as DTensor.

        Raises:
            RuntimeError: If op_name is not registered, or preprocess returns None.
        """
        if op_name not in self.layout_infer_ops:
            return self._handle_unregistered_op(op_name, op_call, args, kwargs)

        cache_manager = LayoutCacheManager.get_instance()
        distribute_op = cache_manager.distributed_op(op_name)

        # Normalize aclop-packed args before any per-op processing. Only the handful
        # of (deprecation-bound) unpack_ops ever use the packed format, so gate the
        # whole normalization behind an O(1) membership test instead of paying two
        # function frames (_normalize_aclop_args + _is_aclop_packed) on every op.
        if op_name in getattr(self, 'unpack_ops', ()):
            packed_call, args = self._normalize_aclop_args(op_name, self.unpack_ops, args)
        else:
            packed_call = None

        result = distribute_op.preprocess(args, kwargs)
        if result is None:
            raise RuntimeError(
                f"Operator '{op_name}' has not been migrated to the three-phase dispatch flow. "
                f"Please implement preprocess() to return (local_args, local_kwargs, cache_values)."
            )
        local_args, local_kwargs, cache_values = result
        cache_key = LayoutCacheKey.from_cache_values(cache_values)

        infer_result, op_impl = OpDispatcher._lookup_or_infer_layout(
            op_call, op_name, cache_key, cache_values, distribute_op, cache_manager
        )

        op_impl = op_call if op_impl is None else op_impl
        py_output = OpDispatcher._call_op_impl(op_impl, packed_call, local_args, local_kwargs)
        output = distribute_op.wrap_output(py_output, infer_result[0])
        return OpDispatcher._restore_inplace_dtensor_result(op_name, args, output)

    @staticmethod
    def _restore_inplace_dtensor_result(op_name: str, args: tuple, output: Any) -> Any:
        """Return the original DTensor wrapper after a local in-place operation."""
        if op_name in {"add_", "sub_"} and args and isinstance(args[0], DTensor):
            return args[0]
        return output

    @staticmethod
    def _lookup_or_infer_layout(func, func_name, cache_key, cache_values, distribute_op, cache_manager):
        """Look up cached layout or compute via distributed op.

        Returns:
            (infer_result, op_impl)
        """
        layout_cache = cache_manager.get_layout_cache()
        if func_name not in layout_cache:
            layout_cache[func_name] = {}
        op_layout_cache = layout_cache[func_name]
        if cache_key in op_layout_cache:
            return op_layout_cache[cache_key]
        infer_result = distribute_op.infer_layout(cache_values)
        op_impl = distribute_op.get_expand_impl(func, infer_result, cache_values)
        op_layout_cache[cache_key] = (infer_result, op_impl)
        return infer_result, op_impl

    def dispatch(self, op_call: callable, args: tuple, kwargs: dict) -> object:
        """Route an op call through the appropriate DTensor dispatch path.

        Args:
            op_call: The raw operator callable.
            args: Positional arguments for op_call.
            kwargs: Keyword arguments for op_call.

        Returns:
            Result of the dispatched op call.
        """
        op_name = platform.get_op_name(op_call)
        if logger.isEnabledFor(logging.DEBUG):
            log_dispatch_enter(op_name, args, kwargs)

        observer = _debug_mode_observer.get()
        if observer is not None:
            observer.on_op_dispatch_enter(op_name, op_call, args, kwargs)

        result = None
        try:
            should_bypass = self._should_bypass_dispatch(op_name)
            ragged_dtensor = None
            if not should_bypass or op_name not in _RAGGED_METADATA_BYPASS_OPS:
                ragged_dtensor = self._validate_ragged_dispatch(op_name, args, kwargs)
            if ragged_dtensor is not None:
                result = self._dispatch_ragged_elementwise(
                    op_call, args, kwargs, ragged_dtensor
                )
                return result

            if should_bypass:
                self._validate_inplace_partial_inputs(op_name, args, kwargs)
                result = op_call(*self._unwrap_args(args), **self._unwrap_kwargs(kwargs))
                if op_name in self._INPLACE_BYPASS_OPS and args and isinstance(args[0], DTensor):
                    result = args[0]
                return result

            if op_name in self._random_ops or op_name in self._random_ms_ops:
                result = self._dispatch_random_op(op_name, op_call, args, kwargs)
                return result

            self._check_decomposed_ce_op_in_loss_parallel(op_name, args, kwargs)

            if self._should_dispatch_loss_parallel(op_name):
                result = self._dispatch_loss_parallel(op_call, args, kwargs)
                return result

            if op_name not in self.layout_infer_ops and get_distributed_op(op_name) is not None:
                self.layout_infer_ops[op_name] = {}

            result = self._dispatch_layout_infer(op_name, op_call, args, kwargs)
            return result
        finally:
            if logger.isEnabledFor(logging.DEBUG):
                log_dispatch_exit(op_name, result)

            if observer is not None:
                observer.on_op_dispatch_exit(op_name, result)

_OP_DISPATCHER = OpDispatcher()
