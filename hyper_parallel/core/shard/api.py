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
"""shard"""
import inspect
from typing import Union, Callable, Dict, List
from functools import wraps
from hyper_parallel.core.layout import Layout, DeviceMesh
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.placement_types import Placement
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from hyper_parallel.platform import get_platform

platform = get_platform()
Parameter = platform.Parameter
Tensor = platform.Tensor
Module = platform.Module


def _has_kwargs(func):
    """_has_kwargs"""
    sig = inspect.signature(func)
    return any(
        param.default != inspect.Parameter.empty
        for param in sig.parameters.values()
    )


def _get_param_name(func):
    """_get_param_name"""
    sig = inspect.signature(func)
    return list(sig.parameters.keys())


def _convert_sharding_plan(sharding_plan: Dict, device_mesh: DeviceMesh) -> Dict:
    """
    Convert sharding_plan values to Layout objects.

    This function recursively traverses the sharding_plan and converts
    placement tuples (e.g., (Shard(0), Replicate())) to Layout objects.

    Args:
        sharding_plan: The original sharding plan with tuple specifications
        device_mesh: The DeviceMesh to use for conversion

    Returns:
        Dict: Converted sharding plan with Layout objects
    """

    def _is_placement_tuple(value):
        """Check if value is a placement specification tuple.

        A placement tuple contains Placement instances (Shard, Replicate) or
        alias strings ("dp", "None"). It should NOT be a tuple of placement tuples.

        Examples of placement tuples:
            (Shard(0), Replicate(), Shard(1))  -> True
            ("dp", "tp", "None")               -> True
            (("dp", "tp"), "None")             -> True (multi-axis sharding)

        Examples of NON-placement tuples:
            ((Shard(0),), (Shard(1),))         -> False (tuple of placement tuples)
        """
        if not isinstance(value, tuple) or len(value) == 0:
            return False

        for item in value:
            # Placement instance is valid
            if isinstance(item, Placement):
                continue
            # String (alias name) is valid
            if isinstance(item, str):
                continue
            # Nested tuple needs special handling
            if isinstance(item, tuple):
                # Nested tuple of strings is valid (multi-axis sharding)
                if len(item) > 0 and all(isinstance(x, str) for x in item):
                    continue
                # Nested tuple containing Placement means this is a tuple of placement tuples
                if len(item) > 0 and any(isinstance(x, Placement) for x in item):
                    return False
                # Empty tuple or other cases - not valid
                return False
            # Any other type is not valid in a placement tuple
            return False

        return True

    def _to_layout(value):
        """Convert a single sharding specification to Layout."""
        layout = Layout.from_device_mesh(device_mesh)
        result = layout(value)
        return result

    def _convert_value(value, wrap_single_as_list=False):
        """Recursively convert value based on its structure."""
        if value is None:
            return None

        # Case 1: It's a placement tuple - convert to Layout
        if _is_placement_tuple(value):
            layout = _to_layout(value)
            # Wrap single layout in list if required (for input/output in forward)
            return [layout] if wrap_single_as_list else layout

        # Case 2: It's a dict - recursively process each value
        if isinstance(value, dict):
            converted_dict = {}
            for k, v in value.items():
                converted_dict[k] = _convert_value(v, wrap_single_as_list=False)
            return converted_dict

        # Case 3: It's a list - recursively process each element
        if isinstance(value, list):
            return [_convert_value(v, wrap_single_as_list=False) for v in value]

        # Case 4: It's a tuple but not a placement tuple - treat as list
        if isinstance(value, tuple):
            return [_convert_value(v, wrap_single_as_list=False) for v in value]

        # Case 5: Other types (e.g., primitives) - return as is
        return value

    def _convert_forward_plan(forward_plan):
        """Convert forward plan with special handling for input/output."""
        if forward_plan is None:
            return None

        converted = {}
        for key, value in forward_plan.items():
            if key.endswith("input") or key.endswith("output"):
                # input/output need special handling:
                # - dict format: convert each value, keep as dict
                # - list/tuple format: convert each element, keep as list
                # - single placement tuple: convert and wrap in list
                if value is None:
                    converted[key] = None
                elif isinstance(value, dict):
                    # Dict format for kwargs: {"x": placements, "activation": placements}
                    converted[key] = {k: _convert_value(v) for k, v in value.items()}
                elif isinstance(value, (list, tuple)):
                    # Check if it's a single placement tuple or a list/tuple of placement tuples
                    if _is_placement_tuple(value):
                        # Single placement tuple - wrap in list
                        converted[key] = [_to_layout(value)]
                    else:
                        # List/tuple of placements for multiple positional args
                        converted[key] = [_convert_value(v) for v in value]
                else:
                    converted[key] = _convert_value(value, wrap_single_as_list=True)
            else:
                # Other keys in forward plan
                converted[key] = _convert_value(value)
        return converted

    # Main conversion logic
    converted_plan = {}

    for key, value in sharding_plan.items():
        if key == "forward":
            converted_plan[key] = _convert_forward_plan(value)
        elif key.endswith("input") or key.endswith("output"):
            # Top-level input/output (for callable sharding)
            if value is None:
                converted_plan[key] = None
            elif isinstance(value, dict):
                converted_plan[key] = {k: _convert_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                if _is_placement_tuple(value):
                    converted_plan[key] = [_to_layout(value)]
                else:
                    converted_plan[key] = [_convert_value(v) for v in value]
            else:
                converted_plan[key] = _convert_value(value, wrap_single_as_list=True)
        else:
            # parameter and other keys - use standard recursive conversion
            converted_plan[key] = _convert_value(value)

    return converted_plan


def _parallel_in(func, args, kwargs, layouts):
    """_parallel_in"""
    if not isinstance(layouts, (list, dict, tuple)):
        raise ValueError(f"The in_layout must be a list, tuple or dict, but got {type(layouts)}.")

    params_name = _get_param_name(func)
    processed_args = list(args)
    processed_kwargs = dict(kwargs)

    def _get_layout(index, is_list):
        """_get_layout"""
        if is_list:
            return layouts[index]
        param_name = params_name[index]
        return layouts[param_name]

    is_list = isinstance(layouts, (list, tuple))
    for i, arg in enumerate(args):
        if not isinstance(arg, DTensor):
            continue

        to_layout = _get_layout(i, is_list)
        processed_args[i] = arg.redistribute(to_layout.mesh, to_layout.placements)
    for k, v in kwargs.items():
        if not isinstance(v, DTensor) or layouts.get(k) is None:
            processed_kwargs[k] = v
            continue
        to_layout = layouts[k]
        processed_kwargs[k] = v.redistribute(to_layout.mesh, to_layout.placements)

    return tuple(processed_args), processed_kwargs


def _parallel_out(outputs, layouts):
    """_parallel_out"""
    if not isinstance(layouts, (list, tuple)):
        raise ValueError(f"The out_layout must be a list or tuple, but got {type(layouts)}.")
    if isinstance(outputs, (tuple, list)):
        if len(outputs) != len(layouts):
            raise ValueError(f"The size of outputs and out_layout must be equal, but got {len(outputs)} and "
                             f"{len(layouts)}")
        new_outputs = []
        for i, arg in enumerate(outputs):
            if not isinstance(arg, DTensor) or arg is None:
                new_outputs.append(arg)
                continue
            to_layout = layouts[i]
            new_outputs.append(arg.redistribute(to_layout.mesh, to_layout.placements))
        return tuple(new_outputs)
    if len(layouts) != 1:
        raise ValueError(f"The size of outputs and out_layout must be equal, but got 1 and "
                         f"{len(layouts)}")

    return outputs.redistribute(layouts[0].mesh, layouts[0].placements) if isinstance(outputs, DTensor) else outputs


def _forward_pre_hook(cell, args):
    """_forward_pre_hook"""
    if cell.in_layout is None:
        return args
    processed_args, _ = _parallel_in(platform.get_cell_construct(cell), args, {}, cell.in_layout)
    return processed_args


def _forward_pre_with_kwargs_hook(cell, args, kwargs):
    """_forward_pre_with_kwargs_hook"""
    if cell.in_layout is None:
        return args, kwargs
    return _parallel_in(platform.get_cell_construct(cell), args, kwargs, cell.in_layout)


def _forward_hook(cell, inputs, outputs):  # pylint: disable=unused-argument
    """_forward_hook"""
    if cell.out_layout is None:
        return outputs
    return _parallel_out(outputs, cell.out_layout)


def _forward_with_kwargs_hook(cell, inputs, kwargs, outputs):  # pylint: disable=unused-argument
    """_forward_with_kwargs_hook"""
    return _forward_hook(cell, inputs, outputs)


def _register_hook(model: Module, sharding_plan: Dict):
    """_register_hook"""

    def _register_cell_hook(model, has_inputs_layout, has_outputs_layout):
        """_register_cell_hook"""
        has_kwargs = _has_kwargs(platform.get_cell_construct(model))
        pre_hook = _forward_pre_with_kwargs_hook if has_kwargs else _forward_pre_hook
        hook = _forward_with_kwargs_hook if has_kwargs else _forward_hook
        if has_inputs_layout:
            model.register_forward_pre_hook(pre_hook, with_kwargs=has_kwargs)

        if has_outputs_layout:
            model.register_forward_hook(hook, with_kwargs=has_kwargs)

    def _set_layouts(model, layouts, set_inputs_layout, set_outputs_layout):
        """_set_layouts"""
        if set_inputs_layout:
            model.in_layout = layouts

        if set_outputs_layout:
            model.out_layout = layouts

    cell_dict = {}
    for name, cell in platform.get_cells_and_names(model):
        cell_dict[name] = cell

    valid_suffix = ["input", "output"]
    for key, value in sharding_plan.items():
        if value is None:
            continue
        has_dot = '.' in key
        split_key = key.rsplit('.', 1)
        prefix = split_key[0] if has_dot else ""
        suffix = split_key[1] if has_dot else key
        if suffix not in valid_suffix:
            raise ValueError(f"In python shard_module, sharding_plan's forward key must end with input or output, "
                             f"but got type {suffix}")

        set_inputs_layout = suffix == "input"
        set_outputs_layout = not set_inputs_layout
        register_cell = cell_dict[prefix]

        _set_layouts(register_cell, value, set_inputs_layout, set_outputs_layout)
        _register_cell_hook(register_cell, set_inputs_layout, set_outputs_layout)


def _register_local_tensor_hook(cell: Module, return_local_tensor_list: List[str]):
    """_register_local_tensor_hook"""

    def hook_func(cell, inputs, outputs):  # pylint: disable=unused-argument
        def _recursive_to_local(out):
            if isinstance(out, (tuple, list)):
                new_out = []
                for item in out:
                    new_out.append(_recursive_to_local(item))
                return tuple(new_out) if isinstance(out, tuple) else new_out
            if isinstance(out, DTensor):
                return out.to_local()
            return out

        return _recursive_to_local(outputs)

    cell_dict = {}
    for name, sub_cell in platform.get_cells_and_names(cell):
        cell_dict[name] = sub_cell

    for cell_name in return_local_tensor_list:
        register_cell = cell_dict[cell_name]
        register_cell.register_forward_hook(hook_func)


def _shard_callable(func: Callable, sharding_plan: Dict):
    """_shard_callable"""
    forward_sharding_plan = sharding_plan.get("forward")
    if forward_sharding_plan is None:
        return func

    @wraps(func)
    def _shard_wrapper(*args, **kwargs):
        """_shard_wrapper"""
        input_layout = forward_sharding_plan.get("input")
        output_layout = forward_sharding_plan.get("output")
        if input_layout is not None:
            args, kwargs = _parallel_in(func, args, kwargs, input_layout)
        outputs = func(*args, **kwargs)
        if output_layout is not None:
            outputs = _parallel_out(outputs, output_layout)
        return outputs

    return _shard_wrapper


def shard_module(model: Union[Module, Callable], device_mesh: DeviceMesh, sharding_plan: ShardingPlan):
    """
    Defining the input, output and parameters layouts of this cell or Callable.

    Note:
        - It is valid only in pynative mode.

    .. warning::
        The method is currently not supported in Graph mode.

    Args:
        model (Module or Callable): The model to be sharded.
        device_mesh (DeviceMesh): The device mesh for sharding.
        sharding_plan (ShardingPlan): Define the layout for the specified parameters, inputs or outputs.
            The sharding specification can be:
            - tuple of strings for alias format, e.g., ("dp", "None")
            - tuple of Placements, e.g., (Shard(0), Replicate())

    Returns:
        Module or Callable: The sharded model.

    Examples:
        >>> # Usage with device_mesh and alias format
        >>> mesh = DeviceMesh("npu", (2, 2), nesh_dim_names=("dp", "tp"))
        >>> sharding_plan = ShardingPlan(
        ...     plan={"mlp.weight": ("None", "tp")},
        ...     input_plan={"input": ("dp", "None")},
        ...     output_plan={"output": ("dp", "tp")}
        ... )
        >>> model = shard_module(model, mesh, sharding_plan)

        >>> # Usage with device_mesh and Placement format
        >>> mesh = DeviceMesh("npu", (2, 2), nesh_dim_names=("dp", "tp"))
        >>> sharding_plan = ShardingPlan(
        ...     plan={"mlp.weight": (Replicate(), Shard(1))},
        ...     input_plan={"input": (Shard(0), Replicate())},
        ...     output_plan={"output": (Shard(0), Shard(1))}
        ... )
        >>> model = shard_module(model, mesh, sharding_plan)
    """
    if platform.get_world_size() == 1:
        return None

    if not isinstance(sharding_plan, ShardingPlan):
        raise TypeError(f"The 'sharding_plan' must be an instance of ShardingPlan, "
                        f"but got {type(sharding_plan)}. Direct dict input is not supported.")

    normalized_plan = {}
    return_local_tensor_list = None

    if sharding_plan.plan:
        normalized_plan["parameter"] = sharding_plan.plan

    forward_part = {}

    if sharding_plan.input_plan:
        if not isinstance(sharding_plan.input_plan, dict):
            raise TypeError(f"input_plan must be a dict, but got {type(sharding_plan.input_plan)}")
        forward_part.update(sharding_plan.input_plan)

    if sharding_plan.output_plan:
        if not isinstance(sharding_plan.output_plan, dict):
            raise TypeError(f"output_plan must be a dict, but got {type(sharding_plan.output_plan)}")
        forward_part.update(sharding_plan.output_plan)

    if forward_part:
        normalized_plan["forward"] = forward_part

    if sharding_plan.return_local_tensor:
        return_local_tensor_list = sharding_plan.return_local_tensor

    # Convert sharding_plan to Layout objects
    converted_plan = _convert_sharding_plan(normalized_plan, device_mesh)

    if not isinstance(model, Module):
        return _shard_callable(model, converted_plan)

    param_sharding_plan = converted_plan.get("parameter")
    forward_sharding_plan = converted_plan.get("forward")

    if param_sharding_plan is not None:
        for param_name, layout in param_sharding_plan.items():
            if not isinstance(layout, Layout):
                raise ValueError(f"In python shard_module, the type of setting in parameter_plan must be Layout, "
                                 f"but got type {type(layout)}")
            result = platform.search_parameter_by_name(model, param_name)
            if not result:
                raise ValueError(f"{param_name} is configured with a layout, but no instance was found.")
            _, _, param = result
            layout.placement_to_tensor_map(param.dim())
            param = platform.set_layout_into_parameter(param, layout)
            platform.update_parameter_by_name(model, result, param)

    if forward_sharding_plan is not None:
        _register_hook(model, forward_sharding_plan)

    if return_local_tensor_list is not None:
        _register_local_tensor_hook(model, return_local_tensor_list)

    return model


def parallelize_value_and_grad(fn, weights, sens=None):
    """
    A wrapper function to generate the function to calculate forward output and gradient for the parallel scenario.

    Args:
        fn (Union[Cell, Function]): Function to do grad operation.
        weights (Union[ParameterTuple, Parameter, list[Parameter]]):
            The parameters of the training network that need to
            calculate the gradient. `weights` can be got through `weights = net.trainable_params()` .
        sens (Union[list(float), tuple(float)], optional): The sensitivity for grad operation. Default: "None".
            - If the fn only have one output, the sens must be None, and it will be attached automatically.
            - If the fn have multiple outputs:
                1) If the sens is None, only handle the first sensitivity, and set the remaining sensitivity to 0.
                2) If the sens is not None, the lengths of sens and outputs of fn must be equal.

    Returns:
        Function, the derivative function used to compute the gradient of a given function.
        For example, as for `out1, out2 = fn(*args)` , gradient function will return outputs like
        `((out1, out2), gradient)` .

    Raises:
        TypeError: If type of Args does not belong to required ones.

    Supported Platforms:
        ``Ascend``
    """
    from mindspore import ops  # pylint: disable=import-outside-toplevel
    grad_fn = ops.GradOperation(get_by_list=True, sens_param=True)

    # use CellWrapper to solve two problems:
    # 1. avoid running the forward fn or cell twice
    # 2. if the input of parallize_value_and_grad is cell and it is directly used as the input for grad,
    #    the operations before and after its __call__ function will not enter the auto-diff process.
    class CellWrapper(Module):
        def __init__(self, net):
            super().__init__(auto_prefix=False)
            self.network = net

        def construct(self, *args, **kwargs):
            return self.network(*args, **kwargs)

        def forward(self, *args, **kwargs):
            return self.network(*args, **kwargs)

    fn = CellWrapper(fn)
    fn.set_grad()  # avoid running the forward fn or cell twice

    def wrapper(*args, **kwargs):
        loss_value = fn(*args, **kwargs)
        p_sens = None

        if isinstance(loss_value, (list, tuple)):
            # There are multiple outputs, requiring multiple sens
            p_sens = []

            if sens is None:
                # if sens is None, only handle the first sens, and set the remaining sens to 0
                loss_0 = loss_value[0]
                if isinstance(loss_0, DTensor):
                    repeat_num = loss_0.layout.repeat_num()
                    sens_0 = ops.fill(ops.DType()(loss_0), loss_0.local_shape, 1.0 / repeat_num)
                else:
                    sens_0 = ops.fill(ops.DType()(loss_0), loss_0.shape, 1.0)
                p_sens.append(sens_0)

                for i in range(1, len(loss_value)):
                    loss_i = loss_value[i]
                    if isinstance(loss_i, DTensor):
                        sens_i = ops.fill(ops.DType()(loss_i), loss_i.local_shape, 0.0)
                    else:
                        sens_i = ops.fill(ops.DType()(loss_i), loss_i.shape, 0.0)
                    p_sens.append(sens_i)

            else:
                # sens is not None
                if not isinstance(sens, list) and not isinstance(sens, tuple):
                    raise TypeError("if the loss is list or tuple, the sens must be None or list or tuple")

                all_float = all(isinstance(item, float) for item in sens)
                if not all_float:
                    raise TypeError("if sens is not None, it should be list of float or tuple of float")

                if len(sens) != len(loss_value):
                    raise TypeError(f"the len of loss is {len(loss_value)}, but the len of sens is {len(sens)}")

                for _, loss_i in enumerate(loss_value):
                    if isinstance(loss_i, DTensor):
                        repeat_num = loss_i.layout.repeat_num()
                        sens_i = ops.fill(ops.DType()(loss_i), loss_i.local_shape, 1.0 / repeat_num)
                    else:
                        sens_i = ops.fill(ops.DType()(loss_i), loss_i.shape, 1.0)
                    p_sens.append(sens_i)

        else:
            # loss is tensor
            if sens is not None:
                raise TypeError(f"the fn only have one output, the sens must be None, but it is {sens}")
            if isinstance(loss_value, DTensor):
                repeat_num = loss_value.layout.repeat_num()
                p_sens = ops.fill(ops.DType()(loss_value), loss_value.local_shape, 1.0 / repeat_num)

            else:
                p_sens = ops.fill(ops.DType()(loss_value), loss_value.shape, 1.0)

        grads = grad_fn(fn, weights)(*args, **kwargs, sens=p_sens)
        return loss_value, grads

    return wrapper
