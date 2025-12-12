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
from typing import Union, Callable, Dict
from functools import wraps
from hyper_parallel.core.layout import Layout, _infer_slice_shape_by_layout, _get_slice_tensor_by_layout
from hyper_parallel.core.dtensor import DTensor
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
        if not isinstance(arg, Tensor) or arg is None:
            continue

        to_layout = _get_layout(i, is_list)
        processed_args[i] = arg.redistribute(to_layout)
    for k, v in kwargs.items():
        if not isinstance(v, Tensor) or v.layout is None or layouts.get(k) is None:
            processed_kwargs[k] = v
            continue
        to_layout = layouts[k]
        processed_kwargs[k] = v.redistribute(to_layout)

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
            if not isinstance(arg, Tensor) or arg is None:
                new_outputs.append(arg)
                continue
            to_layout = layouts[i]
            new_outputs.append(arg.redistribute(to_layout))
        return tuple(new_outputs)
    if len(layouts) != 1:
        raise ValueError(f"The size of outputs and out_layout must be equal, but got 1 and "
                         f"{len(layouts)}")
    return outputs.redistribute(layouts[0])

def _forward_pre_hook(cell, args):
    """_forward_pre_hook"""
    if cell.in_layout is None:
        return args
    processed_args, _ = _parallel_in(cell.construct, args, {}, cell.in_layout)
    return processed_args

def _forward_pre_with_kwargs_hook(cell, args, kwargs):
    """_forward_pre_with_kwargs_hook"""
    if cell.in_layout is None:
        return args, kwargs
    return _parallel_in(cell.construct, args, kwargs, cell.in_layout)

def _forward_hook(cell, inputs, outputs):
    """_forward_hook"""
    if cell.out_layout is None:
        return outputs
    return _parallel_out(outputs, cell.out_layout)

def _forward_with_kwargs_hook(cell, inputs, kwargs, outputs):
    """_forward_with_kwargs_hook"""
    return _forward_hook(cell, inputs, outputs)

def _register_hook(model: Module, sharding_plan: Dict):
    """_register_hook"""
    def _register_cell_hook(model, has_inputs_layout, has_outputs_layout):
        """_register_cell_hook"""
        has_kwargs = _has_kwargs(model.construct)
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
    for name, cell in model.cells_and_names():
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
            raise ValueError(f"In python shard, sharding_plan's forward key must end with input or output, "
                             f"but got type {suffix}")

        set_inputs_layout = suffix == "input"
        set_outputs_layout = not set_inputs_layout
        register_cell = cell_dict[prefix]

        _set_layouts(register_cell, value, set_inputs_layout, set_outputs_layout)
        _register_cell_hook(register_cell, set_inputs_layout, set_outputs_layout)


def _shard_callable(func: Callable, sharding_plan:Dict):
    """_shard_callable"""
    forward_sharding_plan = sharding_plan.get("forward")
    if forward_sharding_plan is None:
        return func
    @wraps(func)
    def _shard_wrapper(*args, **kwargs):
        """_shard_wrapper"""
        input_layout = sharding_plan.get("input")
        output_layout = sharding_plan.get("output")
        if input_layout is not None:
            args, kwargs = _parallel_in(func, args, kwargs, input_layout)
        outputs = func(*args, **kwargs)
        if output_layout is not None:
            outputs = _parallel_out(outputs, output_layout)
        return outputs
    return _shard_wrapper


def _search_parameter_by_name(cell, param_name: str):
    """
    Find the parent Module of the parameter, the parameter's name in the parent Module, and the parameter object itself
    Return value: (parent Module instance, parameter's name in parent Module, parameter object).
    Returns None if not found.
    """
    # Remove the "self." prefix from param_name (to maintain compatibility with original logic)
    param_name = param_name.replace("self.", "")
    # Case 1: The parameter is a direct parameter of the current Module (not in any sub-Module)
    if param_name in cell._params:
        return (cell, param_name, cell._params[param_name])

    # Case 2: The parameter is in a sub-Module (supports multi-level nesting, e.g., "net_b.dense1.weight")
    if "." in param_name:
        # Split into: sub-Module path + parameter name (e.g., "net_b.dense1" + "weight")
        cell_path, param_key = param_name.rsplit(".", 1)
        try:
            # Locate the sub-Module where the parameter resides (supports multi-level paths)
            target_cell = cell.get_sub_cell(cell_path)
            # Check if the sub-Module directly contains this parameter
            if param_key in target_cell._params:
                return (target_cell, param_key, target_cell._params[param_key])
        except AttributeError:
            # Sub-Module path does not exist or the parameter is not in that sub-Module
            pass

    # Traverse all sub-Modules (recursively) to search for the parameter
    for _, child_cell in cell._cells.items():
        if isinstance(child_cell, Module):
            # Recursively search within the sub-Module
            result = _search_parameter_by_name(child_cell, param_name)
            if result is not None:
                return result

    return None

def _update_parameter_by_name(cell, result: tuple, new_param: Parameter) -> bool:
    """
    Modify the original parameter in a Module or sub-Module using the search result
    Args:
        cell: The cell which parameter is to update
        result: The tuple returned by _search_parameter_by_name (contains parent Module, parameter key, old parameter)
        new_param: New Parameter object (used to replace the original parameter)
    """
    parent_cell, param_key, _ = result
    # Key operation: directly modify the _params dictionary of the parent Module (original storage location)
    parent_cell._params[param_key] = new_param

    if param_key in parent_cell.__dict__:
        parent_cell.__dict__[param_key] = new_param
    parent_cell._params_list[param_key] = new_param


def _set_layout_into_parameter(param, layout):
    """Set layout in to parameter"""
    if isinstance(param, DTensor):
        raise ValueError(f"Parameter {param.name} has been configured layout, cannot be set repeatedly.")
    param_info = param.param_info
    requires_grad = param.requires_grad
    name = param.name
    slice_shape = _infer_slice_shape_by_layout(param.shape, layout)

    if not param.has_init:
        # has been init, get slice data
        param_dtensor = DTensor.from_local(_get_slice_tensor_by_layout(param, layout).value(), layout)
        param = Parameter(param_dtensor, name=name, requires_grad=requires_grad)
        param.param_info = param_info
    else:
        # has not been init, need to modify init shape
        param.init_mode.shape = slice_shape
        param_dtensor = DTensor.from_local(param.init_mode, layout)
        param = Parameter(param_dtensor, name=name, requires_grad=requires_grad)
        param.param_info = param_info
    return param


def shard(model: Union[Module, Callable], sharding_plan: Dict):
    """
        Defining the input, output and parameters layouts of this cell or Callable.

        Note:
            - It is valid only in pynative mode.

        .. warning::
            The method is currently not supported in Graph mode.

        Args:
            model (Module or Callable): The model to be sharded.
            sharding_plan (Dict): Define the layout for the specified parameters, inputs or outputs.

    """
    global platform
    if platform.get_world_size() == 1:
        return None
    if not isinstance(model, Module):
        return _shard_callable(model, sharding_plan)

    param_sharding_plan = sharding_plan.get("parameter")
    forward_sharding_plan = sharding_plan.get("forward")

    if param_sharding_plan is not None:
        for param_name, layout in param_sharding_plan.items():
            if not isinstance(layout, Layout):
                raise ValueError(f"In python shard, the type of setting in parameter_plan must be Layout, "
                                 f"but got type {type(layout)}")
            result = _search_parameter_by_name(model, param_name)
            if not result:
                raise ValueError(f"{param_name} is configured with a layout, but no instance was found.")
            _, _, param = result

            if isinstance(param, DTensor):
                raise ValueError(f"Parameter {param.name} has been configured layout, "
                                 f"cannot be set repeatedly.")
            param = _set_layout_into_parameter(param, layout)
            _update_parameter_by_name(model, result, param)

    if forward_sharding_plan is not None:
        _register_hook(model, forward_sharding_plan)
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
    from mindspore import ops
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
