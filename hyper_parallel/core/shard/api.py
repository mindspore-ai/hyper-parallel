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
from typing import Union, Callable, Dict, Optional
from functools import wraps
from hyper_parallel.core.layout import Layout
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.platform import get_platform
from hyper_parallel.core.shard import _op_dispatch as op_dispatch

platform = get_platform()
Parameter = platform.Parameter
Tensor = platform.Tensor
Module = platform.Module


class DistributedCustomOp:
    """
    Wrapper for custom operators to enable automatic DTensor dispatch and infer_layout.

    Unlike MindSpore built-in operators that automatically trigger DTensor.__fallback__,
    custom operators (e.g., ms_custom_ops.paged_attention) do not go through the C++ _run_op
    path and thus won't trigger fallback. This wrapper bridges that gap by:
    1. Detecting DTensor inputs automatically
    2. Calling _op_dispatch._with_layout_infer to trigger infer_layout
    3. Returning properly wrapped DTensor outputs

    Note: MindSpore's Cell.__call__ converts DTensor to plain Tensor before calling construct.
    To work around this, use input_layouts parameter or cell._current_input_layouts.

    Usage:
        # Create a distributed version of the custom op
        distributed_paged_attention = DistributedCustomOp(ms_custom_ops.paged_attention)

        # Use it in your Cell's construct - it will automatically handle DTensor
        output = distributed_paged_attention(query, key_cache, value_cache, ...)

    Args:
        op: The custom operator callable (e.g., ms_custom_ops.paged_attention)
        infer_layout_suffix: Optional suffix for layout inference method.
            Valid values: None, "WithShape", "Reshape", "WithTupleExpand", "Slice"
        cell: Optional Cell instance to read _current_input_layouts from (set by shard hook)
        cpu_arg_indices: Optional tuple of arg indices that should stay on CPU when wrapped
        cpu_kwarg_names: Optional tuple of kwarg names that should stay on CPU when wrapped
    """

    def __init__(self, op: Callable, infer_layout_suffix: Optional[str] = None, cell=None,
                 cpu_arg_indices: Optional[tuple] = None, cpu_kwarg_names: Optional[tuple] = None):
        self._op = op
        self._infer_layout_suffix = infer_layout_suffix
        self._cell = cell
        self._cpu_arg_indices = cpu_arg_indices or ()
        self._cpu_kwarg_names = cpu_kwarg_names or ()
        # Copy op's attributes for compatibility
        if hasattr(op, 'name'):
            self.name = op.name
        if hasattr(op, '__name__'):
            self.__name__ = op.__name__
        if hasattr(op, '__module__'):
            self.__module__ = op.__module__

    def __call__(self, *args, **kwargs):
        """
        Call the wrapped operator. Automatically detects DTensor inputs and
        routes through infer_layout when needed.
        """
        # Check if any input is a DTensor
        has_dtensor = any(isinstance(arg, DTensor) for arg in args)
        print(f"[DistributedCustomOp] op={self._op}, args_count={len(args)}, "
              f"has_dtensor_in_args={has_dtensor}")
        for i, arg in enumerate(args):
            print(f"  arg[{i}] type={type(arg).__name__}, is_dtensor={isinstance(arg, DTensor)}")

        if not has_dtensor:
            has_dtensor = any(isinstance(v, DTensor) for v in kwargs.values())
            print(f"[DistributedCustomOp] checked kwargs, has_dtensor={has_dtensor}")

        # Check if cell has saved layouts from shard hook
        input_layouts = None
        if not has_dtensor and self._cell is not None:
            input_layouts = getattr(self._cell, '_current_input_layouts', None)
            if input_layouts:
                print(f"[DistributedCustomOp] found _current_input_layouts from cell")
                has_dtensor = any(layout is not None for layout in input_layouts)

        if not has_dtensor:
            # No DTensor inputs and no saved layouts, call operator directly
            print(f"[DistributedCustomOp] no DTensor, calling op directly")
            return self._op(*args, **kwargs)

        # Has DTensor inputs or saved layouts, route through dispatch
        print(f"[DistributedCustomOp] has DTensor/layouts, routing through _with_layout_infer")
        dispatcher = op_dispatch._OP_DISPATCHER
        suffix = self._infer_layout_suffix

        # If inputs are plain Tensor but we have layouts, wrap them temporarily
        if input_layouts and not any(isinstance(arg, DTensor) for arg in args):
            wrapped_args = []
            for i, arg in enumerate(args):
                layout = input_layouts[i] if i < len(input_layouts) else None
                if layout is not None and isinstance(arg, Tensor) and not isinstance(arg, DTensor):
                    device = "CPU" if i in self._cpu_arg_indices else None
                    wrapped_args.append(DTensor.from_local(arg, layout, device=device))
                else:
                    wrapped_args.append(arg)
            args = tuple(wrapped_args)
            print(f"[DistributedCustomOp] wrapped args with layouts")

        if not suffix:
            return dispatcher._with_layout_infer(self._op, *args, **kwargs)
        if suffix == "WithShape":
            return dispatcher._with_layout_infer_with_shape(self._op, *args, **kwargs)
        if suffix == "Reshape":
            return dispatcher._with_layout_infer_reshape(self._op, *args)
        if suffix == "WithTupleExpand":
            return dispatcher._with_layout_infer_with_tuple_expand(self._op, *args, **kwargs)
        if suffix == "Slice":
            return dispatcher._with_layout_infer_slice(self._op, *args)

        raise ValueError(f"Unknown infer_layout_suffix: {suffix}")

    def __repr__(self):
        return f"DistributedCustomOp({self._op})"


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
        if not isinstance(arg, DTensor):
            continue

        to_layout = _get_layout(i, is_list)
        processed_args[i] = arg.redistribute(to_layout)
    for k, v in kwargs.items():
        if not isinstance(v, DTensor) or layouts.get(k) is None:
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
            if not isinstance(arg, DTensor) or arg is None:
                new_outputs.append(arg)
                continue
            to_layout = layouts[i]
            new_outputs.append(arg.redistribute(to_layout))
        return tuple(new_outputs)
    if len(layouts) != 1:
        raise ValueError(f"The size of outputs and out_layout must be equal, but got 1 and "
                         f"{len(layouts)}")
    return outputs.redistribute(layouts[0]) if isinstance(outputs, DTensor) else outputs


def _forward_pre_hook(cell, args):
    """_forward_pre_hook"""
    print(f"[shard._forward_pre_hook] cell={cell.__class__.__name__}, in_layout={cell.in_layout}")
    print(f"[shard._forward_pre_hook] args types: {[type(a).__name__ for a in args]}")
    if cell.in_layout is None:
        return args

    # Save DTensor layouts before MindSpore converts them to plain Tensor
    # This allows DistributedCustomOp to access layout info even when construct receives Tensor
    input_layouts = []
    for i, arg in enumerate(args):
        if isinstance(arg, DTensor):
            input_layouts.append(arg.layout)
        elif cell.in_layout and i < len(cell.in_layout):
            input_layouts.append(cell.in_layout[i])
        else:
            input_layouts.append(None)
    cell._current_input_layouts = tuple(input_layouts)
    print(f"[shard._forward_pre_hook] saved _current_input_layouts to cell")

    processed_args, _ = _parallel_in(platform.get_cell_construct(cell), args, {}, cell.in_layout)
    print(f"[shard._forward_pre_hook] processed_args types: {[type(a).__name__ for a in processed_args]}")
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
            raise ValueError(f"In python shard, sharding_plan's forward key must end with input or output, "
                             f"but got type {suffix}")

        set_inputs_layout = suffix == "input"
        set_outputs_layout = not set_inputs_layout
        register_cell = cell_dict[prefix]

        _set_layouts(register_cell, value, set_inputs_layout, set_outputs_layout)
        _register_cell_hook(register_cell, set_inputs_layout, set_outputs_layout)


def _shard_callable(func: Callable, sharding_plan: Dict):
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
            result = platform.search_parameter_by_name(model, param_name)
            if not result:
                raise ValueError(f"{param_name} is configured with a layout, but no instance was found.")
            _, _, param = result

            param = platform.set_layout_into_parameter(param, layout)
            platform.update_parameter_by_name(model, result, param)

    if forward_sharding_plan is not None:
        _register_hook(model, forward_sharding_plan)
    return model


def distributed_op_call(op_call: Callable, *args, **kwargs):
    """
    Call a custom op with layout inference when DTensor inputs exist.

    This API is intended for ms_custom_ops-style operators that are not built-in
    MindSpore primitives and need explicit layout inference.
    """
    dispatcher = op_dispatch._OP_DISPATCHER
    op_name = platform.get_op_name(op_call)
    has_dtensor = any(isinstance(arg, DTensor) for arg in args) or any(
        isinstance(value, DTensor) for value in kwargs.values()
    )
    if not has_dtensor:
        return op_call(*args, **kwargs)
    if op_name not in dispatcher.layout_infer_ops:
        raise RuntimeError(f"Operator {op_name} does not contain parallel layout infer config.")

    suffix = dispatcher.layout_infer_ops[op_name].get("infer_layout_suffix", "")
    if not suffix:
        return dispatcher._with_layout_infer(op_call, *args, **kwargs)
    if suffix == "WithShape":
        return dispatcher._with_layout_infer_with_shape(op_call, *args, **kwargs)
    if suffix == "Reshape":
        return dispatcher._with_layout_infer_reshape(op_call, *args)
    if suffix == "WithTupleExpand":
        return dispatcher._with_layout_infer_with_tuple_expand(op_call, *args, **kwargs)
    if suffix == "Slice":
        return dispatcher._with_layout_infer_slice(op_call, *args)
    raise RuntimeError(f"Operator {op_name} specified wrong suffix in parallel yaml.")


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
