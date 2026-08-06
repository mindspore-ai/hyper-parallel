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
"""MindFormers PyNative AdamW implementation used by swap optimizer tests."""

import logging

from mindspore import _checkparam as validator, Parameter, Tensor, ParameterTuple, _no_grad
from mindspore.common import dtype as mstype
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.ops import auto_generate as gen
from mindspore.ops import operations as P

from hyper_parallel import DTensor, SkipDTensorDispatch

logger = logging.getLogger(__name__)
op_cast = P.Cast()


def inplace_copy(dst, src):
    """Copy ``src`` into ``dst``, operating on local shards for DTensors."""
    local_src = src.to_local() if isinstance(src, DTensor) else src
    local_dst = dst.to_local() if isinstance(dst, DTensor) else dst
    local_dst.copy_(local_src)
    return dst


def _run_adamw_opt(
        beta1, beta2, eps, lr, weight_decay, parameters, gradients,
        exp_avg, exp_avg_sq, optim_filter, bias_correction1, bias_correction2,
        one_minus_beta2
):
    """Apply AdamW optimizer to the weight parameter."""
    params_dtype = parameters.dtype
    if not optim_filter:
        return op_cast(gradients, params_dtype)

    param_fp32 = op_cast(parameters, mstype.float32)
    grads_fp32 = op_cast(gradients, mstype.float32)
    next_param = param_fp32 * (1.0 - lr * weight_decay)

    inplace_copy(exp_avg, exp_avg.mul(beta1).add(grads_fp32.mul(1.0 - beta1)))
    inplace_copy(exp_avg_sq, exp_avg_sq.mul(beta2).addcmul(grads_fp32, grads_fp32, one_minus_beta2))

    step_size = lr / bias_correction1

    denom = (exp_avg_sq / bias_correction2).sqrt().add(eps)
    return_param = op_cast(next_param - (exp_avg / denom).mul(step_size), params_dtype)

    inplace_copy(parameters, return_param)
    return op_cast(gradients, params_dtype)


def _run_fused_adamw_opt(
        op_adamw, amsgrad, maximize, beta1, beta2, eps, step, lr, weight_decay,
        parameters, gradients, exp_avg, exp_avg_sq, max_exp_avg_sq
):
    """Apply Fused AdamW optimizer to the weight parameter."""
    grads = op_cast(gradients, parameters.dtype)
    op_adamw(
        parameters, exp_avg, exp_avg_sq, max_exp_avg_sq, grads,
        step, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)
    return True


def _check_param_value(betas, eps, weight_decay, prim_name):
    """Check the type of inputs."""
    if eps < 0.0:
        raise ValueError(f"Invalid epsilon value: {eps}, should be >= 0.")
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}, should be >= 0 and < 1.")
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}, should be >= 0 and < 1.")
    if weight_decay < 0.0:
        raise ValueError(f"Invalid weight_decay value: {weight_decay}, should be >= 0.")

    validator.check_value_type('betas', betas, [tuple, list], prim_name)
    validator.check("betas size", len(betas), "", [2], validator.IN, prim_name)
    validator.check_value_type("betas[0]", betas[0], [float], prim_name)
    validator.check_value_type("betas[1]", betas[1], [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_value_type("weight_decay", weight_decay, [float], prim_name)


_LOW_PRECISION_DTYPES = (mstype.float16, mstype.bfloat16)


class AdamW(Optimizer):
    """
    This is the implementation of AdamW.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "lr", "weight_decay", and "order_params"
            are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used. It should be noted that weight
              decay can be a dynamic value only when dynamic weight decay is applied. Dynamic weight decay is similar
              to dynamic learning rate, users need to customize a weight decay schedule only with global step as input,
              and during training, the optimizer calls the instance of WeightDecaySchedule to get the weight decay
              value of current step.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule], optional): Default: ``1e-3``.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        betas (Union[list(float), tuple(float)], optional): The exponential decay rate for the 1st and 2nd moment
            estimations. Default: (0.9, 0.999). Each element should be in range (0.0, 1.0).

        eps (float, optional): Term added to the denominator to improve numerical stability. Default: ``1e-6``.
            Should be greater than 0.

        weight_decay (Union[float, int, Cell], optional): Weight decay (L2 penalty). Default: ``0.0``.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Note:
        fp32 master weight copies are always created for bf16/fp16 parameters; optimizer updates run
        on the fp32 copies and are synced back to the model parameters, aligning with Megatron's
        mixed precision optimizer design.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `betas[0]`, `betas[1]` or `eps` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        ValueError: If `eps` is less than or equal to 0.
        ValueError: If `betas[0]`, `betas[1]` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.
    """

    def __init__(
            self,
            params,
            learning_rate=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            enable_cpu_offload=False,
            enable_fused_opt=False,
            use_fused=False,
            **kwargs
    ):
        _check_param_value(betas, eps, weight_decay, self.cls_name)
        super().__init__(learning_rate, params, weight_decay=weight_decay)

        self.beta1_value = betas[0]
        self.beta2_value = betas[1]
        self.eps_value = eps
        self.beta1 = Tensor(betas[0], dtype=mstype.float32)
        self.beta2 = Tensor(betas[1], dtype=mstype.float32)
        self.eps = Tensor(eps, dtype=mstype.float32)
        self.one_minus_beta2 = Tensor(1.0 - betas[1], dtype=mstype.float32)

        self.enable_fused_opt = bool(enable_fused_opt or use_fused)
        self.enable_cpu_offload = enable_cpu_offload
        if self.enable_cpu_offload:
            raise ValueError("Not support enable_cpu_offload.")

        self._is_low_precision_param, self.fp32_params = self._init_main_params()
        self.exp_avg = self._init_state(prefix="adam_m")
        self.exp_avg_sq = self._init_state(prefix="adam_v")

        self.amsgrad = kwargs.get("amsgrad", False)
        self.maximize = kwargs.get("maximize", False)
        if not self.enable_fused_opt:
            self.max_exp_avg_sq = None
        elif self.amsgrad:
            self.max_exp_avg_sq = self._init_state(prefix="max_exp_avg_sq")
        else:
            self.max_exp_avg_sq = self.exp_avg_sq
        self.fused_adamw_opt = gen.AdamW() if self.enable_fused_opt else None

        if self.enable_fused_opt:
            self.global_step = Parameter(Tensor([-1], mstype.int32), "global_step")
            self.global_step_increase_tensor = Tensor([1], mstype.int32)
        else:
            self.global_step = Parameter(Tensor([0], mstype.int64), "global_step")
            self.global_step_increase_tensor = Tensor([1], mstype.int64)

    def _init_main_params(self):
        """Create fp32 master weight copies for bf16/fp16 model parameters.

        fp32 parameters are re-used directly (no copy needed).
        """
        is_low_precision = []
        fp32_params = []
        lp_count = 0
        for param in self._parameters:
            if param.dtype in _LOW_PRECISION_DTYPES:
                is_low_precision.append(True)
                lp_count += 1
                main_param = Parameter(param.clone().float(), name=f"main_param.{param.name}")
                fp32_params.append(main_param)
            else:
                is_low_precision.append(False)
                fp32_params.append(param)
        logger.info("[MixedPrecision] _init_main_params: %d/%d params with fp32 master copy",
                    lp_count, len(self._parameters))
        return tuple(is_low_precision), ParameterTuple(fp32_params)

    def _init_state(self, prefix):
        """Create a zero-initialized fp32 state tuple, named after the model parameters.

        State is cloned from the fp32 master weights (so it matches their shape/dtype)
        but named with the model parameter name, keeping naming aligned with Muon.
        """
        states = []
        for model_param, fp32_param in zip(self._parameters, self.fp32_params):
            state = fp32_param.clone('zeros')
            state.name = f"{prefix}.{model_param.name}"
            states.append(state)
        return ParameterTuple(states)

    def _copy_main_params_to_model_params(self):
        for model_param, fp32_param, is_lp in zip(
                self._parameters, self.fp32_params, self._is_low_precision_param):
            if is_lp:
                inplace_copy(model_param, op_cast(fp32_param, model_param.dtype))

    def reload_main_params_from_model(self):
        """Refresh fp32 master weights from the model parameters.

        Used on weights-only resume (no_load_optim=True): the model params have just
        been loaded from checkpoint while the fp32 masters still hold their pre-load
        init values, so copy model -> master to align their starting points.

        Runs outside the optimizer's construct(), so it must enter SkipDTensorDispatch
        itself (InplaceCopy has no parallel layout infer func under DTensor dispatch).
        _no_grad avoids the "leaf tensor that requires grad in an inplace operator"
        error: the fp32 master is a leaf Parameter and this copy is not an autograd op.
        """
        with _no_grad(), SkipDTensorDispatch():
            for model_param, fp32_param, is_lp in zip(
                    self._parameters, self.fp32_params, self._is_low_precision_param):
                if is_lp:
                    inplace_copy(fp32_param, op_cast(model_param, mstype.float32))

    def _increase_global_step(self):
        """Increase global step in PyNative mode without static-graph AssignAdd."""
        inplace_copy(self.global_step, self.global_step + self.global_step_increase_tensor)

    def construct(self, gradients):
        """Forward AdamW algorithm."""
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self._increase_global_step()

        lr = [float(x) for x in lr] if (self.is_group and self.is_group_lr) else float(lr)
        weight_decay = [float(x) for x in weight_decay] if self.is_group else float(weight_decay)
        with SkipDTensorDispatch():
            if self.enable_fused_opt:
                result = self.forward_fused_opt(gradients, lr, weight_decay)
            else:
                result = self.forward_opt(gradients, lr, weight_decay)
        return result

    def forward_fused_opt(self, gradients, lr, weight_decay):
        """Run fused AdamW with Python-loop dispatch in PyNative mode."""
        results = []
        step = op_cast(self.global_step, mstype.int64)
        is_lr_list = isinstance(lr, list)
        is_wd_list = isinstance(weight_decay, list)

        for index, (param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, optim_filter) in enumerate(zip(
                self.fp32_params,
                gradients,
                self.exp_avg,
                self.exp_avg_sq,
                self.max_exp_avg_sq,
                self.optim_filter,
        )):
            if not optim_filter:
                results.append(True)
                continue
            results.append(_run_fused_adamw_opt(
                self.fused_adamw_opt,
                self.amsgrad,
                self.maximize,
                self.beta1_value,
                self.beta2_value,
                self.eps_value,
                step,
                lr[index] if is_lr_list else lr,
                weight_decay[index] if is_wd_list else weight_decay,
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                max_exp_avg_sq,
            ))

        self._copy_main_params_to_model_params()

        return tuple(results)

    def forward_opt(self, gradients, lr, weight_decay):
        """Run AdamW with Python-loop dispatch in PyNative mode."""
        results = []
        bias_correction1 = 1.0 - self.beta1 ** self.global_step
        bias_correction2 = 1.0 - self.beta2 ** self.global_step
        is_lr_list = isinstance(lr, list)
        is_wd_list = isinstance(weight_decay, list)
        one_minus_beta2 = self.one_minus_beta2

        for index, (param, grad, exp_avg, exp_avg_sq, optim_filter) in enumerate(zip(
                self.fp32_params,
                gradients,
                self.exp_avg,
                self.exp_avg_sq,
                self.optim_filter
        )):
            results.append(_run_adamw_opt(
                self.beta1,
                self.beta2,
                self.eps,
                lr[index] if is_lr_list else lr,
                weight_decay[index] if is_wd_list else weight_decay,
                param,
                grad,
                exp_avg,
                exp_avg_sq,
                optim_filter,
                bias_correction1,
                bias_correction2,
                one_minus_beta2,
            ))

        self._copy_main_params_to_model_params()

        return tuple(results)
