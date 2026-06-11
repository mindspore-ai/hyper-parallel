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
"""Adamw optimizer."""

from typing import List

import torch


def adamw(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[torch.Tensor],
        step: int,
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        maximize: bool
) -> None:
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """

    step_tensor = torch.tensor(step, dtype=torch.int64, device=torch.npu.current_device())
    state_steps = [step_tensor] * len(params)

    torch._fused_adamw_(  # pylint: disable=protected-access
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs if amsgrad else [],
        state_steps,
        amsgrad=amsgrad,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize
    )


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer implementation."""

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            maximize=False
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Set optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []

            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            group['step'] = (group.get('step') or 0) + 1

            current_rank_params = group['params']
            for p in current_rank_params:
                if p.grad is None:
                    continue

                if p.grad.data.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)

                params_with_grad.append(p)
                grads.append(p.grad)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            if params_with_grad:
                adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    group['step'],
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group['maximize']
                )

        return loss
