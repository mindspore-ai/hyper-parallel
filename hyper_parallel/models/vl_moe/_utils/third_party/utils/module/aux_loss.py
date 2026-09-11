# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Standalone MoE Auxiliary Loss Functions

from typing import Optional, Union

import torch


class AuxLossAutoScaler(torch.autograd.Function):
    """Inject auxiliary-loss gradients without changing the forward loss value."""

    main_loss_backward_scale = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        aux_loss_grad = torch.ones_like(aux_loss) * AuxLossAutoScaler.main_loss_backward_scale
        return grad_output, aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        AuxLossAutoScaler.main_loss_backward_scale = scale


def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    moe_aux_loss_coeff: float,
    aux_level: str,
    micro_batch_size: int = 1,
):
    num_tokens = probs.shape[0]
    num_experts = probs.shape[1]
    if aux_level == "seq":
        num_tokens = probs.shape[0] // micro_batch_size
        aggregated_probs_per_expert = probs.view(micro_batch_size, -1, num_experts).sum(dim=1)
        aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert, dim=1).mean() * (
            num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
        )
    else:
        aggregated_probs_per_expert = probs.sum(dim=0)
        aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
            num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
        )
    return aux_loss


def z_loss_func(logits: torch.Tensor, z_loss_coeff: float):
    num_experts = logits.shape[1]
    mean_value = torch.log(torch.full((), num_experts, dtype=logits.dtype, device=logits.device))
    return torch.mean(torch.square(torch.logsumexp(logits, dim=-1) - mean_value)) * z_loss_coeff


def load_balancing_loss_func(
    gate_logits: Union[tuple[torch.Tensor, ...], None],
    num_experts: int,
    topk: int,
    aux_loss_coeff: float,
    aux_level: str = "microbatch",
    score_function: str = "softmax",
    micro_batch_size: int = 1,
    attention_mask: Optional[torch.Tensor] = None,
    z_loss_coeff: float = 0.0,
) -> Union[torch.Tensor, int]:
    """Compute baseline-equivalent aux and z losses at the model output.

    Each entry contains one layer's router logits and the indices actually selected
    by that layer. Losses stay layer-local, matching the original in-layer path.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple) or len(gate_logits) == 0:
        return 0

    valid_layers = [payload for payload in gate_logits if payload is not None]
    if len(valid_layers) == 0:
        return 0

    total_loss = None
    loss_terms = []
    for payload in valid_layers:
        logits = payload[:, :num_experts]
        indices = payload[:, num_experts:].long()
        if score_function == "softmax":
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        elif score_function == "sigmoid":
            probs = torch.sigmoid(logits.float())
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
            probs = probs.to(logits.dtype)
        else:
            raise ValueError(f"Invalid score_function: {score_function}")

        num_tokens = probs.shape[0]
        if aux_level == "seq":
            num_tokens_per_mb = num_tokens // micro_batch_size
            mb_probs = probs.view(micro_batch_size, num_tokens_per_mb, num_experts)
            aggregated_probs = mb_probs.sum(dim=1)
            mb_indices = indices.view(micro_batch_size, num_tokens_per_mb, topk)
            tokens_per_expert = torch.stack(
                [
                    torch.histc(layer_indices.float(), bins=num_experts, min=0, max=num_experts)
                    for layer_indices in mb_indices
                ]
            )
            layer_aux_loss = torch.sum(aggregated_probs * tokens_per_expert, dim=1).mean() * (
                num_experts * aux_loss_coeff / (num_tokens_per_mb * num_tokens_per_mb * topk)
            )
        else:
            aggregated_probs = probs.sum(dim=0)
            tokens_per_expert = torch.histc(indices.float(), bins=num_experts, min=0, max=num_experts)
            layer_aux_loss = torch.sum(aggregated_probs * tokens_per_expert) * (
                num_experts * aux_loss_coeff / (num_tokens * num_tokens * topk)
            )

        layer_z_loss = None
        if z_loss_coeff:
            mean_value = torch.log(
                torch.full((), num_experts, dtype=logits.dtype, device=logits.device)
            )
            layer_z_loss = (
                torch.mean(torch.square(torch.logsumexp(logits, dim=-1) - mean_value)) * z_loss_coeff
            )

        layer_total_loss = layer_aux_loss if layer_z_loss is None else layer_aux_loss + layer_z_loss
        total_loss = layer_total_loss if total_loss is None else total_loss + layer_total_loss
        if layer_z_loss is not None:
            loss_terms.append(layer_z_loss)
        loss_terms.append(layer_aux_loss)

    return total_loss, tuple(loss_terms)
