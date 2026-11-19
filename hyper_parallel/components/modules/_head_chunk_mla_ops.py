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
"""Native saved-state MLA kernels and a CPU-only mathematical diagnostic."""

import torch

_NPU_CAUSAL_MASKS: dict[tuple[str, int], torch.Tensor] = {}


def _causal_mask(tensor: torch.Tensor, length: int = 2048) -> torch.Tensor:
    key = (str(tensor.device), length)
    mask = _NPU_CAUSAL_MASKS.get(key)
    if mask is None:
        mask = torch.triu(torch.ones((length, length), dtype=torch.bool, device=tensor.device), diagonal=1)
        _NPU_CAUSAL_MASKS[key] = mask
    return mask


class _NpuAttention:
    """The same zero-dropout FA and mask modes as native single-card MLA."""

    @staticmethod
    def _options(query):
        length = query.shape[2]
        return {"atten_mask": _causal_mask(query, min(length, 2048)),
                "sparse_mode": 2 if length > 2048 else 0,
                "pre_tockens": 1048576, "next_tockens": 0, "keep_prob": 1.0, "inner_precise": 0}

    @staticmethod
    def forward_saved(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run native FA and retain only its zero-dropout backward state.

        Args:
            query: Stage-local query in BNSD layout.
            key: Stage-local key in BNSD layout.
            value: Stage-local value in BNSD layout.
            scale: Native attention score scale.

        Returns:
            Output and the output/max/sum tensors consumed by native FA-grad.
        """
        # Optional accelerator kernels must not load during CPU planning.
        import torch_npu  # pylint: disable=import-outside-toplevel

        output, maximum, denominator = torch_npu.npu_fusion_attention(
            query.contiguous(), key.contiguous(), value.contiguous(), query.shape[1], "BNSD",
            scale=scale, **_NpuAttention._options(query),
        )[:3]
        # Dropout is forbidden by HeadChunkMLA, so no RNG state is consumed in backward.
        return output, (output, maximum, denominator)

    @staticmethod
    def backward_saved(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, grad_output: torch.Tensor,
        scale: float, saved: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Consume saved native FA state without replaying its forward.

        Args:
            query: Replayed stage query in BNSD layout.
            key: Replayed stage key in BNSD layout.
            value: Replayed stage value in BNSD layout.
            grad_output: Output gradient in the FA owner layout.
            scale: Original attention score scale.
            saved: Output, softmax maximum and softmax sum from forward.

        Returns:
            Query, key and value gradients in their input layouts.
        """
        # Keep the native kernel optional until accelerator execution.
        import torch_npu  # pylint: disable=import-outside-toplevel

        output, maximum, denominator = saved
        gradients = torch_npu.npu_fusion_attention_grad(
            query.contiguous(), key.contiguous(), value.contiguous(), grad_output.contiguous(),
            query.shape[1], "BNSD",
            softmax_max=maximum, softmax_sum=denominator, attention_in=output, scale_value=scale,
            **_NpuAttention._options(query),
        )
        return gradients[:3]


class _PackedNpuAttention:
    """Invocation-owned global document ends; convert only at the FA ABI boundary."""

    def __init__(self, lengths):
        self.lengths = list(lengths)

    @staticmethod
    def _to_tnd(tensor):
        return tensor.squeeze(0).transpose(0, 1).contiguous()

    @staticmethod
    def _to_bnsd(tensor):
        return tensor.transpose(0, 1).unsqueeze(0).contiguous()

    def forward_saved(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Call native TND FA and preserve its matching backward state.

        Args:
            query: Batch-one BNSD query after Ulysses.
            key: Batch-one global BNSD key for this head stage.
            value: Batch-one global BNSD value for this head stage.
            scale: Original MLA attention scale.

        Returns:
            BNSD output and output/max/sum saved state.
        """
        # Native kernels remain optional for CPU planning.
        import torch_npu  # pylint: disable=import-outside-toplevel

        output, maximum, denominator = torch_npu.npu_fusion_attention(
            *(self._to_tnd(tensor) for tensor in (query, key, value)), query.shape[1], "TND",
            atten_mask=_causal_mask(query), scale=scale, sparse_mode=3,
            actual_seq_qlen=self.lengths, actual_seq_kvlen=self.lengths,
            pre_tockens=1048576, next_tockens=0, keep_prob=1.0, inner_precise=0,
        )[:3]
        output = self._to_bnsd(output)
        return output, (output, maximum, denominator)

    def backward_saved(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        grad_output: torch.Tensor, scale: float, saved: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse native packed statistics without replaying FA forward.

        Args:
            query: Batch-one BNSD query after Ulysses.
            key: Reconstructed global BNSD key.
            value: Reconstructed global BNSD value.
            grad_output: BNSD output gradient.
            scale: Original MLA attention scale.
            saved: BNSD output and unmodified TND max/sum statistics.

        Returns:
            BNSD query, key and value gradients.
        """
        # Preserve native TND statistics instead of treating them as BNSD statistics.
        import torch_npu  # pylint: disable=import-outside-toplevel

        output, maximum, denominator = saved
        gradients = torch_npu.npu_fusion_attention_grad(
            *(self._to_tnd(tensor) for tensor in (query, key, value, grad_output)), query.shape[1], "TND",
            atten_mask=_causal_mask(query), softmax_max=maximum, softmax_sum=denominator,
            attention_in=self._to_tnd(output), scale_value=scale, sparse_mode=3,
            actual_seq_qlen=self.lengths, actual_seq_kvlen=self.lengths,
            pre_tockens=1048576, next_tockens=0, keep_prob=1.0, inner_precise=0,
        )
        return tuple(self._to_bnsd(tensor) for tensor in gradients[:3])



def _eager_scores(query, key, scale):
    compute_dtype = torch.float32 if query.dtype in (torch.bfloat16, torch.float16) else query.dtype
    query, key = query.to(compute_dtype), key.to(compute_dtype)
    scores = query.matmul(key.transpose(-1, -2)) * scale
    mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).triu(1)
    return scores.masked_fill(mask, float("-inf"))


class _EagerAttention:
    """CPU oracle backend with a manual VJP, never a second FA forward call."""

    @staticmethod
    def forward_saved(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Evaluate CPU causal attention and save output/max/sum for a manual VJP."""
        scores = _eager_scores(query, key, scale)
        maximum = scores.amax(-1, keepdim=True)
        unnormalized = (scores - maximum).exp()
        denominator = unnormalized.sum(-1, keepdim=True)
        output = (unnormalized / denominator).matmul(value.to(scores.dtype)).to(query.dtype)
        return output, (output, maximum, denominator)

    @staticmethod
    def backward_saved(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, grad_output: torch.Tensor,
        scale: float, saved: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return the CPU analytical Q/K/V gradients without another forward call."""
        output, maximum, denominator = saved
        scores = _eager_scores(query, key, scale)
        probability = (scores - maximum).exp() / denominator
        query, key, value = query.to(scores.dtype), key.to(scores.dtype), value.to(scores.dtype)
        grad = grad_output.to(scores.dtype)
        grad_probability = grad.matmul(value.transpose(-1, -2))
        delta = (grad * output.to(scores.dtype)).sum(-1, keepdim=True)
        grad_scores = probability * (grad_probability - delta) * scale
        return (grad_scores.matmul(key).to(grad_output.dtype),
                grad_scores.transpose(-1, -2).matmul(query).to(grad_output.dtype),
                probability.transpose(-1, -2).matmul(grad).to(grad_output.dtype))


def _rotate(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, backend: str) -> torch.Tensor:
    if cos.numel() == 0:
        return tensor
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    if cos.shape[0] == 1 and tensor.shape[0] > 1:
        cos = cos.expand(tensor.shape[0], -1, -1)
        sin = sin.expand(tensor.shape[0], -1, -1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if backend == "eager":
        first, second = tensor.chunk(2, dim=-1)
        return tensor * cos + torch.cat((-second, first), dim=-1) * sin
    # This is the same half-RoPE call as the native MLA functional interface.
    import torch_npu  # pylint: disable=import-outside-toplevel

    return torch_npu.npu_rotary_mul(tensor.clone(), cos, sin, rotary_mode="half")
