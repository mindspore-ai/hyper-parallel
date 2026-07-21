"""Gated DeltaNet Triton backend for the Torch platform."""

from .function import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_bwd_finish_saved,
    chunk_gated_delta_rule_bwd_prepare_saved,
    chunk_gated_delta_rule_bwd_saved,
    chunk_gated_delta_rule_bwd_state_saved,
    chunk_gated_delta_rule_fwd_apply_state_saved,
    chunk_gated_delta_rule_fwd_output_saved,
    chunk_gated_delta_rule_fwd_prepare_saved,
    chunk_gated_delta_rule_fwd_saved,
)
from .causal_conv1d import causal_conv1d_triton
from .kernels.state_summary import (
    apply_gdn_state_gradient_summary,
    apply_gdn_state_summary,
    chunk_gated_delta_rule_state_gradient_summary_bwd,
    chunk_gated_delta_rule_state_summary_fwd,
)

__all__ = [
    "causal_conv1d_triton",
    "apply_gdn_state_gradient_summary",
    "apply_gdn_state_summary",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_bwd_finish_saved",
    "chunk_gated_delta_rule_bwd_prepare_saved",
    "chunk_gated_delta_rule_bwd_saved",
    "chunk_gated_delta_rule_bwd_state_saved",
    "chunk_gated_delta_rule_fwd_apply_state_saved",
    "chunk_gated_delta_rule_fwd_output_saved",
    "chunk_gated_delta_rule_fwd_prepare_saved",
    "chunk_gated_delta_rule_fwd_saved",
    "chunk_gated_delta_rule_state_summary_fwd",
    "chunk_gated_delta_rule_state_gradient_summary_bwd",
]
