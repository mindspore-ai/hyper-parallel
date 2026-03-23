"""Utility subpackage for distributed tensor operations."""

__all__ = ["compute_local_shape_and_global_offset", "clip_grad_norm_"]

from hyper_parallel.core.utils.shape_utils import compute_local_shape_and_global_offset
from hyper_parallel.core.utils.clip_grad import clip_grad_norm_
