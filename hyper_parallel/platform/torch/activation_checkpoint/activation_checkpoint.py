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
"""Activation Checkpointing Wrapper for Hyper Parallel."""
import functools

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import \
    checkpoint_wrapper as ckpt_wrapper
from torch.utils.checkpoint import checkpoint, noop_context_fn

from .sac import create_selective_checkpoint_contexts


def _checkpoint(function, *args, policy_fn=None):
    if policy_fn:
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
    else:
        context_fn = noop_context_fn
    return checkpoint(function, *args, context_fn=context_fn, use_reentrant=False)


checkpoint_wrapper = functools.partial(ckpt_wrapper, checkpoint_fn=_checkpoint)
