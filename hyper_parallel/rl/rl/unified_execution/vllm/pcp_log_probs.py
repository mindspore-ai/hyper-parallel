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
"""Return log-probabilities from the PCP worker's inference logits."""

import logging

import torch

logger = logging.getLogger(__name__)


def compute_pcp_token_log_probs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Score native sampled/top-k IDs using this inference worker's own logits.

    Sampling has already selected these IDs. Neither the logits nor IDs are
    modified, and no training-side probability is supplied to this function.

    Args:
        logits: [R, V] inference logits for R selected request rows and vocabulary V.
        token_ids: [R, K] sampled/top-k token IDs in the native return order.

    Returns:
        [R, K] FP32 log-probabilities, retaining duplicate IDs and slot order.
    """
    return logits.float().log_softmax(dim=-1).gather(-1, token_ids.long())


def install_pcp_log_probs() -> None:
    """Bind MRV2 returned log-probs inside an explicitly selected unified PCP worker.

    The model adapter calls this only for PCP>1. Workers own one unified model
    and speculative models are unsupported; PCP1 and native-model processes
    never install this binding. Native sampling, top-k IDs and ranks remain
    owned by the original MRV2 implementation.
    """
    # Ascend MRV2 is optional in training, PCP1 and CPU-only config processes.
    from vllm.v1.worker.gpu.sample import logprob as worker_logprob  # pylint: disable=C0415
    from vllm_ascend.worker.v2.sample import logprob  # pylint: disable=C0415

    # Ascend installs an alias in the shared MRV2 score module before loading
    # the model. Both native call sites must retain the same probability rule.
    for module in (logprob, worker_logprob):
        current = module.compute_token_logprobs
        if current is not compute_pcp_token_log_probs and current.__module__ != logprob.__name__:
            raise RuntimeError("Unified PCP found an unexpected MRV2 log-prob implementation")
    for module in (logprob, worker_logprob):
        module.compute_token_logprobs = compute_pcp_token_log_probs
    logger.info("Unified PCP returns FP32 log_softmax probabilities from inference logits")
