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
"""Train a model built directly from public module-family interfaces."""

import logging

import torch

from examples.torch.module_family import DecoderConfig, FamilyDecoder
from hyper_parallel.models import HyperAutoModelForCausalLM


def main() -> None:
    """Build through the public facade and perform two ordinary AdamW updates."""
    config = DecoderConfig()
    model = HyperAutoModelForCausalLM.from_config(
        config, model_factory=FamilyDecoder, torch_dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for _ in range(2):
        hidden_states = torch.randn(2, 4, config.hidden_size, device=next(model.parameters()).device)
        loss = model(hidden_states).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info("Completed two public-interface training steps")


if __name__ == "__main__":
    main()
