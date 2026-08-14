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
"""Model-specific adapters for token-value Critic capabilities."""
from types import MethodType
from typing import Any
def attach_value_head(model: Any, model_name: str) -> Any:
    """Convert a supported causal LM instance into a token-value backbone.

    The adapter is intentionally model-specific: a model registration must
    prove where its final hidden states live instead of deriving fake values
    from logits.
    """
    if model_name != "qwen3_5":
        raise NotImplementedError(
            f"Critic capability is not registered for Hyper model '{model_name}'"
        )
    text_model = getattr(model, "model", None)
    final_norm = getattr(text_model, "norm", None)
    if final_norm is None or not hasattr(model, "config"):
        raise TypeError("qwen3_5 Critic adapter requires model.norm and model.config")
    hidden_size = int(model.config.hidden_size)
    parameter = next(model.parameters())
    linear_type = type(model.lm_head)
    model.value_head = linear_type(
        hidden_size,
        1,
        bias=False,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    original_forward = model.forward
    def value_forward(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run the backbone and project captured final hidden states to values."""
        captured: list[Any] = []
        def capture_hidden(module: Any, inputs: Any, output: Any) -> None:
            """Capture final normalized hidden states for the value head."""
            del module, inputs
            captured.append(output)
        handle = final_norm.register_forward_hook(capture_hidden)
        try:
            original_forward(*args, **kwargs)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("qwen3_5 Critic did not expose final hidden states")
        return {"values": self.value_head(captured[-1]).squeeze(-1)}
    model.forward = MethodType(value_forward, model)
    return model
__all__ = ["attach_value_head"]
