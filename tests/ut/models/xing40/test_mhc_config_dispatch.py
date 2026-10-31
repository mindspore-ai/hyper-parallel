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
"""Configuration propagation and portable mHC dispatch, including autograd."""
import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch

from hyper_parallel.models.xing40.modeling_xing40 import Xing40Config, Xing40MhcPre, Xing40MhcPost

pre_mod = importlib.import_module("hyper_parallel.components.functional.mhc_pre")
post_mod = importlib.import_module("hyper_parallel.components.functional.mhc_post")


def config(value):
    """Build a small mHC configuration for dispatch tests."""
    return Xing40Config(hidden_size=8, num_residual_streams=4, mhc_sinkhorn_iterations=3,
                        mhc_layernorm_epsilon=1e-6, use_fused_mhc=value)


@pytest.mark.parametrize("enabled", [False, True])
def test_model_propagates_flag_and_backward(enabled):
    cfg = config(enabled)
    pre, post = Xing40MhcPre(cfg), Xing40MhcPost(cfg)
    with torch.no_grad():
        pre.phi.weight.normal_(0, .1)
        pre.branch_alpha.fill_(.1)
        pre.branch_beta.zero_()
    x = torch.randn(2, 3, 32, requires_grad=True)
    with patch("hyper_parallel.models.xing40.modeling_xing40.mhc_pre", wraps=pre_mod.mhc_pre) as a, \
         patch("hyper_parallel.models.xing40.modeling_xing40.mhc_post", wraps=post_mod.mhc_post) as b:
        y, hp, hr, residual = pre(x)
        post(y, residual, hp, hr).square().mean().backward()
    assert a.call_args.kwargs["use_fused"] is enabled, (
        f"expected pre use_fused={enabled}, got={a.call_args.kwargs['use_fused']}"
    )
    assert b.call_args.kwargs["use_fused"] is enabled, (
        f"expected post use_fused={enabled}, got={b.call_args.kwargs['use_fused']}"
    )
    assert torch.isfinite(x.grad).all() and torch.isfinite(pre.phi.weight.grad).all(), (
        f"expected finite gradients=True, got x={torch.isfinite(x.grad).all()}, "
        f"phi={torch.isfinite(pre.phi.weight.grad).all()}"
    )

def test_default_and_invalid_config():
    assert Xing40Config().use_fused_mhc is True, (
        f"expected use_fused_mhc=True, got={Xing40Config().use_fused_mhc}"
    )
    with pytest.raises(ValueError, match="boolean"):
        config("false")

def test_npu_dispatch_spies():
    fake = MagicMock()
    fake.device.type = "npu"
    fake.size.return_value = (1, 1, 8)
    # Sentinel exceptions prove dispatch without invoking or requiring NPU.
    for enabled in (False, True):
        with patch.object(pre_mod, "omni_training_custom_ops", object()), \
             patch.object(pre_mod, "_mhc_pre_reference", side_effect=RuntimeError("portable")), \
             patch.object(pre_mod._MhcPre, "apply", side_effect=RuntimeError("fused")):
            with pytest.raises(RuntimeError, match="fused" if enabled else "portable"):
                pre_mod.mhc_pre(fake, fake, fake, fake, 2, use_fused=enabled)
        with patch.object(post_mod, "omni_training_custom_ops", object()), \
             patch.object(post_mod._MhcPost, "apply", side_effect=RuntimeError("fused")):
            if enabled:
                with pytest.raises(RuntimeError, match="fused"):
                    post_mod.mhc_post(fake, fake, fake, fake, 2, use_fused=True)
            else:
                with patch.object(post_mod.torch, "sum", return_value=fake):
                    post_mod.mhc_post(fake, fake, fake, fake, 2, use_fused=False)
                assert not post_mod._MhcPost.apply.called, (
                    f"expected fused post called=False, got={post_mod._MhcPost.apply.called}"
                )
