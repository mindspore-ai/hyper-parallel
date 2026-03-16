# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""pytest entry — state_dict tests for fully_shard.

Example distributed run:
    pytest tests/torch/fully_shard/test_state_dict.py::test_t5_roundtrip_8cards -v -s

Example onecard run:
    pytest tests/torch/fully_shard/test_state_dict.py::test_t15_nested_extra_state_roundtrip -v -s
"""
import os

from tests.common.mark_utils import arg_mark
from tests.torch.fully_shard import _test_state_dict as _state_dict_cases
from tests.torch.utils import torchrun_case

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_state_dict.py")
_PORT_BASE = 12370


# ---------- allcards tests (T2–T3): basic state_dict / load_state_dict ----------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t2_load_dtensor_2cards():
    """
    Feature: Test state_dict shape and load_state_dict with hyper DTensor.
    Description: After fully_shard on 2 cards, verify state_dict() returns
        DTensors with correct global shape, then load via copy and assign paths.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t2_load_dtensor_2cards", _PORT_BASE + 2, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t3_load_tensor_2cards():
    """
    Feature: Test load_state_dict with plain torch.Tensor values.
    Description: Load local shard tensors, global full tensors, and vanilla
        single-NPU checkpoint into fully_shard model on 2 cards.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t3_load_tensor_2cards", _PORT_BASE + 3, num_proc=2)


# ---------- allcards tests (T5–T8, T11, T13) ----------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t5_roundtrip_8cards():
    """
    Feature: Test full training round-trip on 8 cards.
    Description: Train -> save -> load (DTensor + local + global Tensor)
        -> continue training on 8 cards, verify forward values match.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t5_roundtrip_8cards", _PORT_BASE + 5, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t6_get_model_sd_sharded():
    """
    Feature: Test get_model_state_dict sharded options on 8 cards.
    Description: Call get_model_state_dict with default options and with
        cpu_offload=True on 8 cards, verify DTensor shapes and CPU offload.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t6_get_model_sd_sharded", _PORT_BASE + 6, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t7_get_model_sd_full_cpu():
    """
    Feature: Test get_model_state_dict with full_state_dict=True and cpu_offload=True.
    Description: Call get_model_state_dict on 8 cards, verify rank 0 gets full CPU
        tensors and non-rank0 gets empty dict.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t7_get_model_sd_full_cpu", _PORT_BASE + 7, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t8_get_model_sd_ignore_frozen():
    """
    Feature: Test get_model_state_dict with ignore_frozen_params=True.
    Description: Freeze one parameter, verify frozen param is excluded
        and non-frozen params are present.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t8_get_model_sd_ignore_frozen", _PORT_BASE + 8, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t11_meta_load_backward():
    """
    Feature: Test load_state_dict into meta-init model preserves requires_grad.
    Description: Convert fully_shard model params to meta tensors (simulating
        lazy init), load a global checkpoint, verify requires_grad is preserved
        and forward/backward succeed without 'does not require grad' error.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t11_meta_load_backward", _PORT_BASE + 11, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t13_extra_state_error_paths():
    """
    Feature: Test _extra_state strict-mode error paths via real HSDP path.
    Description: Part A: checkpoint lacks _extra_state, model has overrides -
        strict=True raises, strict=False preserves defaults. Part B: state_dict
        has _extra_state but target has no set override - strict=True raises.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t13_extra_state_error_paths",
                  _PORT_BASE + 13, num_proc=2)


# ---------- onecard tests: _to_dtype_if_needed + _extra_state UT ----------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_t10_to_dtype_if_needed():
    """
    Feature: Test _to_dtype_if_needed cast and no-op behavior.
    Description: Verify same dtype returns same object, None returns same object,
        and different dtype returns cast tensor with correct dtype.
    Expectation: Run success.
    """
    _state_dict_cases.test_t10_to_dtype_if_needed()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t15_nested_extra_state_roundtrip():
    """
    Feature: Test nested _extra_state round-trip via real HSDP load path.
    Description: HSDP integration smoke for _extra_state success path.
        Multi-level module tree with mutated extra state, save and load
        through HSDPModule.load_state_dict(), verify restoration.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t15_nested_extra_state_roundtrip",
                  _PORT_BASE + 15, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_t16_extra_state_prefix_stripped_wrapper():
    """
    Feature: Test prefix-stripped _extra_state mapping as helper-level UT.
    Description: Wrapper (simulating Float16Module) strips 'module.' prefix.
        Verify prefix discovery still maps _extra_state to the owning module.
    Expectation: Run success.
    """
    _state_dict_cases.test_t16_extra_state_prefix_stripped_wrapper()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
def test_t17_asymmetric_extra_state_override():
    """
    Feature: Test asymmetric get/set _extra_state override as helper-level UT.
    Description: Validate unexpected and missing-side classification without
        running distributed torchrun.
    Expectation: Run success.
    """
    _state_dict_cases.test_t17_asymmetric_extra_state_override()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t18_pre_hook_injects_extra_state():
    """
    Feature: Test Phase 0 pre-hook injection via real HSDP load path.
    Description: HSDP integration test for the core bug fix: module's
        pre-hook injects _extra_state when checkpoint lacks it,
        strict=True succeeds through real HSDPModule.load_state_dict().
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t18_pre_hook_injects_extra_state",
                  _PORT_BASE + 18, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t19_pre_hook_with_wrapper_prefix():
    """
    Feature: Test pre-hook + wrapper prefix via real HSDP load path.
    Description: PrefixStrippingWrapper strips prefix AND inner module's
        pre-hook injects _extra_state. Exercises full Phase 0/1/2 pipeline
        through HSDPModule.load_state_dict() with combined scenario.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t19_pre_hook_with_wrapper_prefix",
                  _PORT_BASE + 19, num_proc=2)
