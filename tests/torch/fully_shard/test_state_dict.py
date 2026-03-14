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

Run allcards tests:
    pytest tests/torch/fully_shard/test_state_dict.py -k "allcards" -v -s

Run dryrun test:
    pytest tests/torch/fully_shard/test_state_dict.py::test_t10_to_dtype_if_needed -v -s
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


# ---------- allcards test (T5): round-trip training ----------

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


# ---------- allcards tests (T6–T8): get_model_state_dict ----------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t6_get_model_sd_sharded():
    """
    Feature: Test get_model_state_dict sharded options.
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
    Description: Freeze one parameter, call get_model_state_dict on 8 cards,
        verify frozen param is excluded and non-frozen params are present.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t8_get_model_sd_ignore_frozen", _PORT_BASE + 8, num_proc=8)


# ---------- allcards test (T11): meta init -> load -> backward ----------

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
    torchrun_case(_FILE, "test_t11_meta_load_backward", _PORT_BASE + 11, num_proc=8)


# ---------- dryrun test (T10): _to_dtype_if_needed ----------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="dryrun", essential_mark="essential")
def test_t10_to_dtype_if_needed():
    """
    Feature: Test _to_dtype_if_needed cast and no-op behavior.
    Description: Verify same dtype returns same object, None returns same object,
        and different dtype returns cast tensor with correct dtype.
    Expectation: Run success.
    """
    _state_dict_cases.test_t10_to_dtype_if_needed()


# ---------- allcards tests (T13–T19): _extra_state ----------

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t13_extra_state_error_paths():
    """
    Feature: Test _extra_state strict-mode error paths.
    Description: Part A: checkpoint lacks _extra_state, model has overrides —
        strict=True raises, strict=False preserves defaults. Part B: state_dict
        has _extra_state but target has no set override — strict=True raises.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t13_extra_state_error_paths",
                  _PORT_BASE + 13, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t15_nested_extra_state_roundtrip():
    """
    Feature: Test _extra_state round-trip with nested module tree.
    Description: Model has _ExtraStateLinear at two nesting levels
        (encoder.layer0 and encoder.layer1). Verify both _extra_state keys
        are saved and restored correctly via recursive dispatch.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t15_nested_extra_state_roundtrip",
                  _PORT_BASE + 15, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t16_extra_state_prefix_stripped_wrapper():
    """
    Feature: Test _extra_state round-trip through prefix-stripping wrapper.
    Description: Wrapper (simulating Float16Module) overrides state_dict() to
        strip 'module.' prefix. Verify _extra_state keys are correctly loaded
        despite raw module tree paths diverging from state_dict keys.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t16_extra_state_prefix_stripped_wrapper",
                  _PORT_BASE + 16, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t17_asymmetric_extra_state_override():
    """
    Feature: Test asymmetric get/set _extra_state override.
    Description: Module overrides get_extra_state but not set_extra_state.
        state_dict() includes the key, but load_state_dict should treat it
        as unexpected (matching PyTorch _load_from_state_dict semantics).
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t17_asymmetric_extra_state_override",
                  _PORT_BASE + 17, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t18_pre_hook_injects_extra_state():
    """
    Feature: Test _load_state_dict_pre_hooks inject missing _extra_state.
    Description: Module registers a pre-hook that injects a default
        _extra_state key when the checkpoint lacks it. strict=True should
        succeed because Phase 0 pre-hook replay fires the hook before
        the missing-key check.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t18_pre_hook_injects_extra_state",
                  _PORT_BASE + 18, num_proc=2)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_t19_pre_hook_with_wrapper_prefix():
    """
    Feature: Test pre-hook + wrapper prefix rewrite combined scenario.
    Description: Wrapper uses state_dict hooks to strip/add 'module.'
        prefix AND inner module has a pre-hook that injects _extra_state.
        Combines T16 (wrapper) and T18 (pre-hook) scenarios.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_t19_pre_hook_with_wrapper_prefix",
                  _PORT_BASE + 19, num_proc=2)
