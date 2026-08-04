# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Thin pytest entry for forward_and_gradfn ST (lazy-import mindspore).

Implementations live in ``_backward_impl.py`` so pytest collection of this
module does not load MindSpore / hyper_parallel in the parent process.
"""
from __future__ import annotations

import importlib

from tests.common.mark_utils import arg_mark  # noqa: F401

_IMPL = "tests.mindspore.st.pipeline_parallel._backward_impl"


def _run(name: str):
    """Import impl on first use, then call the named test function."""
    # pylint: disable=C0415
    mod = importlib.import_module(_IMPL)
    return getattr(mod, name)()


def test_forward_output_consistency():
    """See ``_backward_impl.test_forward_output_consistency``."""
    return _run("test_forward_output_consistency")


def test_dx_only_inputs():
    """See ``_backward_impl.test_dx_only_inputs``."""
    return _run("test_dx_only_inputs")


def test_call_returns_dx_and_dw_together():
    """See ``_backward_impl.test_call_returns_dx_and_dw_together``."""
    return _run("test_call_returns_dx_and_dw_together")


def test_dx_dw_split_pipeline_consistency():
    """See ``_backward_impl.test_dx_dw_split_pipeline_consistency``."""
    return _run("test_dx_dw_split_pipeline_consistency")


def test_weight_grad_keep_graph_allows_reuse():
    """See ``_backward_impl.test_weight_grad_keep_graph_allows_reuse``."""
    return _run("test_weight_grad_keep_graph_allows_reuse")


def test_multi_output_intermediate_slot_grad():
    """See ``_backward_impl.test_multi_output_intermediate_slot_grad``."""
    return _run("test_multi_output_intermediate_slot_grad")


def test_shared_intermediate_weight_group_merge_dw():
    """See ``_backward_impl.test_shared_intermediate_weight_group_merge_dw``."""
    return _run("test_shared_intermediate_weight_group_merge_dw")


def test_complex_multilayer_network_dx_dw_accuracy():
    """See ``_backward_impl.test_complex_multilayer_network_dx_dw_accuracy``."""
    return _run("test_complex_multilayer_network_dx_dw_accuracy")


def test_weight_subgraph_meets_input_later_dx_dw_accuracy():
    """See ``_backward_impl.test_weight_subgraph_meets_input_later_dx_dw_accuracy``."""
    return _run("test_weight_subgraph_meets_input_later_dx_dw_accuracy")


def test_shared_weight_used_multiple_places_dx_dw_accuracy():
    """See ``_backward_impl.test_shared_weight_used_multiple_places_dx_dw_accuracy``."""
    return _run("test_shared_weight_used_multiple_places_dx_dw_accuracy")


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_dw_only_weights():
    """See ``_backward_impl.test_dw_only_weights``."""
    return _run("test_dw_only_weights")


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_exceptions_and_edge_cases():
    """See ``_backward_impl.test_exceptions_and_edge_cases``."""
    return _run("test_exceptions_and_edge_cases")


def test_tuple_input_grad_position_all():
    """See ``_backward_impl.test_tuple_input_grad_position_all``."""
    return _run("test_tuple_input_grad_position_all")


def test_tuple_input_grad_position_partial():
    """See ``_backward_impl.test_tuple_input_grad_position_partial``."""
    return _run("test_tuple_input_grad_position_partial")


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tuple_input_grad_position_partial_backward_compat():
    """See ``_backward_impl.test_tuple_input_grad_position_partial_backward_compat``."""
    return _run("test_tuple_input_grad_position_partial_backward_compat")


def test_compute_input_grad_raises_for_none_grad_position():
    """See ``_backward_impl.test_compute_input_grad_raises_for_none_grad_position``."""
    return _run("test_compute_input_grad_raises_for_none_grad_position")


def test_call_accumulates_weight_grads():
    """See ``_backward_impl.test_call_accumulates_weight_grads``."""
    return _run("test_call_accumulates_weight_grads")


def test_call_accumulates_input_grads():
    """See ``_backward_impl.test_call_accumulates_input_grads``."""
    return _run("test_call_accumulates_input_grads")


def test_frozen_weight_gets_no_gradient():
    """See ``_backward_impl.test_frozen_weight_gets_no_gradient``."""
    return _run("test_frozen_weight_gets_no_gradient")


def test_requires_grad_false_weight_raises():
    """See ``_backward_impl.test_requires_grad_false_weight_raises``."""
    return _run("test_requires_grad_false_weight_raises")


def test_dict_input_grad_position_all():
    """See ``_backward_impl.test_dict_input_grad_position_all``."""
    return _run("test_dict_input_grad_position_all")


def test_kwargs_tensor_grad_position_all():
    """See ``_backward_impl.test_kwargs_tensor_grad_position_all``."""
    return _run("test_kwargs_tensor_grad_position_all")


def test_tuple_input_compute_input_grad():
    """See ``_backward_impl.test_tuple_input_compute_input_grad``."""
    return _run("test_tuple_input_compute_input_grad")


def test_dict_output_with_sens():
    """See ``_backward_impl.test_dict_output_with_sens``."""
    return _run("test_dict_output_with_sens")


def test_dict_output_with_weights():
    """See ``_backward_impl.test_dict_output_with_weights``."""
    return _run("test_dict_output_with_weights")


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_parallel_grad_cases_suit():
    """See ``_backward_impl.test_parallel_grad_cases_suit``."""
    return _run("test_parallel_grad_cases_suit")
