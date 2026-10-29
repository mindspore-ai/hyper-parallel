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
"""Single-card DSA CANN precision tests; collection stays framework-free."""

import importlib

import pytest

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("layout", ["BSND", "TND"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_dsa_indexer(layout, dtype):
    """Check causal TopK selection and packed boundaries."""
    importlib.import_module("tests.torch.components.functional._dsa_ops_impl").check_indexer(layout, dtype)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_dsa_topk_selection():
    """Check a context longer than TopK, source views and a nondefault stream."""
    importlib.import_module("tests.torch.components.functional._dsa_ops_impl").check_topk_selection()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("layout", ["BSND", "TND"])
def test_dsa_attention(layout):
    """Check sparse MLA forward and every differentiable input against Torch."""
    importlib.import_module("tests.torch.components.functional._dsa_ops_impl").check_attention(layout)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("layout", ["BSND", "TND"])
@pytest.mark.parametrize("sparse", [True])
def test_dsa_kl(layout, sparse):
    """Check sparse KL loss and indexer gradients against Torch."""
    importlib.import_module("tests.torch.components.functional._dsa_ops_impl").check_kl(layout, sparse)


@arg_mark(plat_marks=["platform_ascend950"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("layout", ["BSND", "TND"])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_dsa_dense_indexer_lse(layout, dtype):
    """Check dense indexer LSE V2 against an independent FP32 reference."""
    importlib.import_module("tests.torch.components.functional._dsa_ops_impl").check_dense_indexer_lse(layout, dtype)


@arg_mark(plat_marks=["platform_ascend950"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_dsa_dense_kl_tnd():
    """Check dense indexer KL forward gradients and normalized autograd path."""
    importlib.import_module("tests.torch.components.functional._dsa_ops_impl").check_kl("TND", False)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="onecard", essential_mark="essential")
def test_dsa_training_chain():
    """Check public BSND entry points, shared K/V gradients and auxiliary loss."""
    importlib.import_module("tests.torch.components.functional._dsa_ops_impl").check_training_chain()
