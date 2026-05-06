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
"""torchrun entry for TP + fully_shard end-to-end tests"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase
from tests.torch.utils import torchrun_case

_TEST_TP_FULLY_SHARD_E2E = "_test_tp_fully_shard_e2e.py"


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_tp_fully_shard_typical_networks():
    """
    Feature: representative TP + fully_shard end-to-end coverage.
    Description:
        1. Cover the standard TP-sharded linear network on a 2D TP + FSDP mesh.
        2. Cover the same-dim non-dim0 network with comm_fusion on a 2D TP + FSDP mesh.
    Expectation: Typical TP + fully_shard layouts match standalone loss and local gradients.
    """
    parallel_run([
        TorchCase(
            _TEST_TP_FULLY_SHARD_E2E,
            "test_tp_plus_fully_shard_loss_and_grad_match_standalone",
            12364,
            4,
        ),
        TorchCase(
            _TEST_TP_FULLY_SHARD_E2E,
            "test_tp_plus_fully_shard_same_dim_non_dim0_comm_fusion_loss_and_grad_match_standalone",
            12483,
            4,
        ),
    ])


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pure_tp_replicate_grad_managed_by_fully_shard_matches_standalone():
    """
    Feature: fully_shard manages pure-TP replicated DTensor parameter gradients.
    Description: Run a 1D TP-only case with fully_shard(mesh=None) and compare loss/grad with standalone.
    Expectation: Distributed loss and gradients match standalone, without passing replicate_params.
    """
    master_port = 12375
    case_name = "test_pure_tp_replicate_grad_managed_by_fully_shard_matches_standalone"
    torchrun_case(_TEST_TP_FULLY_SHARD_E2E, case_name, master_port, num_proc=4)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_hsdp_tp_comm_fusion_replicate_group_guards():
    """
    Feature: representative HSDP + TP + comm_fusion coverage for mixed replicate groups.
    Description:
        1. Cover the mixed TP-sharded plus TP-replicated network in one fully_shard unit.
        2. Cover the same network when the TP-replicated parameter is promoted to replicate_params.
    Expectation: Local shard grads and replicate-only grads both match their standalone reference slices.
    """
    master_port = 12482
    case_name = "test_hsdp_plus_tp_comm_fusion_mixed_replicate_groups_match_standalone"
    torchrun_case(_TEST_TP_FULLY_SHARD_E2E, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_hsdp_tp_comm_fusion_replicate_param_guard():
    """
    Feature: representative HSDP + TP + comm_fusion coverage for replicate_params.
    Description: Cover the same mixed network when the TP-replicated parameter is promoted to replicate_params.
    Expectation: Local shard grads and replicate-only grads both match their standalone reference slices.
    """
    master_port = 12484
    case_name = "test_hsdp_plus_tp_comm_fusion_replicate_params_match_standalone"
    torchrun_case(_TEST_TP_FULLY_SHARD_E2E, case_name, master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_hsdp_tp_comm_fusion_complex_replicate_param_network():
    """
    Feature: representative nonlinear HSDP + TP + comm_fusion coverage.
    Description: Run the nonlinear network with multiple TP-replicated replicate_params around the matmul path.
    Expectation: Distributed loss plus tracked local grads match standalone.
    """
    master_port = 12490
    case_name = "test_hsdp_plus_tp_comm_fusion_complex_replicate_params_match_standalone"
    torchrun_case(_TEST_TP_FULLY_SHARD_E2E, case_name, master_port)
