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
"""CPU failure-path tests for distributed bit-exact acceptance gates."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rl.consistency import gates


class TestConsistencyGates(unittest.TestCase):
    """Use real CPU bit patterns and mocked DP exchange to test failure propagation."""

    def setUp(self) -> None:
        """Keep padding out of the loss while preserving one real action position."""
        self.experience = SimpleNamespace(
            sequences=torch.tensor([[10, 11, 0]]),
            attention_mask=torch.tensor([[True, True, False]]),
            loss_action_mask=torch.tensor([[True, False]]),
            old_log_probs=torch.tensor([[-0.5, 0.0]], dtype=torch.float32),
            worker_policy_version=2, trajectories=(),
        )
        rank = patch.object(gates.dist, "get_rank", return_value=0)
        rank.start()
        self.addCleanup(rank.stop)

    def _validate(self, actor: torch.Tensor, group_size: int = 1) -> dict[str, float]:
        return gates.validate_pre_update_consistency(
            self.experience, actor, expected_policy_version=2, group="dp", group_size=group_size,
        )

    def test_pre_update_rejects_invalid_versions_shapes_and_values(self) -> None:
        """Stale, unaligned, non-FP32, empty, and non-finite comparisons must fail."""
        cases = (
            ("old_log_probs", None, "requires rollout"),
            ("worker_policy_version", 1, "policy version mismatch"),
            ("old_log_probs", torch.zeros(1, 3), "shape mismatch"),
            ("old_log_probs", torch.zeros(1, 2, dtype=torch.float64), "dtype mismatch"),
            ("loss_action_mask", torch.ones(1, 3), "action mask shape mismatch"),
            ("loss_action_mask", torch.zeros(1, 2), "at least one valid"),
            ("old_log_probs", torch.tensor([[float("nan"), 0.0]]), "non-finite"),
        )
        for field, invalid, error in cases:
            with self.subTest(field=field, error=error), patch.object(self.experience, field, invalid):
                with self.assertRaisesRegex(RuntimeError, error):
                    self._validate(torch.tensor([[-0.5, 0.0]]))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            self._validate(torch.tensor([[float("inf"), 0.0]]))
        with patch.object(self.experience, "old_log_probs", torch.zeros(1, 2, dtype=torch.float16)):
            with self.assertRaisesRegex(RuntimeError, "FP32"):
                self._validate(torch.zeros(1, 2, dtype=torch.float16))

    def test_signed_zero_is_a_bit_mismatch_but_padding_is_ignored(self) -> None:
        """Equal numerical zeros with different signs must fail bit-exact comparison."""
        self.experience.old_log_probs = torch.tensor([[0.0, float("nan")]])
        result = self._validate(torch.tensor([[0.0, float("inf")]]))
        self.assertEqual(result["training/pre_update_exact_tokens"], 1.0)
        with self.assertRaisesRegex(RuntimeError, "first_mismatches=.*actor_bits"):
            self._validate(torch.tensor([[-0.0, 0.0]]))

    def test_remote_rank_error_and_mismatch_fail_the_local_rank(self) -> None:
        """A clean local comparison cannot conceal a failure on another DP rank."""
        for failure in ("remote invalid mask", None):
            def gather(output: list, record: dict, group: str, remote_failure=failure) -> None:
                """Inject a failing remote record while leaving the local record intact."""
                self.assertEqual(group, "dp")
                output[:] = [record, dict(record, rank=1, error=remote_failure,
                                          mismatch_count=0 if remote_failure else 1,
                                          first_mismatch={"rank": 1})]
            with self.subTest(failure=failure), patch.object(gates.dist, "all_gather_object", side_effect=gather):
                with self.assertRaisesRegex(RuntimeError, "bit-exact gate failed"):
                    self._validate(self.experience.old_log_probs.clone(), group_size=2)

    def test_forward_preflight_rejects_invalid_padding(self) -> None:
        """Packed attention requires at least two contiguous valid tokens."""
        for mask, error in (
            (torch.tensor([[True, False, True]]), "contiguous right padding"),
            (torch.tensor([[True, False, False]]), "at least two"),
            (torch.tensor([[True, True]]), "aligned two-dimensional"),
        ):
            with self.subTest(mask=mask), patch.object(self.experience, "attention_mask", mask):
                with self.assertRaisesRegex(RuntimeError, error):
                    gates.validate_consistency_forward_inputs(
                        self.experience, group="dp", group_size=1, operation="Actor",
                    )

    def test_forward_preflight_propagates_remote_errors(self) -> None:
        """All ranks observe preflight failures before entering packed collectives."""
        def gather(output: list, local_error: object, group: str) -> None:
            """Expose a remote padding error to the locally valid rank."""
            self.assertIsNone(local_error)
            self.assertEqual(group, "dp")
            output[:] = [None, "remote invalid padding"]
        with patch.object(gates.dist, "all_gather_object", side_effect=gather):
            with self.assertRaisesRegex(RuntimeError, "remote invalid padding"):
                gates.validate_consistency_forward_inputs(
                    self.experience, group="dp", group_size=2, operation="Actor",
                )

    def test_negative_control_counts_only_valid_changed_tokens(self) -> None:
        """An unchanged model or changes confined to padding cannot prove learning."""
        for actor, count in (([[-0.5, 99.0]], 0.0), ([[-0.25, 0.0]], 1.0)):
            with self.subTest(actor=actor):
                result = gates.measure_post_update_old_policy_mismatch(
                    self.experience, torch.tensor(actor), group="dp", group_size=1,
                )
                self.assertEqual(result["training/post_update_old_policy_mismatch_count"], count)
                self.assertEqual(result["training/post_update_negative_control_valid"], float(count > 0))
        self.experience.loss_action_mask.zero_()
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            gates.measure_post_update_old_policy_mismatch(
                self.experience, torch.zeros(1, 2), group="dp", group_size=1,
            )
