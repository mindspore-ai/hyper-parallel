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
"""Native balancing logs follow delivered windows instead of speculative plans."""

import logging
import unittest
from itertools import product
from threading import current_thread
from unittest.mock import Mock, patch

from hyper_parallel.distributed_data.balance_logging import logger
from hyper_parallel.distributed_data.schema import WorkloadCost
from tests.ut.distributed_data.test_accumulation_window import _ids, _loader


class TestNativeBalanceLogging(unittest.TestCase):
    """Check reporting boundaries, modeled costs and native source variants."""

    def test_prefetch_logs_only_when_window_is_delivered(self) -> None:
        """Feature: Consumption-aligned INFO logs.
        Description: Prefetch two windows with online/offline metadata and either planning mode.
        Expectation: Only the active window logs, and costs are evaluated once per sample.
        """
        for metadata_mode, broadcast_plan, accumulation in product((False, True), (False, True), (1, 5)):
            with self.subTest(metadata=metadata_mode, broadcast=broadcast_plan, accumulation=accumulation):
                with self.assertLogs(logger.name, level="INFO") as captured, _loader(
                        metadata_mode=metadata_mode, broadcast_plan=broadcast_plan, accumulation=accumulation,
                        log_balance=True,
                ) as loader:
                    cost_model = Mock(spec=["__call__"], return_value=WorkloadCost(llm=7))
                    loader._planner.cost_model = cost_model
                    loader.prefetch()
                    loader.wait_for_prefetch()
                    self.assertEqual(captured.records, [])
                    self.assertEqual(_ids(next(loader)), [0, 1])
                    loader.wait_for_prefetch()
                    expected_windows = 1 if accumulation == 1 else 2
                    self.assertEqual(len(captured.records), 1)
                    first = captured.records[0].getMessage()
                    self.assertIn("[dp-balance] iteration 1 :", first)
                    self.assertIn(f"micro_batches=1-{min(3, accumulation)}", first)
                    self.assertIn(f"original={14 * min(3, accumulation)},", first)
                    self.assertIn("accepted=False", first)
                    self.assertIn("dp0: pack0: seq_len=1, samples=1, cost=7", first)
                    self.assertNotIn("send", first)
                    self.assertEqual(cost_model.call_count, 4 if accumulation == 1 else 10)
                    for _ in range(accumulation - 1):
                        next(loader)
                    self.assertEqual(len(captured.records), expected_windows)
                    self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 2 * accumulation)
                    next(loader)
                    self.assertEqual(len(captured.records), expected_windows + 1)
                    self.assertIn("[dp-balance] iteration 2 :", captured.records[expected_windows].getMessage())
                    self.assertTrue(all(record.levelno == logging.INFO for record in captured.records))
                    self.assertTrue(all(record.threadName == current_thread().name for record in captured.records))

    def test_resume_tail_and_epoch_logs_preserve_window_numbers(self) -> None:
        """Feature: Report identity across tails and restart.
        Description: Resume after a complete step, consume a short tail and start another epoch.
        Expectation: Window ranges and iteration numbers match consumption without saving log state.
        """
        options = {
            "accumulation": 3, "window": 2, "samples": [{"id": index} for index in range(14)], "log_balance": True,
        }
        with self.assertLogs(logger.name, level="INFO") as captured, _loader(**options) as loader:
            for _ in range(3):
                next(loader)
            state = loader.state_dict()
            self.assertNotIn("balance_logs", state)
            self.assertEqual(len(captured.records), 2)
            expected = [_ids(batch) for batch in loader]
            self.assertEqual(len(captured.records), 5)
            self.assertIn("[dp-balance] iteration 3 :", captured.records[-1].getMessage())
            self.assertIn("micro_batches=1-1", captured.records[-1].getMessage())
            self.assertIsNone(next(loader, None))
            self.assertEqual(len(captured.records), 5)
        with self.assertLogs(logger.name, level="INFO") as captured, _loader(**options) as restored:
            restored.load_state_dict(state)
            self.assertEqual([_ids(batch) for batch in restored], expected)
            self.assertEqual(len(captured.records), 3)
            self.assertIn("[dp-balance] iteration 2 :", captured.records[0].getMessage())
            self.assertIn("micro_batches=3-3", captured.records[1].getMessage())
            restored.set_epoch(1)
            next(restored)
            self.assertIn("[dp-balance] iteration 1 :", captured.records[3].getMessage())
            self.assertIn("epoch=1", captured.records[3].getMessage())

    def test_logs_default_off_even_at_info_level(self) -> None:
        """Feature: Opt-in balancing diagnostics.
        Description: Consume native batches at INFO level without enabling the config switch.
        Expectation: No balancing report is formatted or emitted, and samples are still delivered.
        """
        with self.assertNoLogs(logger.name, level="INFO"), patch(
                "hyper_parallel.distributed_data.distributed_dataloader.format_balance_stats",
        ) as formatter, _loader(accumulation=3) as loader:
            self.assertEqual([_ids(batch) for batch in loader], [list(range(i, i + 2)) for i in range(0, 44, 2)])
            formatter.assert_not_called()

    def test_disabled_info_skips_statistics_formatting(self) -> None:
        """Feature: Standard logging-level control.
        Description: Disable INFO before reading and planning all batches.
        Expectation: Native loading does not format balancing diagnostics.
        """
        with patch.object(logger, "isEnabledFor", return_value=False), patch(
                "hyper_parallel.distributed_data.distributed_dataloader.format_balance_stats",
        ) as formatter, _loader(accumulation=3, log_balance=True) as loader:
            self.assertTrue(list(loader))
            formatter.assert_not_called()

    def test_failed_future_step_does_not_emit_its_planning_logs(self) -> None:
        """Feature: Speculation is not reported as consumption.
        Description: Fail device staging after the successor has already been planned.
        Expectation: Only the current window is logged and the original failure is propagated.
        """
        with self.assertLogs(logger.name, level="INFO") as captured, _loader(
                accumulation=3, log_balance=True,
        ) as loader:
            original = loader._stage_batch

            def stage(batch: object) -> object:
                """Fail staging only after the current window is prepared."""
                if loader._step >= 3:
                    raise ValueError("future staging failed")
                return original(batch)

            with patch.object(loader, "_stage_batch", side_effect=stage):
                for _ in range(3):
                    next(loader)
                with self.assertRaisesRegex(ValueError, "future staging failed"):
                    next(loader)
                self.assertEqual(len(captured.records), 1)
