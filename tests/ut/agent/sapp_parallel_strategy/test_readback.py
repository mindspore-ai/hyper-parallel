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
"""UT for the sapp-parallel-strategy's read-back step: rounds set beside each other, the pick, and ND's order."""
import tempfile
import unittest
from pathlib import Path

import readback
import timing

TIMER = {"varies": ["EP", "OP"]}
HEADER = "tag,ep,op,repeat,status,steps,mean_ms,min_ms,max_ms,peak_alloc_gb,peak_reserved_gb,seconds,log"


def _run(ep, op, mean_ms, ok=True):
    """A launch of a round."""
    return timing.Run(key=(str(ep), str(op)), repeat="1", ok=ok, mean_ms=mean_ms, peak_gib=36.0,
                      reserved_gib=38.0, status="ok" if ok else "failed_1", source="round")


def _measured(ep, op, *means):
    """A layout measured once per mean given."""
    return readback.Measured(key=(str(ep), str(op)), runs=[_run(ep, op, mean) for mean in means])


class PooledTest(unittest.TestCase):
    """When a later round is set beside the earlier ones."""

    def setUp(self) -> None:
        """The earlier round: EP1 OP1 and EP2 OP1, three runs each."""
        self.earlier = (Path("earlier"), [_run(1, 1, 8076.0), _run(1, 1, 8115.0), _run(1, 1, 8077.0),
                                          _run(2, 1, 7638.0), _run(2, 1, 7786.0), _run(2, 1, 7698.0)])

    def test_round_whose_shared_layout_overlaps_joins(self):
        """
        Feature: setting rounds beside each other.
        Description: the later round times EP1 OP1 again within its earlier range, and a new layout.
        Expectation: every launch of both rounds is read together.
        """
        later = (Path("later"), [_run(1, 1, 8100.0), _run(4, 1, 7600.0)])
        runs, notes = readback.pooled([self.earlier, later], TIMER)
        self.assertEqual(len(runs), 8, f"launches read: expected=8, got={len(runs)}")
        self.assertIn("set beside", notes[-1], f"note says the round joined: got={notes[-1]}")

    def test_round_whose_shared_layout_moved_is_read_alone(self):
        """
        Feature: setting rounds beside each other.
        Description: the later round runs EP1 OP1 slower than every earlier run of it.
        Expectation: only the later round is read, and the note says why.
        """
        later = (Path("later"), [_run(1, 1, 8500.0), _run(4, 1, 7600.0)])
        runs, notes = readback.pooled([self.earlier, later], TIMER)
        self.assertEqual(len(runs), 2, f"launches read: expected=2, got={len(runs)}")
        self.assertIn("moved", notes[-1], f"note says the node or code moved: got={notes[-1]}")

    def test_round_sharing_no_layout_is_read_alone(self):
        """
        Feature: setting rounds beside each other.
        Description: the later round times only layouts the earlier one did not.
        Expectation: only the later round is read.
        """
        later = (Path("later"), [_run(4, 1, 7600.0), _run(8, 1, 7700.0)])
        runs, _ = readback.pooled([self.earlier, later], TIMER)
        got = {run.key for run in runs}
        self.assertEqual(got, {("4", "1"), ("8", "1")}, f"layouts read: expected EP4 OP1 and EP8 OP1, got={got}")


class VerdictTest(unittest.TestCase):
    """Whether one more round could change the pick."""

    def test_firm_when_the_slowest_run_beats_every_other_fastest(self):
        """
        Feature: the pick's verdict.
        Description: the 8 October round's two fastest layouts.
        Expectation: firm, by the gap between EP2 OP1's slowest run and EP16 OP16's fastest, 172.0 ms.
        """
        best = _measured(2, 1, 7638.2, 7785.9, 7697.8)
        rival = _measured(16, 16, 7957.9, 8036.8, 7975.8)
        got = readback.verdict(best, [rival], TIMER)
        self.assertIn("by 172.0 ms (EP16 OP16)", got, f"verdict: got={got}")

    def test_not_separated_when_runs_overlap(self):
        """
        Feature: the pick's verdict.
        Description: the second layout's fastest run beats the pick's slowest.
        Expectation: not separated, naming the second layout.
        """
        got = readback.verdict(_measured(2, 1, 7600.0, 7800.0), [_measured(4, 1, 7700.0, 7900.0)], TIMER)
        self.assertTrue(got.startswith("Not separated") and "EP4 OP1" in got, f"verdict: got={got}")

    def test_a_layout_run_once_is_not_called(self):
        """
        Feature: the pick's verdict.
        Description: the fastest layout ran once.
        Expectation: it asks for another run before launching on it.
        """
        got = readback.verdict(_measured(2, 1, 7600.0), [_measured(4, 1, 9000.0, 9100.0)], TIMER)
        self.assertTrue(got.startswith("It ran once"), f"verdict: got={got}")


class OrderTest(unittest.TestCase):
    """ND's order scored against the measured one."""

    def test_spearman_of_the_same_and_the_reverse_order(self):
        """
        Feature: the rank correlation the read-back reports.
        Description: an order against itself and against its reverse.
        Expectation: +1 and -1.
        """
        same = readback.spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
        reverse = readback.spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
        self.assertEqual((same, reverse), (1.0, -1.0), f"correlations: expected=(1.0, -1.0), got={(same, reverse)}")

    def test_measure_attaches_nd_rows_and_keeps_failed_launches_apart(self):
        """
        Feature: one entry a measured layout, with ND's row.
        Description: two launches of EP1 OP1, one failed, and ND's ranking holding it second.
        Expectation: one finished run, one failed run, ND's place 2.
        """
        rows = [{"EP": "2", "OP": "1"}, {"EP": "1", "OP": "1"}]
        items = readback.measure([_run(1, 1, 8000.0), _run(1, 1, 0.0, ok=False)], rows, rows, TIMER)
        got = [(len(item.runs), len(item.failed), item.nd_place) for item in items]
        self.assertEqual(got, [(1, 1, 2)], f"measured: expected=[(1, 1, 2)], got={got}")


class ReadRoundTest(unittest.TestCase):
    """A round as its timer wrote it."""

    def test_reads_finished_and_failed_launches(self):
        """
        Feature: reading a round's runs.csv.
        Description: one finished and one failed launch.
        Expectation: both read, keyed by EP and OP, with their status and peaks.
        """
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "runs.csv").write_text(
                f"{HEADER}\nep2_op1_r1,2,1,1,ok,4,7638.203,0,0,36.880,39.639,82,x\n"
                "ep4_op1_r1,4,1,1,failed_1,0,0,0,0,0,0,30,x\n", encoding="utf-8")
            runs = timing.read_round(Path(folder), TIMER["varies"])
        got = [(run.key, run.ok, run.mean_ms, run.peak_gib) for run in runs]
        expected = [(("2", "1"), True, 7638.203, 36.88), (("4", "1"), False, 0.0, 0.0)]
        self.assertEqual(got, expected, f"runs: expected={expected}, got={got}")


if __name__ == "__main__":
    unittest.main()
