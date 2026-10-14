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
"""UT for the sapp-parallel-strategy's launch step: which layouts a round times, and what each timed run reads."""
import json
import unittest

import yaml

import ask
import launch
import timing

TIMER = {
    "script": "examples/x/repeat.sh",
    "preflight": "examples/x/preflight.sh",
    "launcher": "examples/x/launcher.yaml",
    "dies": 16,
    "varies": ["EP", "OP"],
    "fixed": {"MP": "1", "CP": "1", "PP": "1", "VPP": "1"},
    "sets": ["accelerator.ep_size", "fsdp_config.dp_shard_size", "fsdp_config.edp_shard_size",
             "training.train_iters", "profiling.enabled"],
    "node_paths": ["model.config_path"],
    "train_iters": 6,
    "seconds_a_run": 84,
    "repeats": 3,
    "model_glob": "*deepseek*",
}


def _row(ep, op, score, dp="16", cp="1", mbs="1"):
    """A row of ND's ranking with the degrees the tests vary."""
    return {"DP": dp, "MP": "1", "CP": cp, "PP": "1", "VPP": "1", "EP": str(ep), "OP": str(op), "MB": "1",
            "MBS": mbs, "SP": "False", "recompute": "off", "score": str(score), "memory_mb": "38000"}


def _run(ep, op, mean_ms):
    """A finished launch of a round."""
    return timing.Run(key=(str(ep), str(op)), repeat="1", ok=True, mean_ms=mean_ms, peak_gib=36.0,
                      reserved_gib=38.0, status="ok", source="round")


def _key(row):
    """The layout as the timer names it."""
    return timing.key_of(row, TIMER)


class WorthTimingTest(unittest.TestCase):
    """Which layouts a round times, by the trust record's rule."""

    def test_never_checked_times_every_untimed_layout_in_nd_order(self):
        """
        Feature: the launch step's rule where ND's order was never checked.
        Description: four launchable layouts, one of them timed before.
        Expectation: the three others, in ND's order, each "never timed".
        """
        pool = [_row(1, 1, 1.0), _row(2, 1, 2.0), _row(2, 2, 3.0), _row(4, 1, 4.0)]
        chosen = launch.worth_timing(pool, pool, {"region": None}, 0.79, {("2", "1")}, _key)
        got = [(_key(row), why) for row, why in chosen]
        expected = [(("1", "1"), launch.NEVER_TIMED), (("2", "2"), launch.NEVER_TIMED),
                    (("4", "1"), launch.NEVER_TIMED)]
        self.assertEqual(got, expected, f"chosen layouts: expected={expected}, got={got}")

    def test_checked_region_times_the_launch_its_ties_and_nd_pick_outside(self):
        """
        Feature: the launch step's rule where ND's order was checked inside a region.
        Description: ND's pick lies at CP 2, outside the region; inside, one layout sits within the spread of the
            best and one beyond it.
        Expectation: the best inside, the one within the spread, then ND's own pick.
        """
        outside = _row(2, 32, 100.0, dp="32", cp="2")
        best, close, far = _row(1, 64, 110.0), _row(1, 32, 110.5), _row(2, 64, 120.0)
        rows = [outside, best, close, far]
        trust = {"region": {"CP": ["1"]}}
        chosen = launch.worth_timing(rows, rows, trust, 0.79, set(), lambda row: tuple(row.values()))
        got = [(row["CP"], row["EP"], row["OP"], why) for row, why in chosen]
        expected = [("1", "1", "64", "the layout to launch"), ("1", "1", "32", "within 0.79% of it by ND"),
                    ("2", "2", "32", "ND's own pick, outside the checked region")]
        self.assertEqual(got, expected, f"chosen layouts: expected={expected}, got={got}")


class AnchorsTest(unittest.TestCase):
    """The layouts timed before that a new round times again first."""

    def test_nd_best_of_the_timed_then_the_fastest_measured(self):
        """
        Feature: the anchors of a new round.
        Description: ND ranks EP1 OP1 first of the timed layouts; EP2 OP1 measured fastest.
        Expectation: EP1 OP1 first, then EP2 OP1; a layout never timed is not an anchor.
        """
        pool = [_row(1, 1, 1.0), _row(1, 2, 2.0), _row(2, 1, 3.0), _row(4, 4, 4.0)]
        runs = [_run(1, 1, 8089.0), _run(1, 2, 8061.0), _run(2, 1, 7707.0)]
        got = [_key(row) for row, _ in launch.anchors(pool, runs, _key)]
        expected = [("1", "1"), ("2", "1")]
        self.assertEqual(got, expected, f"anchors: expected={expected}, got={got}")

    def test_one_anchor_when_nd_best_is_also_the_fastest(self):
        """
        Feature: the anchors of a new round.
        Description: the layout ND ranks first of the timed ones also measured fastest.
        Expectation: it is the only anchor.
        """
        pool = [_row(1, 1, 1.0), _row(1, 2, 2.0)]
        got = [_key(row) for row, _ in launch.anchors(pool, [_run(1, 1, 7000.0), _run(1, 2, 8000.0)], _key)]
        self.assertEqual(got, [("1", "1")], f"anchors: expected=[('1', '1')], got={got}")


class LauncherTest(unittest.TestCase):
    """What a timed run reads, and when the timer would run another job than the one priced."""

    def setUp(self) -> None:
        """A priced yaml and the launcher it was made from."""
        self.launcher = {"model": {"config_path": "/path/to/model", "layers": 40},
                         "accelerator": {"ep_size": 16}, "fsdp_config": {"dp_shard_size": 16},
                         "training": {"train_iters": 1, "global_batch_size": 16, "micro_batch_size": 1},
                         "dataset": {"data_transform": {"max_seq_len": 4096}}}
        self.doc = {"model": {"config_path": "/laptop/model", "layers": 40},
                    "accelerator": {"ep_size": 1}, "fsdp_config": {"dp_shard_size": 1},
                    "training": {"train_iters": 1, "global_batch_size": 16, "micro_batch_size": 1},
                    "dataset": {"data_transform": {"max_seq_len": 4096}},
                    "context": {"device_num": 16, "expert_shard": "group"}}

    def test_no_difference_beyond_what_the_timer_sets(self):
        """
        Feature: the check that a round times the job ND priced.
        Description: the yamls differ only in fields the timer sets for each run, in node paths and in context.
        Expectation: no difference.
        """
        got = launch.launcher_differences(self.doc, TIMER, self.launcher)
        self.assertEqual(got, [], f"differences: expected=[], got={got}")

    def test_a_priced_field_the_launcher_states_otherwise_is_a_difference(self):
        """
        Feature: the check that a round times the job ND priced.
        Description: ND priced 8192 tokens a sequence and the launcher states 4096.
        Expectation: one difference, naming the field.
        """
        self.doc["dataset"]["data_transform"]["max_seq_len"] = 8192
        got = launch.launcher_differences(self.doc, TIMER, self.launcher)
        self.assertEqual(len(got), 1, f"differences: expected one, got={got}")
        self.assertTrue(got[0].startswith("dataset.data_transform.max_seq_len"),
                        f"difference names the field: got={got[0]}")

    def test_timed_yaml_states_the_layout_as_the_timer_launches_it(self):
        """
        Feature: the trainer yaml of one timed run.
        Description: EP4 OP2 on 16 dies with the expert shard over its group.
        Expectation: EP 4, shard width 2, expert shard 16 / 4, the timer's steps, the profiler off, the
            launcher's node path, and no context.
        """
        got = launch.timed_yaml(self.doc, _row(4, 2, 1.0), TIMER, self.launcher)
        facts = {"ep": got["accelerator"]["ep_size"], "op": got["fsdp_config"]["dp_shard_size"],
                 "edp": got["fsdp_config"]["edp_shard_size"], "iters": got["training"]["train_iters"],
                 "profiler": got["profiling"]["enabled"], "config": got["model"]["config_path"],
                 "context": "context" in got}
        expected = {"ep": 4, "op": 2, "edp": 4, "iters": 6, "profiler": False, "config": "/path/to/model",
                    "context": False}
        self.assertEqual(facts, expected, f"timed yaml: expected={expected}, got={facts}")

    def test_command_names_the_points_and_the_batch(self):
        """
        Feature: the one command that times a round.
        Description: two layouts, three repeats, the run's batch.
        Expectation: the preflight runs before the timer, which runs detached with the points, repeats, batch and
            run root, its log beside the run root.
        """
        lines = launch.command_lines(TIMER, "1:1 2:4", 3, "output/rounds/r", self.doc)
        text = "\n".join(lines)
        for part in ('POINTS="1:1 2:4"', "REPEATS=3", "GLOBAL_BATCH=16", "MICRO_BATCH=1", "nohup env",
                     "RUN_ROOT=output/rounds/r", 'bash examples/x/repeat.sh "$M" > output/rounds/r.log 2>&1 &',
                     "mkdir -p output/rounds"):
            self.assertIn(part, text, f"command lacks {part}: got={text}")
        self.assertLess(text.index("preflight.sh"), text.index("repeat.sh"),
                        f"the preflight runs first: got={text}")


class LaunchableTest(unittest.TestCase):
    """The rows of ND's ranking a timer can launch."""

    def test_only_rows_with_the_timer_degrees_and_the_run_micro_batch(self):
        """
        Feature: the layouts a timer launches.
        Description: a CP 2 row, a micro-batch 2 row and two rows the timer launches.
        Expectation: the two, in ND's order.
        """
        rows = [_row(1, 1, 1.0, dp="8", cp="2"), _row(1, 2, 2.0), _row(2, 1, 3.0, mbs="2"), _row(2, 2, 4.0)]
        doc = {"training": {"micro_batch_size": 1}, "activation_checkpoint": {"mode": "off"}}
        got = [_key(row) for row in timing.launchable(rows, TIMER, doc)]
        self.assertEqual(got, [("1", "2"), ("2", "2")], f"launchable: expected=[('1', '2'), ('2', '2')], got={got}")


class MissingScriptsTest(unittest.TestCase):
    """A checkout without the round's own scripts must say so, not hand over a command that fails."""

    def test_a_checkout_without_the_scripts_is_named(self) -> None:
        """
        Feature: the check that this checkout can run the round it is about to print.
        Description: a timer naming a script and a preflight no checkout holds.
        Expectation: both are reported, by path.
        """
        problems = launch.missing_scripts(TIMER)
        self.assertEqual(len(problems), 2, problems)
        self.assertIn(TIMER["script"], problems[0])
        self.assertIn(TIMER["preflight"], problems[1])

    def test_the_shipped_timers_name_scripts_this_checkout_has(self) -> None:
        """
        Feature: the same check against the timer records as shipped.
        Description: every recorded timer names a script and a preflight in this checkout.
        Expectation: nothing missing, so a round's command is written.
        """
        timers = json.loads((ask.SKILL / "references" / "timers.json").read_text(encoding="utf-8"))
        for name, timer in timers.items():
            if name.startswith("_"):
                continue
            self.assertEqual(launch.missing_scripts(timer), [], name)


class ShippedLauncherTest(unittest.TestCase):
    """Every template with a timer must ship a launcher that states the job the cost model prices."""

    def test_each_timer_launcher_states_what_nd_prices(self) -> None:
        """
        Feature: the launcher a timer reads, as the checkout ships it.
        Description: read each timer record, its template and the launcher it names, and compare them
                     the way the launch step does, ignoring what the timer sets per run.
        Expectation: no difference, so the round times the job that was priced and a command is written.
        """
        skill = ask.SKILL
        timers = json.loads((skill / "references" / "timers.json").read_text(encoding="utf-8"))
        checked = 0
        for name, timer in timers.items():
            if name.startswith("_"):
                continue
            family = name.split("/")[0]
            template = skill / "templates" / f"{family}.yaml"
            launcher = ask.repo_root() / timer["launcher"]
            if not template.is_file() or not launcher.is_file():
                continue
            priced = yaml.safe_load(template.read_text(encoding="utf-8"))
            states = yaml.safe_load(launcher.read_text(encoding="utf-8"))
            self.assertEqual(launch.launcher_differences(priced, timer, states), [],
                             f"{timer['launcher']} does not state what {template.name} prices")
            checked += 1
        self.assertGreater(checked, 0, "no timer record could be checked in this checkout")


if __name__ == "__main__":
    unittest.main()
