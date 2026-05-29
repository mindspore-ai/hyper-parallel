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
"""Unit tests for SAPP-ND `run_nd` end-to-end pipeline.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_run_nd.py
"""
import os
import tempfile
import unittest
from unittest.mock import patch

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd.common import hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level

WORK_PATH = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(WORK_PATH, "deepseek.yaml")


class TestSappNDRunND(unittest.TestCase):
    """A test class for the SAPP-ND ``run_nd`` end-to-end pipeline."""

    def test_run_nd_deepseek_smoke(self):
        """
        Feature: TestSappNDRunND.
        Description: Run the ND search-space generation and performance-ordering
                     pipeline on the shipped DeepSeek yaml, mirroring what
                     ``run_nd.py`` does at the CLI entry point.
        Expectation: ``run_generation_to_ordering`` returns a non-empty scored
                     space whose entries have the documented
                     ``(parallel_config, mem_mb, perf_score, debug_parts)`` shape,
                     and the entries are monotonically sorted by score.
        """
        self.assertTrue(
            os.path.exists(config_path),
            f"shipped DeepSeek yaml missing: path={config_path!r}",
        )

        top_k = 3

        # Redirect matplotlib config dir to a temp path to avoid polluting $HOME
        # and silence the ND logger so that no debug plot/CSV is emitted into
        # the source tree (see `enable_debug` gating in parallelize.py).
        with tempfile.TemporaryDirectory() as mpl_tmp, \
                patch.dict(os.environ, {"MPLCONFIGDIR": mpl_tmp}):
            set_verbose_level(0)

            machine = Hard.Machine(None, "A2")
            dims = Dim.get_dims(["DP", "MP", "PP", "EP", "MB"])

            runner = Par.Parallelize(
                "mindformers",
                config_path,
                machine,
                global_batch_size=None,
                dimensions=dims,
                swap_os=False,
                mppb=False,
                model=None,
                max_mem=None,
                mem_for_ppb=Memory.from_string("0GB"),
            )

            scored_space = runner.run_generation_to_ordering(
                None,
                threads_num=None,
                top_num=top_k,
                cache_file=None,
            )

        self.assertIsInstance(
            scored_space, list,
            f"scored_space must be list, got type={type(scored_space).__name__}",
        )
        self.assertGreater(
            len(scored_space), 0,
            f"scored_space must be non-empty for DeepSeek yaml, got len={len(scored_space)}",
        )

        top = scored_space[:top_k]
        self.assertGreaterEqual(
            len(top), 1,
            f"expected at least 1 top configuration, got len={len(top)}",
        )

        for idx, entry in enumerate(top):
            self.assertGreaterEqual(
                len(entry), 3,
                (f"scored entry must be a tuple of at least "
                 f"(config, mem, score, ...): idx={idx}, entry={entry!r}"),
            )
            mem_mb = entry[1]
            perf_score = entry[2]
            self.assertGreater(
                mem_mb, 0,
                f"entry[{idx}].mem_mb must be > 0, got mem_mb={mem_mb}",
            )
            self.assertGreater(
                perf_score, 0,
                f"entry[{idx}].perf_score must be > 0, got perf_score={perf_score}",
            )

        scores = [entry[2] for entry in top]
        self.assertTrue(
            scores == sorted(scores) or scores == sorted(scores, reverse=True),
            (f"top-{top_k} scored configurations are not monotonically "
             f"sorted by score: scores={scores}"),
        )
