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
"""Config loading must use the current Transformers dtype argument.

Transformers >= 5 renamed the config-loading ``torch_dtype`` argument to
``dtype`` and keeps the old spelling only as a deprecated alias that logs a
deprecation line on the first load of a process. ``get_hf_config`` feeds every
``from_pretrained`` call, so that alias printed on every bring-up run. These
tests pin the argument name rather than the log line, so they do not depend on
the logger's once-per-process cache.
"""
import unittest
from unittest.mock import patch

from hyper_parallel.models._transformers import config_resolver
from tests.common.mark_utils import arg_mark


class TestConfigDtypeArgument(unittest.TestCase):
    """``get_hf_config`` forwards the dtype under the argument name HF reads."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_explicit_dtype_uses_current_argument_name(self):
        """Feature: dtype injection into the loaded config.

        Description: Load a config with an explicit non-auto dtype.
        Expectation: ``dtype`` is forwarded and the deprecated alias is not.
        """
        captured = {}

        def _from_pretrained(path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs
            return "config-sentinel"

        with patch.object(config_resolver.AutoConfig, "from_pretrained", staticmethod(_from_pretrained)):
            config = config_resolver.get_hf_config("/weights", "sdpa", "bfloat16")

        self.assertEqual(config, "config-sentinel")
        self.assertEqual(captured["path"], "/weights")
        self.assertEqual(captured["kwargs"]["dtype"], "bfloat16")
        self.assertNotIn("torch_dtype", captured["kwargs"])
        self.assertEqual(captured["kwargs"]["attn_implementation"], "sdpa")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_auto_dtype_forwards_no_dtype_override(self):
        """Feature: "auto" keeps the checkpoint's own dtype.

        Description: Load a config without a dtype override.
        Expectation: Neither ``dtype`` nor the deprecated alias is forwarded.
        """
        captured = {}

        def _from_pretrained(_path, **kwargs):
            captured["kwargs"] = kwargs
            return "config-sentinel"

        with patch.object(config_resolver.AutoConfig, "from_pretrained", staticmethod(_from_pretrained)):
            config_resolver.get_hf_config("/weights")

        self.assertNotIn("dtype", captured["kwargs"])
        self.assertNotIn("torch_dtype", captured["kwargs"])
        self.assertEqual(captured["kwargs"]["attn_implementation"], "sdpa")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_caller_dtype_kwarg_wins(self):
        """Feature: caller overrides are not replaced.

        Description: Pass an explicit ``dtype`` next to the positional one.
        Expectation: The ``**kwargs`` value survives the ``setdefault``.
        """
        captured = {}

        def _from_pretrained(_path, **kwargs):
            captured["kwargs"] = kwargs
            return "config-sentinel"

        with patch.object(config_resolver.AutoConfig, "from_pretrained", staticmethod(_from_pretrained)):
            config_resolver.get_hf_config("/weights", "sdpa", "bfloat16", dtype="float16")

        self.assertEqual(captured["kwargs"]["dtype"], "float16")


if __name__ == "__main__":
    unittest.main()
