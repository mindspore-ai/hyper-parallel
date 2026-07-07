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
"""Logging callback configuration tests."""
from types import SimpleNamespace

from hyper_parallel.trainer.callbacks.base import LoggingCallback


def _trainer(args):
    return SimpleNamespace(args=args, lr_scheduler=None)


def _args_with_train_logging(logging_cfg):
    train = SimpleNamespace(
        logging=logging_cfg,
        global_batch_size=4,
    )
    data = SimpleNamespace(max_seq_len=128)
    return SimpleNamespace(train=train, data=data)


def test_logging_callback_reads_nested_train_logging():
    """
    Feature: nested logging config
    Description: LoggingCallback reads logging settings from train.logging.
    Expectation: per-step logging config is applied from the nested config.
    """
    logging_cfg = SimpleNamespace(
        log_steps=1,
        report_global_loss=True,
        report_throughput=False,
        model_flops_per_token=None,
        peak_tflops=None,
    )

    callback = LoggingCallback(_trainer(_args_with_train_logging(logging_cfg)))

    assert callback.log_steps == 1
    assert callback.report_global_loss is True
    assert callback.report_throughput is False


def test_logging_callback_keeps_top_level_fallback():
    """
    Feature: logging config fallback
    Description: Legacy top-level logging config remains supported.
    Expectation: callback uses args.logging when train.logging is absent.
    """
    logging_cfg = SimpleNamespace(
        log_steps=5,
        report_global_loss=False,
        report_throughput=True,
        model_flops_per_token=None,
        peak_tflops=None,
    )
    train = SimpleNamespace(global_batch_size=4)
    data = SimpleNamespace(max_seq_len=128)
    args = SimpleNamespace(train=train, data=data, logging=logging_cfg)

    callback = LoggingCallback(_trainer(args))

    assert callback.log_steps == 5
    assert callback.report_global_loss is False
    assert callback.report_throughput is True
