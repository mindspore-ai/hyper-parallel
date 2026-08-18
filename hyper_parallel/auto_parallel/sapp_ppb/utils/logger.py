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
"""logger for pipeline balance"""
import logging

DEFAULT_STDOUT_FORMAT = '%(levelname)s %(asctime)s %(filename)s:%(lineno)d - %(message)s'
FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_HANDLER_NAME = 'sapp_ppb_stream_handler'


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create a namespaced logger and register the ``output`` convenience level.

    Args:
        name: Logger name (typically a package or module name).
        level: Minimum log level accepted by both the logger and the handler.

    Returns:
        The configured :class:`logging.Logger` instance.
    """
    def output(self: logging.Logger, message: str, *args: object) -> None:
        """Emit ``message`` at the warning level."""
        self.warning(message, *args)

    logging.Logger.output = output
    ppb_logger = logging.getLogger(name)
    ppb_logger.setLevel(level)
    handler = next(
        (item for item in ppb_logger.handlers if item.get_name() == _HANDLER_NAME),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        ppb_logger.addHandler(handler)
    handler.setLevel(level)
    handler.setFormatter(FORMATTER)

    return ppb_logger

logger = setup_logger('sapp_ppb', level=logging.INFO)
