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
"""Error types and input validation helpers for SAPP-PPB."""
from typing import Union

Number = Union[int, float]


class SAPPError(ValueError):
    """Raised when SAPP-PPB detects an invalid input or configuration."""


def assert_sapp(test: bool, msg: str) -> None:
    """Raise :class:`SAPPError` with ``msg`` when ``test`` is false.

    Args:
        test: Condition that must hold.
        msg: Human-readable error message attached to the raised exception.

    Raises:
        SAPPError: If ``test`` is ``False``.
    """
    if not test:
        raise SAPPError(msg)


def check_in_bounds(n: Number, n_desc: str, lower_bound: Number, higher_bound: Number) -> None:
    """Check that ``n`` lies in the inclusive range ``[lower_bound, higher_bound]``.

    Args:
        n: The value being validated.
        n_desc: Short description of ``n`` used in the error message.
        lower_bound: Lower bound (inclusive).
        higher_bound: Upper bound (inclusive).

    Raises:
        SAPPError: If ``n`` falls outside ``[lower_bound, higher_bound]``.
    """
    assert_sapp(n >= lower_bound,
                f"{n_desc} {n} should be higher than {lower_bound}")
    assert_sapp(n <= higher_bound,
                f"{n_desc} {n} should be lower than {higher_bound}")
