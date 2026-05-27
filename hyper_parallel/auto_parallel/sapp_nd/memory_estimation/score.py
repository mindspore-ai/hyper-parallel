# Copyright 2025 Huawei Technologies Co., Ltd
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
"""test score functions"""
import numpy as np


def mape(pred, real):
    """Mean absolute percentage error.

    Args:
        pred (list): Predicted values.
        real (list): Real values.

    Returns:
        float: Mean absolute percentage error, or None if no valid pair exists.
    """
    valid_pairs = [(p, r) for p, r in zip(pred, real) if p > 0 and r > 0]
    if not valid_pairs:
        return None
    return 100 / len(valid_pairs) * sum(
        abs((r - p) / r) for p, r in valid_pairs
    )


def r2(pred, real):
    """Coefficient of determination.

    Args:
        pred (list): Predicted values.
        real (list): Real values.

    Returns:
        float: Coefficient of determination, or None if unavailable.
    """
    pairs = list(zip(pred, real))
    if len(pairs) < 2:
        return None
    real_values = [r for _, r in pairs]
    m = np.mean(real_values)
    denominator = sum((r - m) ** 2 for _, r in pairs)
    if denominator == 0:
        return None
    return 1 - sum((r - p) ** 2 for p, r in pairs) / denominator
