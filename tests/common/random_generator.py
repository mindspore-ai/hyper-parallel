# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""generate numpy random input"""
import numpy as np


def _generate_numpy_random_input(shape, dtype, name='', method='', *, low=None, high=None, seed=None):
    """
    Description: Generate numpy.ndarray with random seed.
    Args:
        shape(tuple[int]): The shape of numpy.ndarray.
        dtype(np.dtype):   The dtype of numpy.ndarray.
        name(str):         The name of numpy.ndarray will be used in test cases, eg: 'input', 'grad'.
        method(str):       Use np.random.'method' to generate numpy.ndarray.
        low(number):       Lowest number.
        high(number):      Highest number.
        seed(number):      Set random seed, will be used fo debugging.
    Output:
        numpy.ndarray
    """
    if seed is None:
        seed = get_numpy_global_seed()
    np.random.seed(seed)
    print(f"random_generator: generate a numpy.ndarray(shape={shape}, dtype={dtype}, seed={seed}) by "
          f"numpy.random.{method}, will be used as {name}")
    if method == "randn":
        return np.random.randn(*shape).astype(dtype)
    if method == "randint":
        return np.random.randint(low, high, shape, dtype=dtype)
    if method == "uniform":
        return np.random.uniform(low, high, shape).astype(dtype)
    raise TypeError(f"numpy.random.{method} is not impl.")


def get_numpy_global_seed():
    return 1967515154


def set_numpy_global_seed():
    np.random.seed(get_numpy_global_seed())


def get_numpy_random_seed():
    ii32 = np.iinfo(np.int32)
    seed = np.random.randint(0, ii32.max, size=1).astype(np.int32)
    return seed


def set_numpy_random_seed(seed=None):
    if seed is None:
        set_numpy_global_seed()
    else:
        print(f"random_generator: set random seed={seed} for numpy.random.method.")
        np.random.seed(seed)


def generate_numpy_ndarray_by_randn(shape, dtype, name='', seed=None):
    return _generate_numpy_random_input(shape, dtype, name, "randn", seed=seed)


def generate_numpy_ndarray_by_randint(shape, dtype, low, high=None, name='', seed=None):
    return _generate_numpy_random_input(shape, dtype, name, "randint", low=low, high=high, seed=seed)


def generate_numpy_ndarray_by_uniform(shape, dtype, low=0.0, high=1.0, name='', seed=None):
    return _generate_numpy_random_input(shape, dtype, name, "uniform", low=low, high=high, seed=seed)
