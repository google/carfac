# Copyright 2013 The CARFAC Authors. All Rights Reserved.
#
# This file is part of an implementation of Lyon's cochlear model:
# "Cascade of Asymmetric Resonators with Fast-Acting Compression"
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test utilities for Python CARFAC."""


import importlib.resources
from typing import Callable

import numpy as np

_TEST_DATA_PACKAGE = 'carfac.test_data'


def _read_test_file(full_name: str) -> np.ndarray:
  """Reads a text file written using the Matlab dlmwrite function.

  Args:
    full_name: The name of the test file to load.

  Returns:
    The content of the file as an np.ndarray.
  """
  data = (
      importlib.resources.files(_TEST_DATA_PACKAGE)
      .joinpath(full_name)
      .read_bytes()
  )
  res = []
  for row in data.decode('utf-8').split('\n'):
    if not row.strip():
      continue
    res.append([float(x) for x in row.split(' ')])
  return np.array(res)


def assert_matlab_compatibility(
    test_name: str,
    fun: Callable[[np.ndarray], np.ndarray],
    rtol: float = 0,
    atol: float = 7e-3,
):
  """Asserts that the provided function conforms to the Matlab CARFAC.

  Use this function to verify that the output from a (CARFAC) function returns
  the same (or close) results as those precomputed (by the Matlab version).

  Args:
    test_name: The name of the test data to verify against.
    fun: A function that takes [n_steps, 2] where the last dimension is
      [left channel, right channel] as input and returns [2, n_channels, 2]
      where the first dimension is [left ear, right ear] and the last dimension
      is [BM output, NAP output].
    rtol: Relative tolerance of differences between precomputed Matlab results
      and the tested function. Defaults to 7e-3 just like the C++ tests does for
      float32 testing.
    atol: Absolute tolerarance of differences between precomputed Matlab results
      and the tested function. Defaults to 0 just like the C++ tests does for
      float32 testing.
  """  # fmt: skip
  audio = _read_test_file(f'{test_name}-audio.txt')
  nap1 = _read_test_file(f'{test_name}-matlab-nap1.txt')
  nap2 = _read_test_file(f'{test_name}-matlab-nap2.txt')
  bm1 = _read_test_file(f'{test_name}-matlab-bm1.txt')
  bm2 = _read_test_file(f'{test_name}-matlab-bm2.txt')
  output = fun(audio)
  np.testing.assert_allclose(output[:, 0, :, 0], bm1, rtol=rtol, atol=atol)
  np.testing.assert_allclose(output[:, 0, :, 1], nap1, rtol=rtol, atol=atol)
  np.testing.assert_allclose(output[:, 1, :, 0], bm2, rtol=rtol, atol=atol)
  np.testing.assert_allclose(output[:, 1, :, 1], nap2, rtol=rtol, atol=atol)
