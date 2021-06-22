# Lint as: python3
#!/usr/bin/env python

# Copyright 2021 The CARFAC Authors. All Rights Reserved.
#
# This file is part of an implementation of Lyon's cochlear model:
# "Cascade of Asymmetric Resonators with Fast-Acting Compression"
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for carfac.tf.pz."""

import unittest
from absl import app
import numpy as np
import tensorflow as tf

from . import car


class CARTest(unittest.TestCase):

  def testNonlinear(self):
    for dtype in [tf.float32, tf.float64]:
      car_cell = car.CARCell(linear=False, erb_per_step=6, dtype=dtype)
      car_layer = tf.keras.layers.RNN(car_cell,
                                      return_sequences=True,
                                      dtype=dtype)
      impulse: np.ndarray = np.zeros([1, 4, 1], dtype=np.float64)
      impulse[:, 0, :] = 1
      output: tf.Tensor = car_layer(tf.cast(impulse, dtype=dtype))
      # These numbers are from a prior run of this code, and haven't been
      # verified with more than a visual inspection (see "CARFAC channels
      # frequency response" in visualizations.ipynb) of the frequency response
      # outputs of same code.
      expected_output = [[[0.9483489394187927, 0.5269975066184998,
                           0.26684099435806274, 0.1336478292942047,
                           0.06692484766244888],
                          [0.1864028424024582, 0.43243372440338135,
                           0.24397090077400208, 0.1234208270907402,
                           0.06182999163866043],
                          [-0.3168385624885559, 0.2738230526447296,
                           0.2063443660736084, 0.10690627992153168,
                           0.053610894829034805],
                          [0.3689427077770233, 0.30905500054359436,
                           0.27259621024131775, 0.14332109689712524,
                           0.07191740721464157]]]

      # Only testing up to 4 decimals since float32 and float64 produce slightly
      # different output.
      np.testing.assert_array_almost_equal(output,
                                           expected_output,
                                           decimal=4)

  def testLinear(self):
    for dtype in [tf.float32, tf.float64]:
      car_cell = car.CARCell(linear=True, erb_per_step=6, dtype=dtype)
      car_layer = tf.keras.layers.RNN(car_cell,
                                      return_sequences=True,
                                      dtype=dtype)
      impulse: np.ndarray = np.zeros([1, 4, 1], dtype=np.float64)
      impulse[:, 0, :] = 1
      output: tf.Tensor = car_layer(tf.cast(impulse, dtype=dtype))
      # These numbers are from a prior run of this code, and haven't been
      # verified with more than a visual inspection (see "CARFAC channels
      # frequency response" in visualizations.ipynb) of the frequency response
      # outputs of same code.
      expected_output = [[[0.9483489394187927, 0.5269975066184998,
                           0.26684099435806274, 0.1336478292942047,
                           0.06692484766244888],
                          [0.18639203906059265, 0.4324178695678711,
                           0.2439626008272171, 0.12341666966676712,
                           0.06182790920138359],
                          [-0.3167409896850586, 0.27375346422195435,
                           0.20630639791488647, 0.10688719898462296,
                           0.05360133945941925],
                          [0.36875051259994507, 0.3089130222797394,
                           0.2725130617618561, 0.14327912032604218,
                           0.07189638167619705]]]

      # Only testing up to 4 decimals since float32 and float64 produce slightly
      # different output.
      np.testing.assert_array_almost_equal(output,
                                           expected_output,
                                           decimal=4)

  def testTFFunction(self):
    car_cell = car.CARCell(linear=False, erb_per_step=6)
    car_layer = tf.keras.layers.RNN(car_cell,
                                    return_sequences=True)
    impulse: np.ndarray = np.zeros([1, 4, 1], dtype=np.float32)
    impulse[:, 0, :] = 1

    @tf.function
    def compute(inputs):
      return car_layer(inputs)

    output = compute(impulse)
    # These numbers are from a prior run of this code, and haven't been verified
    # with more than a visual inspection of the frequency response outputs of
    # same code.
    expected_output = [[[0.9483488202095032,
                         0.526997447013855, 0.26684102416038513,
                         0.1336478441953659, 0.06692485511302948],
                        [0.18640300631523132, 0.4324337840080261,
                         0.24397099018096924, 0.12342087179422379,
                         0.06183001399040222],
                        [-0.31683874130249023, 0.27382296323776245,
                         0.2063443809747696, 0.10690628737211227,
                         0.053610898554325104],
                        [0.36894288659095764, 0.30905500054359436,
                         0.27259624004364014, 0.14332111179828644,
                         0.07191741466522217]]]

    np.testing.assert_array_almost_equal(output, expected_output)

  def testGradients(self):
    car_cell = car.CARCell(linear=False, erb_per_step=6)
    car_layer = tf.keras.layers.RNN(car_cell, return_sequences=True)
    impulse: np.ndarray = np.zeros([1, 4, 1], dtype=np.float32)
    impulse[:, 0, :] = 1
    tape = tf.GradientTape(persistent=True)
    with tape:
      output: tf.Tensor = car_layer(impulse)
    for tfvar in [
        car_cell._high_f_damping_compression,
        car_cell._zero_ratio,
        car_cell._min_zeta,
        car_cell._max_zeta,
        car_cell._erb_break_freq,
        car_cell._erb_q,
        car_cell._v_offset,
        car_cell._velocity_scale,
    ]:
      self.assertIsNotNone(tape.gradient(output, tfvar))


def main(_):
  unittest.main()

if __name__ == '__main__':
  app.run(main)
