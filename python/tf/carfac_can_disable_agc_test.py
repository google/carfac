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
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for carfac.python.tf.carfac."""

import numpy as np
import tensorflow as tf

from . import carfac


class CARFACCanDisableAGCTest(tf.test.TestCase):

  def testCanDisableAGC(self):
    # Design two different carfac objects, one normal and one with the AGC
    # turned off. For a 2s long sinusoid input make sure that the AGC output
    # changes over time, and the version without the AGC does NOT change.
    sample_rate = 8000.0
    num_samples = int(8000)
    sound_data = np.reshape(
        np.sin(np.linspace(0, 2 * sample_rate * np.pi, num_samples)),
        (1, num_samples, 1, 1))
    carfac_cell_with_agc = carfac.CARFACCell(
        num_ears=1,
        car_params=carfac.CARParams(sample_rate_hz=tf.constant(sample_rate)),
        outputs=(carfac.CARFACOutput.AGC,))
    carfac_cell_with_agc.call = tf.function(carfac_cell_with_agc.call)
    layer_with_agc = tf.keras.layers.RNN(carfac_cell_with_agc,
                                         return_sequences=True)
    layer_with_agc = tf.function(layer_with_agc)
    agc_enabled_output = layer_with_agc(sound_data).numpy()

    carfac_cell_without_agc = carfac.CARFACCell(
        num_ears=1,
        # This is the way the corresponding C++ test turns off AGC.
        # Setting open_loop=True should have the same effect.
        agc_params=carfac.AGCParams(decimation=tf.zeros([], tf.float32)),
        car_params=carfac.CARParams(sample_rate_hz=tf.constant(sample_rate)),
        outputs=(carfac.CARFACOutput.AGC,))
    carfac_cell_without_agc.call = tf.function(carfac_cell_without_agc.call)
    layer_without_agc = tf.keras.layers.RNN(carfac_cell_without_agc,
                                            return_sequences=True)
    layer_without_agc = tf.function(layer_without_agc)
    agc_disabled_output = layer_without_agc(sound_data).numpy()

    self.assertEqual(agc_enabled_output.shape, agc_disabled_output.shape,
                     'AGC enabled output shape != AGC disabled output shape')
    self.assertTrue(np.any(agc_enabled_output != agc_disabled_output),
                    'AGC enabled output == AGC disabled output')
    for channel in range(agc_disabled_output.shape[3]):
      # With the AGC disabled, the agc output for a given channel
      # should be identical at all times.
      self.assertTrue(np.all(agc_disabled_output[0, :, 0, channel, 0] ==
                             agc_disabled_output[0, 0, 0, channel, 0]),
                      f'AGC disabled output for chan {channel} is not constant')
      self.assertTrue(np.any(agc_enabled_output[0, :, 0, channel, 0] !=
                             agc_enabled_output[0, 0, 0, channel, 0]),
                      f'AGC enabled output for chan {channel} is constant')


if __name__ == '__main__':
  tf.test.main()
