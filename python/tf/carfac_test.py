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

from typing import Optional

import numpy as np
from parameterized import parameterized
import tensorflow as tf

from .. import testing as carfac_testing
from . import carfac


class _TestCallable:

  def __init__(self, cell: carfac.CARFACCell):
    cell.call = tf.function(cell.call)
    self.cell = cell
    self.layer = tf.function(tf.keras.layers.RNN(
        self.cell, return_sequences=True))

  def __call__(self, audio: np.ndarray) -> np.ndarray:
    return self.layer(np.reshape(audio, (1,
                                         audio.shape[0],
                                         audio.shape[1],
                                         1))).numpy()[0]


class CARFACTest(tf.test.TestCase):

  def testConvolvers(self):
    # Verifies that all convolver options produce the same values.
    for kernel_size in [3, 5]:
      previous_values: Optional[tf.Tensor] = None
      kernel = tf.ones((kernel_size,), dtype=tf.float64) * 1.3
      data = (tf.ones((3, 4, 92), dtype=tf.float64) *
              tf.range(92, dtype=tf.float64))
      for convolver in carfac.convolution_methods.values():
        output = convolver(tf.constant(5, dtype=tf.int32), kernel, data)
        if previous_values is None:
          previous_values = output
        else:
          np.testing.assert_allclose(previous_values, output)

  def testRecurrenceExpanders(self):
    # Verifies that all recurrence expanders produce the same values.
    previous_value: Optional[tf.Tensor] = None
    a_0 = (tf.ones((3, 4), dtype=tf.float64) *
           tf.constant(0.51, dtype=tf.float64))
    f = (0.9 + (0.2 / 92) * tf.ones((3, 4, 92), dtype=tf.float64) *
         tf.range(92, dtype=tf.float64))
    g = (0.01 + (2.0 / 92) * tf.ones((3, 4, 92), dtype=tf.float64) *
         tf.range(92, dtype=tf.float64))
    for expander in carfac.recurrence_expansion_methods.values():
      output = expander(a_0, f, g)
      if previous_value is None:
        previous_value = output
      else:
        np.testing.assert_allclose(previous_value, output)

  def testMatchesMatlabWithAGCOff(self):
    carfac_testing.assert_matlab_compatibility(
        'agc_test', _TestCallable(
            carfac.CARFACCell(
                open_loop=True,
                num_ears=2,
                outputs=[carfac.CARFACOutput.BM,
                         carfac.CARFACOutput.NAP],
                car_params=carfac.CARParams(
                    sample_rate_hz=tf.constant(44100.0)))))

  def testMatchesMatlabOnBinauralData(self):
    carfac_testing.assert_matlab_compatibility(
        'binaural_test', _TestCallable(
            carfac.CARFACCell(
                num_ears=2,
                outputs=[carfac.CARFACOutput.BM,
                         carfac.CARFACOutput.NAP],
                car_params=carfac.CARParams(
                    sample_rate_hz=tf.constant(22050.0)))))

  def testMatchesMatlabOnLongBinauralData(self):
    carfac_testing.assert_matlab_compatibility(
        'long_test', _TestCallable(
            carfac.CARFACCell(
                num_ears=2,
                outputs=[carfac.CARFACOutput.BM,
                         carfac.CARFACOutput.NAP],
                car_params=carfac.CARParams(
                    sample_rate_hz=tf.constant(44100.0)))))

  def testMatchesMatlabWithIHCJustHalfWaveRectifyOn(self):
    carfac_testing.assert_matlab_compatibility(
        'ihc_just_hwr_test', _TestCallable(
            carfac.CARFACCell(
                num_ears=2,
                ihc_params=carfac.IHCParams(
                    just_half_wave_rectify=tf.constant(1.0)),
                outputs=[carfac.CARFACOutput.BM,
                         carfac.CARFACOutput.NAP],
                car_params=carfac.CARParams(
                    sample_rate_hz=tf.constant(44100.0)))))

  def testAGCDesignAtLowSampleRate(self):
    for i, spread in enumerate([0.0, 1.0, 1.4, 2.0, 2.8]):
      carfac_cell = carfac.CARFACCell(
          car_params=carfac.CARParams(sample_rate_hz=tf.constant(8000.0)),
          agc_params=carfac.AGCParams(agc1_scales0=tf.constant(1.0 * spread),
                                      agc2_scales0=tf.constant(1.65 * spread)))
      agc_coeffs = carfac_cell._design_agc_coeffs()
      if i == 0:
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_n_taps.numpy() == 3))
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_iterations.numpy() == 0))
      elif i == 1:
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_n_taps.numpy() == 5))
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_iterations.numpy() == 1))
      elif i == 2:
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_n_taps.numpy() == 5))
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_iterations.numpy() == 2))
      elif i == 3:
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_n_taps.numpy() == 5))
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_iterations.numpy() == 4))
      elif i == 4:
        self.assertTrue(
            np.all(
                agc_coeffs.agc_spatial_iterations.numpy() == -1))

  def testAGCDesignBehavesSensibly(self):
    for i, spread in enumerate([0.0, 1.0, 1.4, 2.0, 2.8, 3.5, 4.0]):
      carfac_cell = carfac.CARFACCell(
          car_params=carfac.CARParams(sample_rate_hz=tf.constant(22050.0)),
          agc_params=carfac.AGCParams(agc1_scales0=tf.constant(1.0 * spread),
                                      agc2_scales0=tf.constant(1.65 * spread)))
      agc_coeffs = carfac_cell._design_agc_coeffs()
      if i == 0:
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_n_taps.numpy() == 3))
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_iterations.numpy() == 0))
      elif i == 1:
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_n_taps.numpy() == 3))
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_iterations.numpy() == 1))
      elif i == 2:
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_n_taps.numpy() == 5))
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_iterations.numpy() == 1))
      elif i == 3:
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_n_taps.numpy() == 5))
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_iterations.numpy() == 2))
      elif i == 4:
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_n_taps.numpy() == 5))
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_iterations.numpy() == 3))
      elif i == 5:
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_n_taps.numpy() == 5))
        self.assertTrue(
            np.all(agc_coeffs.agc_spatial_iterations.numpy() == 4))
      elif i == 6:
        self.assertTrue(
            np.all(
                agc_coeffs.agc_spatial_iterations.numpy() == -1))
    pass

  @parameterized.expand((
      (1.0, 1.0, True, tf.float32),
      (1.0, 1.0, False, tf.float32),
      (1.0, 0.0, True, tf.float32),
      (1.0, 0.0, False, tf.float32),
      (0.0, 1.0, True, tf.float32),
      (0.0, 1.0, False, tf.float32),
      (0.0, 0.0, True, tf.float32),
      (0.0, 0.0, False, tf.float32),
      (1.0, 1.0, True, tf.float64),
      (1.0, 1.0, False, tf.float64),
      (1.0, 0.0, True, tf.float64),
      (1.0, 0.0, False, tf.float64),
      (0.0, 1.0, True, tf.float64),
      (0.0, 1.0, False, tf.float64),
      (0.0, 0.0, True, tf.float64),
      (0.0, 0.0, False, tf.float64)))
  def testModes(self,
                just_half_wave_rectify: float,
                one_capacitor: float,
                graph_mode: bool,
                dtype: tf.DType):
    ihc_params = carfac.IHCParams()
    ihc_params.one_capacitor = tf.constant(one_capacitor)
    ihc_params.just_half_wave_rectify = tf.constant(just_half_wave_rectify)
    car_params = carfac.CARParams()
    car_params.erb_per_step = tf.constant(3.0)
    carfac_cell = carfac.CARFACCell(ihc_params=ihc_params,
                                    car_params=car_params,
                                    num_ears=3,
                                    dtype=dtype)
    carfac_layer = tf.keras.layers.RNN(carfac_cell)
    model = tf.keras.Sequential([carfac_layer])
    impulse: np.ndarray = np.zeros([1, 128, 3, 1], dtype=np.float32)
    impulse[:, 0, 0, :] = 1
    def compute(data):
      return model(data)
    o = tf.function(compute)(impulse) if graph_mode else compute(impulse)
    self.assertEqual(o.shape, tf.TensorShape((1, 3, 12, 1)))


if __name__ == '__main__':
  tf.test.main()
