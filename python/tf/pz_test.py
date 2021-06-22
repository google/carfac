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

from typing import Callable
import unittest
from absl import app
import numpy as np
import tensorflow as tf

from . import pz


class CoeffTest(unittest.TestCase):

  def testCoeffs(self):
    # We have a filter H, with poles P and zeros Q:
    # H = g * np.prod(Q - z) / np.prod(P - z)
    # Assuming Q = [1, 2, 3, 4, 5]:
    # H = g * (1 - z) * (2 - z) * (3 - z) * (4 - z) * (5 - z) / np.prod(P - z)
    #   = Y / X
    # Y = X * g * (1 - z) * (2 - z) * (3 - z) * (4 - z) * (5 - z) /
    #   np.prod(P - z)
    # Y = X * g * (z^-1 - 1) * (2 * z^-1 - 1) * (3 * z^-1 - 1) * (4 * z^-1 - 1)
    #   * (5 * z^-1 - 1) / (np.prod(P - z) * z^-5)
    # Y * np.prod(P - z) * z^-5 = X * (z^-1 - 1) * (2 * z^-1 - 1) *
    #   (3 * z^-1 - 1) * (4 * z^-1 - 1) * (5 * z^-1 - 1)
    # Y * np.prod(P - z) * z^-5 = X * (-1 + 15 * z^-1 - 85 * z^-2 + 225 * z^-3
    #   - 274 * z^-4 + 120 * z^-5)
    # Where (-1 + 15 * z^-1 - 85 * z^-2 + 225 * z^-3 - 274 * z^-4 + 120 * z^-5)
    #   = -(qc0 + qc1 * z^-1 + qc2 * z^-2 + qc3 * z^-3 + qc4 * z^-4 + qc5 *
    #   z^-5)
    # And coeffs_from_zeros returns [qc0, qc1, qc2, qc3, qc4, qc5] =>
    # [1, -15, 85, -225, 274, -120]
    inputs: tf.Tensor = tf.constant([1, 2, 3, 4, 5], dtype=tf.complex128)
    outputs: tf.Tensor = pz.coeffs_from_zeros(inputs)
    expected_outputs = [1, -15, 85, -225, 274, -120]
    np.testing.assert_array_almost_equal(outputs, expected_outputs)


class PZTest(unittest.TestCase):

  def assert_impulse_response(self,
                              filt: Callable[[tf.Tensor],
                                             tf.Tensor],
                              dtype: tf.DType,
                              gain: tf.Tensor,
                              poles: tf.Tensor,
                              zeros: tf.Tensor):
    window_size = 64
    impulse: np.ndarray = np.zeros([window_size], dtype=np.float32)
    impulse[0] = 1
    impulse_spectrum: np.ndarray = np.fft.fft(impulse)

    z: np.ndarray = np.exp(np.linspace(0,
                                       2 * np.pi,
                                       window_size,
                                       endpoint=False) * 1j)
    transfer_function: np.ndarray = (
        tf.cast(gain, tf.complex128) *
        np.prod(zeros[None, :] - z[:, None],
                axis=1) /
        np.prod(poles[None, :] - z[:, None],
                axis=1))

    expected_impulse_response: np.ndarray = np.fft.ifft(
        impulse_spectrum * transfer_function)

    # Since the filter requires batch and cell i/o dimensions.
    impulse_response = filt(tf.cast(impulse[None, :, None], dtype))[0, :, 0]

    np.testing.assert_array_almost_equal(impulse_response,
                                         expected_impulse_response)

  def testPZCell(self):
    for dtype in [tf.float32, tf.float64]:
      poles: np.ndarray = 0.5 * np.exp([np.pi * 0.5j])
      poles: tf.Tensor = tf.concat([poles, tf.math.conj(poles)], axis=0)
      zeros: np.ndarray = 0.75 * np.exp([np.pi * 0.25j])
      zeros: tf.Tensor = tf.concat([zeros, tf.math.conj(zeros)], axis=0)
      gain: tf.Tensor = tf.constant(1.5)
      pz_cell = pz.PZCell(gain,
                          poles,
                          zeros,
                          dtype=dtype)
      pz_layer = tf.keras.layers.RNN(pz_cell,
                                     return_sequences=True,
                                     dtype=dtype)
      self.assert_impulse_response(pz_layer, dtype, gain, poles, zeros)

  def testTFFunction(self):
    for dtype in [tf.float32, tf.float64]:
      poles: np.ndarray = 0.1 * np.exp(np.pi * np.array([0.7j]))
      poles: tf.Tensor = tf.concat([poles, tf.math.conj(poles)], axis=0)
      zeros: np.ndarray = 0.75 * np.exp(np.pi * np.array([0.25j]))
      zeros: tf.Tensor = tf.concat([zeros, tf.math.conj(zeros)], axis=0)
      gain: tf.Tensor = tf.constant(2.4)
      pz_cell = pz.PZCell(gain,
                          poles,
                          zeros,
                          dtype=dtype)
      pz_layer = tf.keras.layers.RNN(pz_cell,
                                     return_sequences=True,
                                     dtype=dtype)
      @tf.function
      def compute(inputs):
        # pylint: disable=cell-var-from-loop
        return pz_layer(inputs)

      self.assert_impulse_response(compute, dtype, gain, poles, zeros)

  def testGradients(self):
    tape = tf.GradientTape(persistent=True)
    pz_cell = pz.PZCell(1,
                        0.5 * np.exp([np.pi * 0.2j, np.pi * 0.5j]),
                        0.3 * np.exp([np.pi * 0.6j]))
    with tape:
      current: tf.Tensor = tf.ones([2, 1], dtype=pz_cell.dtype)
      state = tuple(tf.zeros(shape=[current.shape[0], size],
                             dtype=pz_cell.dtype)
                    for size in pz_cell.state_size)
      for _ in range(6):
        current, state = pz_cell.call(current, state)
    for v in [pz_cell.poles, pz_cell.zeros, pz_cell.gain]:
      self.assertTrue(np.isfinite(tape.gradient(current, v)).all())


def main(_):
  unittest.main()

if __name__ == '__main__':
  app.run(main)
