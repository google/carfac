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

from absl.testing import absltest
import numpy as np
import tensorflow as tf

from . import carfac


class CARFACGradientTest(absltest.TestCase):

  def testGradients(self):
    # Execute one step of the CARFAC implementation and make sure that all the
    # parameters show up on the gradient tape.
    ihc_params = carfac.IHCParams()
    ihc_params.one_capacitor = tf.constant(False)
    car_params = carfac.CARParams()
    car_params.erb_per_step = tf.constant(3.0)
    carfac_cell = carfac.CARFACCell(ihc_params=ihc_params,
                                    car_params=car_params,
                                    num_ears=1)
    carfac_cell.call = tf.function(carfac_cell.call)
    carfac_layer = tf.keras.layers.RNN(carfac_cell)
    impulse: np.ndarray = np.zeros([1, 64, 1, 1], dtype=np.float32)
    impulse[:, 0, 0, :] = 1
    tape = tf.GradientTape(persistent=True)
    with tape:
      output = carfac_layer(impulse)
    for tfvar in [
        carfac_cell._car_params.velocity_scale,
        carfac_cell._car_params.v_offset,
        carfac_cell._car_params.min_zeta_at_half_erb_per_step,
        carfac_cell._car_params.max_zeta_at_half_erb_per_step,
        carfac_cell._car_params.zero_ratio,
        carfac_cell._car_params.high_f_damping_compression,
        carfac_cell._car_params.erb_break_freq,
        carfac_cell._car_params.erb_q,
        carfac_cell._ihc_params.tau_lpf,
        carfac_cell._ihc_params.tau1_out,
        carfac_cell._ihc_params.tau1_in,
        carfac_cell._ihc_params.tau2_out,
        carfac_cell._ihc_params.tau2_in,
        carfac_cell._ihc_params.ac_corner_hz,
        carfac_cell._agc_params.agc_stage_gain,
        carfac_cell._agc_params.agc_mix_coeff,
        carfac_cell._agc_params.time_constants0,
        carfac_cell._agc_params.time_constants_mul,
        carfac_cell._agc_params.agc1_scales0,
        carfac_cell._agc_params.agc1_scales_mul,
        carfac_cell._agc_params.agc2_scales0,
        carfac_cell._agc_params.agc2_scales_mul,
    ]:
      self.assertIsNotNone(tape.gradient(output, tfvar),
                           f'No gradient to {tfvar}')


if __name__ == '__main__':
  absltest.main()
