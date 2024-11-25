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

import pathlib
import tempfile
import numpy as np
import tensorflow as tf

from . import carfac


class CARFACGoldenDataTest(tf.test.TestCase):

  def testEagerModeAgainstGoldenData(self):
    # This tests most of CARFAC (it doesn't run with all possible filter or IHC
    # configurations etc, but it runs a complete default CARFAC) against golden
    # data produced after the TF code was verified to behave exactly like the
    # C++ code with the same configuration.
    # It's the simplest way to verify that refactoring the code doesn't change
    # the numerical behaviour, but beware that it doesn't test _everything_.
    carfac_cell = carfac.CARFACCell(
        num_ears=3,
        convolver=carfac.conv1d_convolver,
        recurrence_expander=carfac.recurrence_relation_recurrence_expansion)
    carfac_layer = tf.keras.layers.RNN(carfac_cell, return_sequences=True,
                                       dtype=tf.float32)
    model = tf.keras.Sequential([carfac_layer])
    impulse = np.zeros([1, 512, 3, 1], dtype=np.float32)
    impulse[:, 0, 0, :] = 1
    impulse[:, 10, 1, :] = 1
    impulse[:, 20, 2, :] = 1
    output = model(impulse).numpy()
    output_file = (pathlib.Path(tempfile.gettempdir()) /
                   'tf_carfac_golden_data_output.npz')
    np.savez(output_file, data=output[:, :, :, :, 0])
    print(f'Golden data test saved produced output in {output_file}')
   golden = np.load(pathlib.Path(__file__).parent / 'golden_data.npz')['data']
    np.testing.assert_allclose(golden, output[:, :, :, :, 0], atol=6e-5)


if __name__ == '__main__':
  tf.test.main()
