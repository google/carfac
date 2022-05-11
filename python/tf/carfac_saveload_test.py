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

import tempfile
from absl.testing import absltest
import numpy as np
import tensorflow as tf

from . import carfac


class CARFACSaveloadTest(absltest.TestCase):

  def testSaveLoad(self):
    carfac_cell = carfac.CARFACCell(num_ears=4)
    carfac_layer = tf.keras.layers.RNN(carfac_cell, return_sequences=True,
                                       dtype=tf.float32)
    model = tf.keras.Sequential([carfac_layer])
    impulse: np.ndarray = np.zeros([3, 10, 4, 1], dtype=np.float32)
    impulse[:, 0, :, :] = 1
    model(impulse)
    with tempfile.TemporaryDirectory() as savefile:
      model.save(savefile)
      loaded_model = tf.keras.models.load_model(
          savefile, custom_objects={'CARFACCell': carfac.CARFACCell})
      np.testing.assert_array_almost_equal(model(impulse),
                                           loaded_model(impulse))


if __name__ == '__main__':
  absltest.main()
