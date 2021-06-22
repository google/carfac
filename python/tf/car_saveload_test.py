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

import tempfile
import unittest
from absl import app
import numpy as np
import tensorflow as tf

from . import car


class CARTest(unittest.TestCase):

  def testSaveLoad(self):
    car_cell = car.CARCell()
    car_layer = tf.keras.layers.RNN(car_cell, return_sequences=True)
    model = tf.keras.Sequential()
    model.add(car_layer)
    impulse: np.ndarray = np.zeros([3, 10, 1], dtype=np.float32)
    impulse[:, 0, :] = 1
    impulse: tf.Tensor = tf.constant(impulse)
    model.build(impulse.shape)
    with tempfile.TemporaryDirectory() as savefile:
      model.save(savefile)
      loaded_model: tf.keras.models.Model = tf.keras.models.load_model(
          savefile, custom_objects={'CARCell': car.CARCell})
      np.testing.assert_array_almost_equal(model(impulse),
                                           loaded_model(impulse))


def main(_):
  unittest.main()

if __name__ == '__main__':
  app.run(main)
