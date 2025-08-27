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

import time
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from carfac.tf import carfac


class CARFACPerformanceTest(tf.test.TestCase):

  def testPerformance(self):
    # Not functional tests. Prints run time in eager and graph mode.
    # Runs a test input through different combinations of construction arguments
    # and prints the run time in eager and graph mode for each.
    #
    # Some times measured on 2022-04-14 are:
    #
    ### Graph mode: True, expander: TensorArray
    ### Build + run: 27.07395839691162s => 1.0
    ### Run: 1.2907299995422363s => 1.0
    ### Graph mode: True, expander: RecurrenceRelation
    ### Build + run: 20.0101580619812s => 0.739092443322355
    ### Run: 0.8090441226959229s => 0.6268112796501625
    ### Graph mode: True, expander: MatMul
    ### Build + run: 21.251548528671265s => 0.7849442706942872
    ### Run: 0.7959232330322266s => 0.6166457999074203
    #
    ### Graph mode: False, expander: TensorArray
    ### Build + run: 20.33016085624695s => 1.0
    ### Run: 20.33016085624695s => 1.0
    ### Graph mode: False, expander: RecurrenceRelation
    ### Build + run: 15.327446460723877s => 0.7539264725499768
    ### Run: 15.327446699142456s => 0.7539264842773103
    ### Graph mode: False, expander: MatMul
    ### Build + run: 18.29426860809326s => 0.8998585273107611
    ### Run: 18.29426884651184s => 0.8998585390380948
    #
    ### Graph mode: True, convolver: conv1d
    ### Build + run: 19.960277795791626s => 0.9880280505903286
    ### Run: 0.8232896327972412s => 1.0
    ### Graph mode: True, convolver: concat_add
    ### Build + run: 20.202136754989624s => 1.0
    ### Run: 0.8157484531402588s => 0.9908401863007066
    #
    ### Graph mode: False, convolver: conv1d
    ### Build + run: 15.202692985534668s => 0.9623748907069896
    ### Run: 15.202693700790405s => 0.9623748924105708
    ### Graph mode: False, convolver: concat_add
    ### Build + run: 15.797059059143066s => 1.0
    ### Run: 15.797059774398804s => 1.0
    def run(
        expander: carfac.RecurrenceExpansionCallable,
        convolver: carfac.ConvolverCallable,
        graph_mode: bool,
    ) -> Tuple[float, float]:
      ihc_params = carfac.IHCParams()
      car_params = carfac.CARParams()
      car_params.erb_per_step = tf.constant(3.0)
      carfac_cell = carfac.CARFACCell(
          ihc_params=ihc_params,
          car_params=car_params,
          num_ears=3,
          recurrence_expander=expander,
          convolver=convolver,
      )
      carfac_layer = tf.keras.layers.RNN(carfac_cell)
      model = tf.keras.Sequential([carfac_layer])
      impulse: np.ndarray = np.zeros([1, 128, 3, 1], dtype=np.float32)
      impulse[:, 0, 0, :] = 1

      @tf.function
      def graph_compute(data):
        return model(data)

      if graph_mode:
        total_start = time.time()
        o = graph_compute(impulse)
        total_done = time.time()
        run_start = time.time()
        o = graph_compute(impulse)
        run_done = time.time()
      else:
        total_start = time.time()
        run_start = time.time()
        o = model(impulse)
        total_done = time.time()
        run_done = time.time()
      self.assertEqual(o.shape, tf.TensorShape((1, 3, 12, 1)))
      return (total_done - total_start, run_done - run_start)

    for graph_mode in [True, False]:
      total_times: Dict[str, float] = {}
      run_times: Dict[str, float] = {}
      for name, expander in carfac.recurrence_expansion_methods.items():
        total_t, run_t = run(
            expander=expander,
            graph_mode=graph_mode,
            convolver=carfac.conv1d_convolver,
        )
        total_times[name] = total_t
        run_times[name] = run_t
      total_m = np.max(list(total_times.values()))
      run_m = np.max(list(run_times.values()))
      for name in carfac.recurrence_expansion_methods.keys():
        desc = f'Graph mode: {graph_mode}, expander: {name}'
        print(f'### {desc}')
        print(
            f'### Build + run: {total_times[name]}s =>'
            f' {total_times[name] / total_m}'
        )
        print(f'### Run: {run_times[name]}s => {run_times[name] / run_m}')
    for graph_mode in [True, False]:
      total_times: Dict[str, float] = {}
      run_times: Dict[str, float] = {}
      for name, convolver in carfac.convolution_methods.items():
        total_t, run_t = run(
            expander=carfac.recurrence_relation_recurrence_expansion,
            graph_mode=graph_mode,
            convolver=convolver,
        )
        total_times[name] = total_t
        run_times[name] = run_t
      total_m = np.max(list(total_times.values()))
      run_m = np.max(list(run_times.values()))
      for name in carfac.convolution_methods.keys():
        desc = f'Graph mode: {graph_mode}, convolver: {name}'
        print(f'### {desc}')
        print(
            f'### Build + run: {total_times[name]}s =>'
            f' {total_times[name] / total_m}'
        )
        print(f'### Run: {run_times[name]}s => {run_times[name] / run_m}')


if __name__ == '__main__':
  tf.test.main()
