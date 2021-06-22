# Lint as: python3
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

"""TF Keras Layers to compute output from CARFAC models.

This file is part of an implementation of Lyon's cochlear model:
"Cascade of Asymmetric Resonators with Fast-Acting Compression"

See "Human and Machine Hearing at http://dicklyon.com/hmh/ for more details.
"""

from typing import Dict, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import pz


StatesTuple = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


class CARCell(tf.keras.layers.Layer):
  """A CAR cell for a tf.keras.layers.RNN layer.

  Computes output samples for a set of cochlear places given an input sample.

  It implements the cascade of asymmetric resonators, the core of CARFAC.
  See Chapter 16 of Lyon's book Human and Machine Hearing
  (http://dicklyon.com/hmh/). The poles and zeros are coupled, but the
  pole locations and their width are trainable parameters.
  (The IHC, OHC and AGC are not part of this implementation.)

  The expected use case is to wrap it inside a tf.keras.layers.RNN  and then
  feed audio sample sequences to the layers.

  See also the GitHub repository https://github.com/google/carfac/.

  Attributes:
    output_size: Shape of output. Required by
      https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN.

    state_size: Shape of state. Required by
      https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN.
  """

  def __init__(self, sample_rate_hz: float = 48000.0, erb_per_step: float = 0.5,
               linear: bool = False, max_frequency: float = 20400.0,
               min_frequency: float = 30.0, **kwargs):
    """Initializes a CAR cell.

    Args:
      sample_rate_hz: Sample rate for the input samples.
      erb_per_step: Number of ERBs (see
        https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth)
        between each cochlear place.
      linear: Whether the CAR cell should incorporate nonlinearities or
        not.
      max_frequency: The pole of the highest frequency channel.
      min_frequency: Guaranteed lower than than any channel pole frequency.
      **kwargs: Forwarded to superclass.
    """
    super().__init__(**kwargs)
    self._erb_per_step = erb_per_step
    self._sample_rate_hz = sample_rate_hz
    self._linear = linear
    self._max_freq = max_frequency
    self._min_freq = min_frequency

    def compute_zeta(zeta_at_default_erb_per_step: float) -> float:
      """Computes a reasonable zeta value for a given erb_per_step.

      Based on the assumtion that max small-signal gain at the passband peak
      will be on the order of (0.5/min_zeta)**(1/erb_per_step), and we need
      the start value of that in the same region or the loss function becomes
      too uneven to optimize.

      Args:
        zeta_at_default_erb_per_step: Which zeta this should correspond to at
          default value for erb_per_step.

      Returns:
        The corresponding zeta for the actually used erb_per_step.
      """
      default_erb_per_step = 0.5
      max_small_signal_gain: float = (
          (0.5 / zeta_at_default_erb_per_step) ** (1 / default_erb_per_step))
      return 0.5 / (max_small_signal_gain ** self._erb_per_step)

    # Controls r (pole and zero abs value) which controls damping relative to
    # frequency.
    self._high_f_damping_compression: tf.Variable = self._add_weight(
        'high_f_damping_compression', 0.5)
    # Controls distance from pole to zero.
    self._zero_ratio: tf.Variable = self._add_weight(
        'zero_ratio', 2.0 ** 0.5)
    # min/max zeta controls max damping.
    self._min_zeta: tf.Variable = self._add_weight('min_zeta',
                                                   compute_zeta(0.1))
    self._max_zeta: tf.Variable = self._add_weight('max_zeta',
                                                   compute_zeta(0.35))
    # Controls how we convert from Hz to Cams.
    # The Greenwood map's break frequency in Hertz.
    self._erb_break_freq: tf.Variable = self._add_weight('erb_break_freq',
                                                         165.3)
    # Glassberg and Moore's high-cf ratio
    self._erb_q: tf.Variable = self._add_weight('erb_q', 1000.0 / (24.7 * 4.37))
    # v_offset and velocity_scale controls the nonlinearity in the fast acting
    # compression described in chapter 17 of the book.
    self._v_offset: tf.Variable = self._add_weight('v_offset', 0.04)
    self._velocity_scale: tf.Variable = self._add_weight('velocity_scale', 0.1)

    # This loop over the pole frequencies is done early here in __init__ to
    # ensure that we know the state_size and output_size fields post-init.
    curr_freq: float = self._max_freq
    self._n_poles = 0
    while curr_freq > self._min_freq:
      self._n_poles += 1
      curr_freq -= self._erb_per_step * self.erb(curr_freq)

    self.output_size: int = self._n_poles
    # [u^-1, v^-1, v^-2]
    self.state_size: Tuple[int, int, int] = (
        self.output_size,
        self.output_size,
        self.output_size)

  def _add_weight(self, name: str, value: float) -> tf.Variable:
    return self.add_weight(
        name=name,
        dtype=self.dtype,
        initializer=tf.keras.initializers.Constant(value))

  def get_config(self) -> Dict[str, float]:
    """Required by superclass for serialization.

    See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

    Returns:
      A dictionary with parameters for creating an identical instance.
    """
    return {
        'sample_rate_hz': self._sample_rate_hz,
        'erb_per_step': self._erb_per_step,
        'linear': self._linear,
    }

  def erb(self, f: Union[int, float, tf.Tensor]) -> tf.Tensor:
    """Calculates the equivalent rectangular bandwidth at a given frequency.

    Args:
      f: Float. Frequency in Hz to calculate ERB at.

    Returns:
      Float. The equivalent rectangular bandwidth at f.
    """
    return (self._erb_break_freq + f) / self._erb_q

  def call(self,
           input_at_t: tf.Tensor,
           states_at_t: StatesTuple) -> Tuple[
               tf.Tensor,
               StatesTuple]:
    """Computes output_at_t given input_at_t and states_at_t.

    Args:
      input_at_t: A [batch_size, 1]-tensor with input at this step.
      states_at_t: A tuple (prev_u, prev_v, prev_prev_v) with state tensors at
        this step, where each element is a [batch_size]-complex128-tensor.

    Returns:
      A tuple (output_amplitudes, states), where `output_amplitudes` is a
        [batch_size, n_channels]-complex128-tensor, and `states` is a tuple
        (u, v, prev_v) with state tensors at the next step, where each element
        is a [batch_size]-tensor.
    """

    # This loop over the pole frequencies is done to let AutoGraph track their
    # gradients, which wouldn't have happened if it was done in __init__.
    pole_freqs_ta: tf.TensorArray = tf.TensorArray(size=self._n_poles,
                                                   dtype=self.dtype)
    curr_freq: float = self._max_freq
    for channel_number in range(self._n_poles):
      pole_freqs_ta = pole_freqs_ta.write(channel_number, curr_freq)
      curr_freq -= self._erb_per_step * self.erb(curr_freq)
    pole_freqs: tf.Tensor = pole_freqs_ta.stack()

    # Quoted from ../matlab/CARFAC_Design.m:
    # zero_ratio comes in via h.  In book's circuit D, zero_ratio is 1/sqrt(a),
    # and that a is here 1 / (1+f) where h = f*c.
    # solve for f:  1/zero_ratio^2 = 1 / (1+f)
    # zero_ratio^2 = 1+f => f = zero_ratio^2 - 1
    # We are casting zero_ratio to float64 to achieve numerical equivalence with
    # the C++ version.
    f: tf.Tensor = tf.math.square(self._zero_ratio) - 1
    x: tf.Tensor = pole_freqs * 2 / self._sample_rate_hz
    pole_thetas: tf.Tensor = x * np.pi

    # The book assigns a0 and c0 thus to simplify the equations.
    a0: tf.Tensor = tf.math.cos(pole_thetas)
    c0: tf.Tensor = tf.math.sin(pole_thetas)

    # Quoted from ../matlab/CARFAC_Design.m:
    # When high_f_damping_compression is 0 this is just theta, when
    # high_f_damping_compression is 1 it approaches 0 as theta approaches pi.
    # Also, see picture 17.1 in the book, where zr_coeffs1 * NLF is added
    # to r.
    zr_coeffs1: tf.Tensor = np.pi * (x - self._high_f_damping_compression *
                                     tf.math.pow(x, 3))

    # Here the base value for r (r1), and the initial value for the fast-acting
    # compression (the zr_coeffs1) are initialized.
    # The book is not super easy to follow here, so I have mostly
    # implemented the same math as the matlab and c++ code.
    r1: tf.Tensor = 1 - zr_coeffs1 * self._max_zeta
    min_zetas: tf.Tensor = (self._min_zeta +
                            (0.25 * ((self.erb(pole_freqs) / pole_freqs) -
                                     self._min_zeta)))
    zr_coeffs: tf.Tensor = zr_coeffs1 * (self._max_zeta - min_zetas)

    # These are the previous state contributions, which are the delays in the
    # circuit at picture 17.1.
    # prev_prev_v is called zA_memory in the matlab code, but it is a delay of
    # prev_v.
    prev_u, prev_v, prev_prev_v = states_at_t

    r1_plus_zr_coeffs: tf.Tensor = (r1 + zr_coeffs)[tf.newaxis, :]
    h: tf.Tensor = c0 * f
    g0: tf.Tensor = (1 -
                     2 * r1_plus_zr_coeffs * a0 +
                     tf.math.square(r1_plus_zr_coeffs)) / (
                         1 -
                         2 * r1_plus_zr_coeffs * a0 +
                         h * r1_plus_zr_coeffs * c0 +
                         tf.math.square(r1_plus_zr_coeffs))

    # Velocity is the velocity of the pressure differential.
    velocity: tf.Tensor = prev_v - prev_prev_v
    # NLF is the nonlinearity doing the fast acting compression.
    nlf_out: tf.Tensor = 1.0 / (1.0 +
                                tf.math.square(velocity *
                                               self._velocity_scale +
                                               self._v_offset))

    r: tf.Tensor = r1
    # The undampening is controlled via the NLF.
    if self._linear:
      r = (r + zr_coeffs)[tf.newaxis, :]
    else:
      r += zr_coeffs * nlf_out

    # Following is based on matlab code and figures 16.1 and 17.1 from the book.
    # In figure 17.1 I have used the names U and V in the same way as in figure
    # 16.1. In the C++ code, partial_u is called z1_memory and v is called
    # z2_memory at this point.

    r_mul_prev_u: tf.Tensor = r * prev_u
    r_mul_prev_v: tf.Tensor = r * prev_v

    # This doesn't include the input X
    partial_u: tf.Tensor = a0 * r_mul_prev_u - c0 * r_mul_prev_v
    # V doesn't yet include the input, it will be added in the for loop below.
    v: tf.Tensor = c0 * r_mul_prev_u + a0 * r_mul_prev_v

    # This doesn't include the input X
    partial_y: tf.Tensor = h * v
    # Here is the ripple through the channels, where each channel depends on the
    # previous one.
    u_builder: tf.TensorArray = tf.TensorArray(size=self._n_poles,
                                               dtype=self.dtype)
    output_at_t_builder: tf.TensorArray = tf.TensorArray(size=self._n_poles,
                                                         dtype=self.dtype)
    in_out: tf.Tensor = input_at_t[:, 0]
    for ch in range(self._n_poles):
      # Add this in_out to partial_u to get actual u (in the shape of
      # u_builder).
      u_builder = u_builder.write(ch, partial_u[:, ch] + in_out)
      # Add this input to partial_output to get actual output.
      in_out = tf.math.real(g0[:, ch]) * (in_out + partial_y[:, ch])
      # Save this actual output.
      output_at_t_builder = output_at_t_builder.write(ch, in_out)

    u: tf.Tensor = u_builder.stack()
    u: tf.Tensor = tf.transpose(u, [1, 0])
    states_at_t_plus_1: Tuple[tf.Tensor, tf.Tensor, tf.Tensor] = (u, v, prev_v)
    output_at_t: tf.Tensor = output_at_t_builder.stack()
    output_at_t: tf.Tensor = tf.transpose(output_at_t, [1, 0])
    return output_at_t, states_at_t_plus_1


def plot_car_channels(car_cell: CARCell,
                      window_size: int = 2048,
                      frequency_log_scale: bool = True) -> plt.Figure:
  """Plots the frequency response of the output channels of a CARCell.

  Args:
    car_cell: A CARCell to plot the output of.
    window_size: The window size for the frequency domain conversion.
    frequency_log_scale: Whether to plot the frequency axis in log scale.
  Returns:
    A matplotlib.Figure.
  """
  car_layer: tf.keras.layers.RNN = tf.keras.layers.RNN(car_cell,
                                                       return_sequences=True)
  @tf.function
  def call_car_layer(arg: tf.Tensor) -> tf.Tensor:
    return car_layer(arg)
  impulse: np.ndarray = np.zeros([1, window_size, 1], dtype=car_cell.dtype)
  impulse[:, 0, :] = 1
  got: tf.Tensor = call_car_layer(impulse)
  got = tf.transpose(got, [0, 2, 1])[0]
  return pz.plot_z(np.fft.fft(got), frequency_log_scale=frequency_log_scale)


