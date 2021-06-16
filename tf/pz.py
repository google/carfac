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

"""Classes to compute pole-zero filter output in the time domain."""

import itertools
from typing import Dict, Tuple, Union
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# TODO(zond): replace this with an equivalent from np.typing when the
# internal numpy version is updated.
# https://github.com/numpy/numpy/blob/main/numpy/typing/_array_like.py
ArrayLike = Union[np.ndarray, tf.Tensor]

FloatLike = Union[float, tf.Tensor]


def plot_z(z: ArrayLike,
           sample_rate: float = 48000,
           frequency_log_scale: bool = True) -> plt.Figure:
  """Plots a number of transfer functions in dB FS on a log frequency scale.

  Args:
    z: [num_transfer_functions, num_samples]-complex array of FFT coefficients.
    sample_rate: The sample rate used when generating the window.
    frequency_log_scale: Whether the frequency axis of the plot should be in log
      scale.
  Returns:
    A matplotlib.Figure.
  """
  num_transfer_functions, num_samples = z.shape
  xaxis: np.ndarray = np.tile(np.linspace(0,
                                          (num_samples - 1) * sample_rate *
                                          0.5 / num_samples,
                                          num_samples),
                              [num_transfer_functions, 1])
  fig, ax = plt.subplots()
  if frequency_log_scale:
    ax.set_xscale('log')
  ax.set_xlim((10, 20000))
  ax.set_ylim((-20, 70))
  x: np.ndarray = (xaxis[:, :xaxis.shape[1]//2]).T
  y: np.ndarray = (20 * np.log10(1e-20+np.abs(z[:, :z.shape[1]//2]))).T
  ax.plot(x, y)
  return fig


def plot_pz(poles: ArrayLike,
            zeros: ArrayLike,
            figsize: Tuple[float, float] = (4.4, 4)) -> plt.Figure:
  """Creates a pole/zero plot.

  Args:
    poles: [num_poles]-complex array of poles to plot.
    zeros: [num-zeros]-complex array of zeros to plot.
    figsize: (width, height)-int tuple with the matplotlib figure size to use
      when plotting.
  Returns:
    A matplotlib.Figure.
  """
  fig, ax = plt.subplots(figsize=figsize)
  ax.add_patch(patches.Circle((0, 0),
                              radius=1,
                              fill=False,
                              color='black',
                              ls='solid',
                              alpha=0.1))
  ax.axvline(0, color='0.7')
  ax.axhline(0, color='0.7')
  ax.set_xlim((-1.1, 1.1))
  ax.set_ylim((-1.1, 1.1))

  ax.plot(tf.reshape(tf.math.real(poles), [-1, 1]),
          tf.reshape(tf.math.imag(poles), [-1, 1]),
          'x', markersize=9, alpha=0.5)
  ax.plot(tf.reshape(tf.math.real(zeros), [-1, 1]),
          tf.reshape(tf.math.imag(zeros), [-1, 1]),
          'o', color='none', markeredgecolor='red',
          markersize=9, alpha=0.5)
  return fig


def coeffs_from_zeros(
    polynomial_zeros: Union[tf.Tensor, tf.Variable]) -> tf.Tensor:
  """Computes the coefficients of a polynomial, given the zeroes.

    Assuming we have a filter H, with poles P and zeros Q:
    H = g * np.prod(Q - z) / np.prod(P - z) = Y / X

    Assuming Q and P are [5]-tensors:
    Y / X = g * np.prod(Q * z^-1 - 1) / np.prod(P * z^-1 - 1)
    Y = X * g * np.prod(Q * z^-1 - 1) / np.prod(P * z^-1 - 1)
    Y * np.prod(P * z^-1 - 1) = X * g * np.prod(Q * z^-1 - 1)
    Y * sum(Pc[num] * z^-num for num in range(len(P)+1)) =
      X * g * sum(Qc[num] * z^-num for num in range(len(Q)+1))

    coeffs_from_zeros computes Qc/Pc from Q/P.

  Args:
    polynomial_zeros: [num_zeros]-complex tensor with the zeros to convert to
      coefficients.

  Returns:
    [num_zeros + 1]-complex tensor with the coefficients to use when creating
    the difference equation for the given zeros.
  """
  length: tf.TensorShape = polynomial_zeros.shape[0]
  res: tf.TensorArray = tf.TensorArray(size=length+1,
                                       dtype=polynomial_zeros.dtype)
  c0: tf.Tensor = tf.constant(0.0, dtype=polynomial_zeros.dtype)
  c1: tf.Tensor = tf.constant(1.0, dtype=polynomial_zeros.dtype)
  for num in range(length+1):
    # s representes Qc[num] or Pc[num] in the doc comment above.
    s: tf.Tensor = c0
    for parts in itertools.combinations(np.arange(length), num):
      # prod represents the contribution to each coefficient from multiplying
      # P0 or -1 of (P0 * z^-1 - 1) with P1 or -1 of (P1 * z^-1 - 1).
      prod: tf.Tensor = c1
      for part in parts:
        prod *= -polynomial_zeros[part]
      s += prod
    res = res.write(num, s)
  return res.stack()


class PZCell(tf.keras.layers.Layer):
  """A pole-zero filter cell for a tf.keras.layers.RNN layer.

  Each PZCell has a gain, and a number of poles and zeros.

  The transfer function for the cell is:

  H = Y/X = gain * (q0 - z) * ... * (qn - z) / ( (p0 - z) * ... * (pn - z) )

  where the q's are zeros and the p's are poles.

  This class computes the difference equation for said filter, and generates
  y[n] when fed x[n].

  The cell has trainable pole and zero locations, and is constructed so that all
  gradients back to the gain, poles, and zeros are preserved for training
  purposes.

  Attributes:
    gain: The Keras weight containing the gain of the filter.
    poles: [num_poles]-complex tensor. The Keras weight containing the poles of
      the filter.
    zeros: [num-zeros]-complex tensor. The Keras weight containing the zeros of
      the filter.
    output_size: Shape of output. Required by
      https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN.
    state_size: (input_state_size, output_state_size)-tuple of integers
      containing number of previous steps to keep in state, i.e. shape of state.
      Required by
      https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN.
  """

  def __init__(self,
               gain: FloatLike,
               poles: ArrayLike,
               zeros: ArrayLike,
               **kwargs):
    """Initializes the instance.

    Args:
      gain: The initial gain of the filter.
      poles: [n_poles]-complex tensor with the initial poles of the filter.
      zeros: [n_zeros]-complex tensor with the initial zeros of the filter.
      **kwargs: Forwarded to superclass.
    Raises:
      TypeError: If the dtype isn't either tf.float32 or tf.float64.
    """
    super().__init__(**kwargs)
    if self.dtype != tf.float32 and self.dtype != tf.float64:
      raise TypeError(f'Got `dtype` parameter {self.dtype}, expected to be '
                      'either tf.float32 or tf.float64')
    self._complex_dtype: tf.dtype = tf.complex64
    if self.dtype == tf.float64:
      self._complex_dtype = tf.complex128
    self.gain: tf.Variable = self.add_weight(
        name='gain',
        dtype=self._complex_dtype,
        initializer=tf.keras.initializers.Constant(
            tf.cast(gain, self._complex_dtype)))
    self.poles: tf.Variable = self.add_weight(
        name='poles',
        shape=poles.shape,
        dtype=self._complex_dtype,
        initializer=tf.keras.initializers.Constant(
            tf.cast(poles, self._complex_dtype)))
    self._n_poles: int = poles.shape[0]
    self.zeros: tf.Variable = self.add_weight(
        name='zeros',
        shape=zeros.shape,
        dtype=self._complex_dtype,
        initializer=tf.keras.initializers.Constant(
            tf.cast(zeros, self._complex_dtype)))
    self._n_zeros: int = zeros.shape[0]
    self.output_size: int = 1
    self.state_size: Tuple[int, int] = (self.poles.shape[0]+1,
                                        self.poles.shape[0]+1)

  def get_config(self) -> Dict[str, float]:
    return {
        'n_poles': self.poles.shape[0],
        'n_zeros': self.zeros.shape[0],
    }

  @classmethod
  def from_config(cls, config: Dict[str, float]):
    return cls(0,
               np.zeros(shape=[config.pop('n_poles')]),
               np.zeros(shape=[config.pop('n_zeros')]))

  def call(self,
           input_at_t: tf.Tensor,
           states_at_t: Tuple[tf.Tensor, tf.Tensor]):
    """Computes output_at_t given input_at_t and states_at_t.

    Args:
      input_at_t: A [batch_size, 1]-self.dtype tensor with input at this step.
      states_at_t: A ([len(poles) + 1], [len(poles) + 1])-tuple of tensors with
        state at this step.

    Returns:
      (output_at_t, states_at_t_plus_1):
        output_at_t: A (batch_size, 1)-tensor with output at this step.
        states_at_t_plus_1: A ([len(poles) + 1], [len(poles) + 1])-tuple of
          tensors with state for next step.
    """
    input_dtype: tf.dtype = input_at_t.dtype
    states_dtype: tf.dtype = states_at_t[0].dtype
    input_at_t: tf.Tensor = tf.cast(input_at_t, dtype=self._complex_dtype)
    x_memory: tf.Tensor = tf.cast(states_at_t[0], self._complex_dtype)
    y_memory: tf.Tensor = tf.cast(states_at_t[1], self._complex_dtype)
    x_memory = tf.concat(
        [input_at_t, x_memory[:, :x_memory.shape[1]-1]],
        axis=1)
    pole_coeffs: tf.Tensor = coeffs_from_zeros(self.poles)
    zero_coeffs: tf.Tensor = coeffs_from_zeros(self.zeros)
    zero_offset: tf.Tensor = tf.math.maximum(0, self._n_poles - self._n_zeros)
    output_at_t: tf.Tensor = tf.constant(0,
                                         dtype=self._complex_dtype) * input_at_t
    zero_components: tf.Tensor = (x_memory[:, zero_offset:] *
                                  self.gain *
                                  zero_coeffs)
    output_at_t += tf.math.reduce_sum(zero_components, axis=1)[:, None]
    pole_components: tf.Tensor = (y_memory[:, :y_memory.shape[1]-1] *
                                  pole_coeffs[1:])
    output_at_t -= tf.math.reduce_sum(pole_components, axis=1)[:, None]
    output_at_t: tf.Tensor = tf.math.divide_no_nan(output_at_t, pole_coeffs[0])
    y_memory: tf.Tensor = tf.concat(
        [output_at_t, y_memory[:, :y_memory.shape[1]-1]],
        axis=1)
    states_at_t_plus_1: Tuple[tf.Tensor, tf.Tensor] = (
        tf.cast(tf.math.real(x_memory), states_dtype),
        tf.cast(tf.math.real(y_memory), states_dtype))
    output_at_t: tf.Tensor = tf.cast(output_at_t, input_dtype)
    return output_at_t, states_at_t_plus_1
