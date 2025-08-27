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

"""TF Keras Layers to compute output from CARFAC models.

This file is part of an implementation of Lyon's cochlear model:
"Cascade of Asymmetric Resonators with Fast-Acting Compression"

This implementation uses TF differentiability to enable tracking of gradients
all the way back to design parameters, to enable tuning them for specific tasks.

See "Human and Machine Hearing at http://dicklyon.com/hmh/ for more details.
"""

from collections.abc import Callable
import dataclasses
import enum
from typing import Any, Mapping, Optional, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from carfac.tf import pz


@dataclasses.dataclass
class CARParams:
  """Parameters for the CAR step.

  Mirrors CARParams in cpp/car.h.
  """

  velocity_scale: tf.Tensor = tf.constant(0.1)
  v_offset: tf.Tensor = tf.constant(0.04)
  min_zeta_at_half_erb_per_step: tf.Tensor = tf.constant(0.1)
  max_zeta_at_half_erb_per_step: tf.Tensor = tf.constant(0.35)
  zero_ratio: tf.Tensor = tf.constant(2.0**0.5)
  high_f_damping_compression: tf.Tensor = tf.constant(0.5)
  erb_break_freq: tf.Tensor = tf.constant(165.3)
  erb_q: tf.Tensor = tf.constant(1000.0 / (24.7 * 4.37))
  erb_per_step: tf.Tensor = tf.constant(0.5)
  min_pole_hz: tf.Tensor = tf.constant(30.0)
  sample_rate_hz: tf.Tensor = tf.constant(48000.0)
  max_pole_ratio: tf.Tensor = tf.constant(0.85 * 0.5)

  def erb_hz(self, f: tf.Tensor) -> tf.Tensor:
    """Calculates the equivalent rectangular bandwidth at a given frequency.

    Args:
      f: Frequency in Hz to calculate ERB at.

    Returns:
      The equivalent rectangular bandwidth at f.
    """
    return (self.erb_break_freq + f) / self.erb_q

  def compute_zeta(self, zeta_at_half_erb_per_step: tf.Tensor) -> tf.Tensor:
    """Computes a reasonable zeta value for a given erb_per_step.

    Based on the assumtion that max small-signal gain at the passband peak
    will be on the order of (0.5/min_zeta)**(1/erb_per_step), and we need
    the start value of that in the same region or the loss function becomes
    too uneven to optimize.

    Args:
      zeta_at_half_erb_per_step: Which zeta this should correspond to at default
        value for erb_per_step.

    Returns:
      The corresponding zeta for the actually used erb_per_step.
    """
    default_erb_per_step = 0.5
    max_small_signal_gain = (0.5 / zeta_at_half_erb_per_step) ** (
        1 / default_erb_per_step
    )
    return 0.5 / (max_small_signal_gain**self.erb_per_step)


# Functions used here are expected to take (a_0, f, g) where a_0 is a tensor
# with shape A containing the initial value of a recurrence relation series,
# and f and g are tensors with shape A + [num_steps], and return a series of
# values like a[n+1] = f[n+1] * a[n] + g[n+1], where a[0] is a_0.
# Note that the function will return a[1:] (a_0 will not be included).
RecurrenceExpansionCallable = Callable[
    [tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor
]


@dataclasses.dataclass
class IHCParams:
  """Parameters for the IHC step.

  Mirrors IHCParams in cpp/ihc.h.
  """

  # The attributes `just_half_wave_rectify` and `one_capacitor` are treated as
  # booleans.
  # The reason they are actually floats is that loading a saved model converts
  # the bool to a float, which introduces some unintuitive conversion
  # complications.
  just_half_wave_rectify: tf.Tensor = tf.constant(0.0)
  one_capacitor: tf.Tensor = tf.constant(1.0)
  tau_lpf: tf.Tensor = tf.constant(0.000080)
  tau1_out: tf.Tensor = tf.constant(0.0005)
  tau1_in: tf.Tensor = tf.constant(0.010)
  tau2_out: tf.Tensor = tf.constant(0.0025)
  tau2_in: tf.Tensor = tf.constant(0.005)
  ac_corner_hz: tf.Tensor = tf.constant(20.0)


@dataclasses.dataclass
class AGCParams:
  """Parameters for the AGC step.

  Mirrors AGCParams in cpp/agc.h.
  """

  agc_stage_gain: tf.Tensor = tf.constant(2.0)
  agc_mix_coeff: tf.Tensor = tf.constant(0.5)
  time_constants0: tf.Tensor = tf.constant(0.002)
  time_constants_mul: tf.Tensor = tf.constant(4.0)
  agc1_scales0: tf.Tensor = tf.constant(1.0)
  agc1_scales_mul: tf.Tensor = tf.constant(2.0**0.5)
  agc2_scales0: tf.Tensor = tf.constant(1.65)
  agc2_scales_mul: tf.Tensor = tf.constant(2.0**0.5)
  decimation: tf.Tensor = tf.constant([8.0, 2.0, 2.0, 2.0])

  def linear_growth(
      self,
      recurrence_expander: RecurrenceExpansionCallable,
      v: tf.Tensor,
      scale: tf.Tensor,
      num: tf.Tensor,
  ) -> tf.Tensor:
    """Produces a tensor containing a linear scale.

    Args:
      recurrence_expander: The method to expand the recurrence relation series.
      v: The first value of the return value.
      scale: The multiplier for each step of the return value.
      num: The length of the return value.

    Returns:
      A (num,)-shaped tensor containing [v, v*scale, v*scale*scale, ...].
    """
    return tf.concat(
        [
            [v],
            recurrence_expander(
                v,
                tf.ones((num - 1,), dtype=v.dtype) * scale,
                tf.zeros((num - 1,), dtype=v.dtype),
            ),
        ],
        axis=0,
    )


T = TypeVar('T')


@dataclasses.dataclass
class _CARCoeffs:
  """Mirrors CARCoeffs in cpp/car.h.

  All fields are initialized as zero value tensors to simplify instantiation and
  gradual setup one field at a time.

  Field order and type must match _CARCoeffsET for conversion between them to
  work.
  """

  velocity_scale: tf.Tensor = tf.constant(0.0)
  v_offset: tf.Tensor = tf.constant(0.0)
  r1_coeffs: tf.Tensor = tf.constant(0.0)
  a0_coeffs: tf.Tensor = tf.constant(0.0)
  c0_coeffs: tf.Tensor = tf.constant(0.0)
  h_coeffs: tf.Tensor = tf.constant(0.0)
  g0_coeffs: tf.Tensor = tf.constant(0.0)
  zr_coeffs: tf.Tensor = tf.constant(0.0)
  max_zeta: tf.Tensor = tf.constant(0.0)
  min_zeta: tf.Tensor = tf.constant(0.0)

  def convert(self) -> '_CARCoeffsET':
    return _CARCoeffsET(**vars(self))

  @classmethod
  def tensor_shapes_except_batch(
      cls, num_channels: int
  ) -> Sequence[tf.TensorShape]:
    return tuple(
        tf.TensorShape([num_channels]) for _ in dataclasses.fields(cls)
    )


class _CARCoeffsET(tf.experimental.ExtensionType):
  """Immutable extension type based on _CARCoeffs."""

  velocity_scale: tf.Tensor = tf.constant(0.0)
  v_offset: tf.Tensor = tf.constant(0.0)
  r1_coeffs: tf.Tensor = tf.constant(0.0)
  a0_coeffs: tf.Tensor = tf.constant(0.0)
  c0_coeffs: tf.Tensor = tf.constant(0.0)
  h_coeffs: tf.Tensor = tf.constant(0.0)
  g0_coeffs: tf.Tensor = tf.constant(0.0)
  zr_coeffs: tf.Tensor = tf.constant(0.0)
  max_zeta: tf.Tensor = tf.constant(0.0)
  min_zeta: tf.Tensor = tf.constant(0.0)

  def convert(self) -> _CARCoeffs:
    return _CARCoeffs(**tf.experimental.extension_type.as_dict(self))

  def to_tensors(self) -> Sequence[tf.Tensor]:
    return tuple(tf.experimental.extension_type.as_dict(self).values())

  @classmethod
  def from_tensors(cls, tensors: Sequence[tf.Tensor]) -> '_CARCoeffsET':
    return cls(*tensors)


@dataclasses.dataclass
class _IHCCoeffs:
  """Mirrors IHCCoeffs in cpp/ihc.h.

  All fields are initialized as zero value tensors to simplify instantiation and
  gradual setup one field at a time.

  Field order and type must match _IHCCoeffsET for conversion between them to
  work.
  """

  lpf_coeff: tf.Tensor = tf.constant(0.0)
  out1_rate: tf.Tensor = tf.constant(0.0)
  in1_rate: tf.Tensor = tf.constant(0.0)
  out2_rate: tf.Tensor = tf.constant(0.0)
  in2_rate: tf.Tensor = tf.constant(0.0)
  output_gain: tf.Tensor = tf.constant(0.0)
  rest_output: tf.Tensor = tf.constant(0.0)
  rest_cap1: tf.Tensor = tf.constant(0.0)
  rest_cap2: tf.Tensor = tf.constant(0.0)
  ac_coeff: tf.Tensor = tf.constant(0.0)
  cap1_voltage: tf.Tensor = tf.constant(0.0)
  cap2_voltage: tf.Tensor = tf.constant(0.0)

  # Disclaimer: This type of metaprogramming is not good practice.
  # This function is used to simplify casting all fields of the instance to the
  # provided dtype.
  def cast(self, dtype: tf.DType) -> '_IHCCoeffs':
    params = {}
    for name, value in vars(self).items():
      params[name] = tf.cast(value, dtype)
    return _IHCCoeffs(**params)

  def convert(self) -> '_IHCCoeffsET':
    return _IHCCoeffsET(**vars(self))

  @classmethod
  def tensor_shapes_except_batch(cls) -> Sequence[tf.TensorShape]:
    return tuple(tf.TensorShape(()) for _ in dataclasses.fields(cls))


class _IHCCoeffsET(tf.experimental.ExtensionType):
  """Immutable extension type based on _IHCCoeffs."""

  lpf_coeff: tf.Tensor = tf.constant(0.0)
  out1_rate: tf.Tensor = tf.constant(0.0)
  in1_rate: tf.Tensor = tf.constant(0.0)
  out2_rate: tf.Tensor = tf.constant(0.0)
  in2_rate: tf.Tensor = tf.constant(0.0)
  output_gain: tf.Tensor = tf.constant(0.0)
  rest_output: tf.Tensor = tf.constant(0.0)
  rest_cap1: tf.Tensor = tf.constant(0.0)
  rest_cap2: tf.Tensor = tf.constant(0.0)
  ac_coeff: tf.Tensor = tf.constant(0.0)
  cap1_voltage: tf.Tensor = tf.constant(0.0)
  cap2_voltage: tf.Tensor = tf.constant(0.0)

  def convert(self) -> _IHCCoeffs:
    return _IHCCoeffs(**tf.experimental.extension_type.as_dict(self))

  def to_tensors(self) -> Sequence[tf.Tensor]:
    # Returns a sequence of tensors in the same order as the field declarations
    # in this class, which also matches the argument order in the class
    # constructor.
    # Note that this - and all `to_tensors` - depends on some subtleties:
    # - The dictionary returned by as_dict iterates in insertion order, which is
    #   true for Python 3.7+.
    # - The dictionary returned by as_dict is populated in the order of the
    #   internal field list, which is expected to be true, and also defines the
    #   order of the constructor arguments.
    return tuple(tf.experimental.extension_type.as_dict(self).values())

  @classmethod
  def from_tensors(cls, tensors: Sequence[tf.Tensor]) -> '_IHCCoeffsET':
    return cls(*tensors)


@dataclasses.dataclass
class _AGCCoeffs:
  """Mirrors AGCCoeffs in cpp/agc.h.

  All fields are initialized as zero value tensors to simplify instantiation and
  gradual setup one field at a time.

  Field order and type must match _AGCCoeffsET for conversion between them to
  work.
  """

  agc_stage_gain: tf.Tensor = tf.constant(0.0)
  agc_epsilon: tf.Tensor = tf.constant(0.0)
  decimation: tf.Tensor = tf.constant(0.0)
  agc_pole_z1: tf.Tensor = tf.constant(0.0)
  agc_pole_z2: tf.Tensor = tf.constant(0.0)
  agc_spatial_iterations: tf.Tensor = tf.constant(0.0)
  agc_spatial_fir_left: tf.Tensor = tf.constant(0.0)
  agc_spatial_fir_mid: tf.Tensor = tf.constant(0.0)
  agc_spatial_fir_right: tf.Tensor = tf.constant(0.0)
  agc_spatial_n_taps: tf.Tensor = tf.constant(0.0)
  agc_mix_coeffs: tf.Tensor = tf.constant(0.0)
  agc_gain: tf.Tensor = tf.constant(0.0)
  detect_scale: tf.Tensor = tf.constant(0.0)
  decim: tf.Tensor = tf.constant(0.0)

  @classmethod
  def concat(cls, coeffs: Sequence['_AGCCoeffs']) -> '_AGCCoeffs':
    """Returns concatenated _AGCCoeffs.

    Args:
      coeffs: A list of _AGCCoeffs to concatenate.

    Returns:
      _AGCCoeffs where each attribute is the concatenation of the corresponding
        attributes in the argument list of _AGCCoeffs.
    """
    params = {}
    for field in dataclasses.fields(cls):
      parts = []
      for coeff in coeffs:
        parts.append(getattr(coeff, field.name))
      params[field.name] = tf.stack(parts, axis=0)
    return cls(**params)

  def convert(self) -> '_AGCCoeffsET':
    return _AGCCoeffsET(**vars(self))

  @classmethod
  def tensor_shapes_except_batch(cls, stages: int) -> Sequence[tf.TensorShape]:
    return tuple(tf.TensorShape([stages]) for _ in dataclasses.fields(cls))


class _AGCCoeffsET(tf.experimental.ExtensionType):
  """Immutable extension type based on _AGCCoeffs."""

  agc_stage_gain: tf.Tensor = tf.constant(0.0)
  agc_epsilon: tf.Tensor = tf.constant(0.0)
  decimation: tf.Tensor = tf.constant(0.0)
  agc_pole_z1: tf.Tensor = tf.constant(0.0)
  agc_pole_z2: tf.Tensor = tf.constant(0.0)
  agc_spatial_iterations: tf.Tensor = tf.constant(0.0)
  agc_spatial_fir_left: tf.Tensor = tf.constant(0.0)
  agc_spatial_fir_mid: tf.Tensor = tf.constant(0.0)
  agc_spatial_fir_right: tf.Tensor = tf.constant(0.0)
  agc_spatial_n_taps: tf.Tensor = tf.constant(0.0)
  agc_mix_coeffs: tf.Tensor = tf.constant(0.0)
  agc_gain: tf.Tensor = tf.constant(0.0)
  detect_scale: tf.Tensor = tf.constant(0.0)
  decim: tf.Tensor = tf.constant(0.0)

  def __getitem__(self, stage: tf.Tensor) -> '_AGCCoeffsET':
    """Returns _AGCCoeffsET for only the given stage."""
    values = tf.experimental.extension_type.as_dict(self)
    sliced_values = {name: tensor[stage] for name, tensor in values.items()}
    return _AGCCoeffsET(**sliced_values)

  def convert(self) -> _AGCCoeffs:
    return _AGCCoeffs(**tf.experimental.extension_type.as_dict(self))

  def to_tensors(self) -> Sequence[tf.Tensor]:
    return tuple(tf.experimental.extension_type.as_dict(self).values())

  @classmethod
  def from_tensors(cls, tensors: Sequence[tf.Tensor]) -> '_AGCCoeffsET':
    return cls(*tensors)


@dataclasses.dataclass
class _CARState:
  """Mirrors CARState in cpp/car.h.

  All fields are initialized as zero value tensors to simplify instantiation and
  gradual setup one field at a time.
  """

  z1_memory: tf.Tensor = tf.constant(0.0)
  z2_memory: tf.Tensor = tf.constant(0.0)
  za_memory: tf.Tensor = tf.constant(0.0)
  zb_memory: tf.Tensor = tf.constant(0.0)
  dzb_memory: tf.Tensor = tf.constant(0.0)
  zy_memory: tf.Tensor = tf.constant(0.0)
  g_memory: tf.Tensor = tf.constant(0.0)
  dg_memory: tf.Tensor = tf.constant(0.0)

  @classmethod
  def tensor_shapes_except_batch(
      cls, num_ears: int, num_channels: int
  ) -> Sequence[tf.TensorShape]:
    return tuple(
        tf.TensorShape([num_ears, num_channels])
        for _ in dataclasses.fields(cls)
    )

  def convert(self) -> '_CARStateET':
    return _CARStateET(**vars(self))


class _CARStateET(tf.experimental.ExtensionType):
  """Immutable extension type based on _CARState."""

  z1_memory: tf.Tensor = tf.constant(0.0)
  z2_memory: tf.Tensor = tf.constant(0.0)
  za_memory: tf.Tensor = tf.constant(0.0)
  zb_memory: tf.Tensor = tf.constant(0.0)
  dzb_memory: tf.Tensor = tf.constant(0.0)
  zy_memory: tf.Tensor = tf.constant(0.0)
  g_memory: tf.Tensor = tf.constant(0.0)
  dg_memory: tf.Tensor = tf.constant(0.0)

  def convert(self) -> _CARState:
    return _CARState(**tf.experimental.extension_type.as_dict(self))

  def to_tensors(self) -> Sequence[tf.Tensor]:
    return tuple(tf.experimental.extension_type.as_dict(self).values())

  @classmethod
  def from_tensors(cls, tensors: Sequence[tf.Tensor]) -> '_CARStateET':
    return cls(*tensors)


@dataclasses.dataclass
class _IHCState:
  """Mirrors IHCState in cpp/ihc.h.

  All fields are initialized as zero value tensors to simplify instantiation and
  gradual setup one field at a time.
  """

  ihc_out: tf.Tensor = tf.constant(0.0)
  cap1_voltage: tf.Tensor = tf.constant(0.0)
  cap2_voltage: tf.Tensor = tf.constant(0.0)
  lpf1_state: tf.Tensor = tf.constant(0.0)
  lpf2_state: tf.Tensor = tf.constant(0.0)
  ac_coupler: tf.Tensor = tf.constant(0.0)

  @classmethod
  def tensor_shapes_except_batch(
      cls, num_ears: int, num_channels: int
  ) -> Sequence[tf.TensorShape]:
    return tuple(
        tf.TensorShape([num_ears, num_channels])
        for _ in dataclasses.fields(cls)
    )

  def convert(self) -> '_IHCStateET':
    return _IHCStateET(**vars(self))


class _IHCStateET(tf.experimental.ExtensionType):
  """Immutable extension type based on _CARState."""

  ihc_out: tf.Tensor = tf.constant(0.0)
  cap1_voltage: tf.Tensor = tf.constant(0.0)
  cap2_voltage: tf.Tensor = tf.constant(0.0)
  lpf1_state: tf.Tensor = tf.constant(0.0)
  lpf2_state: tf.Tensor = tf.constant(0.0)
  ac_coupler: tf.Tensor = tf.constant(0.0)

  def convert(self) -> _IHCState:
    return _IHCState(**tf.experimental.extension_type.as_dict(self))

  def to_tensors(self) -> Sequence[tf.Tensor]:
    return tuple(tf.experimental.extension_type.as_dict(self).values())

  @classmethod
  def from_tensors(cls, tensors: Sequence[tf.Tensor]) -> '_IHCStateET':
    return cls(*tensors)


@dataclasses.dataclass
class _AGCState:
  """Mirrors AGCState in cpp/agc.h.

  All fields are initialized as zero value tensors to simplify instantiation and
  gradual setup one field at a time.
  """

  agc_memory: tf.Tensor = tf.constant(0.0)
  input_accum: tf.Tensor = tf.constant(0.0)
  decim_phase: tf.Tensor = tf.constant(0.0)

  def convert(self) -> '_AGCStateET':
    return _AGCStateET(**vars(self))

  def __getitem__(self, stage: tf.Tensor) -> '_AGCState':
    """Returns _AGCState for only the given stage."""
    sliced_values = {
        name: tensor[:, :, :, stage] for name, tensor in vars(self).items()
    }
    return _AGCState(**sliced_values)

  @classmethod
  def concat(cls, states: Sequence['_AGCState']) -> '_AGCState':
    """Returns concatenated _AGCStates.

    Args:
      states: A list of _AGCStates to concatenate.

    Returns:
      _AGCState where each attribute is the concatenation of the corresponding
        attributes in the argument list of _AGCStates.
    """
    params = {}
    for field in dataclasses.fields(cls):
      parts = []
      for state in states:
        parts.append(getattr(state, field.name))
      params[field.name] = tf.stack(parts, axis=3)
    return cls(**params)

  def update(self, stage: tf.Tensor, state: '_AGCState'):
    """Updates a stage for an _AGCState.

    Args:
      stage: The stage to update.
      state: An _AGCState for a single stage to update with.
    """
    for name in [field.name for field in dataclasses.fields(self)]:
      old_value = getattr(self, name)
      new_value = tf.reshape(
          tf.concat(
              [
                  old_value[:, :, :, :stage],
                  getattr(state, name)[:, :, :, tf.newaxis],
                  old_value[:, :, :, stage + 1 :],
              ],
              axis=3,
          ),
          (
              old_value.shape[0] or -1,
              old_value.shape[1],
              old_value.shape[2],
              old_value.shape[3],
          ),
      )
      setattr(self, name, new_value)

  @classmethod
  def tensor_shapes_except_batch(
      cls, num_ears: int, num_channels: int, num_stages: int
  ) -> Sequence[tf.TensorShape]:
    return tuple(
        tf.TensorShape([num_ears, num_channels, num_stages])
        for _ in dataclasses.fields(cls)
    )


class _AGCStateET(tf.experimental.ExtensionType):
  """Immutable extension type based on _AGCState."""

  agc_memory: tf.Tensor = tf.constant(0.0)
  input_accum: tf.Tensor = tf.constant(0.0)
  decim_phase: tf.Tensor = tf.constant(0.0)

  def __getitem__(self, stage: tf.Tensor) -> '_AGCStateET':
    """Returns _AGCStateET for only the given stage."""
    values = tf.experimental.extension_type.as_dict(self)
    sliced_values = {
        name: tensor[:, :, :, stage] for name, tensor in values.items()
    }
    return _AGCStateET(**sliced_values)

  def convert(self) -> _AGCState:
    return _AGCState(**tf.experimental.extension_type.as_dict(self))

  def to_tensors(self) -> Sequence[tf.Tensor]:
    return tuple(tf.experimental.extension_type.as_dict(self).values())

  @classmethod
  def from_tensors(cls, tensors: Sequence[tf.Tensor]) -> '_AGCStateET':
    return cls(*tensors)

  def spec_with_unknown_batch(self) -> '_AGCStateET.Spec':
    """Returns a Spec for the object with unknown batch dimension.

    Returns: A type(self).Spec spec matching this object, but with the batch
      dimension unknown.
    """
    params = {}
    for name, value in tf.experimental.extension_type.as_dict(self).items():
      current_spec = tf.TensorSpec.from_tensor(value)
      shape_as_list = current_spec.shape.as_list()
      shape_as_list[0] = None
      params[name] = tf.TensorSpec(shape_as_list, current_spec.dtype)
    return _AGCStateET.Spec(**params)


# Functions used here are expected to take (steps, kernel, data) where steps is
# a scalar int tensor, kernel is a ([3|5])-shaped int tensor, and data is a
# tensor that will be convolved in the last index steps times with the kernel.
ConvolverCallable = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]


def conv1d_convolver(
    steps: tf.Tensor, kernel: tf.Tensor, data: tf.Tensor
) -> tf.Tensor:
  """Convolves the input with the kernel using tf.nn.conv1d.

  Args:
    steps: The number of convolutions to run the input through.
    kernel: The kernel to use, must be (3,) or (5,) shaped.
    data: The input to convolve.

  Returns:
    The convolved input.
  """
  # pylint: disable=g-long-lambda
  # The multi-line lambda is much more readable than defining a new inner
  # function above the tf.while_loop.
  padding_steps = (tf.shape(kernel)[0] - 1) // 2
  padding = tf.stack([[0, 0], [padding_steps, padding_steps], [0, 0]])
  unrolled_shape = (-1, data.shape[-1], 1)
  expanded_kernel = kernel[:, tf.newaxis, tf.newaxis]
  unrolled_output, _, _ = tf.while_loop(
      lambda a, b, step: step < steps,
      lambda unrolled_input, padding, iteration: (
          tf.reshape(
              tf.nn.conv1d(
                  tf.pad(unrolled_input, padding, mode='SYMMETRIC'),
                  filters=expanded_kernel,
                  stride=1,
                  padding='VALID',
              ),
              unrolled_shape,
          ),
          padding,
          iteration + 1,
      ),
      (tf.reshape(data, unrolled_shape), padding, tf.constant(0, tf.int32)),
  )
  return tf.reshape(unrolled_output, tf.shape(data))


# pylint: disable=g-long-lambda
def concat_add_convolver(
    steps: tf.Tensor, kernel: tf.Tensor, data: tf.Tensor
) -> tf.Tensor:
  """Convolves the input with the kernel using tf.concat and addition.

  This function does the padding and addition on its own, and has been observed
  to be ~5% faster than the conv1d function in graph mode, about same speed in
  eager mode.

  Args:
    steps: The number of convolutions to run the input through.
    kernel: The kernel to use, must be (3,) or (5,) shaped.
    data: The input to convolve.

  Returns:
    The convolved input.
  """

  def conv(unrolled_data: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    padding_steps = (tf.shape(kernel)[0] - 1) // 2
    padded = tf.concat(
        [
            unrolled_data[:, padding_steps - 1 :: -1],
            unrolled_data,
            unrolled_data[:, : -1 - padding_steps : -1],
        ],
        axis=-1,
    )
    return tf.reshape(
        tf.cond(
            tf.shape(kernel)[0] == 3,
            lambda: (
                (padded * kernel[0])[:, :-2]
                + (padded * kernel[1])[:, 1:-1]
                + (padded * kernel[2])[:, 2:]
            ),
            lambda: (
                (padded * kernel[0])[:, :-4]
                + (padded * kernel[1])[:, 1:-3]
                + (padded * kernel[2])[:, 2:-2]
                + (padded * kernel[3])[:, 3:-1]
                + (padded * kernel[4])[:, 4:]
            ),
        ),
        tf.shape(unrolled_data),
    )

  output, _ = tf.while_loop(
      lambda _, step: step < steps,
      lambda unrolled_data, step: (conv(unrolled_data, kernel), step + 1),
      (tf.reshape(data, [-1, data.shape[-1]]), 0),
  )
  return tf.reshape(output, tf.shape(data))


convolution_methods: dict[str, ConvolverCallable] = {
    'conv1d': conv1d_convolver,
    'concat_add': concat_add_convolver,
}


def tensor_array_recurrence_expansion(
    a_0: tf.Tensor, f: tf.Tensor, g: tf.Tensor
) -> tf.Tensor:
  # This is the slowest way to expand a recurrence relation, but most closely
  # resembles the way the C++ code does the same thing.
  """Expands a recurrence series using a while loop and TensorArrays.

  Assumes the relation a[n+1] = f[n+1] * a[n] + g[n+1], where a[0] = a_0.

  Args:
    a_0: a[0], the start value of a.
    f: The scale factor of the series. Needs to have exactly one more index than
      a_0.
    g: The offset factor of the series. Needs to have the same shape as f.

  Returns:
    The a[1:], the series without a_0, with the same shape as f and g.
  """

  def next_in_series(
      prev: tf.Tensor,
      f: tf.Tensor,
      g: tf.Tensor,
      res_ta: tf.TensorArray,
      idx: tf.Tensor,
  ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.TensorArray, tf.Tensor]:
    new = tf.squeeze(
        tf.gather(f, axis=-1, indices=[idx]), axis=-1
    ) * prev + tf.squeeze(tf.gather(g, axis=-1, indices=[idx]), axis=-1)
    return (new, f, g, res_ta.write(idx, new), idx + 1)

  *_, res_ta, _ = tf.while_loop(
      lambda a, b, c, d, idx: idx < f.shape[-1],
      next_in_series,
      (a_0, f, g, tf.TensorArray(size=f.shape[-1], dtype=f.dtype), 0),
  )
  stacked_ta = res_ta.stack()
  indices = list(range(len(stacked_ta.shape)))
  new_indices = indices[1:] + indices[:1]
  return tf.transpose(stacked_ta, new_indices)


def recurrence_relation_recurrence_expansion(
    a_0: tf.Tensor, f: tf.Tensor, g: tf.Tensor
) -> tf.Tensor:
  """Expands the recurrence series using cumulative products and sums.

  Assumes the relation is a[n+1] = f[n+1] * a[n] + g[n+1], where a[0] = a_0.

  This solution is observed to be 20-30% faster than
  tensor_array_recurrence_expansion method, but may have problems with numerical
  stability due to division by cumulative products.

  Args:
    a_0: a[0], the start value of a.
    f: The scale factor of the series. Needs to have exactly one more index than
      a_0.
    g: The offset factor of the series. Needs to have the same shape as f.

  Returns:
    a[1:], the series without a_0, with the same shape as f and g.
  """
  f_cp = tf.math.cumprod(f, axis=-1)
  return f_cp * (a_0[..., tf.newaxis] + tf.math.cumsum(g / f_cp, axis=-1))


def mat_mul_recurrence_expansion(
    a_0: tf.Tensor, f: tf.Tensor, g: tf.Tensor
) -> tf.Tensor:
  """Expands the recurrence series using matrix multiplication.

  Assumes the relation is a[n+1] = f[n+1] * a[n] + g[n+1], where a[0] = a_0.

  This solution is observed to be almost as fast as
  recurrence_relation_recurrence_expansion but doesn't have the same problems
  with numerical instability.

  Args:
    a_0: a[0], the start value of a.
    f: The scale factor of the series. Needs to have exactly one more index than
      a_0.
    g: The offset factor of the series. Needs to have the same shape as f.

  Returns:
    a[1:], the series without a_0, with the same shape as f and g.
  """
  f_log_cs = tf.math.cumsum(tf.math.log(f), axis=-1)
  dim_range = tf.range(f.shape[-1])
  f_mult = tf.where(
      dim_range[:, tf.newaxis] > dim_range[tf.newaxis, :],
      tf.constant(0, dtype=f_log_cs.dtype),
      tf.math.exp(f_log_cs[..., tf.newaxis, :] - f_log_cs[..., tf.newaxis]),
  )
  return tf.math.exp(f_log_cs) * a_0[..., tf.newaxis] + tf.squeeze(
      g[..., tf.newaxis, :] @ f_mult, -2
  )


# These are alternative implementations of expanding a series of values like
# a[n+1] = f[n+1] * a[n] + g[n+1] with known f, g, and a[0].
# They are available to use for comparing performance of different solutions
# on different platforms.
# Note that they all return a[1:], i.e. a[0] is not included.
# TensorArray is the slowest method, but is the one closest to the way the
# original Matlab and the C++ versions do it (basically a for-loop that updates)
# one value at a time.
# RecurrenceRelation uses a very small number of TF ops to do the same thing
# via cumulative sums and products, and has been observed to produce ~30% faster
# runs than TensorArray on CPU. It does a division by a cumulative product that
# may lead to numerical instability.
# MatMul builds a large matrix to multiply the values with, and has been
# observed to produce ~%25 faster runs than TensorArray on CPU, but may provide
# more benefit on GPU or TPU. It also logarithms to avoid the numerical that
# RecurrenceRelations suffers from.
recurrence_expansion_methods: dict[str, RecurrenceExpansionCallable] = {
    'TensorArray': tensor_array_recurrence_expansion,
    'RecurrenceRelation': recurrence_relation_recurrence_expansion,
    'MatMul': mat_mul_recurrence_expansion,
}


class CARFACOutput(enum.Enum):
  """The different types of possible output a CARFACCell can produce.

  BM: The basilar membrane output of the model.
  OHC: The outer hair cell output of the model.
  AGC: The automatic gain control output of the model.
  NAP: The neural activation pattern output of the model.
  """

  BM = 0
  OHC = 1
  AGC = 2
  NAP = 3


class CARFACCell(tf.keras.layers.Layer):
  """A CAR cell for a tf.keras.layers.RNN layer.

  Computes output samples for a set of cochlear places given an input sample.

  It implements the CARFAC model, see Lyon's book Human and Machine Hearing
  (http://dicklyon.com/hmh/).

  It tracks gradients both in eager and graph mode all the way from parameters
  to ouput, so all parameters can be tuned for the wanted use case.

  The expected use case is to wrap it inside a tf.keras.layers.RNN  and then
  feed audio sample sequences to the layers. The test cases in in carfac_test.py
  can be used for reference.

  See also the GitHub repository https://github.com/google/carfac/.

  The tensor shape used for input is [batch_idx, ear_idx, 1] and for
  output [batch_idx, ear_idx, channel_idx, output_idx], where the last dimension
  contains the selected output values.

  When wrapped in a tf.keras.layers.RNN layer the input and output respectively
  becomes [batch_idx, sample_idx, ear_idx, 1] and
  [batch_idx, sample_idx, ear_idx, channel_idx, output_idx].

  Note that with erb_per_step smaller than ~0.3 the CARFACCell needs to use
  tf.float64 instead of the default tf.float32, or numerical problems will cause
  not-a-number gradients.

  Attributes:
    output_size: Shape of output. Required by
      https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN.
    state_size: Shape of state. Required by
      https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN.
    car_params: The CARParams used by this CARFACCell.
    ihc_params: The IHCParams used by this CARFACCell.
    agc_params: The AGCParams used by this CARFACCell.
  """

  output_size: tf.TensorShape
  state_size: tuple[tf.TensorShape, ...]

  _recurrence_expander: RecurrenceExpansionCallable
  _linear: bool
  _open_loop: bool
  _outputs: tuple[CARFACOutput, ...]

  car_params: Optional[CARParams]
  ihc_params: Optional[IHCParams]
  agc_params: Optional[AGCParams]

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> 'CARFACCell':
    """Required by superclass for serialization.

    See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

    Args:
      config: A Map of config to use when unserializing a CARFACCell.

    Returns:
      A CARFACCell using the provided config.
    """
    car_params = CARParams(**config['car_params_init_args'])
    ihc_params = IHCParams(**config['ihc_params_init_args'])
    agc_params = AGCParams(**config['agc_params_init_args'])
    return cls(
        car_params=car_params,
        ihc_params=ihc_params,
        agc_params=agc_params,
        outputs=tuple(CARFACOutput(id) for id in config['outputs']),
        num_ears=config['num_ears'],
        linear=config['linear'],
        open_loop=config['open_loop'],
    )

  def __init__(
      self,
      car_params: CARParams = CARParams(),
      ihc_params: IHCParams = IHCParams(),
      agc_params: AGCParams = AGCParams(),
      num_ears: int = 1,
      outputs: Sequence[CARFACOutput] = (CARFACOutput.NAP,),
      linear: bool = False,
      open_loop: bool = False,
      recurrence_expander: RecurrenceExpansionCallable = mat_mul_recurrence_expansion,
      convolver: ConvolverCallable = conv1d_convolver,
      **kwargs
  ):
    """Initializes a CAR cell.

    Args:
      car_params: CARParams defining the CAR step of this cell.
      ihc_params: IHCParams defining the IHC step of this cell.
      agc_params: AGCParams defining the AGC step of this cell.
      num_ears: Number of simulated ears. Input has shape [batch_idx, ear_idx,
        1], and output has shape [batch_idx, channel_idx, ear_idx, output_idx].
        Note that after wrapping a CARFACCell in an RNN layer, the input and
        output respectively will have shapes [batch_idx, step_idx, ear_idx, 1]
        and [batch_idx, step_idx, ear_idx, channel_idx, output_idx].
      outputs: The requested output. The output dimension will be populated with
        the outputs specified in this parameter.
      linear: Whether the cell should incorporate nonlinearities or not.
      open_loop: Whether to run in open loop, which will turn off IHC/AGC
        feedback.
      recurrence_expander: The method to expand recurrence relation series.
      convolver: The method to convolve over 1 dimension.
      **kwargs: Forwarded to superclass.
    """
    super().__init__(**kwargs)
    self._recurrence_expander = recurrence_expander
    self._convolver = convolver
    self._linear = linear
    self._open_loop = open_loop
    self._outputs = tuple(outputs)

    curr_freq = car_params.max_pole_ratio * car_params.sample_rate_hz
    num_channels = 0
    while curr_freq > car_params.min_pole_hz:
      curr_freq -= car_params.erb_per_step * car_params.erb_hz(curr_freq)
      num_channels += 1

    self.car_params = self._copy_params_from(car_params)
    self.ihc_params = self._copy_params_from(ihc_params)
    self.agc_params = self._copy_params_from(agc_params)

    self.output_size = tf.TensorShape((num_ears, num_channels, len(outputs)))

    state_size = [
        *_CARCoeffs.tensor_shapes_except_batch(num_channels),
        *_CARState.tensor_shapes_except_batch(
            self.output_size[0], self.output_size[1]
        ),
        *_IHCCoeffs.tensor_shapes_except_batch(),
        *_IHCState.tensor_shapes_except_batch(
            self.output_size[0], self.output_size[1]
        ),
    ]
    if self.agc_params.decimation.shape.as_list():
      state_size.extend(
          _AGCCoeffs.tensor_shapes_except_batch(
              self.agc_params.decimation.shape[0]
          )
      )
      state_size.extend(
          _AGCState.tensor_shapes_except_batch(
              self.output_size[0],
              self.output_size[1],
              self.agc_params.decimation.shape[0],
          )
      )

    self.state_size = tuple(state_size)

  def _copy_params_from(self, values: tf.experimental.ExtensionType):
    params = {}
    for k, v in vars(values).items():
      if k in [
          'erb_per_step',
          'min_pole_hz',
          'sample_rate_hz',
          'max_pole_hz',
          'just_half_wave_rectify',
          'one_capacitor',
      ]:
        params[k] = tf.cast(v, name=k, dtype=self.dtype)
      else:
        params[k] = self._add_weight(k, v)
    return type(values)(**params)

  def get_initial_state(
      self,
      inputs: Any = None,
      batch_size: Any = None,
      dtype: Optional[tf.DType] = None,
  ) -> Sequence[tf.Tensor]:
    """Required by tf.keras.layers.RNN to initialize sequential processing.

    See https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN

    Either inputs or (batch_size and dtype) must be provided.

    Args:
      inputs: Optional batch of inputs.
      batch_size: Optional batch size of future inputs.
      dtype: Optional dtype of future inputs.

    Returns:
      A tuple useful as state of first step in the RNN processing.
    """
    car_coeffs = self._design_car_coeffs()

    if batch_size is None:
      batch_size = inputs.shape[0]
    if dtype is None:
      dtype = inputs.dtype
    car_state = _CARState()
    zeros = tf.zeros(
        [batch_size, self.output_size[0], self.output_size[1]], dtype=self.dtype
    )
    ones_without_channels = tf.ones(
        [batch_size, self.output_size[0], 1], dtype=self.dtype
    )
    ones = tf.ones(
        [batch_size, self.output_size[0], self.output_size[1]], dtype=self.dtype
    )
    car_state.z1_memory = zeros
    car_state.z2_memory = zeros
    car_state.za_memory = zeros
    car_state.zb_memory = ones_without_channels * car_coeffs.zr_coeffs
    car_state.dzb_memory = zeros
    car_state.zy_memory = zeros
    car_state.g_memory = ones_without_channels * car_coeffs.g0_coeffs
    car_state.dg_memory = zeros

    ihc_coeffs = self._design_ihc_coeffs()

    ihc_state = _IHCState()
    ihc_state.ac_coupler = zeros
    ihc_state.ihc_out = zeros
    ihc_state.lpf1_state = ones * ihc_coeffs.rest_output
    ihc_state.lpf2_state = ones * ihc_coeffs.rest_output
    ihc_state.cap1_voltage = ones * ihc_coeffs.rest_cap1
    ihc_state.cap2_voltage = ones * ihc_coeffs.rest_cap2

    initial_state = [
        *car_coeffs.to_tensors(),
        *car_state.convert().to_tensors(),
        *ihc_coeffs.to_tensors(),
        *ihc_state.convert().to_tensors(),
    ]

    if self.agc_params.decimation.shape.as_list():
      agc_coeffs = self._design_agc_coeffs()

      agc_state = _AGCState()
      agc_state.decim_phase = zeros
      agc_state.agc_memory = zeros
      agc_state.input_accum = zeros
      agc_state = _AGCState.concat(
          [agc_state for _ in range(self.agc_params.decimation.shape[0])]
      )

      initial_state.extend(agc_coeffs.to_tensors())
      initial_state.extend(agc_state.convert().to_tensors())

    return tuple(initial_state)

  def _add_weight(self, name: str, value: tf.Tensor) -> tf.Variable:
    return self.add_weight(
        name=name,
        dtype=self.dtype,
        shape=tf.convert_to_tensor(value).shape,
        initializer=tf.keras.initializers.Constant(tf.cast(value, self.dtype)),
    )

  def get_config(self) -> dict[str, Any]:
    """Required by superclass for serialization.

    See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

    Returns:
      A dictionary with parameters for creating an identical instance.
    """
    return {
        'car_params_init_args': {
            k: v.numpy() for k, v in vars(self.car_params).items()
        },
        'ihc_params_init_args': {
            k: v.numpy() for k, v in vars(self.ihc_params).items()
        },
        'agc_params_init_args': {
            k: v.numpy() for k, v in vars(self.agc_params).items()
        },
        'outputs': self._outputs,
        'linear': self._linear,
        'open_loop': self._open_loop,
        'num_ears': self.output_size[0],
    }

  # pylint: disable=cell-var-from-loop,g-long-lambda
  def _design_agc_coeffs(self) -> _AGCCoeffsET:
    # Many comments in this function copied from cpp/carfac.cc:
    agc_coeffs = []
    previous_stage_gain = tf.constant(0.0, dtype=self.dtype)
    decim = tf.constant(1.0, dtype=self.dtype)
    delay = tf.constant(0.0, dtype=self.dtype)
    spread_sq = tf.constant(0.0, dtype=self.dtype)
    for stage in range(self.agc_params.decimation.shape[0]):
      agc_coeff = _AGCCoeffs()
      agc_coeffs.append(agc_coeff)
      time_constants = self.agc_params.linear_growth(
          self._recurrence_expander,
          self.agc_params.time_constants0,
          self.agc_params.time_constants_mul,
          self.agc_params.decimation.shape[0],
      )
      agc1_scales = self.agc_params.linear_growth(
          self._recurrence_expander,
          self.agc_params.agc1_scales0,
          self.agc_params.agc1_scales_mul,
          self.agc_params.decimation.shape[0],
      )
      agc2_scales = self.agc_params.linear_growth(
          self._recurrence_expander,
          self.agc_params.agc2_scales0,
          self.agc_params.agc2_scales_mul,
          self.agc_params.decimation.shape[0],
      )
      agc_coeff.agc_stage_gain = self.agc_params.agc_stage_gain
      agc_coeff.decimation = self.agc_params.decimation[stage]
      total_dc_gain = previous_stage_gain
      # Calculate the parameters for the current stage.
      tau = time_constants[stage]
      agc_coeff.decim = agc_coeff.decimation * decim
      agc_coeff.agc_epsilon = 1.0 - tf.math.exp(
          (-1.0 * agc_coeff.decim) / (tau * self.car_params.sample_rate_hz)
      )
      n_times = tau * (self.car_params.sample_rate_hz / agc_coeff.decim)
      delay = (agc2_scales[stage] - agc1_scales[stage]) / n_times
      spread_sq = (
          tf.math.square(agc1_scales[stage])
          + tf.math.square(agc2_scales[stage])
      ) / n_times
      u = 1.0 + (1.0 / spread_sq)
      p = u - tf.math.sqrt(tf.math.square(u) - 1.0)
      dp = delay * (1.0 - (2.0 * p) + tf.math.square(p)) / 2.0
      agc_coeff.agc_pole_z1 = p - dp
      agc_coeff.agc_pole_z2 = p + dp
      n_taps = tf.constant(0, self.dtype)
      done = tf.constant(False)
      n_iterations = tf.constant(1, self.dtype)
      # Initialize the FIR coefficient settings at each stage.
      fir_left = tf.constant(0, self.dtype)
      fir_mid = tf.constant(1, self.dtype)
      fir_right = tf.constant(0, self.dtype)
      (n_iterations, n_taps, done) = tf.cond(
          spread_sq == 0,
          lambda: (
              tf.constant(0, self.dtype),
              tf.constant(3, self.dtype),
              tf.constant(True),
          ),
          lambda: (n_iterations, n_taps, done),
      )

      def compute_filter(
          n_iterations: tf.Tensor,
          n_taps: tf.Tensor,
          done: tf.Tensor,
          fir_left: tf.Tensor,
          fir_mid: tf.Tensor,
          fir_right: tf.Tensor,
      ) -> tuple[
          tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
      ]:
        def taps_5() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
          return tf.cond(
              n_iterations > 3,
              lambda: (n_taps, tf.constant(-1, self.dtype), tf.constant(True)),
              lambda: (n_taps, n_iterations + 1, done),
          )

        (n_taps, n_iterations, done) = tf.switch_case(
            tf.cast(n_taps, tf.int32),
            branch_fns={
                0: lambda: (tf.constant(3, self.dtype), n_iterations, done),
                1: lambda: (n_taps, n_iterations, done),
                2: lambda: (n_taps, n_iterations, done),
                3: lambda: (tf.constant(5, self.dtype), n_iterations, done),
                4: lambda: (n_taps, n_iterations, done),
                5: taps_5,
            },
        )

        def not_done(
            n_iterations: tf.Tensor,
            n_taps: tf.Tensor,
            fir_left: tf.Tensor,
            fir_mid: tf.Tensor,
            fir_right: tf.Tensor,
        ) -> tuple[
            tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor
        ]:
          # The smoothing function is a space-domain smoothing, but it
          # considered here by analogy to time-domain smoothing, which is why
          # its potential off-centeredness is called a delay.  Since it's a
          # smoothing filter, it is also analogous to a discrete probability
          # distribution (a p.m.f.), with mean corresponding to delay and
          # variance corresponding to squared spatial spread (in samples, or
          # channels, and the square thereof,
          # respecitively). Here we design a filter implementation's coefficient
          # via the method of moment matching, so we get the intended delay and
          # spread, and don't worry too much about the shape of the
          # distribution, which will be some kind of blob not too far from
          # Gaussian if we run several FIR iterations.
          n_iterations_float = tf.cast(n_iterations, self.dtype)
          delay_variance = spread_sq / n_iterations_float
          mean_delay = delay / n_iterations_float
          sq_mean_delay = tf.square(mean_delay)

          def taps_3(
              delay_variance: tf.Tensor, mean_delay: tf.Tensor
          ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            a = (delay_variance + sq_mean_delay - mean_delay) / 2.0
            b = (delay_variance + sq_mean_delay + mean_delay) / 2.0
            fir_left = a
            fir_mid = 1 - a - b
            fir_right = b
            done = fir_mid >= 0.25
            return (fir_left, fir_mid, fir_right, done)

          def taps_5(
              delay_variance: tf.Tensor, mean_delay: tf.Tensor
          ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
            a = (
                ((delay_variance + sq_mean_delay) * 2.0 / 5.0)
                - (mean_delay * 2.0 / 3.0)
            ) / 2.0
            b = (
                ((delay_variance + sq_mean_delay) * 2.0 / 5.0)
                + (mean_delay * 2.0 / 3.0)
            ) / 2.0
            fir_left = a / 2.0
            fir_mid = 1 - a - b
            fir_right = b / 2.0
            done = fir_mid >= 0.15
            return (fir_left, fir_mid, fir_right, done)

          (fir_left, fir_mid, fir_right, done) = tf.switch_case(
              tf.cast(n_taps, tf.int32),
              branch_fns={
                  0: lambda: (fir_left, fir_mid, fir_right, tf.constant(False)),
                  1: lambda: (fir_left, fir_mid, fir_right, tf.constant(False)),
                  2: lambda: (fir_left, fir_mid, fir_right, tf.constant(False)),
                  3: lambda: taps_3(delay_variance, mean_delay),
                  4: lambda: (fir_left, fir_mid, fir_right, tf.constant(False)),
                  5: lambda: taps_5(delay_variance, mean_delay),
              },
          )
          return (n_iterations, n_taps, done, fir_left, fir_mid, fir_right)

        return tf.cond(
            done,
            lambda: (n_iterations, n_taps, done, fir_left, fir_mid, fir_right),
            lambda: not_done(
                n_iterations, n_taps, fir_left, fir_mid, fir_right
            ),
        )

      (n_iterations, n_taps, done, fir_left, fir_mid, fir_right) = (
          tf.while_loop(
              lambda _a, _b, done, _c, _d, _e: tf.logical_not(done),
              compute_filter,
              (n_iterations, n_taps, done, fir_left, fir_mid, fir_right),
          )
      )
      # Once we have the FIR design for this stage we can assign it to the
      # appropriate data members.
      agc_coeff.agc_spatial_iterations = n_iterations
      agc_coeff.agc_spatial_n_taps = n_taps
      agc_coeff.agc_spatial_fir_left = fir_left
      agc_coeff.agc_spatial_fir_mid = fir_mid
      agc_coeff.agc_spatial_fir_right = fir_right
      total_dc_gain += tf.math.pow(agc_coeff.agc_stage_gain, stage)
      agc_coeff.agc_mix_coeffs = tf.constant(0, dtype=self.dtype)
      if stage != 0:
        agc_coeff.agc_mix_coeffs = self.agc_params.agc_mix_coeff / (
            tau * (self.car_params.sample_rate_hz / agc_coeff.decim)
        )
      agc_coeff.agc_gain = total_dc_gain
      agc_coeff.detect_scale = 1 / total_dc_gain
      previous_stage_gain = agc_coeff.agc_gain
      decim = agc_coeff.decim
    return _AGCCoeffs.concat(agc_coeffs).convert()

  def _carfac_detect(self, input_output: tf.Tensor) -> tf.Tensor:
    # Many comments in this function copied from cpp/carfac_util.h.
    a = 0.175
    b = 0.1
    # This offsets the low-end tail into negative x territory.
    # The parameter a is adjusted for the book, to make the 20% DC response
    # threshold at 0.1.
    c = tf.math.maximum(input_output, -a) + a
    # Zero is the final answer for many points.
    return tf.math.pow(c, 3) / (tf.math.pow(c, 3) + tf.math.square(c) + b)

  def _design_ihc_coeffs(self) -> _IHCCoeffsET:
    ihc_coeffs = _IHCCoeffs().cast(self.dtype)
    ihc_coeffs.ac_coeff = (
        2
        * np.pi
        * self.ihc_params.ac_corner_hz
        / self.car_params.sample_rate_hz
    )

    def build_capacitors() -> _IHCCoeffsET:
      conduct_at_10 = self._carfac_detect(tf.constant(10.0, dtype=self.dtype))
      conduct_at_0 = self._carfac_detect(tf.constant(0.0, dtype=self.dtype))

      def one_capacitor() -> _IHCCoeffsET:
        ro = 1.0 / conduct_at_10
        c = self.ihc_params.tau1_out / ro
        ri = self.ihc_params.tau1_in / c
        saturation_output = 1.0 / ((2.0 * ro) + ri)
        r0 = 1.0 / conduct_at_0
        current = 1.0 / (ri + r0)
        ihc_coeffs.cap1_voltage = 1.0 - (current * ri)
        ihc_coeffs.lpf_coeff = 1.0 - tf.math.exp(
            -1.0 / (self.ihc_params.tau_lpf * self.car_params.sample_rate_hz)
        )
        ihc_coeffs.out1_rate = ro / (
            self.ihc_params.tau1_out * self.car_params.sample_rate_hz
        )
        ihc_coeffs.in1_rate = 1.0 / (
            self.ihc_params.tau1_in * self.car_params.sample_rate_hz
        )
        ihc_coeffs.output_gain = 1.0 / (saturation_output - current)
        ihc_coeffs.rest_output = current / (saturation_output - current)
        ihc_coeffs.rest_cap1 = ihc_coeffs.cap1_voltage
        # Setting rest_cap2/out2_rate/in2_rate/cap2_voltage just to make
        # autograph accept that they have values with the correct shape.
        ihc_coeffs.rest_cap2 = ihc_coeffs.rest_cap1
        ihc_coeffs.out2_rate = ihc_coeffs.out1_rate
        ihc_coeffs.in2_rate = ihc_coeffs.in1_rate
        ihc_coeffs.cap2_voltage = ihc_coeffs.cap1_voltage
        return ihc_coeffs.convert()

      def two_capacitors() -> _IHCCoeffsET:
        ro = 1.0 / conduct_at_10
        c2 = self.ihc_params.tau2_out / ro
        r2 = self.ihc_params.tau2_in / c2
        c1 = self.ihc_params.tau1_out / r2
        r1 = self.ihc_params.tau1_in / c1
        saturation_output = 1.0 / (2.0 * ro + r2 + r1)
        r0 = 1.0 / conduct_at_0
        current = 1.0 / (r1 + r2 + r0)
        ihc_coeffs.cap1_voltage = 1.0 - (current * r1)
        ihc_coeffs.cap2_voltage = ihc_coeffs.cap1_voltage - (current * r2)
        ihc_coeffs.lpf_coeff = 1.0 - tf.math.exp(
            -1.0 / (self.ihc_params.tau_lpf * self.car_params.sample_rate_hz)
        )
        ihc_coeffs.out1_rate = 1.0 / (
            self.ihc_params.tau1_out * self.car_params.sample_rate_hz
        )
        ihc_coeffs.in1_rate = 1.0 / (
            self.ihc_params.tau1_in * self.car_params.sample_rate_hz
        )
        ihc_coeffs.out2_rate = ro / (
            self.ihc_params.tau2_out * self.car_params.sample_rate_hz
        )
        ihc_coeffs.in2_rate = 1.0 / (
            self.ihc_params.tau2_in * self.car_params.sample_rate_hz
        )
        ihc_coeffs.output_gain = 1.0 / (saturation_output - current)
        ihc_coeffs.rest_output = current / (saturation_output - current)
        ihc_coeffs.rest_cap1 = ihc_coeffs.cap1_voltage
        ihc_coeffs.rest_cap2 = ihc_coeffs.cap2_voltage
        return ihc_coeffs.convert()

      return tf.cond(
          self.ihc_params.one_capacitor != 0.0, one_capacitor, two_capacitors
      )

    return tf.cond(
        self.ihc_params.just_half_wave_rectify != 0.0,
        ihc_coeffs.convert,
        build_capacitors,
    )

  def _design_car_coeffs(self) -> _CARCoeffsET:
    max_hz = self.car_params.max_pole_ratio * self.car_params.sample_rate_hz
    ones = tf.ones(shape=(self.output_size[1],), dtype=self.dtype)
    pole_freqs = tf.concat(
        [
            [max_hz],
            self._recurrence_expander(
                max_hz,
                ones[1:]
                * (1 - self.car_params.erb_per_step / self.car_params.erb_q),
                ones[1:]
                * (
                    -self.car_params.erb_per_step
                    * self.car_params.erb_break_freq
                    / self.car_params.erb_q
                ),
            ),
        ],
        axis=0,
    )

    car_coeffs = _CARCoeffs()
    theta = pole_freqs * 2 * np.pi / self.car_params.sample_rate_hz
    car_coeffs.velocity_scale = ones * self.car_params.velocity_scale
    car_coeffs.v_offset = ones * self.car_params.v_offset
    car_coeffs.a0_coeffs = tf.math.cos(theta)
    car_coeffs.c0_coeffs = tf.math.sin(theta)
    car_coeffs.min_zeta = ones * self.car_params.compute_zeta(
        self.car_params.min_zeta_at_half_erb_per_step
    )
    car_coeffs.max_zeta = ones * self.car_params.compute_zeta(
        self.car_params.max_zeta_at_half_erb_per_step
    )
    x = theta / np.pi
    car_coeffs.zr_coeffs = np.pi * (
        x - (self.car_params.high_f_damping_compression * tf.math.pow(x, 3))
    )
    car_coeffs.r1_coeffs = 1.0 - (car_coeffs.zr_coeffs * car_coeffs.max_zeta)
    min_zetas = car_coeffs.min_zeta + (
        0.25
        * (
            (self.car_params.erb_hz(pole_freqs) / pole_freqs)
            - car_coeffs.min_zeta
        )
    )
    car_coeffs.zr_coeffs *= car_coeffs.max_zeta - min_zetas
    car_coeffs.h_coeffs = car_coeffs.c0_coeffs * (
        tf.math.square(self.car_params.zero_ratio) - 1.0
    )
    r = car_coeffs.r1_coeffs + car_coeffs.zr_coeffs
    car_coeffs.g0_coeffs = (
        1.0 - (2.0 * r * car_coeffs.a0_coeffs) + tf.math.square(r)
    ) / (
        1.0
        - (2.0 * r * car_coeffs.a0_coeffs)
        + (car_coeffs.h_coeffs * r * car_coeffs.c0_coeffs)
        + tf.math.square(r)
    )
    return car_coeffs.convert()

  def _ihc_step(
      self,
      ihc_coeffs: _IHCStateET,
      ihc_state_et: _IHCStateET,
      car_state_et: _CARStateET,
  ) -> _IHCStateET:
    car_out = car_state_et.zy_memory
    ihc_state = ihc_state_et.convert()
    ac_diff = car_out - ihc_state.ac_coupler
    ihc_state.ac_coupler += ihc_coeffs.ac_coeff * ac_diff

    def just_half_wave_rectify(
        ihc_state_et: _IHCStateET, ac_diff: tf.Tensor
    ) -> _IHCStateET:
      ihc_state = ihc_state_et.convert()
      ihc_state.ihc_out = tf.clip_by_value(ac_diff, 0, 2)
      return ihc_state.convert()

    def lpf_step(ihc_state_et: _IHCStateET, ac_diff: tf.Tensor) -> _IHCStateET:
      conductance = self._carfac_detect(ac_diff)

      def one_capacitor(ihc_state_et: _IHCStateET) -> _IHCStateET:
        ihc_state = ihc_state_et.convert()
        ihc_state.ihc_out = conductance * ihc_state.cap1_voltage
        ihc_state.cap1_voltage = (
            ihc_state.cap1_voltage
            - (ihc_state.ihc_out * ihc_coeffs.out1_rate)
            + ((1.0 - ihc_state.cap1_voltage) * ihc_coeffs.in1_rate)
        )
        return ihc_state.convert()

      def two_capacitors(ihc_state_et: _IHCStateET) -> _IHCStateET:
        ihc_state = ihc_state_et.convert()
        ihc_state.ihc_out = conductance * ihc_state.cap2_voltage
        ihc_state.cap1_voltage = (
            ihc_state.cap1_voltage
            - (
                (ihc_state.cap1_voltage - ihc_state.cap2_voltage)
                * ihc_coeffs.out1_rate
            )
            + ((1.0 - ihc_state.cap1_voltage) * ihc_coeffs.in1_rate)
        )
        ihc_state.cap2_voltage = (
            ihc_state.cap2_voltage
            - (ihc_state.ihc_out * ihc_coeffs.out2_rate)
            + (
                (ihc_state.cap1_voltage - ihc_state.cap2_voltage)
                * ihc_coeffs.in2_rate
            )
        )
        return ihc_state.convert()

      ihc_state_et = tf.cond(
          self.ihc_params.one_capacitor != 0.0,
          lambda: one_capacitor(ihc_state_et),
          lambda: two_capacitors(ihc_state_et),
      )
      ihc_state = ihc_state_et.convert()
      # Smooth the output twice using an LPF.
      ihc_state.lpf1_state += ihc_coeffs.lpf_coeff * (
          ihc_state.ihc_out * ihc_coeffs.output_gain - ihc_state.lpf1_state
      )
      ihc_state.lpf2_state += ihc_coeffs.lpf_coeff * (
          ihc_state.lpf1_state - ihc_state.lpf2_state
      )
      ihc_state.ihc_out = ihc_state.lpf2_state - ihc_coeffs.rest_output
      return ihc_state.convert()

    return tf.cond(
        self.ihc_params.just_half_wave_rectify != 0.0,
        lambda: just_half_wave_rectify(ihc_state.convert(), ac_diff),
        lambda: lpf_step(ihc_state.convert(), ac_diff),
    )

  def _agc_step(
      self,
      agc_coeffs: _AGCCoeffsET,
      agc_state_et: _AGCStateET,
      ihc_state_et: _IHCStateET,
  ) -> tuple[_AGCStateET, tf.Tensor]:
    if self.agc_params.decimation.shape[0] == 0:
      return (agc_state_et, tf.constant(False))

    # First copy the ihc_out to the input accumulator of the first stage,
    # and if the stage has reached its decimation phase, copy the input
    # accumulator (after decimation) forward to the next stage and repeat.
    def feed_stage_forward(
        agc_state_et: _AGCStateET,
        stage: tf.Tensor,
        last_decimated_stage: tf.Tensor,
        done: tf.Tensor,
        agc_in_out: tf.Tensor,
    ) -> tuple[_AGCStateET, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
      agc_states = agc_state_et.convert()
      agc_state = agc_states[stage]
      agc_state.input_accum += agc_in_out
      agc_state.decim_phase += 1
      agc_states.update(stage, agc_state)

      def decimate(
          agc_state_et: _AGCStateET, stage: tf.Tensor
      ) -> tuple[_AGCStateET, tf.Tensor, tf.Tensor, tf.Tensor]:
        agc_states = agc_state_et.convert()
        agc_state = agc_states[stage]
        agc_state.decim_phase = tf.zeros_like(agc_state.decim_phase)
        agc_states.update(stage, agc_state)
        return (
            agc_states.convert(),  # pytype: disable=bad-return-type  # dynamic-method-lookup
            stage >= self.agc_params.decimation.shape[0] - 1,
            stage,
            agc_state.input_accum / agc_coeffs[stage].decimation,
        )

      (agc_state_et, done, last_decimated_stage, agc_in_out) = tf.cond(
          agc_state.decim_phase[0, 0, 0]
          >= tf.cast(agc_coeffs[stage].decimation, agc_state.decim_phase.dtype),
          lambda: decimate(agc_states.convert(), stage),
          lambda: (
              agc_states.convert(),
              tf.constant(True),
              last_decimated_stage,
              agc_in_out,
          ),
      )
      stage = tf.cond(done, lambda: stage, lambda: stage + 1)
      return (agc_state_et, stage, last_decimated_stage, done, agc_in_out)

    agc_in_out = (
        ihc_state_et.ihc_out
        * agc_coeffs[self.agc_params.decimation.shape[0] - 1].detect_scale
    )
    (agc_state_et, _, last_decimated_stage, _, _) = tf.while_loop(
        lambda _a, _b, c, done, d: tf.logical_not(done),
        feed_stage_forward,
        (
            agc_state_et,
            tf.constant(0, tf.int32),
            tf.constant(-1, tf.int32),
            tf.constant(False),
            agc_in_out,
        ),
        shape_invariants=(
            agc_state_et.spec_with_unknown_batch(),
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((), tf.int32),
            tf.TensorSpec((), tf.bool),
            tf.TensorSpec(
                (None, agc_in_out.shape[1], agc_in_out.shape[2]),
                agc_in_out.dtype,
            ),
        ),
    )

    # Then move backwards from the last decimated stage and apply the input
    # accumulator of this tage along with the agc memory of the next stage
    # to the agc memory of this stage, and then smooth it.
    def feed_stage_backward(
        agc_state_et: _AGCStateET, stage: tf.Tensor, _: tf.Tensor
    ) -> tuple[_AGCStateET, tf.Tensor, tf.Tensor]:
      agc_states = agc_state_et.convert()
      agc_state = agc_states[stage]
      decim_accum = agc_state.input_accum / agc_coeffs[stage].decimation
      agc_states.update(stage, agc_state)

      def add_next(
          agc_states_et: _AGCStateET, stage: tf.Tensor, decim_accum: tf.Tensor
      ) -> tf.Tensor:
        return (
            decim_accum
            + agc_coeffs[stage].agc_stage_gain
            * agc_states_et[stage + 1].agc_memory
        )

      decim_accum = tf.cond(
          stage < agc_coeffs.decimation.shape[0] - 1,
          lambda: add_next(agc_states.convert(), stage, decim_accum),
          lambda: decim_accum,
      )
      agc_state.input_accum = tf.zeros_like(agc_state.input_accum)
      agc_state.agc_memory += agc_coeffs[stage].agc_epsilon * (
          decim_accum - agc_state.agc_memory
      )
      agc_state.agc_memory = self._agc_spatial_smooth(
          agc_coeffs, stage, agc_state.agc_memory
      )
      agc_states.update(stage, agc_state)
      return (
          agc_states.convert(),  # pytype: disable=bad-return-type  # dynamic-method-lookup
          stage - 1,
          stage < 1,
      )

    (agc_state_et, _, _) = tf.while_loop(
        lambda _a, _b, done: tf.logical_not(done),
        feed_stage_backward,
        (agc_state_et, last_decimated_stage, last_decimated_stage < 0),
    )
    return (agc_state_et, last_decimated_stage > -1)

  def _agc_spatial_smooth(
      self, agc_coeffs: _AGCCoeffsET, stage: tf.Tensor, agc_memory: tf.Tensor
  ) -> tf.Tensor:
    def spatial_smooth(stage: tf.Tensor, agc_memory: tf.Tensor) -> tf.Tensor:
      stage_coeffs = agc_coeffs[stage]
      kernel = tf.switch_case(
          tf.cast(stage_coeffs.agc_spatial_n_taps, tf.int32),
          {
              0: lambda: tf.zeros((3, 1, 1), dtype=self.dtype),
              1: lambda: tf.zeros((3, 1, 1), dtype=self.dtype),
              2: lambda: tf.zeros((3, 1, 1), dtype=self.dtype),
              3: lambda: tf.stack([
                  stage_coeffs.agc_spatial_fir_left,
                  stage_coeffs.agc_spatial_fir_mid,
                  stage_coeffs.agc_spatial_fir_right,
              ]),
              4: lambda: tf.zeros((3, 1, 1), dtype=self.dtype),
              5: lambda: tf.stack([
                  stage_coeffs.agc_spatial_fir_left,
                  stage_coeffs.agc_spatial_fir_left,
                  stage_coeffs.agc_spatial_fir_mid,
                  stage_coeffs.agc_spatial_fir_right,
                  stage_coeffs.agc_spatial_fir_right,
              ]),
          },
      )
      return self._convolver(
          tf.cast(stage_coeffs.agc_spatial_iterations, tf.int32),
          kernel,
          agc_memory,
      )

    return tf.cond(
        agc_coeffs[stage].agc_spatial_iterations >= 0,
        lambda: spatial_smooth(stage, agc_memory),
        lambda: self._agc_smooth_double_exponential(
            agc_coeffs[stage].agc_pole_z1,
            agc_coeffs[stage].agc_pole_z2,
            agc_memory,
        ),
    )

  def _agc_smooth_double_exponential(
      self, pole_z1: tf.Tensor, pole_z2: tf.Tensor, agc_memory: tf.Tensor
  ) -> tf.Tensor:
    ones = tf.ones_like(agc_memory)
    state1 = self._recurrence_expander(
        tf.zeros_like(agc_memory)[:, :, 0],
        ones[:, :, -11:] * pole_z1,
        (1 - pole_z1) * agc_memory[:, :, -11:],
    )[:, :, -1]
    state2 = self._recurrence_expander(
        state1, ones * pole_z2, (1 - pole_z2) * agc_memory[:, :, ::-1]
    )[:, :, -1]
    return self._recurrence_expander(
        state2, ones * pole_z1, (1 - pole_z1) * agc_memory
    )

  def _cross_couple(
      self, agc_coeffs: _AGCCoeffsET, agc_state_et: _AGCStateET
  ) -> _AGCStateET:
    def next_stage(
        agc_state_et: _AGCStateET, stage: tf.Tensor, done: tf.Tensor
    ) -> tuple[_AGCStateET, tf.Tensor, tf.Tensor]:
      def mix_channels(
          agc_state_et: _AGCStateET, stage: tf.Tensor
      ) -> _AGCStateET:
        agc_states = agc_state_et.convert()
        agc_state = agc_states[stage]
        mean_agc_state = (
            tf.math.reduce_sum(agc_state.agc_memory, axis=1)
            / agc_state.agc_memory.shape[1]
        )
        # Reintroduce the ear dimension.
        mean_agc_state = mean_agc_state[:, tf.newaxis, :]
        # Resize the ear dimension to the right number of ears.
        mean_agc_state = tf.tile(
            mean_agc_state, [1, agc_state.agc_memory.shape[1], 1]
        )
        # Return the old state for this stage + agc_mix_coeffs for this
        # stage * (mean - current).
        agc_state.agc_memory = agc_state.agc_memory + agc_coeffs[
            stage
        ].agc_mix_coeffs * (mean_agc_state - agc_state.agc_memory)
        agc_states.update(stage, agc_state)
        return agc_states.convert()

      agc_state_et = tf.cond(
          agc_coeffs[stage].agc_mix_coeffs > 0,
          lambda: mix_channels(agc_state_et, stage),
          lambda: agc_state_et,
      )
      stage += 1
      done = tf.cond(
          stage >= self.agc_params.decimation.shape[0],
          lambda: tf.constant(True),
          lambda: agc_state_et.decim_phase[0, 0, 0, stage] > 0,
      )
      return (agc_state_et, stage, done)

    (agc_state_et, _, _) = tf.while_loop(
        lambda _a, _b, done: tf.logical_not(done),
        next_stage,
        (agc_state_et, 0, agc_state_et.decim_phase[0, 0, 0, 0] > 0),
    )
    return agc_state_et

  def _close_agc_loop(
      self,
      agc_coeffs: _AGCCoeffsET,
      car_coeffs: _CARCoeffsET,
      car_state_et: _CARStateET,
      agc_state_et: _AGCStateET,
  ) -> _CARStateET:
    car_state = car_state_et.convert()
    if self._open_loop:
      # Zero the deltas to make the parameters not keep changing.
      car_state.dzb_memory = tf.zeros_like(car_state.dzb_memory)
      car_state.dg_memory = tf.zeros_like(car_state.dg_memory)
    else:
      # Scale factor to get the deltas to update in this many steps.
      scaling = 1.0 / agc_coeffs[tf.constant(0, tf.int32)].decimation
      undamping = 1.0 - agc_state_et[tf.constant(0, tf.int32)].agc_memory
      # This sets the delta for the damping zb.
      car_state.dzb_memory = (
          car_coeffs.zr_coeffs * undamping - car_state.zb_memory
      ) * scaling
      # Find new stage gains to go with new dampings.
      r = car_coeffs.r1_coeffs + car_coeffs.zr_coeffs * undamping
      g_values = (1 - 2 * r * car_coeffs.a0_coeffs + tf.math.square(r)) / (
          1
          - 2 * r * car_coeffs.a0_coeffs
          + car_coeffs.h_coeffs * r * car_coeffs.c0_coeffs
          + tf.math.square(r)
      )
      # This updates the target stage gain.
      car_state.dg_memory = (g_values - car_state.g_memory) * scaling
    return car_state.convert()

  def call(
      self,
      input_at_t: tf.Tensor,
      states_at_t: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
  ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """Computes output_at_t given input_at_t and states_at_t.

    Args:
      input_at_t: A [batch_size, n_ears, 1]-tensor with input at this step.
      states_at_t: A tuple of state tensors at this step, where each element is
        a [batch_size]-tensor.

    Returns:
      A tuple (output_amplitudes, states), where `output_amplitudes` is a
        [batch_size, n_ears, n_channels]-tensor, and `states` is a
        tuple of with state tensors at the next step.
    """

    car_coeffs = _CARCoeffsET.from_tensors(
        states_at_t[: len(dataclasses.fields(_CARCoeffs))]
    )
    states_at_t = states_at_t[len(dataclasses.fields(_CARCoeffs)) :]
    car_state = _CARStateET.from_tensors(
        states_at_t[: len(dataclasses.fields(_CARState))]
    )
    states_at_t = states_at_t[len(dataclasses.fields(_CARState)) :]
    ihc_coeffs = _IHCCoeffsET.from_tensors(
        states_at_t[: len(dataclasses.fields(_IHCCoeffs))]
    )
    states_at_t = states_at_t[len(dataclasses.fields(_IHCCoeffs)) :]
    ihc_state = _IHCStateET.from_tensors(
        states_at_t[: len(dataclasses.fields(_IHCState))]
    )
    states_at_t = states_at_t[len(dataclasses.fields(_IHCState)) :]
    if self.agc_params.decimation.shape.as_list():
      agc_coeffs = _AGCCoeffsET.from_tensors(
          states_at_t[: len(dataclasses.fields(_AGCCoeffs))]
      )
      states_at_t = states_at_t[len(dataclasses.fields(_AGCCoeffs)) :]
      agc_state = _AGCStateET.from_tensors(
          states_at_t[: len(dataclasses.fields(_AGCState))]
      )

    car_state = self._car_step(car_coeffs, input_at_t, car_state)
    if not self._linear:
      ihc_state = self._ihc_step(ihc_coeffs, ihc_state, car_state)
    agc_memory_updated = tf.constant(False)
    if (
        not self._open_loop
        and not self._linear
        and self.agc_params.decimation.shape.as_list()
    ):
      # `agc_coeffs` and `agc_state` are defined iff
      # `agc_params.decimation.shape.as_list()`, which is true in this branch.
      agc_state, agc_memory_updated = self._agc_step(
          agc_coeffs,
          agc_state,  # pylint: disable=used-before-assignment
          ihc_state,
      )

      def agc_was_updated(
          car_state: _CARStateET, agc_state: _AGCStateET
      ) -> tuple[_CARStateET, _AGCStateET]:
        if self.output_size[1] > 1:
          agc_state = self._cross_couple(agc_coeffs, agc_state)
        car_state = self._close_agc_loop(
            agc_coeffs, car_coeffs, car_state, agc_state
        )
        return (car_state, agc_state)

      (car_state, agc_state) = tf.cond(
          agc_memory_updated,
          lambda: agc_was_updated(car_state, agc_state),
          lambda: (car_state, agc_state),
      )

    next_state = [
        *car_coeffs.to_tensors(),
        *car_state.to_tensors(),
        *ihc_coeffs.to_tensors(),
        *ihc_state.to_tensors(),
    ]
    if self.agc_params.decimation.shape.as_list():
      next_state.extend(agc_coeffs.to_tensors() + agc_state.to_tensors())

    outputs: list[tf.Tensor] = []
    for output in self._outputs:
      if output == CARFACOutput.BM:
        outputs.append(car_state.zy_memory)
      elif output == CARFACOutput.OHC:
        outputs.append(car_state.za_memory)
      elif output == CARFACOutput.AGC:
        outputs.append(car_state.zb_memory)
      elif output == CARFACOutput.NAP:
        outputs.append(ihc_state.ihc_out)

    return (tf.stack(outputs, axis=3), tuple(next_state))

  def _car_step(
      self,
      car_coeffs: _CARCoeffsET,
      input_at_t: tf.Tensor,
      car_state_et: _CARStateET,
  ) -> _CARStateET:
    car_state = car_state_et.convert()
    # Many comments in this function from here copied from cpp/ear.cc.
    # Interpolates g.
    car_state.g_memory += car_state.dg_memory
    # Calculates the AGC interpolation state.
    car_state.zb_memory += car_state.dzb_memory
    # This updates the nonlinear function of 'velocity' along with zA, which is
    # a delay of z2.
    r = car_coeffs.r1_coeffs
    if self._linear:
      r += car_coeffs.zr_coeffs
    else:
      r += (
          # This product is the "undamping" delta r.
          car_state.zb_memory
          *
          # OHC nonlinear function.
          # We start with a quadratic nonlinear function, and limit it via a
          # rational function. This makes the result go to zero at high
          # absolute velocities, so it will do nothing there.
          (
              1.0
              / (
                  1.0
                  + tf.math.square(
                      (
                          car_coeffs.velocity_scale
                          * (car_state.z2_memory - car_state.za_memory)
                      )  # velocities
                      + car_coeffs.v_offset
                  )
              )
          )
      )
    car_state.za_memory = car_state.z2_memory
    # Here we reduce the CAR state by r and then rotate with the fixed cos/sin
    # coeffs, using both temp arrays, ending the scope of tmp1_ as r.
    tmp2: tf.tensor = r * car_state.z2_memory
    # But first stop using tmp1_ for r, and use it for r * z1 instead.
    tmp1 = r * car_state.z1_memory
    car_state.z1_memory = (  # This still needs stage inputs to be added.
        car_coeffs.a0_coeffs * tmp1 - car_coeffs.c0_coeffs * tmp2
    )
    car_state.z2_memory = (
        car_coeffs.c0_coeffs * tmp1 + car_coeffs.a0_coeffs * tmp2
    )
    car_state.zy_memory = car_coeffs.h_coeffs * car_state.z2_memory

    # This is the big ripple of the zy_memory. The recurrence expander computes
    # zy_memory[n] = g_memory[n] * zy_memory[n-1] + g_memory[n] * zy_memory[n].
    car_state.zy_memory = self._recurrence_expander(
        input_at_t[:, :, 0],
        car_state.g_memory,
        car_state.g_memory * car_state.zy_memory,
    )
    car_state.z1_memory = car_state.z1_memory + tf.concat(
        [input_at_t[:, :, 0:1], car_state.zy_memory[:, :, :-1]], axis=-1
    )

    return car_state.convert()


def plot_car_channels(
    cell: CARFACCell,
    window_size: int = 8192,
    figsize: tuple[float, float] = (10, 5),
    frequency_log_scale: bool = True,
) -> plt.Figure:
  """Plots the frequency response of the CAR layer of a CARFACCell.

  Will create a new linear and open-loop CARFACCell using the parameters of the
  provided cell and plot the frequency response of the BM output of the new
  cell.

  Args:
    cell: A CARFACCell to plot the output of.
    window_size: The window size for the frequency domain conversion.
    figsize: The size of the returned figure.
    frequency_log_scale: Whether to plot the frequency axis in log scale.

  Returns:
    A matplotlib.Figure.
  """
  linear_cell = CARFACCell(
      linear=True,
      outputs=(CARFACOutput.BM,),
      open_loop=True,
      car_params=cell.car_params,
      ihc_params=cell.ihc_params,
      agc_params=cell.agc_params,
  )
  layer = tf.keras.layers.RNN(linear_cell, return_sequences=True)
  layer.call = tf.function(layer.call)
  impulse: np.ndarray = np.zeros([1, window_size, 1, 1], dtype=cell.dtype)
  impulse[:, 0, :, :] = 1
  got = layer(impulse)
  transposed_got = tf.transpose(got[0, :, 0, :, 0], [1, 0])
  return pz.plot_z(
      np.fft.fft(transposed_got),
      frequency_log_scale=frequency_log_scale,
      figsize=figsize,
  )
