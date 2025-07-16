"""Implement Dick Lyon's cascade of asymmetric resonators in JAX.

Copyright 2023 The CARFAC Authors. All Rights Reserved.
Author: Richard F. Lyon

This file is part of an implementation of Lyon's cochlear model:
"Cascade of Asymmetric Resonators with Fast-Acting Compression"

Currently this package is not open sourced.
"""

import dataclasses
import functools
import math
import numbers
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

# This file contains all the implementation of CARFAC-JAX and includes the
# following sections.
# 1. Data structure definitions.
#    First, different data structures (`DesignParameters`, `Hypers`, `Weights`
#    and `State`) are defined for each step ("CAR", "AGC", "IHC"), each "ear"
#    and the whole CARFAC. The definitions include part of the default values.
#
#    Explanation and meanings of each type of data classes
#     - `DesignParameters`: used as the input to the design and init functions.
#       They are set manually by users and won't be changed in any functions.
#     - `Hypers`: the hyperparameters. They are produced from the design
#       functions and define the configurations of the model. They are tagged
#       `static` in `jax.jit` so one can use their values in conditionals etc.
#       Model functions or parameter optimization should never change them.
#       These are typically boolean and integer configuration parameters that
#       change the structure of the graph and thus are not differentiable.
#       Notice that, for convenience, we use `Hypers` to contain all the
#       parameters that are produced from design and won't be changed from then
#       on (i.e. in training and model computation). So its parameters may not
#       change the computation graph or even may not be used in any model
#       computation.
#     - `Weights`: the trainable weights. They are created and initialised by
#       the design and init functions. They are part of the inputs of model
#       functions. These values shouldn't be changed in the model functions and
#       should be updated by optimization algorithms.
#     - `State`: the state variables. These are implementation specific and keep
#       the internal state of the model. For the benefit of JAX, which
#       prefers/mandates a purely functional programming style, they are passed
#       as additional function inputs, and their new values are returned as
#       function outputs.
# 2. Design and init functions.
#    Second, design and init functions for each step ("CAR", "AGC", "IHC"), each
#    "ear" and the whole CARFAC are defined. These functions are not JITed so
#    one can use arbitrary python code in it.
#    Normally, users only need to use the `design_and_init_carfac` function to
#    obtained the hyperparameters, trainable weights (initalized) and states
#    (initalized). All the other design and init functions are called by
#    `design_and_init_carfac` function.
# 3. Model functions.
#    Third, the model functions for each step and the `run_segment` function are
#    defined. These functions are meant to be used after being JITted so one
#    must ensure they are JITable. A helper function `run_segment_jit` is also
#    defined to help users run the JITted version of `run_segment` conveniently.
#    But notice that `run_segment_jit` will donate the input state for speed.


###################
# Data Structures #
###################


# Because operations of `jax.numpy.ndarray` like `max/min`, `[]` will generate
# `jax.numpy.ndarray` of size 1 rather than numeric types like `float` or `int`,
# for convenience, we annotate some parameters as a union of numeric type and
# `jax.numpy.ndarray`. For example `Union[float, jnp.ndarray]`.
# TODO(honglinyu): should we avoid the unions and only annotate `jnp.ndarray`?
# Specifically, as pointed by @malcolmslaney, will this matter when we run it on
# TPU/GPU?


# TODO(honglinyu): can we use inheritance instead? Also, can we define a general
# hashing function too?
def _are_two_equal_hypers(hypers1, hypers2):
  """Used in comparing whether two `*Hypers` objects are equal in `__eq__()`."""
  if not isinstance(hypers1, type(hypers2)):
    return False
  children1, _ = hypers1.tree_flatten()
  children2, _ = hypers2.tree_flatten()
  for child1, child2 in zip(children1, children2):
    # Notice that when `__eq__` returns true for two objects, their `__hash__`
    # must return the same. So if `__hash__` hashes member variables' `id`, we
    # must compare `id` in `__eq__` too.
    if isinstance(child1, jnp.ndarray):
      if id(child1) != id(child2):
        return False
    else:
      if child1 != child2:
        return False
  return True


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class CarDesignParameters:
  """All the parameters manually set to design the CAR filterbank.

  This object is part of the `CarfacDesignParameter`. It should be assigned by
  the user. It is the input to the design/init functions and shouldn't be used
  in any model functions.
  """

  velocity_scale: float = 0.1  # for the velocity nonlinearity
  v_offset: float = 0.04  # offset gives a quadratic part
  min_zeta: float = 0.10  # minimum damping factor in mid-freq channels
  max_zeta: float = 0.35  # maximum damping factor in mid-freq channels
  first_pole_theta: float = 0.85 * math.pi
  zero_ratio: float = math.sqrt(2)  # how far zero is above pole
  high_f_damping_compression: float = 0.5  # 0 to 1 to compress zeta
  erb_per_step: float = 0.5  # assume G&M's ERB formula
  min_pole_hz: float = 30
  erb_break_freq: float = 165.3  # Greenwood map's break freq.
  erb_q: float = 1000 / (24.7 * 4.37)  # Glasberg and Moore's high-cf ratio
  use_delay_buffer: bool = False
  linear_car: bool = False
  ac_corner_hz: float = 20  # AC couple at 20 Hz corner

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.velocity_scale,
        self.v_offset,
        self.min_zeta,
        self.max_zeta,
        self.first_pole_theta,
        self.zero_ratio,
        self.high_f_damping_compression,
        self.erb_per_step,
        self.min_pole_hz,
        self.erb_break_freq,
        self.erb_q,
        self.use_delay_buffer,
        self.linear_car,
        self.ac_corner_hz,
    )
    aux_data = (
        'velocity_scale',
        'v_offset',
        'min_zeta',
        'max_zeta',
        'first_pole_theta',
        'zero_ratio',
        'high_f_damping_compression',
        'erb_per_step',
        'min_pole_hz',
        'erb_break_freq',
        'erb_q',
        'use_delay_buffer',
        'linear_car',
        'ac_corner_hz',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class CarHypers:
  """The hyperparameters of CAR (tagged as `static` in `jax.jit`).

  This object will be created and assigned in the design phase. It will be
  passed to the model functions as static arguments (as part of the
  `CarfacHypers`). It often contains booleans or integer configuration
  parameters that would change the structure of the graph, and thus are
  undifferentiable.
  """

  n_ch: int = 0
  use_delay_buffer: bool = False
  linear_car: bool = False

  # After https://github.com/google/carfac/commit/559a2f83, `r1_coeffs`,
  # `a0_coeffs`, `c0_coeffs`, `h_coeffs`, `g0_coeffs` and `zr_coeffs` are not
  # used in model computation so they are moved to `CarHypers`.
  r1_coeffs: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.zeros(())
  )
  a0_coeffs: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.zeros(())
  )
  c0_coeffs: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.zeros(())
  )
  h_coeffs: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.zeros(())
  )
  g0_coeffs: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.zeros(())
  )
  zr_coeffs: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.zeros(())
  )

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.n_ch,
        self.use_delay_buffer,
        self.linear_car,
        self.r1_coeffs,
        self.a0_coeffs,
        self.c0_coeffs,
        self.h_coeffs,
        self.g0_coeffs,
        self.zr_coeffs,
    )
    aux_data = (
        'n_ch',
        'use_delay_buffer',
        'linear_car',
        'r1_coeffs',
        'a0_coeffs',
        'c0_coeffs',
        'h_coeffs',
        'g0_coeffs',
        'zr_coeffs',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)

  # Needed by `static_argnums` of `jax.jit`.
  def __hash__(self):
    return hash((
        self.n_ch,
        self.use_delay_buffer,
        self.linear_car,
        id(self.r1_coeffs),
        id(self.a0_coeffs),
        id(self.c0_coeffs),
        id(self.h_coeffs),
        id(self.g0_coeffs),
        id(self.zr_coeffs),
    ))

  def __eq__(self, other):
    return _are_two_equal_hypers(self, other)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class CarWeights:
  """The trainable weights of the filters.

  As part of the `CarfacWeights`, it will be passed to the model functions but
  shouldn't be modified in the model computations.
  """

  velocity_scale: float = 0.0
  v_offset: float = 0.0

  ohc_health: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.zeros(())
  )
  # The parameters for `stage_g`.
  ga_coeffs: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.zeros(())
  )
  gb_coeffs: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.zeros(())
  )
  gc_coeffs: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.zeros(())
  )

  ac_coeff: float = 0.0

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.velocity_scale,
        self.v_offset,
        self.ohc_health,
        self.ga_coeffs,
        self.gb_coeffs,
        self.gc_coeffs,
        self.ac_coeff,
    )
    aux_data = (
        'velocity_scale',
        'v_offset',
        'ohc_health',
        'ga_coeffs',
        'gb_coeffs',
        'gc_coeffs',
        'ac_coeff',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class CarState:
  """All the state variables for the CAR filterbank."""

  z1_memory: jnp.ndarray
  z2_memory: jnp.ndarray
  za_memory: jnp.ndarray
  zb_memory: jnp.ndarray
  dzb_memory: jnp.ndarray
  zy_memory: jnp.ndarray
  g_memory: jnp.ndarray
  dg_memory: jnp.ndarray
  ac_coupler: jnp.ndarray

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.z1_memory,
        self.z2_memory,
        self.za_memory,
        self.zb_memory,
        self.dzb_memory,
        self.zy_memory,
        self.g_memory,
        self.dg_memory,
        self.ac_coupler,
    )
    aux_data = (
        'z1_memory',
        'z2_memory',
        'za_memory',
        'zb_memory',
        'dzb_memory',
        'zy_memory',
        'g_memory',
        'dg_memory',
        'ac_coupler',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class AgcDesignParameters:
  """All the parameters set manually to design the AGC filters."""

  n_stages: int = 4
  time_constants: jnp.ndarray = dataclasses.field(
      default_factory=lambda: 0.002 * 4.0 ** jnp.arange(4, dtype=float)
  )
  agc_stage_gain: float = 2.0  # gain from each stage to next slower stage
  # how often to update the AGC states
  decimation: Tuple[int, ...] = (8, 2, 2, 2)
  agc1_scales: jnp.ndarray = dataclasses.field(
      default_factory=lambda: 1.0
      * math.sqrt(2) ** jnp.arange(4, dtype=float)  # 1 per channel
  )
  agc2_scales: jnp.ndarray = dataclasses.field(
      default_factory=lambda: 1.65 * math.sqrt(2) ** jnp.arange(4, dtype=float)
  )
  agc_mix_coeffs: float = 0.5

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.n_stages,
        self.time_constants,
        self.agc_stage_gain,
        self.decimation,
        self.agc1_scales,
        self.agc2_scales,
        self.agc_mix_coeffs,
    )
    aux_data = (
        'n_stages',
        'time_constants',
        'agc_stage_gain',
        'decimation',
        'agc1_scales',
        'agc2_scales',
        'agc_mix_coeffs',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class AgcHypers:
  """The hyperparameters (tagged as `static` in `jax.jit`) of AGC step.

  As part of the `Coefficients`, it is designed/inited based on the `Parameters`
  In the design/init functions. It will be passed to the model functions as
  static arguments.
  Each AGC stage will have 1 `AgcCoefficients`.

  Attributes:
    n_ch: number of channels.
    n_agc_stages: number of AGC stages.
    decimation: the decimation of this AGC stage.
    agc_spatial_iterations: how many times FIR smoothings will be run.
    agc_spatial_n_taps: number of taps of the FIR filter.
    reverse_cumulative_decimation: the cumulative decimation of each stage in
      reverse order.
    max_cumulative_decimation: the maximum cumulative decimation across all AGC
      stages.
  """

  n_ch: int
  n_agc_stages: int
  decimation: int = 0
  agc_spatial_iterations: int = 0
  agc_spatial_n_taps: int = 0
  reverse_cumulative_decimation: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.array([64, 32, 16, 8], dtype=int)
  )
  max_cumulative_decimation: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.array(64, dtype=int)
  )

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.n_ch,
        self.n_agc_stages,
        self.decimation,
        self.agc_spatial_iterations,
        self.agc_spatial_n_taps,
        self.reverse_cumulative_decimation,
        self.max_cumulative_decimation,
    )
    aux_data = (
        'n_ch',
        'n_agc_stages',
        'decimation',
        'agc_spatial_iterations',
        'agc_spatial_n_taps',
        'reverse_cumulative_decimation',
        'max_cumulative_decimation',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)

  # Needed by `static_argnums` of `jax.jit`.
  def __hash__(self):
    # Notice that we hash the `id` of `jnp.ndarray` rather than its content for
    # speed. This should always be correct because `jnp.ndarray` is immutable.
    # But this also means the hash will be different if this field is assigned
    # to a different array with exactly the same value. We think such case
    # should be very rare in usage.
    return hash((
        self.n_ch,
        self.n_agc_stages,
        self.decimation,
        self.agc_spatial_iterations,
        self.agc_spatial_n_taps,
        id(self.reverse_cumulative_decimation),
        id(self.max_cumulative_decimation),
    ))

  def __eq__(self, other):
    return _are_two_equal_hypers(self, other)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class AgcWeights:
  """Trainable weights of the AGC step."""

  agc_epsilon: float = 0.0
  agc_stage_gain: float = 0.0
  agc_spatial_fir: Optional[List[Union[float, jnp.ndarray]]] = None
  detect_scale: float = 1.0
  agc_mix_coeffs: Union[float, jnp.ndarray] = 0.0
  AGC_polez1: Union[float, jnp.ndarray] = 0.0  # pylint: disable=invalid-name
  AGC_polez2: Union[float, jnp.ndarray] = 0.0  # pylint: disable=invalid-name

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.agc_epsilon,
        self.agc_stage_gain,
        self.agc_spatial_fir,
        self.detect_scale,
        self.agc_mix_coeffs,
        self.AGC_polez1,
        self.AGC_polez2,
    )
    aux_data = (
        'agc_epsilon',
        'agc_stage_gain',
        'agc_spatial_fir',
        'detect_scale',
        'agc_mix_coeffs',
        'AGC_polez1',
        'AGC_polez2',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class AgcState:
  """All the state variables for one stage of the AGC."""

  decim_phase: int
  agc_memory: jnp.ndarray
  input_accum: jnp.ndarray

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (self.decim_phase, self.agc_memory, self.input_accum)
    aux_data = ('decim_phase', 'agc_memory', 'input_accum')
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class IhcDesignParameters:
  """Variables needed for the inner hair cell implementation."""

  ihc_style: str = 'two_cap'
  tau_lpf: float = 0.000080  # 80 microseconds smoothing twice
  tau_out: float = 0.0005  # depletion tau is pretty fast
  tau_in: float = 0.010  # recovery tau is slower
  tau1_out: float = 0.00050  # depletion tau is pretty fast
  tau1_in: float = 0.00020  # recovery tau is very fast 200 us
  tau2_out: float = 0.001  # depletion tau is pretty fast 1 ms
  tau2_in: float = 0.010  # recovery tau is slower 10ms

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.ihc_style,
        self.tau_lpf,
        self.tau_out,
        self.tau_in,
        self.tau1_out,
        self.tau1_in,
        self.tau2_out,
        self.tau2_in,
    )
    aux_data = (
        'ihc_style',
        'tau_lpf',
        'tau_out',
        'tau_in',
        'tau1_out',
        'tau1_in',
        'tau2_out',
        'tau2_in',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SynDesignParameters:
  """Variables needed for the synapse implementation."""

  n_classes: int = 3  # Default. Modify params and redesign to change.
  ihcs_per_channel: dataclasses.InitVar[int] = 10  # Maybe 20 would be better?
  healthy_n_fibers: jnp.ndarray = dataclasses.field(
      default_factory=lambda: (jnp.array([50.0, 35.0, 25.0], dtype=float))
  )
  spont_rates: jnp.ndarray = dataclasses.field(
      default_factory=lambda: (jnp.array([50.0, 6.0, 1.0], dtype=float))
  )
  sat_rates: float = 200.0
  sat_reservoir: float = 0.2
  v_width: float = 0.02
  tau_lpf: float = 0.000080
  reservoir_tau: float = 0.02
  # Tweaked.
  # The weights 1.2 were picked before correctly account for sample rate
  # and number of fibers. This way works for more different numbers.
  agc_weights: jnp.ndarray = dataclasses.field(
      default_factory=lambda: (
          jnp.array([1.2, 1.2, 1.2], dtype=float) / (float(22050))
      )
  )

  def __post_init__(self, ihcs_per_channel):
    self.healthy_n_fibers *= ihcs_per_channel
    self.agc_weights /= ihcs_per_channel

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  # Note that ihcs_per_channel is set to 1 for flattening/unflattening,
  # to avoid repeat multiplication in __post_init__.
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.n_classes,
        1,
        self.healthy_n_fibers,
        self.spont_rates,
        self.sat_rates,
        self.sat_reservoir,
        self.v_width,
        self.tau_lpf,
        self.reservoir_tau,
        self.agc_weights,
    )
    aux_data = (
        'n_classes',
        'ihcs_per_channel',
        'healthy_n_fibers',
        'spont_rates',
        'sat_rates',
        'sat_reservoir',
        'v_width',
        'tau_lpf',
        'reservoir_tau',
        'agc_weights',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SynHypers:
  """Hyperparameters for the IHC synapse. Tagged `static` in `jax.jit`."""

  do_syn: bool = False

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):
    children = [self.do_syn]
    aux_data = ('do_syn',)
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)

  # Needed by `static_argnums` of `jax.jit`.
  def __hash__(self):
    children, _ = self.tree_flatten()
    return hash(children)

  def __eq__(self, other):
    return _are_two_equal_hypers(self, other)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SynWeights:
  """Trainable weights of the IHC synapse."""

  n_fibers: jnp.ndarray
  v_widths: jnp.ndarray
  v_halfs: jnp.ndarray  # Same units as v_recep and v_widths.
  a1: float  # Feedback gain
  a2: float  # Output gain
  agc_weights: jnp.ndarray  # For making a nap out to agc in.
  spont_p: jnp.ndarray  # Used only to init the output LPF
  spont_sub: jnp.ndarray
  res_lpf_inits: jnp.ndarray
  res_coeff: float
  lpf_coeff: float

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):
    """Flatten tree for pytree."""
    children = (
        self.n_fibers,
        self.v_widths,
        self.v_halfs,
        self.a1,
        self.a2,
        self.agc_weights,
        self.spont_p,
        self.spont_sub,
        self.res_lpf_inits,
        self.res_coeff,
        self.lpf_coeff,
    )
    aux_data = (
        'n_fibers',
        'v_widths',
        'v_halfs',
        'a1',
        'a2',
        'agc_weights',
        'spont_p',
        'spont_sub',
        'res_lpf_inits',
        'res_coeff',
        'lpf_coeff',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SynState:
  """All the state variables for the IHC synapse."""

  reservoirs: jnp.ndarray
  lpf_state: jnp.ndarray

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):
    children = (self.reservoirs, self.lpf_state)
    aux_data = ('reservoirs', 'lpf_state')
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class IhcHypers:
  """Hyperparameters for the inner hair cell. Tagged `static` in `jax.jit`."""

  n_ch: int
  # 0 is just_hwr, 1 is one_cap, 2 is two_cap.
  ihc_style: int

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):
    children = (self.n_ch, self.ihc_style)
    aux_data = ('n_ch', 'ihc_style')
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)

  # Needed by `static_argnums` of `jax.jit`.
  def __hash__(self):
    children, _ = self.tree_flatten()
    return hash(tuple(children))

  def __eq__(self, other):
    return _are_two_equal_hypers(self, other)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class IhcWeights:
  """Trainable weights of the IHC step."""

  lpf_coeff: float = 0.0
  out1_rate: Union[float, jnp.ndarray] = 0.0
  in1_rate: float = 0.0
  out2_rate: Union[float, jnp.ndarray] = 0.0
  in2_rate: Union[float, jnp.ndarray] = 0.0
  output_gain: Union[float, jnp.ndarray] = 0.0
  rest_output: Union[float, jnp.ndarray] = 0.0
  rest_cap2: Union[float, jnp.ndarray] = 0.0
  rest_cap1: Union[float, jnp.ndarray] = 0.0

  rest_cap: Union[float, jnp.ndarray] = 0.0
  out_rate: Union[float, jnp.ndarray] = 0.0
  in_rate: float = 0.0

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.lpf_coeff,
        self.out1_rate,
        self.in1_rate,
        self.out2_rate,
        self.in2_rate,
        self.output_gain,
        self.rest_output,
        self.rest_cap2,
        self.rest_cap1,
        self.rest_cap,
        self.out_rate,
        self.in_rate,
    )
    aux_data = (
        'lpf_coeff',
        'out1_rate',
        'in1_rate',
        'out2_rate',
        'in2_rate',
        'output_gain',
        'rest_output',
        'rest_cap2',
        'rest_cap1',
        'rest_cap',
        'out_rate',
        'in_rate',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


# States can be donated so we must use default_factory to create a freshly new
# one every time.
@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class IhcState:
  """All the state variables for the inner-hair cell implementation."""

  ihc_accum: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.array(0, dtype=float)
  )
  cap_voltage: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.array(0, dtype=float)
  )
  lpf1_state: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.array(0, dtype=float)
  )
  lpf2_state: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.array(0, dtype=float)
  )

  cap1_voltage: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.array(0, dtype=float)
  )
  cap2_voltage: jnp.ndarray = dataclasses.field(
      default_factory=lambda: jnp.array(0, dtype=float)
  )

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.ihc_accum,
        self.cap_voltage,
        self.lpf1_state,
        self.lpf2_state,
        self.cap1_voltage,
        self.cap2_voltage,
    )
    aux_data = (
        'ihc_accum',
        'cap_voltage',
        'lpf1_state',
        'lpf2_state',
        'cap1_voltage',
        'cap2_voltage',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class EarDesignParameters:
  """Parameters manually set to design Carfac for 1 ear."""

  car: CarDesignParameters = dataclasses.field(
      default_factory=CarDesignParameters
  )
  agc: AgcDesignParameters = dataclasses.field(
      default_factory=AgcDesignParameters
  )
  ihc: IhcDesignParameters = dataclasses.field(
      default_factory=IhcDesignParameters
  )
  syn: SynDesignParameters = dataclasses.field(
      default_factory=SynDesignParameters
  )

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (self.car, self.agc, self.ihc)
    aux_data = ('car', 'agc', 'ihc')
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class EarHypers:
  """Hyperparameters (tagged as static in `jax.jit`) of 1 ear."""

  n_ch: int
  pole_freqs: jnp.ndarray
  max_channels_per_octave: float
  car: CarHypers
  agc: List[AgcHypers]  # One element per AGC layer.
  ihc: IhcHypers
  syn: SynHypers

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (
        self.n_ch,
        self.pole_freqs,
        self.max_channels_per_octave,
        self.car,
        self.agc,
        self.ihc,
        self.syn,
    )
    aux_data = (
        'n_ch',
        'pole_freqs',
        'max_channels_per_octave',
        'car',
        'agc',
        'ihc',
        'syn',
    )
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)

  # Needed by `static_argnums` of `jax.jit`.
  def __hash__(self):
    # Notice that we hash the `id` of `jnp.ndarray` rather than its content for
    # speed. This should always be correct because `jnp.ndarray` is immutable.
    # But this also means the hash will be different if this field is assigned
    # to a different array with exactly the same value. We think such case
    # should be very rare in usage.
    return hash((
        self.n_ch,
        id(self.pole_freqs),
        self.max_channels_per_octave,
        self.car,
        tuple(self.agc),
        self.ihc,
    ))

  def __eq__(self, other):
    return _are_two_equal_hypers(self, other)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class EarWeights:
  """Trainable weights of 1 ear."""

  car: CarWeights
  agc: List[AgcWeights]
  ihc: IhcWeights
  syn: SynWeights

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (self.car, self.agc, self.ihc, self.syn)
    aux_data = ('car', 'agc', 'ihc', 'syn')
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class EarState:
  """The state of 1 ear."""

  car: CarState
  ihc: IhcState
  agc: List[AgcState]
  syn: SynState

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = (self.car, self.ihc, self.agc, self.syn)
    aux_data = ('car', 'ihc', 'agc', 'syn')
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class CarfacDesignParameters:
  """All the parameters set manually for designing CARFAC."""

  fs: float = 22050.0
  n_ears: int = 1
  ears: List[EarDesignParameters] = dataclasses.field(
      default_factory=lambda: [EarDesignParameters()]
  )

  def __init__(self, fs=22050.0, n_ears=1, use_delay_buffer=False):
    """Initialize the Design Parameters dataclass.

    Args:
      fs: Samples per second.
      n_ears: Number of ears to design for.
      use_delay_buffer: Whether to use the delay buffer implementation for the
        car_step.
    """
    self.fs = fs
    self.n_ears = n_ears
    self.ears = [EarDesignParameters() for _ in range(n_ears)]
    for ear in self.ears:
      ear.car.use_delay_buffer = use_delay_buffer

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):
    children = (self.fs, self.n_ears, self.ears)
    aux_data = ('fs', 'n_ears', 'ears')
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class CarfacHypers:
  """All the static variables (tagged as `static` in jax.jit)."""

  ears: List[EarHypers] = dataclasses.field(default_factory=list)
  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html

  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = [self.ears]
    aux_data = ['ears']
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)

  # Needed by `static_argnums` of `jax.jit`.
  def __hash__(self):
    children, _ = self.tree_flatten()
    return hash(tuple(map(tuple, children)))

  def __eq__(self, other):
    return _are_two_equal_hypers(self, other)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class CarfacWeights:
  """All the trainable weights."""

  ears: List[EarWeights] = dataclasses.field(default_factory=list)

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):
    children = [self.ears]  # pylint: disable=missing-function-docstring
    aux_data = ['ears']
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class CarfacState:
  """All the state variables."""

  ears: List[EarState] = dataclasses.field(default_factory=list)

  # The following 2 functions are boiler code required by pytree.
  # Reference: https://jax.readthedocs.io/en/latest/pytrees.html
  def tree_flatten(self):  # pylint: disable=missing-function-docstring
    children = [self.ears]
    aux_data = ['ears']
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)


#############################
# Design and Init Functions #
#############################


# Design/init functions shouldn't be JIT'ed.
# TODO(honglinyu): Now design and init are combined in a single function. This
# is mainly for convenience because `Hypers`, `Weights` and `State` are all set
# up together in the MATLAB and numpy versions. Should we decouple "design" and
# "init" into separate functions?


def hz_to_erb(
    cf_hz: Union[float, jnp.ndarray],
    erb_break_freq: float = 1000 / 4.37,
    erb_q: float = 1000 / (24.7 * 4.37),
):
  """Auditory filter nominal Equivalent Rectangular Bandwidth.

  Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
  ERB = (break_frequency + CF ) / Q, where the break_frequency might default's
  to Greenwood's 165.3, but could be Glasberg Moore's 228.83.




  Args:
    cf_hz: A scalar or a vector of frequencies (CF) to convert to ERB scale
    erb_break_freq: The corner frequency where we go from linear to log
      bandwidth
    erb_q: The width of one filter (Q = cf/Bandwidth)

  Returns:
    A scalar or vector with the ERB scale for the input frequencies.
  """
  return (erb_break_freq + cf_hz) / erb_q


def design_and_init_filters(
    ear: int, params: CarfacDesignParameters, pole_freqs: jnp.ndarray
) -> Tuple[CarHypers, CarWeights, CarState]:
  """Design the actual CAR filters.

  Args:
    ear: the index of the ear to be designed and initialised.
    params: The design parameters.
    pole_freqs: Where to put the poles for each channel.

  Returns:
    The CAR coefficient structure which allows the filters be computed quickly.
  """

  ear_params = params.ears[ear]

  n_ch = len(pole_freqs)

  # Declares return values.
  car_hypers = CarHypers(
      n_ch=n_ch,
      use_delay_buffer=ear_params.car.use_delay_buffer,
      linear_car=ear_params.car.linear_car,
  )
  car_weights = CarWeights()
  car_weights.velocity_scale = ear_params.car.velocity_scale
  car_weights.v_offset = ear_params.car.v_offset
  car_weights.ac_coeff = 2 * jnp.pi * ear_params.car.ac_corner_hz / params.fs

  car_weights.ohc_health = jnp.ones(n_ch)

  # zero_ratio comes in via h.  In book's circuit D, zero_ratio is 1/sqrt(a),
  # and that a is here 1 / (1+f) where h = f*c.
  # solve for f:  1/zero_ratio^2 = 1 / (1+f)
  # zero_ratio^2 = 1+f => f = zero_ratio^2 - 1
  f = ear_params.car.zero_ratio**2 - 1  # nominally 1 for half-octave

  # Make pole positions, s and c coeffs, h and g coeffs, etc.,
  # which mostly depend on the pole angle theta:
  theta = pole_freqs * (2 * math.pi / params.fs)

  c0 = jnp.sin(theta)
  a0 = jnp.cos(theta)

  # different possible interpretations for min-damping r:
  # r = exp(-theta * car_params.min_zeta).
  # Compress theta to give somewhat higher Q at highest thetas:
  ff = ear_params.car.high_f_damping_compression  # 0 to 1; typ. 0.5
  x = theta / math.pi
  zr_coeffs = math.pi * (x - ff * x**3)  # when ff is 0, this is just theta,
  #                       and when ff is 1 it goes to zero at theta = pi.
  max_zeta = ear_params.car.max_zeta
  car_hypers.r1_coeffs = (
      1 - zr_coeffs * max_zeta
  )  # "r1" for the max-damping condition
  min_zeta = ear_params.car.min_zeta
  # Increase the min damping where channels are spaced out more, by pulling
  # 25% of the way toward hz_to_erb/pole_freqs (close to 0.1 at high f)
  min_zetas = min_zeta + 0.25 * (
      hz_to_erb(
          pole_freqs,
          ear_params.car.erb_break_freq,
          ear_params.car.erb_q,
      )
      / pole_freqs
      - min_zeta
  )
  car_hypers.zr_coeffs = zr_coeffs * (max_zeta - min_zetas)
  # how r relates to undamping

  # undamped coupled-form coefficients:
  car_hypers.a0_coeffs = a0
  car_hypers.c0_coeffs = c0

  # the zeros follow via the h_coeffs
  h = c0 * f
  car_hypers.h_coeffs = h

  # Efficient approximation with g as quadratic function of undamping.
  # First get g at both ends and the half-way point. (ref:
  # https://github.com/google/carfac/commit/559a2f83)
  undamping = 0.0
  g0 = design_stage_g(car_hypers, undamping)
  undamping = 1.0
  g1 = design_stage_g(car_hypers, undamping)
  undamping = 0.5
  ghalf = design_stage_g(car_hypers, undamping)
  # Store fixed coefficients for A*undamping.^2 + B^undamping + C
  car_weights.ga_coeffs = 2 * (g0 + g1 - 2 * ghalf)
  car_weights.gb_coeffs = 4 * ghalf - 3 * g0 - g1
  car_weights.gc_coeffs = g0
  # Set up initial stage gains.
  # Maybe re-do this at Init time?
  undamping = car_weights.ohc_health
  # Avoid running this model function at Design time; see tests.
  car_hypers.g0_coeffs = design_stage_g(car_hypers, undamping)  # pytype: disable=wrong-arg-types  # jnp-type

  # Init car states
  car_state = CarState(
      z1_memory=jnp.zeros((n_ch,)),
      z2_memory=jnp.zeros((n_ch,)),
      za_memory=jnp.zeros((n_ch,)),
      zb_memory=jnp.copy(car_hypers.zr_coeffs),  # "States" will be donated.
      dzb_memory=jnp.zeros((n_ch,)),
      zy_memory=jnp.zeros((n_ch,)),
      g_memory=jnp.copy(car_hypers.g0_coeffs),  # "States" will be donated.
      dg_memory=jnp.zeros((n_ch,)),
      ac_coupler=jnp.zeros((n_ch,)),
  )
  return car_hypers, car_weights, car_state


def design_stage_g(
    car_hypers: CarHypers, relative_undamping: float
) -> jnp.ndarray:
  """Return the stage gain g needed to get unity gain at DC.

  This function is called only in design functions. So it isn't JITed.
  Args:
    car_hypers: The hyperparameters for the filterbank.
    relative_undamping:  The r variable, defined in section 17.4 of Lyon's book
      to be  (1-b) * NLF(v). The first term is undamping (because b is the
      gain).

  Returns:
    A float vector with the gain for each AGC stage.
  """

  r1 = car_hypers.r1_coeffs  # at max damping
  a0 = car_hypers.a0_coeffs
  c0 = car_hypers.c0_coeffs
  h = car_hypers.h_coeffs
  zr = car_hypers.zr_coeffs
  r = r1 + zr * relative_undamping
  g = (1 - 2 * r * a0 + r**2) / (1 - 2 * r * a0 + h * r * c0 + r**2)

  return g


def ihc_detect(x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
  """An IHC-like sigmoidal detection nonlinearity for the CARFAC.

  Resulting conductance is in about [0...1.3405]
  Args:
    x: the input (BM motion) to the inner hair cell

  Returns:
    The IHC conductance
  """

  if isinstance(x, numbers.Number):
    x_in = jnp.array((float(x),))
  else:
    x_in = x

  # offset of low-end tail into neg x territory
  # this parameter is adjusted for the book, to make the 20% DC
  # response threshold at 0.1
  a = 0.175

  z = jnp.maximum(0.0, x_in + a)
  conductance = z**3 / (z**3 + z**2 + 0.1)

  if isinstance(x, numbers.Number):
    return conductance[0]
  return conductance


def design_and_init_ihc_syn(
    ear: int, params: CarfacDesignParameters, n_ch: int
) -> Tuple[SynHypers, Optional[SynWeights], Optional[SynState]]:
  """Design the IHC Synapse Hypers/Weights/State.

  Only returns State and Weights if Params and CarfacHypers are set up to use
  synapse.

  Args:
    ear: the index of the ear.
    params: all the design parameters
    n_ch: How many channels there are in the filterbank.

  Returns:
    A tuple containing the newly created and initialised synapse coefficients,
    weights and state.
  """
  ear_params = params.ears[ear]
  ihc_params = ear_params.ihc
  syn_params = ear_params.syn
  if ihc_params.ihc_style != 'two_cap_with_syn':
    hypers = SynHypers(do_syn=False)
    return hypers, None, None

  hypers = SynHypers(do_syn=True)
  fs = params.fs
  n_classes = syn_params.n_classes
  v_widths = syn_params.v_width * jnp.ones((1, n_classes))
  # Do some design.  First, gains to get sat_rate when sigmoid is 1, which
  # involves the reservoir steady-state solution.
  # Most of these are not per-channel, but could be expanded that way
  # later if desired.
  # Mean sat prob of spike per sample per neuron, likely same for all
  # classes.
  # Use names 1 for sat and 0 for spont in some of these.
  p1 = syn_params.sat_rates / fs
  p0 = syn_params.spont_rates / fs
  w1 = syn_params.sat_reservoir
  q1 = 1 - w1
  # Assume the sigmoid is switching between 0 and 1 at 50% duty cycle, so
  # normalized mean value is 0.5 at saturation.
  s1 = 0.5
  r1 = s1 * w1
  # solve q1 = a1*r1 for gain coeff a1:
  a1 = q1 / r1
  # solve p1 = a2*r1 for gain coeff a2:
  a2 = p1 / r1
  # Now work out how to get the desired spont.
  r0 = p0 / a2
  q0 = r0 * a1
  w0 = 1 - q0
  s0 = r0 / w0
  # Solve for (negative) sigmoid midpoint offset that gets s0 right.
  offsets = jnp.log((1 - s0) / s0)
  spont_p = a2 * w0 * s0  # should match p0; check it; yes it does
  agc_weights = fs * syn_params.agc_weights
  spont_sub = (syn_params.healthy_n_fibers * spont_p) @ jnp.transpose(
      agc_weights
  )
  weights = SynWeights(
      n_fibers=jnp.ones((n_ch, 1)) * syn_params.healthy_n_fibers[None, :],
      v_widths=v_widths,
      v_halfs=offsets[None, :] * v_widths,
      a1=a1,
      a2=a2,
      agc_weights=agc_weights,
      spont_p=spont_p,
      spont_sub=spont_sub,
      res_lpf_inits=q0,
      res_coeff=1 - math.exp(-1 / (syn_params.reservoir_tau * fs)),
      lpf_coeff=1 - math.exp(-1 / (syn_params.tau_lpf * fs)),
  )
  state = SynState(
      reservoirs=jnp.ones((n_ch, 1)) * weights.res_lpf_inits[None, :],
      lpf_state=jnp.ones((n_ch, 1)) * weights.spont_p[None, :],
  )
  return hypers, weights, state


def design_and_init_ihc(
    ear: int, params: CarfacDesignParameters, n_ch: int
) -> Tuple[IhcHypers, IhcWeights, IhcState]:
  """Design the inner hair cell implementation from parameters.

  Args:
    ear: the index of the ear.
    params: all the design parameters
    n_ch: How many channels there are in the filterbank.

  Returns:
    A tuple containing the newly created and initialised IHC coefficients,
    weights and state.
  """
  ear_params = params.ears[ear]

  ihc_params = ear_params.ihc

  ihc_style_num = 0
  if ihc_params.ihc_style == 'just_hwr':
    ihc_style_num = 0
  elif ihc_params.ihc_style == 'one_cap':
    ihc_style_num = 1
  elif (
      ihc_params.ihc_style == 'two_cap'
      or ihc_params.ihc_style == 'two_cap_with_syn'
  ):
    ihc_style_num = 2
  else:
    raise NotImplementedError
  ihc_hypers = IhcHypers(n_ch=n_ch, ihc_style=ihc_style_num)
  if ihc_params.ihc_style == 'just_hwr':
    ihc_weights = IhcWeights()
    ihc_state = IhcState(ihc_accum=jnp.zeros((n_ch,)))
  elif ihc_params.ihc_style == 'one_cap':
    ro = 1 / ihc_detect(10)  # output resistance at a very high level
    c = ihc_params.tau_out / ro
    ri = ihc_params.tau_in / c
    # to get steady-state average, double ro for 50# duty cycle
    saturation_output = 1 / (2 * ro + ri)
    # also consider the zero-signal equilibrium:
    r0 = 1 / ihc_detect(0)
    current = 1 / (ri + r0)
    cap_voltage = 1 - current * ri
    ihc_weights = IhcWeights()
    ihc_weights.lpf_coeff = 1 - math.exp(-1 / (ihc_params.tau_lpf * params.fs))
    ihc_weights.out_rate = ro / (ihc_params.tau_out * params.fs)
    ihc_weights.in_rate = 1 / (ihc_params.tau_in * params.fs)
    ihc_weights.output_gain = 1 / (saturation_output - current)
    ihc_weights.rest_output = current / (saturation_output - current)
    ihc_weights.rest_cap = cap_voltage
    ihc_state = IhcState(
        ihc_accum=jnp.zeros((n_ch,)),
        cap_voltage=ihc_weights.rest_cap * jnp.ones((n_ch,)),
        lpf1_state=ihc_weights.rest_output * jnp.ones((n_ch,)),
        lpf2_state=ihc_weights.rest_output * jnp.ones((n_ch,)),
    )
  elif (
      ihc_params.ihc_style == 'two_cap'
      or ihc_params.ihc_style == 'two_cap_with_syn'
  ):
    g1_max = ihc_detect(10)  # receptor conductance at high level

    r1min = 1 / g1_max
    c1 = ihc_params.tau1_out * g1_max  # capacitor for min depletion tau
    r1 = ihc_params.tau1_in / c1  # resistance for recharge tau
    # to get approx steady-state average, double r1min for 50% duty cycle
    # also consider the zero-signal equilibrium:
    g10 = ihc_detect(0)
    r10 = 1 / g10
    rest_current1 = 1 / (r1 + r10)
    cap1_voltage = 1 - rest_current1 * r1  # quiescent/initial state

    # Second cap similar, but using receptor voltage as detected signal.
    max_vrecep = r1 / (r1min + r1)  # Voltage divider from 1.
    # Identity from receptor potential to neurotransmitter conductance:
    g2max = max_vrecep  # receptor resistance at very high level.
    r2min = 1 / g2max
    c2 = ihc_params.tau2_out * g2max
    r2 = ihc_params.tau2_in / c2
    saturation_current2 = 1 / (2 * r2min + r2)
    # Also consider the zero-signal equlibrium
    rest_vrecep = r1 * rest_current1
    g20 = rest_vrecep
    r20 = 1 / g20
    rest_current2 = 1 / (r2 + r20)
    cap2_voltage = 1 - rest_current2 * r2  # quiescent/initial state

    ihc_weights = IhcWeights()
    ihc_weights.lpf_coeff = 1 - math.exp(-1 / (ihc_params.tau_lpf * params.fs))
    ihc_weights.out1_rate = r1min / (ihc_params.tau1_out * params.fs)
    ihc_weights.in1_rate = 1 / (ihc_params.tau1_in * params.fs)
    ihc_weights.out2_rate = r2min / (ihc_params.tau2_out * params.fs)
    ihc_weights.in2_rate = 1 / (ihc_params.tau2_in * params.fs)
    ihc_weights.output_gain = 1 / (saturation_current2 - rest_current2)
    ihc_weights.rest_output = rest_current2 / (
        saturation_current2 - rest_current2
    )
    ihc_weights.rest_cap2 = cap2_voltage
    ihc_weights.rest_cap1 = cap1_voltage

    ihc_state = IhcState(
        ihc_accum=jnp.zeros((n_ch,)),
        cap1_voltage=ihc_weights.rest_cap1 * jnp.ones((n_ch,)),
        cap2_voltage=ihc_weights.rest_cap2 * jnp.ones((n_ch,)),
        lpf1_state=ihc_weights.rest_output * jnp.ones((n_ch,)),
        lpf2_state=ihc_weights.rest_output * jnp.ones((n_ch,)),
    )
  else:
    raise NotImplementedError
  return ihc_hypers, ihc_weights, ihc_state


def design_fir_coeffs(n_taps, delay_variance, mean_delay, n_iter):
  """Design the finite-impulse-response filter needed for AGC smoothing.

  The smoothing function is a space-domain smoothing, but it considered
  here by analogy to time-domain smoothing, which is why its potential
  off-centeredness is called a delay.  Since it's a smoothing filter, it is
  also analogous to a discrete probability distribution (a p.m.f.), with
  mean corresponding to delay and variance corresponding to squared spatial
  spread (in samples, or channels, and the square thereof, respecitively).
  Here we design a filter implementation's coefficient via the method of
  moment matching, so we get the intended delay and spread, and don't worry
  too much about the shape of the distribution, which will be some kind of
  blob not too far from Gaussian if we run several FIR iterations.

  Args:
    n_taps: Width of the spatial filter kernel (either 3 or 5).
    delay_variance: Value, in channels-squared, of how much spread (second
      moment) the filter has about its spatial mean output (delay being a
      time-domain concept apply here to spatial channels domain).
    mean_delay: Value, in channels, of how much of the mean of filter output is
      offset toward basal channels (that is, positive delay means put more
      weight on more apical inputs).
    n_iter: Number of times to apply the 3-point or 5-point FIR filter to
      achieve the desired delay and spread.

  Returns:
    List of 3 FIR coefficients, and a Boolean saying the design was done
    successfully.
  """

  # reduce mean and variance of smoothing distribution by n_iterations:
  mean_delay = mean_delay / n_iter
  delay_variance = delay_variance / n_iter
  if n_taps == 3:
    # based on solving to match mean and variance of [a, 1-a-b, b]:
    a = (delay_variance + mean_delay * mean_delay - mean_delay) / 2
    b = (delay_variance + mean_delay * mean_delay + mean_delay) / 2
    fir = [a, 1 - a - b, b]
    ok = fir[2 - 1] >= 0.25
  elif n_taps == 5:
    # based on solving to match [a/2, a/2, 1-a-b, b/2, b/2]:
    a = (
        (delay_variance + mean_delay * mean_delay) * 2 / 5 - mean_delay * 2 / 3
    ) / 2
    b = (
        (delay_variance + mean_delay * mean_delay) * 2 / 5 + mean_delay * 2 / 3
    ) / 2
    # first and last coeffs are implicitly duplicated to make 5-point FIR:
    fir = [a / 2, 1 - a - b, b / 2]
    ok = fir[2 - 1] >= 0.15
  else:
    raise ValueError('Bad n_taps (%d) in agc_spatial_fir' % n_taps)

  return fir, ok


def design_and_init_agc(
    ear: int, params: CarfacDesignParameters, n_ch: int
) -> Tuple[List[AgcHypers], List[AgcWeights], List[AgcState]]:
  """Design the AGC implementation from the parameters.

  Args:
    ear: the index of the ear.
    params: all the design parameters.
    n_ch: How many channels there are in the filterbank.

  Returns:
    The coefficients for all the stages.
  """
  ear_params = params.ears[ear]

  fs = params.fs

  n_agc_stages = ear_params.agc.n_stages

  # AGC1 pass is smoothing from base toward apex;
  # AGC2 pass is back, which is done first now (in double exp. version)
  agc1_scales = ear_params.agc.agc1_scales
  agc2_scales = ear_params.agc.agc2_scales

  decim = 1

  total_dc_gain = 0

  ##
  # Convert to vector of AGC coeffs
  agc_hypers = []
  agc_weights = []
  agc_states = []
  for stage in range(n_agc_stages):
    agc_hypers.append(AgcHypers(n_ch=n_ch, n_agc_stages=n_agc_stages))
    agc_weights.append(AgcWeights())
    agc_states.append(
        AgcState(
            decim_phase=0,
            agc_memory=jnp.zeros((n_ch,)),
            input_accum=jnp.zeros((n_ch,)),
        )
    )

    agc_hypers[stage].decimation = ear_params.agc.decimation[stage]
    tau = ear_params.agc.time_constants[stage]
    # time constant in seconds
    decim = decim * ear_params.agc.decimation[stage]
    # net decim to this stage
    # epsilon is how much new input to take at each update step:
    agc_weights[stage].agc_epsilon = 1 - math.exp(-decim / (tau * fs))

    agc_weights[stage].agc_stage_gain = ear_params.agc.agc_stage_gain

    # effective number of smoothings in a time constant:
    ntimes = tau * (fs / decim)  # typically 5 to 50

    # decide on target spread (variance) and delay (mean) of impulse
    # response as a distribution to be convolved ntimes:
    # TODO(dicklyon): specify spread and delay instead of scales?
    delay = (agc2_scales[stage] - agc1_scales[stage]) / ntimes
    spread_sq = (agc1_scales[stage] ** 2 + agc2_scales[stage] ** 2) / ntimes

    # get pole positions to better match intended spread and delay of
    # [[geometric distribution]] in each direction (see wikipedia)
    u = 1 + 1 / spread_sq  # these are based on off-line algebra hacking.
    p = u - math.sqrt(u**2 - 1)  # pole that would give spread if used twice.
    dp = delay * (1 - 2 * p + p**2) / 2
    polez1 = p - dp
    polez2 = p + dp
    agc_weights[stage].AGC_polez1 = polez1
    agc_weights[stage].AGC_polez2 = polez2

    # try a 3- or 5-tap FIR as an alternative to the double exponential:
    n_taps = 0
    done = False
    n_iterations = 1
    agc_spatial_fir = None
    while not done:
      if n_taps == 0:
        # first attempt a 3-point FIR to apply once:
        n_taps = 3
      elif n_taps == 3:
        # second time through, go wider but stick to 1 iteration
        n_taps = 5
      elif n_taps == 5:
        # apply FIR multiple times instead of going wider:
        n_iterations = n_iterations + 1
        if n_iterations > 16:
          raise ValueError('Too many n_iterations in CARFAC AGC design')
      else:
        # to do other n_taps would need changes in spatial_smooth
        # and in Design_fir_coeffs
        raise ValueError('Bad n_taps (%d) in design_agc' % n_taps)

      [agc_spatial_fir, done] = design_fir_coeffs(
          n_taps, spread_sq, delay, n_iterations
      )

    # When done, store the resulting FIR design in coeffs:
    agc_hypers[stage].agc_spatial_iterations = n_iterations
    agc_weights[stage].agc_spatial_fir = agc_spatial_fir
    agc_hypers[stage].agc_spatial_n_taps = n_taps

    agc_hypers[stage].reverse_cumulative_decimation = jnp.flip(
        jnp.cumprod(jnp.asarray(ear_params.agc.decimation))
    )
    agc_hypers[stage].max_cumulative_decimation = max(
        agc_hypers[stage].reverse_cumulative_decimation
    )

    # accumulate DC gains from all the stages, accounting for stage_gain:
    total_dc_gain = total_dc_gain + ear_params.agc.agc_stage_gain ** (stage)

    # TODO(dicklyon) -- is this the best binaural mixing plan?
    if stage == 0:
      agc_weights[stage].agc_mix_coeffs = 0.0
    else:
      agc_weights[stage].agc_mix_coeffs = ear_params.agc.agc_mix_coeffs / (
          tau * (fs / decim)
      )

  # adjust stage 1 detect_scale to be the reciprocal DC gain of the AGC filters:
  agc_weights[0].detect_scale = 1 / total_dc_gain

  return agc_hypers, agc_weights, agc_states


def design_and_init_carfac(
    params: CarfacDesignParameters,
) -> Tuple[CarfacHypers, CarfacWeights, CarfacState]:
  """Design and init the CARAFAC model.

  Args:
    params: all the design params. Won't be changed in this function.

  Returns:
    Newly created and initialised hypers, trainable weights and state of
    CARFAC model.
  """
  hypers = CarfacHypers()
  weights = CarfacWeights()
  state = CarfacState()

  for ear, ear_params in enumerate(params.ears):
    # first figure out how many filter stages (PZFC/CARFAC channels):
    pole_hz = ear_params.car.first_pole_theta * params.fs / (2 * math.pi)
    n_ch = 0
    while pole_hz > ear_params.car.min_pole_hz:
      n_ch = n_ch + 1
      pole_hz = pole_hz - ear_params.car.erb_per_step * hz_to_erb(
          pole_hz, ear_params.car.erb_break_freq, ear_params.car.erb_q
      )

    # Now we have n_ch, the number of channels, so can make the array
    # and compute all the frequencies again to put into it:
    pole_freqs = jnp.zeros((n_ch,))  # float64 didn't help
    pole_hz = ear_params.car.first_pole_theta * params.fs / (2 * math.pi)
    for ch in range(n_ch):
      if jnp.__name__ == 'jax.numpy':  # for debugging.
        pole_freqs = pole_freqs.at[ch].set(pole_hz)
      else:
        pole_freqs[ch] = pole_hz
      pole_hz = pole_hz - ear_params.car.erb_per_step * hz_to_erb(
          pole_hz, ear_params.car.erb_break_freq, ear_params.car.erb_q
      )
    # Now we have n_ch, the number of channels, and pole_freqs array.
    max_channels_per_octave = 1 / math.log(pole_freqs[0] / pole_freqs[1], 2)

    # Convert to include an ear_array, each w coeffs and state...
    car_hypers, car_weights, car_state = design_and_init_filters(
        ear, params, pole_freqs
    )
    agc_hypers, agc_weights, agc_state = design_and_init_agc(ear, params, n_ch)
    ihc_hypers, ihc_weights, ihc_state = design_and_init_ihc(ear, params, n_ch)
    syn_hypers, syn_weights, syn_state = design_and_init_ihc_syn(
        ear, params, n_ch
    )

    ear_hypers = EarHypers(
        n_ch=n_ch,
        pole_freqs=pole_freqs,
        max_channels_per_octave=max_channels_per_octave,
        car=car_hypers,
        agc=agc_hypers,
        ihc=ihc_hypers,
        syn=syn_hypers,
    )
    hypers.ears.append(ear_hypers)

    ear_weights = EarWeights(
        car=car_weights,
        agc=agc_weights,
        ihc=ihc_weights,
        syn=syn_weights,
    )
    weights.ears.append(ear_weights)

    ear_state = EarState(
        car=car_state,
        agc=agc_state,
        ihc=ihc_state,
        syn=syn_state,
    )
    state.ears.append(ear_state)

  return hypers, weights, state


####################
# Model Functions  #
####################


def stage_g(
    car_weights: CarWeights, undamping: Union[float, jnp.ndarray]
) -> jnp.ndarray:
  """Return the stage gain g needed to get unity gain at DC, using a quadratic approximation.

  Args:
    car_weights: The trainable weights for the filterbank.
    undamping:  The r variable, defined in section 17.4 of Lyon's book to be
      (1-b) * NLF(v). The first term is undamping (because b is the gain).

  Returns:
    A float vector with the gain for each AGC stage.
  """
  return (
      car_weights.ga_coeffs * (undamping**2)
      + car_weights.gb_coeffs * undamping
      + car_weights.gc_coeffs
  )


def ohc_nlf(velocities: jnp.ndarray, car_weights: CarWeights) -> jnp.ndarray:
  """The outer hair cell non-linearity.

  Start with a quadratic nonlinear function, and limit it via a
  rational function; make the result go to zero at high
  absolute velocities, so it will do nothing there. See section 17.2 of Lyon's
  book.

  Args:
    velocities: the BM velocities
    car_weights: CarWeights that contains the data needed for computation.

  Returns:
    A nonlinear nonnegative function of velocities.
  """

  nlf = 1.0 / (
      1 + (velocities * car_weights.velocity_scale + car_weights.v_offset) ** 2
  )

  return nlf


def ear_ch_scan(
    carry: Tuple[jax.Array, jax.Array, jax.Array], ch: int
) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
  """Propagates the BM action down the cochlea, to new channels.

  This function is the guts of a jax.lax.scan call.

  Args:
    carry: the g, zy, and in_out matrices need to implement the CAR filters.
    ch: The current channel number of the current scan step.

  Returns:
    the new carry, and the output for the current channel.
  """
  g, zy, in_out = carry
  new_in_out = g[ch] * (in_out + zy[ch])
  return (g, zy, new_in_out), new_in_out


def car_step(
    x_in: float,
    ear: int,
    hypers: CarfacHypers,
    weights: CarfacWeights,
    car_state: CarState,
):
  """The step function of the cascaded filters."""
  ear_coeffs = hypers.ears[ear]
  ear_weights = weights.ears[ear]
  car_weights = ear_weights.car

  g = car_state.g_memory + car_state.dg_memory  # interp g
  zb = car_state.zb_memory + car_state.dzb_memory  # AGC interpolation car_state
  # update the nonlinear function of "velocity", and za (delay of z2):
  za = car_state.za_memory
  v = car_state.z2_memory - za
  if ear_coeffs.car.linear_car:
    nlf = 1  # To allow testing
  else:
    nlf = ohc_nlf(v, car_weights)
  r = (
      hypers.ears[ear].car.r1_coeffs + zb * nlf
  )  #  zb * nfl is "undamping" delta r.
  za = car_state.z2_memory

  #  now reduce car_state by r and rotate with the fixed cos/sin coeffs:
  z1 = r * (
      hypers.ears[ear].car.a0_coeffs * car_state.z1_memory
      - hypers.ears[ear].car.c0_coeffs * car_state.z2_memory
  )
  #  z1 = z1 + inputs
  z2 = r * (
      hypers.ears[ear].car.c0_coeffs * car_state.z1_memory
      + hypers.ears[ear].car.a0_coeffs * car_state.z2_memory
  )

  if ear_coeffs.car.use_delay_buffer:
    # Optional fully-parallel update uses a delay per stage.
    zy = car_state.zy_memory
    # Propagate delayed last outputs zy and fill in new input.
    zy = jnp.insert(zy[0:-1], 0, x_in)
    z1 = z1 + zy  # add new stage inputs to z1 states
    zy = g * (hypers.ears[ear].car.h_coeffs * z2 + zy)  # Outputs from z2
  else:
    # Ripple input-output path to avoid delay...
    # this is the only part that doesn't get computed "in parallel":
    zy = hypers.ears[ear].car.h_coeffs * z2  # partial output
    in_out = x_in
    # Copied from @malcolmslaney's version.
    _, zy = jax.lax.scan(  # pytype: disable=wrong-arg-types  # lax-types
        ear_ch_scan, (g, zy, in_out), jnp.arange(g.shape[-1])
    )
    z1 += jnp.hstack((x_in, zy[:-1]))

  #  put new car_state back in place of old
  #  (z1 is a genuine temp; the others can update by reference in C)
  car_state.z1_memory = z1
  car_state.z2_memory = z2
  car_state.za_memory = za
  car_state.zb_memory = zb
  car_state.zy_memory = zy
  car_state.g_memory = g

  ac_diff = zy - car_state.ac_coupler
  car_state.ac_coupler = car_state.ac_coupler + car_weights.ac_coeff * ac_diff

  return ac_diff, car_state


def syn_step(
    v_recep: jnp.ndarray,
    ear: int,
    weights: CarfacWeights,
    syn_state: SynState,
) -> Tuple[jnp.ndarray, jnp.ndarray, SynState]:
  """Step the inner-hair cell model with ont input sample."""
  syn_weights = weights.ears[ear].syn

  # Drive multiple synapse classes with receptor potential from IHC,
  # returning instantaneous spike rates per class, for a group of neurons
  # associated with the CF channel, including reductions due to synaptopathy.
  # Normalized offset position into neurotransmitter release sigmoid.
  x = (v_recep[None, :] - jnp.transpose(syn_weights.v_halfs)) / jnp.transpose(
      syn_weights.v_widths
  )
  x = jnp.transpose(x)
  s = 1 / (1 + jnp.exp(-x))  # Between 0 and 1; positive at rest.
  q = syn_state.reservoirs  # aka 1 - w, between 0 and 1; positive at rest.
  r = (1 - q) * s  # aka w*s, between 0 and 1, proportional to release rate.

  # Smooth once with LPF (receptor potential was already smooth), after
  # applying the gain coeff a2 to convert to firing prob per sample.
  syn_state.lpf_state = syn_state.lpf_state + syn_weights.lpf_coeff * (
      syn_weights.a2 * r - syn_state.lpf_state
  )  # this is firing probs.
  firing_probs = syn_state.lpf_state  # Poisson rate per neuron per sample.
  # Include number of effective neurons per channel here, and interval T;
  # so the rates (instantaneous action potentials per second) can be huge.
  firings = syn_weights.n_fibers * firing_probs

  # Feedback, to update reservoir state q for next time.
  syn_state.reservoirs = q + syn_weights.res_coeff * (syn_weights.a1 * r - q)
  # Make an output that resembles ihc_out, to go to agc_in
  # (collapse over classes).
  # Includes synaptopathy's presumed effect of reducing feedback via n_fibers.
  # But it's relative to the healthy nominal spont, so could potentially go
  # a bit negative in quiet is there was loss of high-spont or medium-spont
  # units.
  # The weight multiplication is an inner product, reducing n_classes
  # columns to 1 column (first transpose the agc_weights row to a column).
  syn_out = (
      syn_weights.n_fibers * firing_probs
  ) @ syn_weights.agc_weights - syn_weights.spont_sub

  return syn_out, firings, syn_state


def ihc_step(
    bm_out: jnp.ndarray,
    ear: int,
    hypers: CarfacHypers,
    weights: CarfacWeights,
    ihc_state: IhcState,
) -> Tuple[jnp.ndarray, jnp.ndarray, IhcState]:
  """Step the inner-hair cell model with ont input sample.

  One sample-time update of inner-hair-cell (IHC) model, including the
  detection nonlinearity and one or two capacitor state variables.

  Args:
    bm_out: The output from the CAR filterbank
    ear: the index of the ear.
    hypers: all the static coefficients.
    weights: all the trainable weights.
    ihc_state: The ihc state.

  Returns:
    The firing probability of the hair cells in each channel,
    the receptor potential output and the new state.
  """

  ihc_weights = weights.ears[ear].ihc
  ihc_hypers = hypers.ears[ear].ihc
  v_recep = jnp.zeros(hypers.ears[ear].car.n_ch)
  if ihc_hypers.ihc_style == 0:
    ihc_out = jnp.minimum(2, jnp.maximum(0, bm_out))
    #  limit it for stability
  else:
    conductance = ihc_detect(bm_out)  # rectifying nonlinearity

    if ihc_hypers.ihc_style == 1:
      ihc_out = conductance * ihc_state.cap_voltage
      ihc_state.cap_voltage = (
          ihc_state.cap_voltage
          - ihc_out * ihc_weights.out_rate
          + (1 - ihc_state.cap_voltage) * ihc_weights.in_rate
      )
      #  smooth it twice with LPF:
      ihc_out = ihc_out * ihc_weights.output_gain
      ihc_state.lpf1_state = ihc_state.lpf1_state + ihc_weights.lpf_coeff * (
          ihc_out - ihc_state.lpf1_state
      )
      ihc_state.lpf2_state = ihc_state.lpf2_state + ihc_weights.lpf_coeff * (
          ihc_state.lpf1_state - ihc_state.lpf2_state
      )
      ihc_out = ihc_state.lpf2_state - ihc_weights.rest_output
    else:
      # Change to 2-cap version mediated by receptor potential at cap1:
      # Geisler book fig 8.4 suggests 40 to 800 Hz corner.
      receptor_current = conductance * ihc_state.cap1_voltage
      # "out" means charge depletion; "in" means restoration toward 1.
      ihc_state.cap1_voltage = (
          ihc_state.cap1_voltage
          - receptor_current * ihc_weights.out1_rate
          + (1 - ihc_state.cap1_voltage) * ihc_weights.in1_rate
      )
      receptor_potential = 1 - ihc_state.cap1_voltage
      ihc_out = receptor_potential * ihc_state.cap2_voltage
      ihc_state.cap2_voltage = (
          ihc_state.cap2_voltage
          - ihc_out * ihc_weights.out2_rate
          + (1 - ihc_state.cap2_voltage) * ihc_weights.in2_rate
      )
      ihc_out = ihc_out * ihc_weights.output_gain
      ihc_state.lpf1_state = ihc_state.lpf1_state + ihc_weights.lpf_coeff * (
          ihc_out - ihc_state.lpf1_state
      )
      ihc_out = ihc_state.lpf1_state - ihc_weights.rest_output
      v_recep = ihc_weights.rest_cap1 - ihc_state.cap1_voltage

  # for where decimated output is useful
  ihc_state.ihc_accum = ihc_state.ihc_accum + ihc_out

  return ihc_out, v_recep, ihc_state


def _agc_step_jit_helper(
    depth: int,
    n_agc_stages: int,
    ear: int,
    hypers: CarfacHypers,
    weights: CarfacWeights,
    agc_in: jax.Array,
    state: List[AgcState],
) -> List[AgcState]:
  """Compute agc step using a loop.

  Args:
    depth: how many smoothing stages do we need to compute.
    n_agc_stages: total number of stages. It equals to coeffs[0].n_agc_stages.
      We make it a standalone arg to make it static.
    ear: the index of the ear.
    hypers: the details of the AGC design.
    weights: all the trainable weights.
    agc_in: the input data for this stage, a vector of channel values
    state: the state of each channel's AGC.

  Returns:
    A list of updated AgcState objects.
  """
  agc_hypers = hypers.ears[ear].agc

  # [1st loop]: update `input_accum`. This needs to be from stage 1 to stage 4.

  # Stage 0 always accumulates input. And its `agc_in` is the input argument.
  state[0].input_accum = state[0].input_accum + agc_in
  # For the other stages, its `agc_in` comes from the average of previous stage.
  # Here "a smooth stage will be computed" means two things,
  #   1) Its OWN `input_accum` will be accumulated because it will sample a new
  #      data point from the previous stage.
  #   2) Its NEXT stage's `input_accum` will be accumulated because the next
  #      smooth stage will sample a new data point too.
  for stage in range(depth):
    # Checking `stage < n_agc_stages - 1` is needed because,
    #   1) `depth` can equal `n_agc_stages` which is out-of-bounds.
    #   2) No matter `depth=n_agc_stages-1` or `depth=n_agc_stages`, we both and
    #      only need to update the `state[n_agc_stages-1].input_accum` which is
    #     of the last stage. And `state[0].input_accum` has been updated above.
    if stage < n_agc_stages - 1:
      state[stage + 1].input_accum = (
          state[stage + 1].input_accum
          + state[stage].input_accum / agc_hypers[stage].decimation
      )

  # [2nd loop]: compute `agc_memory`. This needs to be from stage 4 to stage 1.
  for stage in reversed(range(depth)):
    agc_in = state[stage].input_accum / agc_hypers[stage].decimation
    state[stage].input_accum = jnp.zeros(
        state[stage].input_accum.shape
    )  # reset accumulator
    if stage < n_agc_stages - 1:
      agc_in = (
          agc_in
          + weights.ears[ear].agc[stage].agc_stage_gain
          * state[stage + 1].agc_memory
      )

    agc_stage_state = state[stage].agc_memory
    # first-order recursive smoothing filter update, in time:
    agc_stage_state = agc_stage_state + weights.ears[ear].agc[
        stage
    ].agc_epsilon * (agc_in - agc_stage_state)
    # spatial smooth:
    agc_stage_state = spatial_smooth_jit(
        agc_hypers[stage], weights.ears[ear].agc[stage], agc_stage_state
    )
    # and store the state back (in C++, do it all in place?)
    state[stage].agc_memory = agc_stage_state

  return state


def agc_step(
    detects: jax.Array,
    ear: int,
    hypers: CarfacHypers,
    weights: CarfacWeights,
    state: List[AgcState],
) -> Tuple[bool, List[AgcState]]:
  """One time step of the AGC state update; decimates internally.

  Args:
    detects: The output of the IHC stage, input to AGC.
    ear: the index of the ear.
    hypers: all the coefficients.
    weights: all the trainable weights.
    state: A list of AGC states.

  Returns:
    A bool to indicate whether the AGC output has updated, and the new state.
  """
  # Figures out how many smoothing stages we need to run without using
  # conditionals. The basic idea is to avoid using loop "break".
  # # Version 1: Use "python loop + arithmetics".
  # depth = 0  # how many stages of smoothings should we run.
  # if_inc = 1  # denotes whether to increase the depth.
  # for stage in range(n_agc_stages):
  #   new_decim_phase = jnp.mod(
  #       state[stage].decim_phase + 1, coeffs[stage].decimation
  #   )
  #   # This is equivalent to
  #   # `decim_phase = if_inc == 0 ? decim_phase : new_decim_phase`.
  #   # When `if_inc` becomes 0, it means the current stage can't receive a
  #   # sample this time.
  #   state[stage].decim_phase = (
  #       state[stage].decim_phase
  #       + (new_decim_phase - state[stage].decim_phase) * if_inc
  #   )
  #   # This is equivalent to `if_inc = if_inc and (new_decim_phase == 0)`.
  #   # We should only increase `depth` when the current layer receives sample.
  #   # (i.e. `new_decim_phase == 0`). And whenever `if_inc = 0`, it should be 0
  #   # for in the following.
  #   if_inc = if_inc * (new_decim_phase == 0)
  #   # `if_inc` can only be 0 or 1.
  #   depth = depth + if_inc

  # Version 2: Use "python loop + jnp logicals"
  depth = 0
  if_inc = True
  agc_hypers = hypers.ears[ear].agc
  agc_weights = weights.ears[ear].agc
  n_agc_stages = agc_hypers[0].n_agc_stages
  for stage in range(n_agc_stages):
    state[stage].decim_phase = jax.lax.cond(
        if_inc,
        lambda x, y: jnp.mod(x + 1, y),
        lambda x, y: x,
        state[stage].decim_phase,
        agc_hypers[stage].decimation,
    )
    if_inc = jnp.logical_and(if_inc, state[stage].decim_phase == 0)
    depth = jax.lax.cond(if_inc, lambda: depth + 1, lambda: depth)

  # TODO(honglinyu): Version 3: "jnp.argmax"
  # Currently, we store each stage's own decimation and sample index separately.
  # We can change that to, e.g.,
  #   1. Store a single sample index like sample_index.
  #   2. Store the overall decimation of each stage. For example, currently, the
  #      default "per-stage decimation" is [8, 2, 2, 2]. We can store something
  #      like `overall_decimation = [8, 16, 32, 64]` or its reverse-order
  #      version `rev_overall_decimation = [64, 32, 16, 8]`. Then we can get the
  #      `depth` by something like,
  #   `n_agc_stages - jnp.argmin(jnp.mod(rev_overall_decimation, sample_index))`
  # Notice that using "jnp.lax.while_loop" to "early stopping" shouldn't be
  # possible.

  updated = depth != 0

  agc_in = agc_weights[0].detect_scale * detects

  branch_fns = [
      functools.partial(_agc_step_jit_helper, i, n_agc_stages, ear, hypers)
      for i in range(n_agc_stages + 1)
  ]
  state = jax.lax.switch(depth, branch_fns, weights, agc_in, state)

  return updated, state


def _shift_right(s: jax.Array, amount: int) -> jax.Array:
  """Rotate a vector to the right by amount, or to the left if negative."""
  if amount > 0:
    return jnp.concatenate((s[0:amount, ...], s[:-amount, ...]), axis=0)
  elif amount < 0:
    return jnp.concatenate(
        (s[-amount:, ...], jnp.flip(s[amount:, ...])), axis=0
    )
  else:
    return s


def spatial_smooth_jit(
    agc_hypers: AgcHypers, weights: AgcWeights, stage_state: jax.Array
) -> jax.Array:
  """Does the spatial smoothing.

  Args:
    agc_hypers: The AGC coefficients for this state.
    weights: all the trainable weights.
    stage_state: The state of the AGC right now.

  Returns:
    A new stage stage object.
  """
  # Always do the default case to unblock the work on agc_step.
  # TODO(honglinyu): add the other cases back.
  n_iterations = agc_hypers.agc_spatial_iterations
  fir_coeffs = weights.agc_spatial_fir
  return jax.lax.fori_loop(
      0,
      n_iterations,
      lambda _, stage_state: (  # pylint: disable=g-long-lambda
          fir_coeffs[0] * _shift_right(stage_state, 1)
          + fir_coeffs[1] * _shift_right(stage_state, 0)
          + fir_coeffs[2] * _shift_right(stage_state, -1)
      ),
      stage_state,
  )


def cross_couple(
    hypers: CarfacHypers, weights: CarfacWeights, state: CarfacState
) -> CarfacState:
  """This function cross couples gain between multiple ears.

  There's no impact for this function in the case of monoaural input.

  Args:
    hypers: The hyperparameters for this carfac instance.
    weights: Calculated weights for this instance.
    state: Current state for this carfac instance.

  Returns:
    The state after AGC cross coupling between all the ears.
  """
  n_ears = len(hypers.ears)
  if n_ears <= 1:
    return state

  n_stages = len(hypers.ears[0].agc)

  # now cross-ear mix the stage that updated (leading stage at phase 0)
  this_stage_continue = True

  for stage in range(n_stages):
    this_stage_continue = jnp.logical_and(
        this_stage_continue, state.ears[0].agc[stage].decim_phase == 0
    )

    def _calc_stage_mix(stt, loop_stage=stage):
      # If the agc_mix_coeffs are <= 0, in numpy we continue through this loop.
      # in Jax we simply set the mix_coeff to 0 which makes this loop a no-op.
      mix_coeff = jnp.maximum(weights.ears[0].agc[loop_stage].agc_mix_coeffs, 0)
      this_stage_sum = 0
      for ear_index in range(n_ears):
        this_stage_sum += stt.ears[ear_index].agc[loop_stage].agc_memory
      this_stage_mean = this_stage_sum / len(hypers.ears)
      # now move everything to the mean.
      for ear_index in range(n_ears):
        stage_state = stt.ears[ear_index].agc[loop_stage].agc_memory
        stt.ears[ear_index].agc[loop_stage].agc_memory = (
            stage_state + mix_coeff * (this_stage_mean - stage_state)
        )
      return stt

    state = jax.lax.cond(
        this_stage_continue, _calc_stage_mix, lambda x: x, state
    )
  return state


def close_agc_loop(
    hypers: CarfacHypers, weights: CarfacWeights, state: CarfacState
) -> CarfacState:
  """Fastest decimated rate determines interp needed."""
  decim1 = hypers.ears[0].agc[0].decimation
  for ear, hypers_ear in enumerate(hypers.ears):
    undamping = 1 - state.ears[ear].agc[0].agc_memory  # stage 1 result
    undamping = undamping * weights.ears[ear].car.ohc_health
    # Update the target stage gain for the new damping:
    new_g = stage_g(weights.ears[ear].car, undamping)
    # set the deltas needed to get to the new damping:
    state.ears[ear].car.dzb_memory = (
        hypers_ear.car.zr_coeffs * undamping - state.ears[ear].car.zb_memory
    ) / decim1
    state.ears[ear].car.dg_memory = (
        new_g - state.ears[ear].car.g_memory
    ) / decim1
  return state


def run_segment(
    input_waves: jnp.ndarray,
    hypers: CarfacHypers,
    weights: CarfacWeights,
    state: CarfacState,
    open_loop: bool = False,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, CarfacState, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
  """This function runs the entire CARFAC model.

  That is, filters a 1 or more channel
  sound input segment to make one or more neural activity patterns (naps)
  it can be called multiple times for successive segments of any length,
  as long as the returned cfp with modified state is passed back in each
  time.

  input_waves is a column vector if there's just one audio channel
  more generally, it has a row per time sample, a column per audio channel.

  naps has a row per time sample, a column per filterbank channel, and
  a layer per audio channel if more than 1.
  BM is basilar membrane motion (filter outputs before detection).

  the input_waves are assumed to be sampled at the same rate as the
  CARFAC is designed for; a resampling may be needed before calling this.

  The function works as an outer iteration on time, updating all the
  filters and AGC states concurrently, so that the different channels can
  interact easily.  The inner loops are over filterbank channels, and
  this level should be kept efficient.

  Args:
    input_waves: the audio input.
    hypers: all the coefficients of the model. It will be passed to all the
      JIT'ed functions as static variables.
    weights: all the trainable weights. It will not be changed.
    state: all the state of the CARFAC model. It will be updated and returned.
    open_loop: whether to run CARFAC without the feedback.

  Returns:
    naps: neural activity pattern
    naps_fibers: neural activity of different fibers
        (only populated with non-zeros when ihc_style equals "two_cap_with_syn")
    state: the updated state of the CARFAC model.
    BM: The basilar membrane motion
    seg_ohc & seg_agc are optional extra outputs useful for seeing what the
      ohc nonlinearity and agc are doing; both in terms of extra damping.
  """
  if len(input_waves.shape) < 2:
    input_waves = jnp.reshape(input_waves, (-1, 1))
  [n_samp, n_ears] = input_waves.shape
  n_fibertypes = SynDesignParameters.n_classes

  # TODO(honglinyu): add more assertions using checkify.
  # if n_ears != cfp.n_ears:
  #   raise ValueError(
  #       'Bad number of input_waves channels (%d vs %d) passed to Run' %
  #       (n_ears, cfp.n_ears))

  n_ch = hypers.ears[0].car.n_ch
  naps = jnp.zeros((n_samp, n_ch, n_ears))  # allocate space for result
  naps_fibers = jnp.zeros((n_samp, n_ch, n_fibertypes, n_ears))
  bm = jnp.zeros((n_samp, n_ch, n_ears))
  seg_ohc = jnp.zeros((n_samp, n_ch, n_ears))
  seg_agc = jnp.zeros((n_samp, n_ch, n_ears))

  # A 2022 addition to make open-loop running behave:
  if open_loop:
    # zero the deltas:
    for ear in range(n_ears):
      state.ears[ear].car.dzb_memory = jnp.zeros(
          state.ears[ear].car.dzb_memory.shape
      )
      state.ears[ear].car.dg_memory = jnp.zeros(
          state.ears[ear].car.dg_memory.shape
      )

  # Note that we can use naive for loops here because it will make gradient
  # computation very slow.
  def run_segment_scan_helper(carry, k):
    naps, naps_fibers, state, bm, seg_ohc, seg_agc, input_waves = carry
    agc_updated = False
    for ear in range(n_ears):
      # This would be cleaner if we could just get and use a reference to
      # cfp.ears(ear), but Matlab doesn't work that way...
      car_out, state.ears[ear].car = car_step(
          input_waves[k, ear], ear, hypers, weights, state.ears[ear].car
      )

      # update IHC state & output on every time step, too
      ihc_out, v_recep, state.ears[ear].ihc = ihc_step(
          car_out, ear, hypers, weights, state.ears[ear].ihc
      )

      if hypers.ears[ear].syn.do_syn:
        ihc_out, firings, state.ears[ear].syn = syn_step(
            v_recep, ear, weights, state.ears[ear].syn
        )
        naps_fibers = naps_fibers.at[k, :, :, ear].set(firings)
      else:
        naps_fibers = naps_fibers.at[k, :, :, ear].set(
            jnp.zeros([jnp.shape(ihc_out)[0], n_fibertypes])
        )

      # run the AGC update step, decimating internally,
      agc_updated, state.ears[ear].agc = agc_step(
          ihc_out, ear, hypers, weights, state.ears[ear].agc
      )
      # save some output data:
      naps = naps.at[k, :, ear].set(ihc_out)
      bm = bm.at[k, :, ear].set(car_out)
      car_state = state.ears[ear].car
      seg_ohc = seg_ohc.at[k, :, ear].set(car_state.za_memory)
      seg_agc = seg_agc.at[k, :, ear].set(car_state.zb_memory)

    def close_agc_loop_helper(
        hypers: CarfacHypers, weights: CarfacWeights, state: CarfacState
    ) -> CarfacState:
      """A helper function to do some checkings before calling the real one."""
      # Handle multi-ears.
      state = cross_couple(hypers, weights, state)
      if not open_loop:
        return close_agc_loop(hypers, weights, state)
      else:
        return state

    state = jax.lax.cond(
        agc_updated,
        close_agc_loop_helper,
        lambda c, w, s: s,  # If agc isn't updated, returns the original state.
        hypers,
        weights,
        state,
    )

    return (naps, naps_fibers, state, bm, seg_ohc, seg_agc, input_waves), None

  return jax.lax.scan(
      run_segment_scan_helper,
      (naps, naps_fibers, state, bm, seg_ohc, seg_agc, input_waves),
      jnp.arange(n_samp),
  )[0][:-1]


@functools.partial(
    jax.jit,
    static_argnames=(
        'hypers',
        'open_loop',
    ),
)
def run_segment_jit(
    input_waves: jnp.ndarray,
    hypers: CarfacHypers,
    weights: CarfacWeights,
    state: CarfacState,
    open_loop: bool = False,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, CarfacState, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
  """A JITted version of run_segment for convenience.

  Care should be taken with the hyper parameters in hypers. If the hypers object
  is modified, jit recompilation _may_ not occur even though it should. The best
  way to account for this is to always make a deep copy of the hypers and modify
  those. Example usage if modifying the hypers (which most users should not):

  naps, _, _, _, _ = run_segment_jit(input, hypers, weights, state)

  hypers_copy = copy.deepcopy(hypers)
  hypers_jax2.ears[0].car.r1_coeffs /= 2.0
  naps, _, _, _, _ = run_segment_jit(input, hypers_copy, weights, state)

  If no modifications to the CarfacHypers are made, the same hypers object
  should be reused.
  Args:
    input_waves: the audio input.
    hypers: all the coefficients of the model. It will be passed to all the
      JIT'ed functions as static variables.
    weights: all the trainable weights. It will not be changed.
    state: all the state of the CARFAC model. It will be updated and returned.
    open_loop: whether to run CARFAC without the feedback.

  Returns:
    naps: neural activity pattern
    naps_fibers: neural activity of the different fiber types
        (only populated with non-zeros when ihc_style equals "two_cap_with_syn")
    state: the updated state of the CARFAC model.
    BM: The basilar membrane motion
    seg_ohc & seg_agc are optional extra outputs useful for seeing what the
      ohc nonlinearity and agc are doing; both in terms of extra damping.
  """
  return run_segment(input_waves, hypers, weights, state, open_loop)


def run_segment_jit_in_chunks_notraceable(
    input_waves: jnp.ndarray,
    hypers: CarfacHypers,
    weights: CarfacWeights,
    state: CarfacState,
    open_loop: bool = False,
    segment_chunk_length: int = 32 * 48000,
) -> tuple[
    jnp.ndarray, jnp.ndarray, CarfacState, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
  """Runs the jitted segment runner in segment groups.

  Running CARFAC on an audio segment this way is most useful when running
  CARFAC-JAX on a number of audio segments that have varying lengths. The
  implementation is completed by executing CARFAC-JAX on sub-segments in
  descending binary search lengths, and then concatenated back together.

  This implementation minimises the number of recompilations of the JAX code on
  inputs of different shapes, whilst still leveraging JAX compilation.
  Performance is slightly slower than a pure Jitted implementation, roughly
  20 percent on a regular CPU.

  Args:
    input_waves: The audio input.
    hypers: All the coefficients of the model. It will be passed to all the
      JIT'ed functions as static variables.
    weights: All the trainable weights. It will not be changed.
    state: All the state of the CARFAC model. It will be updated and returned.
    open_loop: Whether to run CARFAC without the feedback.
    segment_chunk_length: The chunk length, in samples, to use as the default
      largest chunk.

  Returns:
    naps: Neural activity pattern as a numpy array.
    state: The updated state of the CARFAC model.
    BM: The basilar membrane motion as a numpy array.
    seg_ohc & seg_agc are optional extra outputs useful for seeing what the
      ohc nonlinearity and agc are doing; both in terms of extra damping.

  Raises:
    RuntimeError: If this function is being JITTed, which it should not be.
  """
  # We add a check for tracer, until a superior fix is fixed. Tracked in
  # https://github.com/jax-ml/jax/issues/18544 .
  if isinstance(input_waves, jax.core.Tracer) or isinstance(
      weights, jax.core.Tracer
  ):
    raise RuntimeError("This function shouldn't be transformed by JAX.")
  segment_length = segment_chunk_length
  if len(input_waves.shape) < 2:
    input_waves = jnp.reshape(input_waves, (-1, 1))
  naps_out = []
  naps_fibers_out = []
  bm_out = []
  ohc_out = []
  agc_out = []
  # NOMUTANTS -- This is a performance optimization.
  while segment_length > 16:
    [n_samp, _] = input_waves.shape
    if n_samp >= segment_length:
      [current_waves, input_waves] = jnp.split(input_waves, [segment_length], 0)
      naps_jax, naps_fibers_jax, state, bm_jax, seg_ohc_jax, seg_agc_jax = (
          run_segment_jit(current_waves, hypers, weights, state, open_loop)
      )
      naps_out.append(naps_jax)
      naps_fibers_out.append(naps_fibers_jax)
      bm_out.append(bm_jax)
      ohc_out.append(seg_ohc_jax)
      agc_out.append(seg_agc_jax)
    else:
      segment_length //= 2
  [n_samp, _] = input_waves.shape
  # Take the last few items and just run them.
  if n_samp > 0:
    naps_jax, naps_fibers_jax, state, bm_jax, seg_ohc_jax, seg_agc_jax = (
        run_segment_jit(input_waves, hypers, weights, state, open_loop)
    )
    naps_out.append(naps_jax)
    naps_fibers_out.append(naps_fibers_jax)
    bm_out.append(bm_jax)
    ohc_out.append(seg_ohc_jax)
    agc_out.append(seg_agc_jax)
  naps_out = np.concatenate(naps_out, 0)
  naps_fibers_out = np.concatenate(naps_fibers_out, 0)
  bm_out = np.concatenate(bm_out, 0)
  ohc_out = np.concatenate(ohc_out, 0)
  agc_out = np.concatenate(agc_out, 0)
  return naps_out, naps_fibers_out, state, bm_out, ohc_out, agc_out
