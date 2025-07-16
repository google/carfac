"""Implement Dick Lyon's cascade of asymmetric resonators.


Copyright 2012 The CARFAC Authors. All Rights Reserved.
Author: Richard F. Lyon

This file is part of an implementation of Lyon's cochlear model:
"Cascade of Asymmetric Resonators with Fast-Acting Compression"

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Note this is a pretty direct translation of the Matlab code, preserving
the original variable names. The Matlab code can be found in the `matlab` folder
at https://github.com/google/carfac.
"""

import dataclasses
import math
import numbers
from typing import List, Optional, Tuple, Union

import numpy as np

# Note in general, a functional block (i.e. IHC or AGC) has parameters.  After
# a design phase they become coefficients that describe how to do the
# calculation quickly on audio.  And each block has some state.  In other words:
#  Params -> Coeffs -> State
#
# The CarfacParams structure, usually stored as a cfp, also points to the
# ear(s), and each ear has a pointer to the coeffs structure and those include
# the state variables.

# There are three blocks that act in sequence to create the entire
# CARFAC model.  The three blocks are: CAR, IHC, AGC.

############################################################################
# CAR - Cascade of Asymmetric Resonators
############################################################################


@dataclasses.dataclass
class CarParams:
  """All the parameters needed to define the CAR filters."""
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
  ac_corner_hz: float = 20  # AC couple at 20 Hz corner


@dataclasses.dataclass
class CarCoeffs:
  """Coefficients used by the CAR filter signal-graph implementation."""
  n_ch: int
  velocity_scale: float
  v_offset: float
  ac_coeff: float
  use_delay_buffer: bool = False

  r1_coeffs: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(())
  )
  a0_coeffs: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(())
  )
  c0_coeffs: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(())
  )
  h_coeffs: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(()))
  g0_coeffs: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(())
  )
  ga_coeffs: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(())
  )
  gb_coeffs: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(())
  )
  gc_coeffs: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(())
  )
  zr_coeffs: np.ndarray = dataclasses.field(
      default_factory=lambda: np.zeros(())
  )
  ohc_health: np.ndarray = dataclasses.field(
      default_factory=lambda: np.ones(())
  )


def hz_to_erb(cf_hz: Union[float, np.ndarray],
              erb_break_freq: float = 1000 / 4.37,
              erb_q: float = 1000 / (24.7 * 4.37)):
  """Auditory filter nominal Equivalent Rectangular Bandwidth.

  Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
  ERB = 24.7 * (1 + 4.37 * cf_hz / 1000)

  Args:
    cf_hz: A scalar or a vector of frequencies (CF) to convert to ERB scale
    erb_break_freq: The corner frequency where we go from linear to log
      bandwidth
    erb_q: The width of one filter (Q = cf/Bandwidth)

  Returns:
    A scalar or vector with the ERB scale for the input frequencies.
  """
  return (erb_break_freq + cf_hz) / erb_q


def design_filters(
    car_params: CarParams,
    fs: float,
    pole_freqs: Union[List[float], np.ndarray],
    use_delay_buffer: bool = False,
) -> CarCoeffs:
  """Design the actual CAR filters.

  Args:
    car_params: The parameters of the desired CAR filterbank.
    fs: The sampling rate in Hz
    pole_freqs: Where to put the poles for each channel.
    use_delay_buffer: Whether to use delay buffer in car step.

  Returns:
    The CAR coefficient structure which allows the filters be computed quickly.
  """

  n_ch = len(pole_freqs)
  pole_freqs = np.asarray(pole_freqs)

  # the filter design coeffs:
  # scalars first:
  car_coeffs = CarCoeffs(
      n_ch=n_ch,
      velocity_scale=car_params.velocity_scale,
      v_offset=car_params.v_offset,
      ac_coeff=2 * np.pi * car_params.ac_corner_hz / fs,
      use_delay_buffer=use_delay_buffer,
  )

  car_coeffs.ohc_health = np.ones(n_ch, dtype=np.float32)

  # zero_ratio comes in via h.  In book's circuit D, zero_ratio is 1/sqrt(a),
  # and that a is here 1 / (1+f) where h = f*c.
  # solve for f:  1/zero_ratio^2 = 1 / (1+f)
  # zero_ratio^2 = 1+f => f = zero_ratio^2 - 1
  f = car_params.zero_ratio**2 - 1  # nominally 1 for half-octave

  # Make pole positions, s and c coeffs, h and g coeffs, etc.,
  # which mostly depend on the pole angle theta:
  theta = pole_freqs * (2 * math.pi / fs)

  c0 = np.sin(theta)
  a0 = np.cos(theta)

  # different possible interpretations for min-damping r:
  # r = exp(-theta * car_params.min_zeta).
  # Compress theta to give somewhat higher Q at highest thetas:
  ff = car_params.high_f_damping_compression  # 0 to 1; typ. 0.5
  x = theta / math.pi
  zr_coeffs = math.pi * (x - ff * x**3)  # when ff is 0, this is just theta,
  #                       and when ff is 1 it goes to zero at theta = pi.
  max_zeta = car_params.max_zeta
  car_coeffs.r1_coeffs = (1 - zr_coeffs * max_zeta
                         )  # "r1" for the max-damping condition
  min_zeta = car_params.min_zeta
  # Increase the min damping where channels are spaced out more, by pulling
  # 25% of the way toward hz_to_erb/pole_freqs (close to 0.1 at high f)
  min_zetas = min_zeta + 0.25 * (
      hz_to_erb(pole_freqs, car_params.erb_break_freq, car_params.erb_q) /
      pole_freqs - min_zeta)
  car_coeffs.zr_coeffs = zr_coeffs * (max_zeta - min_zetas)
  # how r relates to undamping

  # undamped coupled-form coefficients:
  car_coeffs.a0_coeffs = a0
  car_coeffs.c0_coeffs = c0

  # the zeros follow via the h_coeffs
  h = c0 * f
  car_coeffs.h_coeffs = h

  # Efficient approximation with g as quadratic function of undamping.
  # First get g at both ends and the half-way point.
  undamping = 0.0
  g0 = design_stage_g(car_coeffs, undamping)
  undamping = 1.0
  g1 = design_stage_g(car_coeffs, undamping)
  undamping = 0.5
  ghalf = design_stage_g(car_coeffs, undamping)
  # Store fixed coefficients for A*undamping.^2 + B^undamping + C
  car_coeffs.ga_coeffs = 2 * (g0 + g1 - 2 * ghalf)
  car_coeffs.gb_coeffs = 4 * ghalf - 3 * g0 - g1
  car_coeffs.gc_coeffs = g0

  # Set up initial stage gains.
  # Maybe re-do this at Init time?
  undamping = car_coeffs.ohc_health
  # Avoid running this model function at Design time; see tests.
  car_coeffs.g0_coeffs = design_stage_g(car_coeffs, undamping)

  return car_coeffs


def design_stage_g(
    car_coeffs: CarCoeffs, relative_undamping: float
) -> np.ndarray:
  """Return the stage gain g needed to get unity gain at DC.

  Args:
    car_coeffs: The already-designed coefficients for the filterbank.
    relative_undamping:  The r variable, defined in section 17.4 of Lyon's book
      to be  (1-b) * NLF(v). The first term is undamping (because b is the
      gain).

  Returns:
    A float vector with the gain for each AGC stage.
  """

  r1 = car_coeffs.r1_coeffs  # at max damping
  a0 = car_coeffs.a0_coeffs
  c0 = car_coeffs.c0_coeffs
  h = car_coeffs.h_coeffs
  zr = car_coeffs.zr_coeffs
  r = r1 + zr * relative_undamping
  g = (1 - 2 * r * a0 + r**2) / (1 - 2 * r * a0 + h * r * c0 + r**2)

  return g


def stage_g(car_coeffs: CarCoeffs, undamping: float) -> np.ndarray:
  """Return the stage gain g needed to get unity gain at DC, using a quadratic approximation.

  Args:
    car_coeffs: The already-designed coefficients for the filterbank.
    undamping:  The r variable, defined in section 17.4 of Lyon's book to be
      (1-b) * NLF(v). The first term is undamping (because b is the gain).

  Returns:
    A float vector with the gain for each AGC stage.
  """
  return (
      car_coeffs.ga_coeffs * (undamping**2)
      + car_coeffs.gb_coeffs * undamping
      + car_coeffs.gc_coeffs
  )


@dataclasses.dataclass
class CarState:
  """All the state variables for the CAR filterbank."""
  z1_memory: np.ndarray
  z2_memory: np.ndarray
  za_memory: np.ndarray
  zb_memory: np.ndarray  # , coeffs.zr_coeffs, ...
  dzb_memory: np.ndarray  # (n_ch, 1), ...
  zy_memory: np.ndarray  # zeros(n_ch, 1), ...
  g_memory: np.ndarray  # coeffs.g0_coeffs, ...
  dg_memory: np.ndarray  #  zeros(n_ch, 1) ...
  ac_coupler: np.ndarray

  def __init__(self, coeffs: CarCoeffs, dtype=np.float32):
    n_ch = coeffs.n_ch
    self.z1_memory = np.zeros((n_ch,), dtype=dtype)
    self.z2_memory = np.zeros((n_ch,), dtype=dtype)
    self.za_memory = np.zeros((n_ch,), dtype=dtype)
    self.zb_memory = coeffs.zr_coeffs
    self.dzb_memory = np.zeros((n_ch,), dtype=dtype)
    self.zy_memory = np.zeros((n_ch,), dtype=dtype)
    self.g_memory = coeffs.g0_coeffs
    self.dg_memory = np.zeros((n_ch,), dtype=dtype)
    self.ac_coupler = np.zeros((n_ch,), dtype=dtype)


def car_init_state(coeffs: CarCoeffs) -> CarState:
  return CarState(coeffs)


def car_step(x_in: float,
             car_coeffs: CarCoeffs,
             car_state: CarState,
             linear: bool = False):
  """One sample-time update step for the filter part of the CARFAC.

  Most of the update is parallel; finally we ripple inputs at the end.
  do the DOHC stuff:

  Args:
    x_in: the input audio
    car_coeffs: the implementation parameters for the filterbank
    car_state: The car_state of the filters before adding this one input sample
    linear: for testing, don't run through the outer hair cell model.

  Returns:
    The filterbank output vector and the new state variables for the filterbank.
  """

  g = car_state.g_memory + car_state.dg_memory  # interp g
  zb = car_state.zb_memory + car_state.dzb_memory  # AGC interpolation car_state
  # update the nonlinear function of "velocity", and za (delay of z2):
  za = car_state.za_memory
  v = car_state.z2_memory - za
  if linear:
    nlf = 1  # To allow testing
  else:
    nlf = ohc_nlf(v, car_coeffs)
  r = car_coeffs.r1_coeffs + zb * nlf  #  zb * nfl is "undamping" delta r.
  za = car_state.z2_memory

  #  now reduce car_state by r and rotate with the fixed cos/sin coeffs:
  z1 = r * (
      car_coeffs.a0_coeffs * car_state.z1_memory -
      car_coeffs.c0_coeffs * car_state.z2_memory)
  #  z1 = z1 + inputs
  z2 = r * (
      car_coeffs.c0_coeffs * car_state.z1_memory +
      car_coeffs.a0_coeffs * car_state.z2_memory)

  if car_coeffs.use_delay_buffer:
    # Optional fully-parallel update uses a delay per stage.
    zy = car_state.zy_memory
    zy[1:] = zy[0:-1]  # Propagate delayed last outputs zy
    zy[0] = x_in  # fill in new input
    z1 = z1 + zy  # add new stage inputs to z1 states
    zy = g * (car_coeffs.h_coeffs * z2 + zy)  # Outputs from z2
  else:
    # Ripple input-output path to avoid delay...
    # this is the only part that doesn't get computed "in parallel":
    zy = car_coeffs.h_coeffs * z2  # partial output
    in_out = x_in
    for ch, output_gain in enumerate(g):
      #  could do this here, or later in parallel:
      z1[ch] += in_out
      # ripple, saving final channel outputs in zy
      in_out = output_gain * (in_out + zy[ch])
      zy[ch] = in_out

  #  put new car_state back in place of old
  #  (z1 is a genuine temp; the others can update by reference in C)
  car_state.z1_memory = z1
  car_state.z2_memory = z2
  car_state.za_memory = za
  car_state.zb_memory = zb
  car_state.zy_memory = zy
  car_state.g_memory = g

  ac_diff = zy - car_state.ac_coupler
  car_state.ac_coupler = car_state.ac_coupler + car_coeffs.ac_coeff * ac_diff
  return ac_diff, car_state


############################################################################
# IHC - Inner Hair Cell
############################################################################


# The next three classes define three different types of inner-hair cell models.
# TODO(malcolmslaney) Perhaps make one superclass?
@dataclasses.dataclass
class IhcJustHwrParams:
  ihc_style: str = 'just_hwr'


@dataclasses.dataclass
class IhcOneCapParams(IhcJustHwrParams):
  ihc_style: str = 'one_cap'
  tau_lpf: float = 0.000080  # 80 microseconds smoothing twice
  tau_out: float = 0.0005  # depletion tau is pretty fast
  tau_in: float = 0.010  # recovery tau is slower


@dataclasses.dataclass
class IhcTwoCapParams(IhcJustHwrParams):
  ihc_style: str = 'two_cap'
  tau_out: float = 0.0005  # depletion tau is pretty fast
  tau_in: float = 0.010  # recovery tau is slower
  tau_lpf: float = 0.000080  # 80 microseconds smoothing twice
  tau1_out: float = 0.000500  # depletion tau is pretty fast
  tau1_in: float = 0.000200  # recovery tau is very fast 200 us
  tau2_out: float = 0.001  # depletion tau is pretty fast 1 ms
  tau2_in: float = 0.010  # recovery tau is slower 10 ms


@dataclasses.dataclass
class IhcSynParams:
  """Parameters for the IHC synapse model."""

  do_syn: bool = False
  n_classes: int = 3  # Default. Modify params and redesign to change.
  ihcs_per_channel: dataclasses.InitVar[int] = 10  # Maybe 20 would be better?
  healthy_n_fibers: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array([50, 35, 25])
  )
  spont_rates: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array([50, 6, 1])
  )
  sat_rates: float = 200.0
  sat_reservoir: float = 0.2
  v_width: float = 0.02
  tau_lpf: float = 0.000080
  reservoir_tau: float = 0.02
  # Tweaked.
  # The weights 1.2 were picked before correctly account for sample rate
  # and number of fibers. This way works for more different numbers.
  agc_weights: np.ndarray = dataclasses.field(
      default_factory=lambda: (np.array([1.2, 1.2, 1.2]) / 22050)
  )

  def __post_init__(self, ihcs_per_channel):
    self.healthy_n_fibers *= ihcs_per_channel
    self.agc_weights /= ihcs_per_channel


# See Section 18.3 (A Digital IHC Model)
def ihc_detect(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
  """An IHC-like sigmoidal detection nonlinearity for the CARFAC.

  Resulting conductance is in about [0...1.3405]
  Args:
    x: the input (BM motion) to the inner hair cell

  Returns:
    The IHC conductance
  """

  if isinstance(x, numbers.Number):
    x_in = np.array((float(x),))
  else:
    x_in = x

  # offset of low-end tail into neg x territory
  # this parameter is adjusted for the book, to make the 20% DC
  # response threshold at 0.1
  a = 0.175

  z = np.maximum(0.0, x_in + a)
  conductance = z**3 / (z**3 + z**2 + 0.1)

  if isinstance(x, numbers.Number):
    return conductance[0]
  return conductance


@dataclasses.dataclass
class IhcCoeffs:
  """Variables needed for the inner hair cell implementation."""
  n_ch: int
  lpf_coeff: float = 0
  out1_rate: float = 0
  in1_rate: float = 0
  out2_rate: float = 0
  in2_rate: float = 0
  output_gain: float = 0
  rest_output: float = 0
  rest_cap2: float = 0
  rest_cap1: float = 0
  rest_cap: float = 0
  out_rate: float = 0
  in_rate: float = 0
  # 0 is just_hwr, 1 is one_cap, 2 is two_cap
  ihc_style: int = 0


@dataclasses.dataclass
class IhcSynCoeffs:
  """Variables needed for the inner hair cell synapse implementation."""

  n_ch: int
  n_classes: int
  n_fibers: np.ndarray
  v_widths: np.ndarray
  v_halfs: np.ndarray  # Same units as v_recep and v_widths.
  a1: float  # Feedback gain
  a2: float  # Output gain
  agc_weights: np.ndarray  # For making a nap out to agc in.
  spont_p: np.ndarray  # Used only to init the output LPF
  spont_sub: np.ndarray
  res_lpf_inits: np.ndarray
  res_coeff: float
  lpf_coeff: float


def design_ihc(
    ihc_params: IhcJustHwrParams | IhcOneCapParams | IhcTwoCapParams,
    fs: float,
    n_ch: int,
) -> IhcCoeffs:
  """Design the inner hair cell implementation from parameters.

  Args:
    ihc_params: The parameters specifying the details of the IHC implementation.
    fs: The sampling rate (Hz) for the DSP operations.
    n_ch: How many channels in the filterbank.

  Returns:
    A IHC coefficient class.
  """
  if isinstance(ihc_params, IhcOneCapParams):
    ro = 1 / ihc_detect(10)  # output resistance at a very high level
    c = ihc_params.tau_out / ro
    ri = ihc_params.tau_in / c
    # to get steady-state average, double ro for 50# duty cycle
    saturation_output = 1 / (2 * ro + ri)
    # also consider the zero-signal equilibrium:
    r0 = 1 / ihc_detect(0)
    current = 1 / (ri + r0)
    cap_voltage = 1 - current * ri
    ihc_coeffs = IhcCoeffs(
        n_ch=n_ch,
        ihc_style=1,
        lpf_coeff=1 - math.exp(-1 / (ihc_params.tau_lpf * fs)),
        out_rate=ro / (ihc_params.tau_out * fs),
        in_rate=1 / (ihc_params.tau_in * fs),
        output_gain=1 / (saturation_output - current),
        rest_output=current / (saturation_output - current),
        rest_cap=cap_voltage,
    )
  elif isinstance(ihc_params, IhcTwoCapParams):
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

    ihc_coeffs = IhcCoeffs(
        n_ch=n_ch,
        ihc_style=2,
        lpf_coeff=1 - math.exp(-1 / (ihc_params.tau_lpf * fs)),
        out1_rate=r1min / (ihc_params.tau1_out * fs),
        in1_rate=1 / (ihc_params.tau1_in * fs),
        out2_rate=r2min / (ihc_params.tau2_out * fs),
        in2_rate=1 / (ihc_params.tau2_in * fs),
        output_gain=1 / (saturation_current2 - rest_current2),
        rest_output=rest_current2 / (saturation_current2 - rest_current2),
        rest_cap2=cap2_voltage,
        rest_cap1=cap1_voltage,
    )
  elif isinstance(ihc_params, IhcJustHwrParams):
    ihc_coeffs = IhcCoeffs(n_ch=n_ch, ihc_style=0)
  else:
    raise NotImplementedError
  return ihc_coeffs


def design_ihc_syn(
    ihc_syn_params: IhcSynParams, fs: float, pole_freqs: np.ndarray
) -> Optional[IhcSynCoeffs]:
  """Design the inner hair cell synapse implementation from parameters."""
  if not ihc_syn_params.do_syn:
    return None
  n_ch = pole_freqs.shape[0]
  n_classes = ihc_syn_params.n_classes
  v_widths = ihc_syn_params.v_width * np.ones((1, n_classes))
  # Do some design.  First, gains to get sat_rate when sigmoid is 1, which
  # involves the reservoir steady-state solution.
  # Most of these are not per-channel, but could be expanded that way
  # later if desired.
  # Mean sat prob of spike per sample per neuron, likely same for all
  # classes.
  # Use names 1 for sat and 0 for spont in some of these.
  p1 = ihc_syn_params.sat_rates / fs
  p0 = ihc_syn_params.spont_rates / fs
  w1 = ihc_syn_params.sat_reservoir
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
  offsets = np.log((1 - s0) / s0)
  spont_p = a2 * w0 * s0  # should match p0; check it; yes it does.
  agc_weights = fs * (ihc_syn_params.agc_weights)
  spont_sub = (ihc_syn_params.healthy_n_fibers * spont_p) @ np.transpose(
      agc_weights
  )
  return IhcSynCoeffs(
      n_ch=n_ch,
      n_classes=n_classes,
      n_fibers=np.ones((n_ch, 1)) * ihc_syn_params.healthy_n_fibers,
      v_widths=v_widths,
      v_halfs=offsets * v_widths,
      a1=a1,
      a2=a2,
      agc_weights=agc_weights,
      spont_p=spont_p,
      spont_sub=spont_sub,
      res_lpf_inits=q0,
      res_coeff=1 - np.exp(-1 / (ihc_syn_params.reservoir_tau * fs)),
      lpf_coeff=1 - np.exp(-1 / (ihc_syn_params.tau_lpf * fs)),
  )


@dataclasses.dataclass
class IhcState:
  """All the state variables for the inner-hair cell implementation."""
  ihc_accum: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array(0)
  )  # Placeholders
  cap_voltage: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array(0)
  )
  lpf1_state: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array(0)
  )
  lpf2_state: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array(0)
  )

  cap1_voltage: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array(0)
  )
  cap2_voltage: np.ndarray = dataclasses.field(
      default_factory=lambda: np.array(0)
  )

  def __init__(self, coeffs, dtype=np.float32):
    n_ch = coeffs.n_ch
    if coeffs.ihc_style == 0:
      self.ihc_accum = np.zeros((n_ch,), dtype=dtype)
    elif coeffs.ihc_style == 1:
      self.ihc_accum = np.zeros((n_ch,), dtype=dtype)
      self.cap_voltage = coeffs.rest_cap * np.ones((n_ch,), dtype=dtype)
      self.lpf1_state = coeffs.rest_output * np.ones((n_ch,), dtype=dtype)
      self.lpf2_state = coeffs.rest_output * np.ones((n_ch,), dtype=dtype)
    else:
      self.ihc_accum = np.zeros((n_ch,), dtype=dtype)
      self.cap1_voltage = coeffs.rest_cap1 * np.ones((n_ch,), dtype=dtype)
      self.cap2_voltage = coeffs.rest_cap2 * np.ones((n_ch,), dtype=dtype)
      self.lpf1_state = coeffs.rest_output * np.ones((n_ch,), dtype=dtype)


def ihc_init_state(coeffs):
  return IhcState(coeffs)


@dataclasses.dataclass
class IhcSynState:
  """All the state variables for the synapse implementation."""

  reservoirs: np.ndarray
  lpf_state: np.ndarray

  def __init__(self, coeffs: IhcSynCoeffs, dtype=np.float32):
    n_ch = coeffs.n_ch
    self.reservoirs = np.ones((n_ch, 1), dtype=dtype) * coeffs.res_lpf_inits
    self.lpf_state = np.ones((n_ch, 1), dtype=dtype) * coeffs.spont_p


def syn_init_state(coeffs: IhcSynCoeffs):
  return IhcSynState(coeffs)


def ohc_nlf(velocities: np.ndarray, car_coeffs: CarCoeffs) -> np.ndarray:
  """The outer hair cell non-linearity.

  Start with a quadratic nonlinear function, and limit it via a
  rational function; make the result go to zero at high
  absolute velocities, so it will do nothing there. See section 17.2 of Lyon's
  book.

  Args:
    velocities: the BM velocities
    car_coeffs: an object that describes the CAR filterbank implementation

  Returns:
    A nonlinear nonnegative function of velocities.
  """

  nlf = 1.0 / (
      1 + (velocities * car_coeffs.velocity_scale + car_coeffs.v_offset)**2)

  return nlf


def ihc_step(
    bm_out: np.ndarray, ihc_coeffs: IhcCoeffs, ihc_state: IhcState
) -> Tuple[np.ndarray, IhcState, np.ndarray]:
  """Step the inner-hair cell model with ont input sample.

  One sample-time update of inner-hair-cell (IHC) model, including the
  detection nonlinearity and one or two capacitor state variables.

  The receptor potential output will be empty unless in two_cap mode. It's used
  as an input for the syn_step.

  Args:
    bm_out: The output from the CAR filterbank
    ihc_coeffs: The run-time parameters for the inner hair cells
    ihc_state: The run-time state

  Returns:
    The firing probability of the hair cells in each channel, the new state and
    the receptor potential output.
  """
  v_recep = np.zeros(ihc_coeffs.n_ch)
  if ihc_coeffs.ihc_style == 0:
    ihc_out = np.minimum(2, np.maximum(0, bm_out))
    #  limit it for stability
  else:
    conductance = ihc_detect(bm_out)  # rectifying nonlinearity

    if ihc_coeffs.ihc_style == 1:
      ihc_out = conductance * ihc_state.cap_voltage
      ihc_state.cap_voltage = (
          ihc_state.cap_voltage - ihc_out * ihc_coeffs.out_rate +
          (1 - ihc_state.cap_voltage) * ihc_coeffs.in_rate)
      #  smooth it twice with LPF:
      ihc_out = ihc_out * ihc_coeffs.output_gain
      ihc_state.lpf1_state = (
          ihc_state.lpf1_state + ihc_coeffs.lpf_coeff *
          (ihc_out - ihc_state.lpf1_state))
      ihc_state.lpf2_state = (
          ihc_state.lpf2_state + ihc_coeffs.lpf_coeff *
          (ihc_state.lpf1_state - ihc_state.lpf2_state))
      ihc_out = ihc_state.lpf2_state - ihc_coeffs.rest_output
    else:
      # Change to 2-cap version mediated by receptor potential at cap1:
      # Geisler book fig 8.4 suggests 40 to 800 Hz corner.
      receptor_current = conductance * ihc_state.cap1_voltage
      # "out" means charge depletion; "in" means restoration toward 1.
      ihc_state.cap1_voltage = (
          ihc_state.cap1_voltage
          - receptor_current * ihc_coeffs.out1_rate
          + (1 - ihc_state.cap1_voltage) * ihc_coeffs.in1_rate
      )
      receptor_potential = 1 - ihc_state.cap1_voltage
      ihc_out = receptor_potential * ihc_state.cap2_voltage
      ihc_state.cap2_voltage = (
          ihc_state.cap2_voltage
          - ihc_out * ihc_coeffs.out2_rate
          + (1 - ihc_state.cap2_voltage) * ihc_coeffs.in2_rate
      )
      ihc_out = ihc_out * ihc_coeffs.output_gain
      ihc_state.lpf1_state = ihc_state.lpf1_state + ihc_coeffs.lpf_coeff * (
          ihc_out - ihc_state.lpf1_state
      )
      ihc_out = ihc_state.lpf1_state - ihc_coeffs.rest_output
      # Return a modified receptor potential that's zero at rest, for SYN.
      v_recep = ihc_coeffs.rest_cap1 - ihc_state.cap1_voltage

  # for where decimated output is useful
  # Leaving this for v2 cochleagram compatibility, but no v3/SYN version:
  ihc_state.ihc_accum = ihc_state.ihc_accum + ihc_out

  return ihc_out, ihc_state, v_recep


def ihc_model_run(input_data: np.ndarray, fs: float) -> np.ndarray:
  """Design and run the inner hair cell model for some input audio.

  Args:
    input_data: Sound data to run through the IHC model.
    fs: The sampling rate (Hz) for the data,

  Returns:
    The output after ther IHC model.
  """
  cfp = design_carfac(fs=fs)
  cfp = carfac_init(cfp)

  output = input_data * 0.0
  ihc_state = cfp.ears[0].ihc_state
  for i in range(input_data.shape[0]):
    car_out = input_data[i]
    # does not work for SYN model.
    ihc_out, ihc_state, _ = ihc_step(car_out, cfp.ears[0].ihc_coeffs, ihc_state)
    output[i] = ihc_out[0]
  return output


def syn_step(
    v_recep: np.ndarray, syn_coeffs: IhcSynCoeffs, syn_state: IhcSynState
) -> Tuple[np.ndarray, np.ndarray, IhcSynState]:
  """Step the synapse model with one input sample."""
  # Drive multiple synapse classes with receptor potential from IHC,
  # returning instantaneous spike rates per class, for a group of neurons
  # associated with the CF channel, including reductions due to synaptopathy.

  # Normalized offset position into neurotransmitter release sigmoid.
  x = (v_recep - np.transpose(syn_coeffs.v_halfs)) / np.transpose(
      syn_coeffs.v_widths
  )
  x = np.transpose(x)
  s = 1 / (1 + np.exp(-x))  # Between 0 and 1; positive at rest.
  q = syn_state.reservoirs  # aka 1 - w, between 0 and 1; positive at rest.
  r = (1 - q) * s  # aka w*s, between 0 and 1, proportional to release rate.

  # Smooth once with LPF (receptor potential was already smooth), after
  # applying the gain coeff a2 to convert to firing prob per sample.
  syn_state.lpf_state = syn_state.lpf_state + syn_coeffs.lpf_coeff * (
      syn_coeffs.a2 * r - syn_state.lpf_state
  )  # this is firing probs.
  firing_probs = syn_state.lpf_state  # Poisson rate per neuron per sample.
  # Include number of effective neurons per channel here, and interval T;
  # so the rates (instantaneous action potentials per second) can be huge.
  firings = syn_coeffs.n_fibers * firing_probs

  # Feedback, to update reservoir state q for next time.
  syn_state.reservoirs = q + syn_coeffs.res_coeff * (syn_coeffs.a1 * r - q)

  # Make an output that resembles ihc_out, to go to agc_in
  # (collapse over classes).
  # Includes synaptopathy's presumed effect of reducing feedback via n_fibers.
  # But it's relative to the healthy nominal spont, so could potentially go
  # a bit negative in quiet is there was loss of high-spont or medium-spont
  # units.
  # The weight multiplication is an inner product, reducing n_classes
  # columns to 1 column (first transpose the agc_weights row to a column).
  syn_out = (
      syn_coeffs.n_fibers * firing_probs
  ) @ syn_coeffs.agc_weights - syn_coeffs.spont_sub

  return syn_out, firings, syn_state


############################################################################
# AGC - Automatic Gain Control
############################################################################


@dataclasses.dataclass
class AgcParams:
  """All the parameters needed to define the behavior of the AGC filters."""
  n_stages: int = 4
  time_constants: np.ndarray = dataclasses.field(
      default_factory=lambda: 0.002 * 4 ** np.arange(4)
  )
  agc_stage_gain: float = 2  # gain from each stage to next slower stage
  # how often to update the AGC states
  decimation: Tuple[int, ...] = (8, 2, 2, 2)
  agc1_scales: List[float] = dataclasses.field(
      default_factory=lambda: 1.0 * np.sqrt(2) ** np.arange(4)
  )  # 1 per channel
  agc2_scales: List[float] = dataclasses.field(
      default_factory=lambda: 1.65 * math.sqrt(2) ** np.arange(4)
  )  # spread more toward base
  agc_mix_coeff: float = 0.5


@dataclasses.dataclass
class AgcCoeffs:
  n_ch: int
  n_agc_stages: int
  agc_stage_gain: float
  agc_epsilon: float = 0  # Will be updated during the design
  decimation: int = 0
  agc_spatial_iterations: int = 0
  agc_spatial_fir: Optional[List[float]] = None
  agc_spatial_n_taps: int = 0
  detect_scale: float = 1
  agc_mix_coeffs: float = 0


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
    a = ((delay_variance + mean_delay * mean_delay) * 2 / 5 -
         mean_delay * 2 / 3) / 2
    b = ((delay_variance + mean_delay * mean_delay) * 2 / 5 +
         mean_delay * 2 / 3) / 2
    # first and last coeffs are implicitly duplicated to make 5-point FIR:
    fir = [a / 2, 1 - a - b, b / 2]
    ok = fir[2 - 1] >= 0.15
  else:
    raise ValueError('Bad n_taps (%d) in agc_spatial_fir' % n_taps)

  return fir, ok


def design_agc(agc_params: AgcParams, fs: float, n_ch: int) -> List[AgcCoeffs]:
  """Design the AGC implementation from the parameters.

  Args:
    agc_params: The parameters desired for this AGC block
    fs: The sampling rate (Hz)
    n_ch: The number of channels in the filterbank.

  Returns:
    The coefficients for all the stages.
  """
  n_agc_stages = agc_params.n_stages

  # AGC1 pass is smoothing from base toward apex;
  # AGC2 pass is back, which is done first now (in double exp. version)
  agc1_scales = agc_params.agc1_scales
  agc2_scales = agc_params.agc2_scales

  decim = 1

  total_dc_gain = 0

  ##
  # Convert to vector of AGC coeffs
  agc_coeffs = []
  for stage in range(n_agc_stages):
    agc_coeffs.append(AgcCoeffs(n_ch, n_agc_stages, agc_params.agc_stage_gain))

    agc_coeffs[stage].decimation = agc_params.decimation[stage]
    tau = agc_params.time_constants[stage]
    # time constant in seconds
    decim = decim * agc_params.decimation[stage]
    # net decim to this stage
    # epsilon is how much new input to take at each update step:
    agc_coeffs[stage].agc_epsilon = 1 - math.exp(-decim / (tau * fs))

    # effective number of smoothings in a time constant:
    ntimes = tau * (fs / decim)  # typically 5 to 50

    # decide on target spread (variance) and delay (mean) of impulse
    # response as a distribution to be convolved ntimes:
    # TODO(dicklyon): specify spread and delay instead of scales?
    delay = (agc2_scales[stage] - agc1_scales[stage]) / ntimes
    spread_sq = (agc1_scales[stage]**2 + agc2_scales[stage]**2) / ntimes

    # get pole positions to better match intended spread and delay of
    # [[geometric distribution]] in each direction (see wikipedia)
    u = 1 + 1 / spread_sq  # these are based on off-line algebra hacking.
    p = u - math.sqrt(u**2 - 1)  # pole that would give spread if used twice.
    dp = delay * (1 - 2 * p + p**2) / 2
    polez1 = p - dp
    polez2 = p + dp
    agc_coeffs[stage].AGC_polez1 = polez1
    agc_coeffs[stage].AGC_polez2 = polez2

    # try a 3- or 5-tap FIR as an alternative to the double exponential:
    n_taps = 0
    done = False
    n_iterations = 1
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

      [agc_spatial_fir, done] = design_fir_coeffs(n_taps, spread_sq, delay,
                                                  n_iterations)

    # When done, store the resulting FIR design in coeffs:
    agc_coeffs[stage].agc_spatial_iterations = n_iterations
    agc_coeffs[stage].agc_spatial_fir = agc_spatial_fir
    agc_coeffs[stage].agc_spatial_n_taps = n_taps

    # accumulate DC gains from all the stages, accounting for stage_gain:
    total_dc_gain = total_dc_gain + agc_params.agc_stage_gain**(stage)

    # TODO(dicklyon) -- is this the best binaural mixing plan?
    if stage == 0:
      agc_coeffs[stage].agc_mix_coeffs = 0
    else:
      agc_coeffs[stage].agc_mix_coeffs = agc_params.agc_mix_coeff / (
          tau * (fs / decim)
      )

  # adjust stage 1 detect_scale to be the reciprocal DC gain of the AGC filters:
  agc_coeffs[0].detect_scale = 1 / total_dc_gain

  return agc_coeffs


@dataclasses.dataclass
class AgcState:
  """All the state variables for one stage of the AGC."""
  agc_memory: np.ndarray
  input_accum: np.ndarray
  decim_phase: int = 0

  def __init__(self, coeffs: List[AgcCoeffs]):
    n_ch = coeffs[0].n_ch

    self.agc_memory = np.zeros((n_ch,))
    self.input_accum = np.zeros((n_ch,))


def agc_init_state(coeffs: List[AgcCoeffs]) -> List[AgcState]:
  n_agc_stages = coeffs[0].n_agc_stages
  state = []
  for _ in range(n_agc_stages):
    state.append(AgcState(coeffs))

  return state


def agc_recurse(coeffs: List[AgcCoeffs], agc_in, stage: int,
                state: List[AgcState]) -> Tuple[bool, List[AgcState]]:
  """Compute the AGC output for one stage, doing the recursion.

  Args:
    coeffs: the details of the AGC design
    agc_in: the input data for this stage, a vector of channel values
    stage: Which stage are we computing?
    state: The state of each channel's AGC.

  Returns:
    The new state and a flag indicating whether the outputs have been updated
    (often not because of decimation).
  """

  # Design consistency checks
  if len(coeffs) != len(state):
    raise ValueError('Length of coeffs (%d) and state (%d) do not agree.' %
                     (len(coeffs), len(state)))
  if len(state[stage].agc_memory) != coeffs[stage].n_ch:
    raise ValueError('Width of agc_memory (%d) and n_ch (%d) do not agree.' %
                     (len(state[stage].agc_memory), coeffs[stage].n_ch))
  if len(agc_in) != coeffs[stage].n_ch:
    raise ValueError('Width of AGC (%d) and n_ch (%d) do not agree.' %
                     (len(agc_in), coeffs[stage].n_ch))
  assert len(agc_in) == coeffs[stage].n_ch

  # decim factor for this stage, relative to input or prev. stage:
  decim = coeffs[stage].decimation

  #  decim phase of this stage (do work on phase 0 only):
  def mod(a: int, b: int):
    return a % b

  decim_phase = mod(state[stage].decim_phase + 1, decim)
  state[stage].decim_phase = decim_phase

  # accumulate input for this stage from detect or previous stage:
  state[stage].input_accum = state[stage].input_accum + agc_in

  # nothing else to do if it's not the right decim_phase
  if decim_phase == 0:
    # Do lots of work, at decimated rate.
    # decimated inputs for this stage, and to be decimated more for next:
    agc_in = state[stage].input_accum / decim
    state[stage].input_accum[:] = 0  # reset accumulator

    if stage < coeffs[0].n_agc_stages - 1:
      updated, state = agc_recurse(coeffs, agc_in, stage + 1, state)
      # and add its output to this stage input, whether it updated or not:
      agc_in = agc_in + coeffs[stage].agc_stage_gain * state[stage +
                                                             1].agc_memory

    agc_stage_state = state[stage].agc_memory
    # first-order recursive smoothing filter update, in time:
    agc_stage_state = agc_stage_state + coeffs[stage].agc_epsilon * (
        agc_in - agc_stage_state)
    # spatial smooth:
    agc_stage_state = spatial_smooth(coeffs[stage], agc_stage_state)
    # and store the state back (in C++, do it all in place?)
    state[stage].agc_memory = agc_stage_state

    updated = True  # bool to say we have new state
  else:
    updated = False

  return updated, state


def agc_step(detects: np.ndarray, coeffs: List[AgcCoeffs],
             state: List[AgcState]) -> Tuple[bool, List[AgcState]]:
  """One time step of the AGC state update; decimates internally.

  Args:
    detects: The output of the IHC stage.
    coeffs: A list of AGC coefficients, one per stage (typically 4)
    state: A list of AGC states

  Returns:
    A bool to indicate whether the AGC output has updated, and the new state.
  """

  stage = 0
  agc_in = coeffs[0].detect_scale * detects
  return agc_recurse(coeffs, agc_in, stage, state)


############################################################################
# CARFAC - Cascade of asymmetric resonators with fast acting compression
############################################################################


@dataclasses.dataclass
class CarfacCoeffs:
  car_coeffs: CarCoeffs
  agc_coeffs: List[AgcCoeffs]  # One element per AGC layer (typically 4)
  ihc_coeffs: IhcCoeffs
  syn_coeffs: Optional[IhcSynCoeffs] = None
  car_state: Optional[CarState] = None
  ihc_state: Optional[IhcState] = None
  agc_state: Optional[List[AgcState]] = None
  syn_state: Optional[IhcSynState] = None


@dataclasses.dataclass
class CarfacParams:
  fs: float
  max_channels_per_octave: float
  car_params: CarParams
  agc_params: AgcParams
  ihc_params: Optional[IhcJustHwrParams | IhcOneCapParams | IhcTwoCapParams]
  syn_params: Optional[IhcSynParams]
  n_ch: int
  pole_freqs: float
  ears: List[CarfacCoeffs]
  n_ears: int


def design_carfac(
    n_ears: int = 1,
    fs: float = 22050,
    car_params: Optional[CarParams] = None,
    agc_params: Optional[AgcParams] = None,
    ihc_params: Optional[
        Union[IhcJustHwrParams, IhcOneCapParams, IhcTwoCapParams]
    ] = None,
    syn_params: Optional[IhcSynParams] = None,
    ihc_style: str = 'two_cap',
    use_delay_buffer: bool = False,
) -> CarfacParams:
  """This function designs the CARFAC filterbank.

  CARFAC is a Cascade of Asymmetric Resonators with Fast-Acting Compression);
  that is, it take bundles of parameters and computes all the filter
  coefficients needed to run it.

  See other functions for designing and characterizing the CARFAC:
    [naps, cfp] = Run(cfp, input_waves)
    transfns = Transfer_Functions(cfp, to_channels, from_channels)

  Defaults to Greenwood map with Q value from Glasberg & Moore's ERB scale:
  erb_break_freq = 165.3;  # Greenwood map's break freq.

  All args are defaultable; for sample/default args see the code; they
  make 96 channels at default fs = 22050, 114 channels at 44100.

  Args:
    n_ears: How many ears (1 or 2, in general) in the simulation
    fs: is sample rate (per second)
    car_params: bundles all the pole-zero filter cascade parameters
    agc_params: bundles all the automatic gain control parameters
    ihc_params: bundles all the inner hair cell parameters
    syn_params: bundles all synaptopathy parameters
    ihc_style: Type of IHC Model to use. Valid avlues are 'one_cap' for Allen
      model, 'two_cap' for the v2 model, 'two_cap_with_syn' for the v3 model and
      'just_hwr' for the simpler HWR model.
    use_delay_buffer: Whether to use the delay buffer implementation in the CAR
      step.

  Returns:
    A Carfac filter structure (for running the calcs.)
  """

  car_params = car_params or CarParams()
  agc_params = agc_params or AgcParams()

  if not ihc_params:
    match ihc_style:
      case 'two_cap' | 'two_cap_with_syn':
        ihc_params = IhcTwoCapParams()
      case 'one_cap':
        ihc_params = IhcOneCapParams()
      case 'just_hwr':
        ihc_params = IhcJustHwrParams()
      case _:
        raise ValueError(f'Unknown IHC style: {ihc_style}')
  if not syn_params and ihc_style == 'two_cap_with_syn':
    syn_params = IhcSynParams(do_syn=True)

  # first figure out how many filter stages (PZFC/CARFAC channels):
  pole_hz = car_params.first_pole_theta * fs / (2 * math.pi)
  n_ch = 0
  while pole_hz > car_params.min_pole_hz:
    n_ch = n_ch + 1
    pole_hz = pole_hz - car_params.erb_per_step * hz_to_erb(
        pole_hz, car_params.erb_break_freq, car_params.erb_q)

  # Now we have n_ch, the number of channels, so can make the array
  # and compute all the frequencies again to put into it:
  pole_freqs = np.zeros((n_ch,), dtype=np.float32)  # float64 didn't help
  pole_hz = car_params.first_pole_theta * fs / (2 * math.pi)
  for ch in range(n_ch):
    pole_freqs[ch] = pole_hz
    pole_hz = pole_hz - car_params.erb_per_step * hz_to_erb(
        pole_hz, car_params.erb_break_freq, car_params.erb_q)
  # Now we have n_ch, the number of channels, and pole_freqs array.

  max_channels_per_octave = 1 / math.log(pole_freqs[0] / pole_freqs[1], 2)

  # Convert to include an ear_array, each w coeffs and state...
  car_coeffs = design_filters(
      car_params, fs, pole_freqs, use_delay_buffer=use_delay_buffer
  )
  agc_coeffs = design_agc(agc_params, fs, n_ch)
  ihc_coeffs = design_ihc(ihc_params, fs, n_ch)
  if syn_params:
    syn_coeffs = design_ihc_syn(syn_params, fs, pole_freqs)
  else:
    syn_coeffs = None
  # Copy same designed coeffs into each ear (can do differently in the
  # future).
  ears = []
  for _ in range(n_ears):
    ears.append(CarfacCoeffs(car_coeffs, agc_coeffs, ihc_coeffs, syn_coeffs))

  cfp = CarfacParams(
      fs,
      max_channels_per_octave,
      car_params,
      agc_params,
      ihc_params,
      syn_params,
      n_ch,
      pole_freqs,
      ears,
      n_ears,
  )
  return cfp


def carfac_init(cfp: CarfacParams) -> CarfacParams:
  """Initialize state for one or more ears of a CARFAC model.

  This allocates and zeros all the state vector storage in the cfp struct.
  Args:
    cfp: the state structure for the filterbank

  Returns:
    A new version of the state structure with all initializations done.
  """

  n_ears = cfp.n_ears

  for ear in range(n_ears):
    # for now there's only one coeffs, not one per ear
    cfp.ears[ear].car_state = car_init_state(cfp.ears[ear].car_coeffs)
    cfp.ears[ear].ihc_state = ihc_init_state(cfp.ears[ear].ihc_coeffs)
    cfp.ears[ear].agc_state = agc_init_state(cfp.ears[ear].agc_coeffs)
    if cfp.syn_params:
      cfp.ears[ear].syn_state = syn_init_state(cfp.ears[ear].syn_coeffs)
  return cfp


def shift_right(s: np.ndarray, amount: int) -> np.ndarray:
  """Rotate a vector to the right by amount, or to the left if negative."""
  if amount > 0:
    return np.concatenate((s[0:amount, ...], s[:-amount, ...]), axis=0)
  elif amount < 0:
    return np.concatenate((s[-amount:, ...], np.flip(s[amount:, ...])), axis=0)
  else:
    return s


def spatial_smooth(coeffs: AgcCoeffs, stage_state: np.ndarray) -> np.ndarray:
  """Design the AGC spatial smoothing filter.

  TODO(dicklyon): Can you say more??

  Args:
    coeffs: The coefficient description for this state
    stage_state: The state of the AGC right now.

  Returns:
    A new stage stage object.
  """

  n_iterations = coeffs.agc_spatial_iterations

  use_fir = n_iterations >= 0

  if use_fir:
    fir_coeffs = coeffs.agc_spatial_fir
    if coeffs.agc_spatial_n_taps == 3:
      for _ in range(n_iterations):
        stage_state = (
            fir_coeffs[0] * shift_right(stage_state, 1) +
            fir_coeffs[1] * shift_right(stage_state, 0) +
            fir_coeffs[2] * shift_right(stage_state, -1))

    #  5-tap smoother duplicates first and last coeffs
    elif coeffs.agc_spatial_n_taps == 5:
      for _ in range(n_iterations):
        stage_state = (
            fir_coeffs[0] *
            (shift_right(stage_state, 2) + shift_right(stage_state, 1)) +
            fir_coeffs[1] * shift_right(stage_state, 0) + fir_coeffs[2] *
            (shift_right(stage_state, -1) + shift_right(stage_state, -2)))
    else:
      raise ValueError('Bad agc_spatial_n_taps (%d) in spatial_smooth' %
                       coeffs.agc_spatial_n_taps)
  else:
    # use IIR method, back-and-forth first-order smoothers:
    raise NotImplementedError
    # TODO(malcolmslaney) Translate smooth_double_exponential()
    # stage_state = smooth_double_exponential(stage_state,
    #   coeffs.AGC_polez1[stage], coeffs.AGC_polez2[stage])

  return stage_state


def smooth_double_exponential(signal_vecs, polez1, polez2, fast_matlab_way):
  # Not sure why this was forgotten in the first conversion of Matlab to np..
  # but not needed for now.
  raise NotImplementedError


def close_agc_loop(cfp: CarfacParams) -> CarfacParams:
  """Fastest decimated rate determines interp needed."""
  decim1 = cfp.agc_params.decimation[0]

  for ear in range(cfp.n_ears):
    undamping = 1 - cfp.ears[ear].agc_state[0].agc_memory  # stage 1 result
    undamping = undamping * cfp.ears[ear].car_coeffs.ohc_health
    # Update the target stage gain for the new damping:
    new_g = stage_g(cfp.ears[ear].car_coeffs, undamping)
    # set the deltas needed to get to the new damping:
    cfp.ears[ear].car_state.dzb_memory = (
        (cfp.ears[ear].car_coeffs.zr_coeffs * undamping -
         cfp.ears[ear].car_state.zb_memory) / decim1)
    cfp.ears[ear].car_state.dg_memory = (
        new_g - cfp.ears[ear].car_state.g_memory) / decim1
  return cfp


def cross_couple(ears: List[CarfacCoeffs]) -> List[CarfacCoeffs]:
  """This function cross couples gain between multiple ears.

  There's no impact for this function in the case of monoaural input.
  Args:
    ears: the list of ear inputs.

  Returns:
    The list of ear inputs, modified to multiaural coupling.
  """
  if len(ears) <= 1:
    return ears
  n_stages = ears[0].agc_coeffs[0].n_agc_stages
  # now cross-ear mix the stages that updated (leading stages at phase 0):
  for stage in range(n_stages):
    if ears[0].agc_state[stage].decim_phase > 0:
      return ears
    else:
      mix_coeff = ears[0].agc_coeffs[stage].agc_mix_coeffs
      if mix_coeff <= 0:
        continue
      this_stage_sum = 0
      # sum up over the ears and get their mean:
      for ear in ears:
        this_stage_sum += ear.agc_state[stage].agc_memory
      this_stage_mean = this_stage_sum / len(ears)
      # now move them all toward the mean:
      for ear in ears:
        stage_state = ear.agc_state[stage].agc_memory
        ear.agc_state[stage].agc_memory = stage_state + mix_coeff * (
            this_stage_mean - stage_state
        )
  return ears


def run_segment(
    cfp: CarfacParams,
    input_waves: np.ndarray,
    open_loop: bool = False,
    linear_car: bool = False
) -> Tuple[np.ndarray, CarfacParams, np.ndarray, np.ndarray, np.ndarray]:
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
    cfp: a structure that descirbes everything we know about this CARFAC.
    input_waves: the audio input
    open_loop: whether to run CARFAC without the feedback.
    linear_car (new over Matlab): use CAR filters without OHC effects.

  Returns:
    naps: neural activity pattern
    cfp: the new structure describing the state of the CARFAC
    BM: The basilar membrane motion
    seg_ohc & seg_agc are optional extra outputs useful for seeing what the
      ohc nonlinearity and agc are doing; both in terms of extra damping.
  """

  do_bm = 1

  if len(input_waves.shape) < 2:
    input_waves = np.reshape(input_waves, (-1, 1))
  [n_samp, n_ears] = input_waves.shape

  if n_ears != cfp.n_ears:
    raise ValueError(
        'Bad number of input_waves channels (%d vs %d) passed to Run' %
        (n_ears, cfp.n_ears))

  n_ch = cfp.n_ch
  naps = np.zeros((n_samp, n_ch, n_ears))  # allocate space for result
  if do_bm:
    bm = np.zeros((n_samp, n_ch, n_ears))
    seg_ohc = np.zeros((n_samp, n_ch, n_ears))
    seg_agc = np.zeros((n_samp, n_ch, n_ears))

  # A 2022 addition to make open-loop running behave:
  if open_loop:
    # zero the deltas:
    for ear in range(cfp.n_ears):
      cfp.ears[ear].car_state.dzb_memory *= 0
      cfp.ears[ear].car_state.dg_memory *= 0

  for k in range(n_samp):
    # at each time step, possibly handle multiple channels
    for ear in range(n_ears):

      # This would be cleaner if we could just get and use a reference to
      # cfp.ears(ear), but Matlab doesn't work that way...
      [car_out, cfp.ears[ear].car_state] = car_step(
          input_waves[k, ear],
          cfp.ears[ear].car_coeffs,
          cfp.ears[ear].car_state,
          linear=linear_car)

      # update IHC state & output on every time step, too
      [ihc_out, cfp.ears[ear].ihc_state, v_recep] = ihc_step(
          car_out, cfp.ears[ear].ihc_coeffs, cfp.ears[ear].ihc_state
      )

      if cfp.syn_params and cfp.syn_params.do_syn:
        # We don't use the firings yet.
        [ihc_syn_out, _, cfp.ears[ear].syn_state] = syn_step(
            v_recep, cfp.ears[ear].syn_coeffs, cfp.ears[ear].syn_state
        )
        nap = ihc_syn_out
      else:
        nap = ihc_out

      # run the AGC update step, decimating internally,
      [agc_updated, cfp.ears[ear].agc_state] = agc_step(
          nap, cfp.ears[ear].agc_coeffs, cfp.ears[ear].agc_state
      )

      # save some output data:
      naps[k, :, ear] = nap  # output to neural activity pattern
      if do_bm:
        bm[k, :, ear] = car_out
        state = cfp.ears[ear].car_state
        seg_ohc[k, :, ear] = state.za_memory
        seg_agc[k, :, ear] = state.zb_memory

    #  connect the feedback from agc_state to car_state when it updates;
    #  all ears together here due to mixing across them:
    if agc_updated:
      if n_ears > 1:
        #  do multi-aural cross-coupling:
        cfp.ears = cross_couple(cfp.ears)
      if not open_loop:
        cfp = close_agc_loop(cfp)

  return naps, cfp, bm, seg_ohc, seg_agc
