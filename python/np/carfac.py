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
the original variable names.
https://github.com/google/carfac/tree/master/matlab
"""

import dataclasses
import math
import numbers
from typing import List, Optional, Tuple, Union

import numpy as np

# TODO(malcolmslaney): Make sure everything is type annotated & has doc string.
# TODO(malcolmslaney): Get rid of bare generic warnings
# TODO(malcolmslaney): Figure out attribute lint errors
# TODO(malcolmslaney, dicklyon): Check out ???s in the documentation.


# Note in general, a functional block (i.e. IHC or AGC) has parameters.  After
# a design phase they become coefficients that describe how to do the
# calculation.  And each block has some state.  In other words:
#  Params -> Coeffs -> State
# The CarfacParams structure, usually stored as a cfp, also points to the
# ear(s), and each ear has a pointer to the coeffs structure and those include
# the state variables.

# In addition there are three blocks that act in sequence to create the entire
# CARFAC model.  The three blocks are: CAR, IHC, AGC.
## CARFAC Design Functions

### CARFAC Parameter Structures
# pylint: disable=g-bare-generic
# pytype: disable=attribute-error

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


@dataclasses.dataclass
class CarCoeffs:
  n_ch: int
  velocity_scale: float
  v_offset: float

  r1_coeffs: np.float64 = None
  a0_coeffs: np.float64 = None
  c0_coeffs: np.float64 = None
  h_coeffs: np.float64 = None
  g0_coeffs: np.float64 = None


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


def design_filters(car_params, fs, pole_freqs):
  """Design the actual CAR filters."""

  n_ch = len(pole_freqs)
  pole_freqs = np.asarray(pole_freqs)

  # the filter design coeffs:
  # scalars first:
  car_coeffs = CarCoeffs(
      n_ch=n_ch,
      velocity_scale=car_params.velocity_scale,
      v_offset=car_params.v_offset)

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

  # for unity gain at min damping, radius r; only used in Init:
  relative_undamping = np.ones((n_ch,))  # max undamping to start
  # this function needs to take car_coeffs even if we haven't finished
  # constucting it by putting in the g0_coeffs:
  car_coeffs.g0_coeffs = stage_g(car_coeffs, relative_undamping)

  return car_coeffs


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


def car_init_state(coeffs):
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
  #  zb * nfl is "undamping" delta r:
  r = car_coeffs.r1_coeffs + zb * nlf
  za = car_state.z2_memory

  #  now reduce car_state by r and rotate with the fixed cos/sin coeffs:
  z1 = r * (
      car_coeffs.a0_coeffs * car_state.z1_memory -
      car_coeffs.c0_coeffs * car_state.z2_memory)
  #  z1 = z1 + inputs
  z2 = r * (
      car_coeffs.c0_coeffs * car_state.z1_memory +
      car_coeffs.a0_coeffs * car_state.z2_memory)

  zy = car_coeffs.h_coeffs * z2  # partial output

  #  Ripple input-output path, instead of parallel, to avoid delay...
  # this is the only part that doesn't get computed "in parallel":
  in_out = x_in
  for ch in range(len(zy)):
    #  could do this here, or later in parallel:
    z1[ch] = z1[ch] + in_out
    # ripple, saving final channel outputs in zy
    in_out = g[ch] * (in_out + zy[ch])
    zy[ch] = in_out

  #  put new car_state back in place of old
  #  (z1 is a genuine temp; the others can update by reference in C)
  car_state.z1_memory = z1
  car_state.z2_memory = z2
  car_state.za_memory = za
  car_state.zb_memory = zb
  car_state.zy_memory = zy
  car_state.g_memory = g

  car_out = zy

  return car_out, car_state


############################################################################
# IHC - Inner Hair Cell
############################################################################


# The next three classes define three different types of inner-hair cell models.
# TODO(malcolmslaney) Perhaps make one superclass?
@dataclasses.dataclass
class IhcJustHwrParams:
  just_hwr: bool = True  # just a simple HWR
  ac_corner_hz: float = 20


@dataclasses.dataclass
class IhcOneCapParams:
  just_hwr: bool = False  # not just a simple HWR
  one_cap: bool = True  # bool; 0 for new two-cap hack
  tau_lpf: float = 0.000080  # 80 microseconds smoothing twice
  tau_out: float = 0.0005  # depletion tau is pretty fast
  tau_in: float = 0.010  # recovery tau is slower
  ac_corner_hz: float = 20


@dataclasses.dataclass
class IhcTwoCapParams:
  just_hwr: bool = False  # not just a simple HWR
  one_cap: bool = False  # bool; 0 for new two-cap hack
  tau_lpf: float = 0.000080  # 80 microseconds smoothing twice
  tau1_out: float = 0.010  # depletion tau is pretty fast
  tau1_in: float = 0.020  # recovery tau is slower
  tau2_out: float = 0.0025  # depletion tau is pretty fast
  tau2_in: float = 0.005  # recovery tau is slower
  ac_corner_hz: float = 20


# See Section 18.3 (A Digital IHC Model)
def ihc_detect(x: Union[float, np.ndarray]):
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
  just_hwr: bool
  lpf_coeff: float = 0
  out1_rate: float = 0
  in1_rate: float = 0
  out2_rate: float = 0
  in2_rate: float = 0
  one_cap: float = 0
  output_gain: float = 0
  rest_output: float = 0
  rest_cap2: float = 0
  rest_cap1: float = 0
  ac_coeff: float = 0

  rest_cap: float = 0
  out_rate: float = 0
  in_rate: float = 0


def design_ihc(ihc_params, fs, n_ch):
  """Design the inner hair cell implementation from parameters."""
  if ihc_params.just_hwr:
    ihc_coeffs = IhcCoeffs(n_ch=n_ch, just_hwr=True)
  else:
    if ihc_params.one_cap:
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
          just_hwr=False,
          lpf_coeff=1 - math.exp(-1 / (ihc_params.tau_lpf * fs)),
          out_rate=ro / (ihc_params.tau_out * fs),
          in_rate=1 / (ihc_params.tau_in * fs),
          one_cap=ihc_params.one_cap,
          output_gain=1 / (saturation_output - current),
          rest_output=current / (saturation_output - current),
          rest_cap=cap_voltage)
    else:
      ro = 1 / ihc_detect(10)  # output resistance at a very high level
      c2 = ihc_params.tau2_out / ro
      r2 = ihc_params.tau2_in / c2
      c1 = ihc_params.tau1_out / r2
      r1 = ihc_params.tau1_in / c1
      # to get steady-state average, double ro for 50# duty cycle
      saturation_output = 1 / (2 * ro + r2 + r1)
      # also consider the zero-signal equilibrium:
      r0 = 1 / ihc_detect(0)
      current = 1 / (r1 + r2 + r0)
      cap1_voltage = 1 - current * r1
      cap2_voltage = cap1_voltage - current * r2
      ihc_coeffs = IhcCoeffs(
          n_ch=n_ch,
          just_hwr=False,
          lpf_coeff=1 - math.exp(-1 / (ihc_params.tau_lpf * fs)),
          out1_rate=1 / (ihc_params.tau1_out * fs),
          in1_rate=1 / (ihc_params.tau1_in * fs),
          out2_rate=ro / (ihc_params.tau2_out * fs),
          in2_rate=1 / (ihc_params.tau2_in * fs),
          one_cap=ihc_params.one_cap,
          output_gain=1 / (saturation_output - current),
          rest_output=current / (saturation_output - current),
          rest_cap2=cap2_voltage,
          rest_cap1=cap1_voltage)
  # one more late addition that applies to all cases:
  ihc_coeffs.ac_coeff = 2 * math.pi * ihc_params.ac_corner_hz / fs
  return ihc_coeffs


@dataclasses.dataclass
class IhcState:
  """All the state variables for the inner-hair cell implementation."""
  ihc_accum: np.ndarray = np.array(0)  # Placeholders
  cap_voltage: np.ndarray = np.array(0)
  lpf1_state: np.ndarray = np.array(0)
  lpf2_state: np.ndarray = np.array(0)
  ac_coupler: np.ndarray = np.array(0)

  cap1_voltage: np.ndarray = np.array(0)
  cap2_voltage: np.ndarray = np.array(0)

  def __init__(self, coeffs):
    n_ch = coeffs.n_ch
    if coeffs.just_hwr:
      self.ihc_accum = np.zeros((n_ch,))
      self.ac_coupler = np.zeros((n_ch,))
    else:
      if coeffs.one_cap:
        self.ihc_accum = np.zeros((n_ch,))
        self.cap_voltge = coeffs.rest_cap * np.ones((n_ch,))
        self.lpf1_state = coeffs.rest_output * np.ones((n_ch,))
        self.lpf2_state = coeffs.rest_output * np.ones((n_ch,))
        self.ac_coupler = np.zeros((n_ch,))
      else:
        self.ihc_accum = np.zeros((n_ch,))
        self.cap1_voltage = coeffs.rest_cap1 * np.ones((n_ch,))
        self.cap2_voltage = coeffs.rest_cap2 * np.ones((n_ch,))
        self.lpf1_state = coeffs.rest_output * np.ones((n_ch,))
        self.lpf2_state = coeffs.rest_output * np.ones((n_ch,))
        self.ac_coupler = np.zeros((n_ch,))


def ihc_init_state(coeffs):
  return IhcState(coeffs)


def ohc_nlf(velocities, car_coeffs: CarCoeffs):
  #  function nlf = ohc_nlf(velocities, car_coeffs)
  # start with a quadratic nonlinear function, and limit it via a
  # rational function; make the result go to zero at high
  #  absolute velocities, so it will do nothing there.

  nlf = 1.0 / (
      1 + (velocities * car_coeffs.velocity_scale + car_coeffs.v_offset)**2)

  return nlf


def ihc_step(filters_out: np.float64, coeffs: IhcCoeffs,
             ihc_state: IhcState):
  """Step the inner-hair cell model with ont input sample.

  One sample-time update of inner-hair-cell (IHC) model, including the
  detection nonlinearity and one or two capacitor state variables.

  Args:
    filters_out: The output from the CAR filterbank
    coeffs: The run-time parameters for the inner hair cells
    ihc_state: The run-time state

  Returns:
    The firing probability (??) for the hair cells in each channel
    and the new state.
  """
  # TODO(malcolmslaney) change coeffs to ihc_coeffs for consistency.

  # AC couple the filters_out, with 20 Hz corner
  ac_diff = filters_out - ihc_state.ac_coupler
  ihc_state.ac_coupler = ihc_state.ac_coupler + coeffs.ac_coeff * ac_diff

  if coeffs.just_hwr:
    ihc_out = np.min(2, np.max(0, ac_diff))
    #  limit it for stability
  else:
    conductance = ihc_detect(ac_diff)  # rectifying nonlinearity

    if coeffs.one_cap:
      ihc_out = conductance * ihc_state.cap_voltage
      ihc_state.cap_voltage = (
          ihc_state.cap_voltage - ihc_out * coeffs.out_rate +
          (1 - ihc_state.cap_voltage) * coeffs.in_rate)
    else:
      # change to 2-cap version more like Meddis's:
      ihc_out = conductance * ihc_state.cap2_voltage
      ihc_state.cap1_voltage = (
          ihc_state.cap1_voltage -
          (ihc_state.cap1_voltage - ihc_state.cap2_voltage) * coeffs.out1_rate +
          (1 - ihc_state.cap1_voltage) * coeffs.in1_rate)

      ihc_state.cap2_voltage = (
          ihc_state.cap2_voltage - ihc_out * coeffs.out2_rate +
          (ihc_state.cap1_voltage - ihc_state.cap2_voltage) * coeffs.in2_rate)

    #  smooth it twice with LPF:
    ihc_out = ihc_out * coeffs.output_gain
    ihc_state.lpf1_state = (
        ihc_state.lpf1_state +
        coeffs.lpf_coeff * (ihc_out - ihc_state.lpf1_state))
    ihc_state.lpf2_state = (
        ihc_state.lpf2_state + coeffs.lpf_coeff *
        (ihc_state.lpf1_state - ihc_state.lpf2_state))
    ihc_out = ihc_state.lpf2_state - coeffs.rest_output

  # for where decimated output is useful
  ihc_state.ihc_accum = ihc_state.ihc_accum + ihc_out

  return ihc_out, ihc_state


def ihc_model_run(input_data, fs):
  """Design and run the inner hair cell model for some input audio."""
  cfp = design_carfac(fs=fs)
  cfp = carfac_init(cfp)

  output = input_data * 0.0
  ihc_state = cfp.ears[0].ihc_state
  for i in range(input_data.shape[0]):
    car_out = input_data[i]
    ihc_out, ihc_state = ihc_step(car_out, cfp.ears[0].ihc_coeffs,
                                  ihc_state)
    output[i] = ihc_out[0]
  return output


############################################################################
# AGC - Automatic Gain Control
############################################################################


@dataclasses.dataclass
class AgcParams:
  """All the parameters needed to define the behavior of the AGC filters."""
  n_stages: int = 4
  time_constants: np.ndarray = 0.002 * 4**np.arange(4)
  agc_stage_gain: float = 2  # gain from each stage to next slower stage
  decimation: tuple = (8, 2, 2, 2)  # how often to update the AGC states
  agc1_scales: list = 1.0 * np.sqrt(2)**np.arange(4)  # in units of channels
  agc2_scales: list = 1.65 * math.sqrt(2)**np.arange(
      4)  # spread more toward base
  agc_mix_coeff: float = 0.5


@dataclasses.dataclass
class AgcCoeffs:
  n_ch: int
  n_agc_stages: int
  agc_stage_gain: float
  decimation: int = 0  # check this type
  agc_spatial_iterations: int = 0
  agc_spatial_fir: Optional[list] = None  # Check this type
  agc_spatial_n_taps: int = 0
  detect_scale: float = 1


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
    n_taps: Width of the spatial filter kernel.
    delay_variance: ???
    mean_delay: ???
    n_iter: ???

  Returns:
    List of FIR coefficients, and a boolen saying the design was done
    correctly.
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


def design_agc(agc_params, fs, n_ch):
  """Design the AGC implementation from the parameters."""
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
    agc_coeffs.append(
        AgcCoeffs(n_ch, n_agc_stages, agc_params.agc_stage_gain))

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
    # TODO(dicklyon): specify spread and delay instead of scales???
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
    done = 0
    n_iterations = 1
    if spread_sq == 0:
      n_iterations = 0
      n_taps = 3
      done = 1
    while ~done:
      if n_taps == 0:
        # first attempt a 3-point FIR to apply once:
        n_taps = 3
      elif n_taps == 3:
        # second time through, go wider but stick to 1 iteration
        n_taps = 5
      elif n_taps == 5:
        # apply FIR multiple times instead of going wider:
        n_iterations = n_iterations + 1
        if n_iterations > 4:
          n_iterations = -1  # Signal to use IIR instead.
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
          tau * (fs / decim))

  # adjust stage 1 detect_scale to be the reciprocal DC gain of the AGC filters:
  agc_coeffs[0].detect_scale = 1 / total_dc_gain

  return agc_coeffs


def stage_g(car_coeffs, relative_undamping):
  """Return the stage gain g needed to get unity gain at DC."""

  r1 = car_coeffs.r1_coeffs  # at max damping
  a0 = car_coeffs.a0_coeffs
  c0 = car_coeffs.c0_coeffs
  h = car_coeffs.h_coeffs
  zr = car_coeffs.zr_coeffs
  r = r1 + zr * relative_undamping
  g = (1 - 2 * r * a0 + r**2) / (1 - 2 * r * a0 + h * r * c0 + r**2)

  return g


@dataclasses.dataclass
class AgcState:
  """All the state variables for one stage of the AGC."""
  agc_memory: np.float64
  input_accum: np.float64
  decim_phase: int = 0

  def __init__(self, coeffs: List[AgcCoeffs]):
    n_ch = coeffs[0].n_ch

    self.agc_memory = np.zeros((n_ch,))
    self.input_accum = np.zeros((n_ch,))


def agc_init_state(coeffs):
  n_agc_stages = coeffs[0].n_agc_stages
  state = []
  for _ in range(n_agc_stages):
    state.append(AgcState(coeffs))

  return state


def agc_recurse(coeffs: List[AgcCoeffs], agc_in, stage: int,
                state: List[AgcState]):
  """Compute the AGC output for one stage, doing the recursion.

  Args:
    coeffs: the details of the AGC design
    agc_in: the input data for this stage, a vector of channel values
    stage: Which stage are we computing?
    state: The state of each channel's AGC.

  Returns:
    The new state and a flag indicating whether the outputs have been updated
    (often not because of decimation.)
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
      state, updated = agc_recurse(coeffs, agc_in, stage + 1, state)
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

    updated = 1  # bool to say we have new state
  else:
    updated = 0

  return state, updated


def agc_step(detects, coeffs: List[AgcCoeffs],
             state: List[AgcState]):
  #  function [state, updated] = agc_step(detects, coeffs, state)
  #
  # one time step of the AGC state update; decimates internally

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

  car_state: Optional[CarState] = None
  ihc_state: Optional[IhcState] = None
  agc_state: Optional[List[AgcState]] = None


@dataclasses.dataclass
class CarfacParams:
  fs: float
  max_channels_per_octave: int
  car_params: CarParams
  agc_params: AgcParams
  ihc_params: Optional[IhcCoeffs]
  n_ch: int
  pole_freqs: np.float64
  ears: List[CarfacCoeffs]
  n_ears: int


def design_carfac(
    n_ears: int = 1,
    fs: float = 22050,
    car_params: Optional[CarParams] = None,
    agc_params: Optional[AgcParams] = None,
    ihc_params: Optional[Union[IhcJustHwrParams,
                               IhcOneCapParams,
                               IhcTwoCapParams]] = None):
  """This function designs the CARFAC filterbank.

  CARFAC is a Cascade of Asymmetric Resonators with Fast-Acting Compression);
  that is, it take bundles of parameters and computes all the filter
  coefficients needed to run it.

  See other functions for designing and characterizing the CARFAC:
    [naps, cfp] = Run(cfp, input_waves)
    transfns = Transfer_Functions(cfp, to_channels, from_channels)

  Defaults to Glasberg & Moore's ERB curve:
    erb_break_freq = 1000/4.37;  # 228.833
    erb_q = 1000/(24.7*4.37);    # 9.2645

  All args are defaultable; for sample/default args see the code; they
  make 96 channels at default fs = 22050, 114 channels at 44100.

  Args:
    n_ears: How many ears (1 or 2, in general) in the simulation
    fs: is sample rate (per second)
    car_params: bundles all the pole-zero filter cascade parameters
    agc_params: bundles all the automatic gain control parameters
    ihc_params: bundles all the inner hair cell parameters

  Returns:
    A Carfac filter structure (for running the calcs.)

  """

  car_params = car_params or CarParams()
  agc_params = agc_params or AgcParams()

  if not ihc_params:
    # HACK: these constant control the defaults
    one_cap = 1  # bool; 1 for Allen model, as text states we use
    just_hwr = 0  # book; 0 for normal/fancy IHC; 1 for HWR
    if just_hwr:
      ihc_params = IhcJustHwrParams()
    else:
      if one_cap:
        ihc_params = IhcOneCapParams()
      else:
        ihc_params = IhcTwoCapParams()

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

  max_channels_per_octave = int(
      math.log(2) / math.log(pole_freqs[1] / pole_freqs[2]))

  # Convert to include an ear_array, each w coeffs and state...
  car_coeffs = design_filters(car_params, fs, pole_freqs)
  agc_coeffs = design_agc(agc_params, fs, n_ch)
  ihc_coeffs = design_ihc(ihc_params, fs, n_ch)

  # Copy same designed coeffs into each ear (can do differently in the
  # future).
  ears = []
  for _ in range(n_ears):
    ears.append(CarfacCoeffs(car_coeffs, agc_coeffs, ihc_coeffs))

  cfp = CarfacParams(fs, max_channels_per_octave, car_params, agc_params,
                     ihc_params, n_ch, pole_freqs, ears, n_ears)
  return cfp


def carfac_init(cfp: CarfacParams):
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

  return cfp


def shift_right(s: np.float, amount: int):
  if amount > 0:
    return np.concatenate((s[0:amount, ...], s[:-amount, ...]), axis=0)
  elif amount < 0:
    return np.concatenate((s[-amount:, ...], np.flip(s[amount:, ...])), axis=0)
  else:
    return s


def spatial_smooth(coeffs: AgcCoeffs,
                   stage_state: AgcState):
  """Design the AGC spatial smoothing filter."""

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


def close_agc_loop(cfp: CarfacParams):
  """Fastest decimated rate determines interp needed."""
  decim1 = cfp.agc_params.decimation[0]

  for ear in range(cfp.n_ears):
    undamping = 1 - cfp.ears[ear].agc_state[0].agc_memory  # stage 1 result
    # Update the target stage gain for the new damping:
    new_g = stage_g(cfp.ears[ear].car_coeffs, undamping)
    # set the deltas needed to get to the new damping:
    cfp.ears[ear].car_state.dzb_memory = (
        (cfp.ears[ear].car_coeffs.zr_coeffs * undamping -
         cfp.ears[ear].car_state.zb_memory) / decim1)
    cfp.ears[ear].car_state.dg_memory = (
        new_g - cfp.ears[ear].car_state.g_memory) / decim1


def run_segment(cfp: CarfacParams,
                input_waves: np.ndarray,
                open_loop=0,
                linear_car=False) -> Tuple[np.ndarray, CarfacParams,
                                           np.ndarray, np.ndarray,
                                           np.ndarray]:
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
      [car_out, _] = car_step(
          input_waves[k, ear],
          cfp.ears[ear].car_coeffs,
          cfp.ears[ear].car_state,
          linear=linear_car)

      # update IHC state & output on every time step, too
      [ihc_out, _] = ihc_step(car_out, cfp.ears[ear].ihc_coeffs,
                              cfp.ears[ear].ihc_state)

      # run the AGC update step, decimating internally,
      [_, updated] = agc_step(ihc_out, cfp.ears[ear].agc_coeffs,
                              cfp.ears[ear].agc_state)

      # save some output data:
      naps[k, :, ear] = ihc_out  # output to neural activity pattern
      if do_bm:
        bm[k, :, ear] = car_out
        state = cfp.ears[ear].car_state
        seg_ohc[k, :, ear] = state.za_memory
        seg_agc[k, :, ear] = state.zb_memory

    #  connect the feedback from agc_state to car_state when it updates;
    #  all ears together here due to mixing across them:
    if updated:
      if n_ears > 1:
        #  do multi-aural cross-coupling:
        raise NotImplementedError
        # TODO(malcolmslaney) Translate Cross_Couple()
        # cfp.ears = Cross_Couple(cfp.ears)

    if not open_loop:
      close_agc_loop(cfp)
  return naps, cfp, bm, seg_ohc, seg_agc
