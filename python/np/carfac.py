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
from typing import Any, List, Optional, Tuple, Union

import numpy as np

# TODO(malcolmslaney): Convert names to proper Google format
# TODO(malcolmslaney): Get rid of bare generic warnings
# TODO(malcolmslaney): Figure out attribute lint errors
# TODO(malcolmslaney, dicklyon): Check out ???s in the documentation.

## CARFAC Design Functions

### CARFAC Parameter Structures
# pylint: disable=invalid-name  # Original Matlab names for initial version
# pylint: disable=g-bare-generic
# pytype: disable=attribute-error


@dataclasses.dataclass
class CF_CAR_param_struct:
  """All the parameters needed to define the CAR filters."""
  velocity_scale: float = 0.1  # for the velocity nonlinearity
  v_offset: float = 0.04  # offset gives a quadratic part
  min_zeta: float = 0.10  # minimum damping factor in mid-freq channels
  max_zeta: float = 0.35  # maximum damping factor in mid-freq channels
  first_pole_theta: float = 0.85 * math.pi
  zero_ratio: float = math.sqrt(2)  # how far zero is above pole
  high_f_damping_compression: float = 0.5  # 0 to 1 to compress zeta
  ERB_per_step: float = 0.5  # assume G&M's ERB formula
  min_pole_Hz: float = 30
  ERB_break_freq: float = 165.3  # Greenwood map's break freq.
  ERB_Q: float = 1000 / (24.7 * 4.37)  # Glasberg and Moore's high-cf ratio


@dataclasses.dataclass
class CF_AGC_param_struct:
  """All the parameters needed to define the behavior of the AGC filters."""
  n_stages: int = 4
  time_constants: np.ndarray = 0.002 * 4**np.arange(4)
  AGC_stage_gain: float = 2  # gain from each stage to next slower stage
  decimation: tuple = (8, 2, 2, 2)  # how often to update the AGC states
  AGC1_scales: list = 1.0 * np.sqrt(2)**np.arange(4)  # in units of channels
  AGC2_scales: list = 1.65 * math.sqrt(2)**np.arange(
      4)  # spread more toward base
  AGC_mix_coeff: float = 0.5


# The next three classes define three different types of inner-hair cell models.
# TODO(malcolmslaney) Perhaps make one superclass?
@dataclasses.dataclass
class CF_IHC_just_hwr_params_struct:
  just_hwr: bool = True  # just a simple HWR
  ac_corner_Hz: float = 20


@dataclasses.dataclass
class CF_IHC_one_cap_params_struct:
  just_hwr: bool = False  # not just a simple HWR
  one_cap: bool = True  # bool; 0 for new two-cap hack
  tau_lpf: float = 0.000080  # 80 microseconds smoothing twice
  tau_out: float = 0.0005  # depletion tau is pretty fast
  tau_in: float = 0.010  # recovery tau is slower
  ac_corner_Hz: float = 20


@dataclasses.dataclass
class CF_IHC_two_cap_params_struct:
  just_hwr: bool = False  # not just a simple HWR
  one_cap: bool = False  # bool; 0 for new two-cap hack
  tau_lpf: float = 0.000080  # 80 microseconds smoothing twice
  tau1_out: float = 0.010  # depletion tau is pretty fast
  tau1_in: float = 0.020  # recovery tau is slower
  tau2_out: float = 0.0025  # depletion tau is pretty fast
  tau2_in: float = 0.005  # recovery tau is slower
  ac_corner_Hz: float = 20


# See Section 18.3 (A Digital IHC Model)
def CARFAC_Detect(x: Union[float, np.ndarray]):
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


### Inner Hair Cell Design


@dataclasses.dataclass
class IHC_coeffs_struct:
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


@dataclasses.dataclass
class IHC_state_struct:
  # One channel state for testing/verification
  cap_voltage: float = 0
  cap1_voltage: float = 0
  cap2_voltage: float = 0
  lpf1_state: float = 0
  lpf2_state: float = 0
  ihc_accum: float = 0


## the IHC design coeffs:
def CARFAC_DesignIHC(IHC_params, fs, n_ch):
  """Design the inner hair cell implementation from parameters."""
  if IHC_params.just_hwr:
    IHC_coeffs = IHC_coeffs_struct(n_ch=n_ch, just_hwr=True)
  else:
    if IHC_params.one_cap:
      ro = 1 / CARFAC_Detect(10)  # output resistance at a very high level
      c = IHC_params.tau_out / ro
      ri = IHC_params.tau_in / c
      # to get steady-state average, double ro for 50# duty cycle
      saturation_output = 1 / (2 * ro + ri)
      # also consider the zero-signal equilibrium:
      r0 = 1 / CARFAC_Detect(0)
      current = 1 / (ri + r0)
      cap_voltage = 1 - current * ri
      IHC_coeffs = IHC_coeffs_struct(
          n_ch=n_ch,
          just_hwr=False,
          lpf_coeff=1 - math.exp(-1 / (IHC_params.tau_lpf * fs)),
          out_rate=ro / (IHC_params.tau_out * fs),
          in_rate=1 / (IHC_params.tau_in * fs),
          one_cap=IHC_params.one_cap,
          output_gain=1 / (saturation_output - current),
          rest_output=current / (saturation_output - current),
          rest_cap=cap_voltage)
    else:
      ro = 1 / CARFAC_Detect(10)  # output resistance at a very high level
      c2 = IHC_params.tau2_out / ro
      r2 = IHC_params.tau2_in / c2
      c1 = IHC_params.tau1_out / r2
      r1 = IHC_params.tau1_in / c1
      # to get steady-state average, double ro for 50# duty cycle
      saturation_output = 1 / (2 * ro + r2 + r1)
      # also consider the zero-signal equilibrium:
      r0 = 1 / CARFAC_Detect(0)
      current = 1 / (r1 + r2 + r0)
      cap1_voltage = 1 - current * r1
      cap2_voltage = cap1_voltage - current * r2
      IHC_coeffs = IHC_coeffs_struct(
          n_ch=n_ch,
          just_hwr=False,
          lpf_coeff=1 - math.exp(-1 / (IHC_params.tau_lpf * fs)),
          out1_rate=1 / (IHC_params.tau1_out * fs),
          in1_rate=1 / (IHC_params.tau1_in * fs),
          out2_rate=ro / (IHC_params.tau2_out * fs),
          in2_rate=1 / (IHC_params.tau2_in * fs),
          one_cap=IHC_params.one_cap,
          output_gain=1 / (saturation_output - current),
          rest_output=current / (saturation_output - current),
          rest_cap2=cap2_voltage,
          rest_cap1=cap1_voltage)
  # one more late addition that applies to all cases:
  IHC_coeffs.ac_coeff = 2 * math.pi * IHC_params.ac_corner_Hz / fs
  return IHC_coeffs


### AGC Design
def Design_FIR_coeffs(n_taps, delay_variance, mean_delay, n_iter):
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
    FIR = [a, 1 - a - b, b]
    OK = FIR[2 - 1] >= 0.25
  elif n_taps == 5:
    # based on solving to match [a/2, a/2, 1-a-b, b/2, b/2]:
    a = ((delay_variance + mean_delay * mean_delay) * 2 / 5 -
         mean_delay * 2 / 3) / 2
    b = ((delay_variance + mean_delay * mean_delay) * 2 / 5 +
         mean_delay * 2 / 3) / 2
    # first and last coeffs are implicitly duplicated to make 5-point FIR:
    FIR = [a / 2, 1 - a - b, b / 2]
    OK = FIR[2 - 1] >= 0.15
  else:
    raise ValueError('Bad n_taps (%d) in AGC_spatial_FIR' % n_taps)

  return FIR, OK


## the AGC design coeffs:


@dataclasses.dataclass
class AGC_coeffs_struct:
  n_ch: int
  n_AGC_stages: int
  AGC_stage_gain: float
  decimation: int = 0  # check this type
  AGC_spatial_iterations: int = 0
  AGC_spatial_FIR: Optional[list] = None  # Check this type
  AGC_spatial_n_taps: int = 0
  detect_scale: float = 1


def CARFAC_DesignAGC(AGC_params, fs, n_ch):
  """Design the AGC implementation from the parameters."""
  n_AGC_stages = AGC_params.n_stages

  # AGC1 pass is smoothing from base toward apex;
  # AGC2 pass is back, which is done first now (in double exp. version)
  AGC1_scales = AGC_params.AGC1_scales
  AGC2_scales = AGC_params.AGC2_scales

  decim = 1

  total_DC_gain = 0

  ##
  # Convert to vector of AGC coeffs
  AGC_coeffs = []
  for stage in range(n_AGC_stages):
    AGC_coeffs.append(
        AGC_coeffs_struct(n_ch, n_AGC_stages, AGC_params.AGC_stage_gain))

    AGC_coeffs[stage].decimation = AGC_params.decimation[stage]
    tau = AGC_params.time_constants[stage]
    # time constant in seconds
    decim = decim * AGC_params.decimation[stage]
    # net decim to this stage
    # epsilon is how much new input to take at each update step:
    AGC_coeffs[stage].AGC_epsilon = 1 - math.exp(-decim / (tau * fs))

    # effective number of smoothings in a time constant:
    ntimes = tau * (fs / decim)  # typically 5 to 50

    # decide on target spread (variance) and delay (mean) of impulse
    # response as a distribution to be convolved ntimes:
    # TODO(dicklyon): specify spread and delay instead of scales???
    delay = (AGC2_scales[stage] - AGC1_scales[stage]) / ntimes
    spread_sq = (AGC1_scales[stage]**2 + AGC2_scales[stage]**2) / ntimes

    # get pole positions to better match intended spread and delay of
    # [[geometric distribution]] in each direction (see wikipedia)
    u = 1 + 1 / spread_sq  # these are based on off-line algebra hacking.
    p = u - math.sqrt(u**2 - 1)  # pole that would give spread if used twice.
    dp = delay * (1 - 2 * p + p**2) / 2
    polez1 = p - dp
    polez2 = p + dp
    AGC_coeffs[stage].AGC_polez1 = polez1
    AGC_coeffs[stage].AGC_polez2 = polez2

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
        # to do other n_taps would need changes in CARFAC_Spatial_Smooth
        # and in Design_FIR_coeffs
        raise ValueError('Bad n_taps (%d) in CARFAC_DesignAGC' % n_taps)

      [AGC_spatial_FIR, done] = Design_FIR_coeffs(n_taps, spread_sq, delay,
                                                  n_iterations)

    # When done, store the resulting FIR design in coeffs:
    AGC_coeffs[stage].AGC_spatial_iterations = n_iterations
    AGC_coeffs[stage].AGC_spatial_FIR = AGC_spatial_FIR
    AGC_coeffs[stage].AGC_spatial_n_taps = n_taps

    # accumulate DC gains from all the stages, accounting for stage_gain:
    total_DC_gain = total_DC_gain + AGC_params.AGC_stage_gain**(stage)

    # TODO(dicklyon) -- is this the best binaural mixing plan?
    if stage == 0:
      AGC_coeffs[stage].AGC_mix_coeffs = 0
    else:
      AGC_coeffs[stage].AGC_mix_coeffs = AGC_params.AGC_mix_coeff / (
          tau * (fs / decim))

  # adjust stage 1 detect_scale to be the reciprocal DC gain of the AGC filters:
  AGC_coeffs[0].detect_scale = 1 / total_DC_gain

  return AGC_coeffs


def CARFAC_Stage_g(CAR_coeffs, relative_undamping):
  """Return the stage gain g needed to get unity gain at DC."""

  r1 = CAR_coeffs.r1_coeffs  # at max damping
  a0 = CAR_coeffs.a0_coeffs
  c0 = CAR_coeffs.c0_coeffs
  h = CAR_coeffs.h_coeffs
  zr = CAR_coeffs.zr_coeffs
  r = r1 + zr * relative_undamping
  g = (1 - 2 * r * a0 + r**2) / (1 - 2 * r * a0 + h * r * c0 + r**2)

  return g


# Test below after we design the stage filters

### CAR Design


def ERB_Hz(CF_Hz: Union[float, np.ndarray],
           ERB_break_freq: float = 1000 / 4.37,
           ERB_Q: float = 1000 / (24.7 * 4.37)):
  """Auditory filter nominal Equivalent Rectangular Bandwidth.

  Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
  ERB = 24.7 * (1 + 4.37 * CF_Hz / 1000)

  Args:
    CF_Hz: A scalar or a vector of frequencies (CF) to convert to ERB scale
    ERB_break_freq: The corner frequency where we go from linear to log
      bandwidth
    ERB_Q: The width of one filter (Q = CF/Bandwidth)

  Returns:
    A scalar or vector with the ERB scale for the input frequencies.
  """
  return (ERB_break_freq + CF_Hz) / ERB_Q


@dataclasses.dataclass
class CAR_coeffs_struct:
  n_ch: int
  velocity_scale: float
  v_offset: float

  r1_coeffs: np.float64 = None
  a0_coeffs: np.float64 = None
  c0_coeffs: np.float64 = None
  h_coeffs: np.float64 = None
  g0_coeffs: np.float64 = None


## Design the filter coeffs:
def CARFAC_DesignFilters(CAR_params, fs, pole_freqs):
  """Design the actual CAR filters."""

  n_ch = len(pole_freqs)
  pole_freqs = np.asarray(pole_freqs)

  # the filter design coeffs:
  # scalars first:
  CAR_coeffs = CAR_coeffs_struct(
      n_ch=n_ch,
      velocity_scale=CAR_params.velocity_scale,
      v_offset=CAR_params.v_offset)

  # don't really need these zero arrays, but it's a clue to what fields
  # and types are need in other language implementations:
  # CAR_coeffs.r1_coeffs = np.zeros(n_ch, 1)
  # CAR_coeffs.a0_coeffs = np.zeros(n_ch, 1)
  # CAR_coeffs.c0_coeffs = np.zeros(n_ch, 1)
  # CAR_coeffs.h_coeffs = np.zeros(n_ch, 1)

  # CAR_coeffs.g0_coeffs = np.zeros(n_ch, 1)

  # zero_ratio comes in via h.  In book's circuit D, zero_ratio is 1/sqrt(a),
  # and that a is here 1 / (1+f) where h = f*c.
  # solve for f:  1/zero_ratio^2 = 1 / (1+f)
  # zero_ratio^2 = 1+f => f = zero_ratio^2 - 1
  f = CAR_params.zero_ratio**2 - 1  # nominally 1 for half-octave

  # Make pole positions, s and c coeffs, h and g coeffs, etc.,
  # which mostly depend on the pole angle theta:
  theta = pole_freqs * (2 * math.pi / fs)

  c0 = np.sin(theta)
  a0 = np.cos(theta)

  # different possible interpretations for min-damping r:
  # r = exp(-theta * CF_CAR_params.min_zeta).
  # Compress theta to give somewhat higher Q at highest thetas:
  ff = CAR_params.high_f_damping_compression  # 0 to 1; typ. 0.5
  x = theta / math.pi

  zr_coeffs = math.pi * (x - ff * x**3)  # when ff is 0, this is just theta,
  #                       and when ff is 1 it goes to zero at theta = pi.
  max_zeta = CAR_params.max_zeta
  CAR_coeffs.r1_coeffs = (1 - zr_coeffs * max_zeta
                         )  # "r1" for the max-damping condition
  min_zeta = CAR_params.min_zeta
  # Increase the min damping where channels are spaced out more, by pulling
  # 25% of the way toward ERB_Hz/pole_freqs (close to 0.1 at high f)
  min_zetas = min_zeta + 0.25 * (
      ERB_Hz(pole_freqs, CAR_params.ERB_break_freq, CAR_params.ERB_Q) /
      pole_freqs - min_zeta)
  CAR_coeffs.zr_coeffs = zr_coeffs * (max_zeta - min_zetas)
  # how r relates to undamping

  # undamped coupled-form coefficients:
  CAR_coeffs.a0_coeffs = a0
  CAR_coeffs.c0_coeffs = c0

  # the zeros follow via the h_coeffs
  h = c0 * f
  CAR_coeffs.h_coeffs = h

  # for unity gain at min damping, radius r; only used in CARFAC_Init:
  relative_undamping = np.ones((n_ch,))  # max undamping to start
  # this function needs to take CAR_coeffs even if we haven't finished
  # constucting it by putting in the g0_coeffs:
  CAR_coeffs.g0_coeffs = CARFAC_Stage_g(CAR_coeffs, relative_undamping)

  return CAR_coeffs


### Overall CARFAC Design


@dataclasses.dataclass
class CF_struct:
  fs: float
  max_channels_per_octave: int
  CAR_params: CF_CAR_param_struct
  AGC_params: CF_AGC_param_struct
  IHC_params: Optional[IHC_coeffs_struct]
  n_ch: int
  pole_freqs: np.float64
  ears: list
  n_ears: int


@dataclasses.dataclass
class ear_struct:
  CAR_coeffs: CAR_coeffs_struct
  AGC_coeffs: List[AGC_coeffs_struct]
  IHC_coeffs: IHC_coeffs_struct

  CAR_state: Any = None  #  CAR_Init_State(CF.ears[ear].CAR_coeffs)
  IHC_state: Any = None  #  IHC_Init_State(CF.ears[ear].IHC_coeffs)
  AGC_state: Any = None  #  AGC_Init_State(CF.ears[ear].AGC_coeffs)


def CARFAC_Design(
    n_ears: int = 1,
    fs=22050,
    CF_CAR_params: Optional[CF_CAR_param_struct] = None,
    CF_AGC_params: Optional[CF_AGC_param_struct] = None,
    CF_IHC_params: Optional[Union[CF_IHC_just_hwr_params_struct,
                                  CF_IHC_one_cap_params_struct,
                                  CF_IHC_two_cap_params_struct]] = None):
  """This function designs the CARFAC filterbank.

  CARFAC is a Cascade of Asymmetric Resonators with Fast-Acting Compression);
  that is, it take bundles of parameters and computes all the filter
  coefficients needed to run it.

  See other functions for designing and characterizing the CARFAC:
    [naps, CF] = CARFAC_Run(CF, input_waves)
    transfns = CARFAC_Transfer_Functions(CF, to_channels, from_channels)

  Defaults to Glasberg & Moore's ERB curve:
    ERB_break_freq = 1000/4.37;  # 228.833
    ERB_Q = 1000/(24.7*4.37);    # 9.2645

  All args are defaultable; for sample/default args see the code; they
  make 96 channels at default fs = 22050, 114 channels at 44100.

  Args:
    n_ears: How many ears (1 or 2, in general) in the simulation
    fs: is sample rate (per second)
    CF_CAR_params: bundles all the pole-zero filter cascade parameters
    CF_AGC_params: bundles all the automatic gain control parameters
    CF_IHC_params: bundles all the inner hair cell parameters

  Returns:
    A Carfac filter structure (for running the calcs.)

  """

  CF_CAR_params = CF_CAR_params or CF_CAR_param_struct()
  CF_AGC_params = CF_AGC_params or CF_AGC_param_struct()

  if not CF_IHC_params:
    # HACK: these constant control the defaults
    one_cap = 1  # bool; 1 for Allen model, as text states we use
    just_hwr = 0  # book; 0 for normal/fancy IHC; 1 for HWR
    if just_hwr:
      CF_IHC_params = CF_IHC_just_hwr_params_struct()
    else:
      if one_cap:
        CF_IHC_params = CF_IHC_one_cap_params_struct()
      else:
        CF_IHC_params = CF_IHC_two_cap_params_struct()

  # first figure out how many filter stages (PZFC/CARFAC channels):
  pole_Hz = CF_CAR_params.first_pole_theta * fs / (2 * math.pi)
  n_ch = 0
  while pole_Hz > CF_CAR_params.min_pole_Hz:
    n_ch = n_ch + 1
    pole_Hz = pole_Hz - CF_CAR_params.ERB_per_step * ERB_Hz(
        pole_Hz, CF_CAR_params.ERB_break_freq, CF_CAR_params.ERB_Q)

  # Now we have n_ch, the number of channels, so can make the array
  # and compute all the frequencies again to put into it:
  pole_freqs = np.zeros((n_ch,), dtype=np.float32)  # float64 didn't help
  pole_Hz = CF_CAR_params.first_pole_theta * fs / (2 * math.pi)
  for ch in range(n_ch):
    pole_freqs[ch] = pole_Hz
    pole_Hz = pole_Hz - CF_CAR_params.ERB_per_step * ERB_Hz(
        pole_Hz, CF_CAR_params.ERB_break_freq, CF_CAR_params.ERB_Q)

  # Now we have n_ch, the number of channels, and pole_freqs array.

  max_channels_per_octave = int(
      math.log(2) / math.log(pole_freqs[1] / pole_freqs[2]))

  # Convert to include an ear_array, each w coeffs and state...
  CAR_coeffs = CARFAC_DesignFilters(CF_CAR_params, fs, pole_freqs)
  AGC_coeffs = CARFAC_DesignAGC(CF_AGC_params, fs, n_ch)
  IHC_coeffs = CARFAC_DesignIHC(CF_IHC_params, fs, n_ch)

  # Copy same designed coeffs into each ear (can do differently in the
  # future).
  ears = []
  for _ in range(n_ears):
    ears.append(ear_struct(CAR_coeffs, AGC_coeffs, IHC_coeffs))

  CF = CF_struct(fs, max_channels_per_octave, CF_CAR_params, CF_AGC_params,
                 CF_IHC_params, n_ch, pole_freqs, ears, n_ears)
  return CF


def CARFAC_Init(CF: CF_struct):
  """Initialize state for one or more ears of CF.

  This allocates and zeros all the state vector storage in the CF struct.
  Args:
    CF: the state structure for the filterbank

  Returns:
    A new version of the state structure with all initializations done.
  """

  n_ears = CF.n_ears

  for ear in range(n_ears):
    # for now there's only one coeffs, not one per ear
    CF.ears[ear].CAR_state = CAR_Init_State(CF.ears[ear].CAR_coeffs)
    CF.ears[ear].IHC_state = IHC_Init_State(CF.ears[ear].IHC_coeffs)
    CF.ears[ear].AGC_state = AGC_Init_State(CF.ears[ear].AGC_coeffs)

  return CF


@dataclasses.dataclass
class car_state_struct:
  """All the state variables for the CAR filterbank."""
  z1_memory: np.ndarray
  z2_memory: np.ndarray
  zA_memory: np.ndarray
  zB_memory: np.ndarray  # , coeffs.zr_coeffs, ...
  dzB_memory: np.ndarray  # (n_ch, 1), ...
  zY_memory: np.ndarray  # zeros(n_ch, 1), ...
  g_memory: np.ndarray  # coeffs.g0_coeffs, ...
  dg_memory: np.ndarray  #  zeros(n_ch, 1) ...

  def __init__(self, coeffs: CAR_coeffs_struct, dtype=np.float32):
    n_ch = coeffs.n_ch
    self.z1_memory = np.zeros((n_ch,), dtype=dtype)
    self.z2_memory = np.zeros((n_ch,), dtype=dtype)
    self.zA_memory = np.zeros((n_ch,), dtype=dtype)
    self.zB_memory = coeffs.zr_coeffs
    self.dzB_memory = np.zeros((n_ch,), dtype=dtype)
    self.zY_memory = np.zeros((n_ch,), dtype=dtype)
    self.g_memory = coeffs.g0_coeffs
    self.dg_memory = np.zeros((n_ch,), dtype=dtype)


def CAR_Init_State(coeffs):
  return car_state_struct(coeffs)


@dataclasses.dataclass
class agc_state_struct:
  """All the state variables for one stage of the AGC."""
  AGC_memory: np.float64
  input_accum: np.float64
  decim_phase: int = 0

  def __init__(self, coeffs: List[AGC_coeffs_struct]):
    n_ch = coeffs[0].n_ch

    self.AGC_memory = np.zeros((n_ch,))
    self.input_accum = np.zeros((n_ch,))


def AGC_Init_State(coeffs):
  n_AGC_stages = coeffs[0].n_AGC_stages
  state = []
  for _ in range(n_AGC_stages):
    state.append(agc_state_struct(coeffs))

  return state


@dataclasses.dataclass
class ihc_state_struct:
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


def IHC_Init_State(coeffs):
  return ihc_state_struct(coeffs)


def CARFAC_OHC_NLF(velocities, CAR_coeffs: CAR_coeffs_struct):
  #  function nlf = CARFAC_OHC_NLF(velocities, CAR_coeffs)
  # start with a quadratic nonlinear function, and limit it via a
  # rational function; make the result go to zero at high
  #  absolute velocities, so it will do nothing there.

  nlf = 1.0 / (
      1 + (velocities * CAR_coeffs.velocity_scale + CAR_coeffs.v_offset)**2)

  return nlf


def CARFAC_CAR_Step(x_in: float,
                    CAR_coeffs: CAR_coeffs_struct,
                    state: car_state_struct,
                    linear: bool = False):
  """One sample-time update step for the filter part of the CARFAC.

  Most of the update is parallel; finally we ripple inputs at the end.
  do the DOHC stuff:

  Args:
    x_in: the input audio
    CAR_coeffs: the implementation parameters for the filterbank
    state: The state of the filters before adding this one input sample
    linear: for testing, don't run through the outer hair cell model.

  Returns:
    The filterbank output vector and the new state variables for the filterbank.
  """

  g = state.g_memory + state.dg_memory  # interp g
  zB = state.zB_memory + state.dzB_memory  # AGC interpolation state
  # update the nonlinear function of "velocity", and zA (delay of z2):
  zA = state.zA_memory
  v = state.z2_memory - zA
  if linear:
    nlf = 1  # To allow testing
  else:
    nlf = CARFAC_OHC_NLF(v, CAR_coeffs)
  #  zB * nfl is "undamping" delta r:
  r = CAR_coeffs.r1_coeffs + zB * nlf
  zA = state.z2_memory

  #  now reduce state by r and rotate with the fixed cos/sin coeffs:
  z1 = r * (
      CAR_coeffs.a0_coeffs * state.z1_memory -
      CAR_coeffs.c0_coeffs * state.z2_memory)
  #  z1 = z1 + inputs
  z2 = r * (
      CAR_coeffs.c0_coeffs * state.z1_memory +
      CAR_coeffs.a0_coeffs * state.z2_memory)

  zY = CAR_coeffs.h_coeffs * z2  # partial output

  #  Ripple input-output path, instead of parallel, to avoid delay...
  # this is the only part that doesn't get computed "in parallel":
  in_out = x_in
  for ch in range(len(zY)):
    #  could do this here, or later in parallel:
    z1[ch] = z1[ch] + in_out
    # ripple, saving final channel outputs in zY
    in_out = g[ch] * (in_out + zY[ch])
    zY[ch] = in_out

  #  put new state back in place of old
  #  (z1 is a genuine temp; the others can update by reference in C)
  state.z1_memory = z1
  state.z2_memory = z2
  state.zA_memory = zA
  state.zB_memory = zB
  state.zY_memory = zY
  state.g_memory = g

  car_out = zY

  return car_out, state


def CARFAC_IHC_Step(filters_out: np.float64, coeffs: IHC_coeffs_struct,
                    state: ihc_state_struct):
  """Step the inner-hair cell model with ont input sample.

  One sample-time update of inner-hair-cell (IHC) model, including the
  detection nonlinearity and one or two capacitor state variables.

  Args:
    filters_out: The output from the CAR filterbank
    coeffs: The run-time parameters for the inner hair cells
    state: The run-time state

  Returns:
    The firing probability (??) for the hair cells in each channel
    and the new state.
  """

  # AC couple the filters_out, with 20 Hz corner
  ac_diff = filters_out - state.ac_coupler
  state.ac_coupler = state.ac_coupler + coeffs.ac_coeff * ac_diff

  if coeffs.just_hwr:
    ihc_out = np.min(2, np.max(0, ac_diff))
    #  limit it for stability
  else:
    conductance = CARFAC_Detect(ac_diff)  # rectifying nonlinearity

    if coeffs.one_cap:
      ihc_out = conductance * state.cap_voltage
      state.cap_voltage = (
          state.cap_voltage - ihc_out * coeffs.out_rate +
          (1 - state.cap_voltage) * coeffs.in_rate)
    else:
      # change to 2-cap version more like Meddis's:
      ihc_out = conductance * state.cap2_voltage
      state.cap1_voltage = (
          state.cap1_voltage -
          (state.cap1_voltage - state.cap2_voltage) * coeffs.out1_rate +
          (1 - state.cap1_voltage) * coeffs.in1_rate)

      state.cap2_voltage = (
          state.cap2_voltage - ihc_out * coeffs.out2_rate +
          (state.cap1_voltage - state.cap2_voltage) * coeffs.in2_rate)

    #  smooth it twice with LPF:
    ihc_out = ihc_out * coeffs.output_gain
    state.lpf1_state = (
        state.lpf1_state + coeffs.lpf_coeff * (ihc_out - state.lpf1_state))
    state.lpf2_state = (
        state.lpf2_state + coeffs.lpf_coeff *
        (state.lpf1_state - state.lpf2_state))
    ihc_out = state.lpf2_state - coeffs.rest_output

  # for where decimated output is useful
  state.ihc_accum = state.ihc_accum + ihc_out

  return ihc_out, state


def IHC_model_run(input_data, fs):
  """Design and run the inner hair cell model for some input audio."""
  CF = CARFAC_Design(fs=fs)
  CF = CARFAC_Init(CF)

  output = input_data * 0.0
  ihc_state = CF.ears[0].IHC_state
  for i in range(input_data.shape[0]):
    car_out = input_data[i]
    ihc_out, ihc_state = CARFAC_IHC_Step(car_out, CF.ears[0].IHC_coeffs,
                                         ihc_state)
    output[i] = ihc_out[0]
  return output


def shift_right(s: np.float, amount: int):
  if amount > 0:
    return np.concatenate((s[0:amount, ...], s[:-amount, ...]), axis=0)
  elif amount < 0:
    return np.concatenate((s[-amount:, ...], np.flip(s[amount:, ...])), axis=0)
  else:
    return s


def CARFAC_Spatial_Smooth(coeffs: AGC_coeffs_struct,
                          stage_state: agc_state_struct):
  """Design the AGC spatial smoothing filter."""

  n_iterations = coeffs.AGC_spatial_iterations

  use_FIR = n_iterations >= 0

  if use_FIR:
    FIR_coeffs = coeffs.AGC_spatial_FIR
    if coeffs.AGC_spatial_n_taps == 3:
      for _ in range(n_iterations):
        stage_state = (
            FIR_coeffs[0] * shift_right(stage_state, 1) +
            FIR_coeffs[1] * shift_right(stage_state, 0) +
            FIR_coeffs[2] * shift_right(stage_state, -1))

    #  5-tap smoother duplicates first and last coeffs
    elif coeffs.AGC_spatial_n_taps == 5:
      for _ in range(n_iterations):
        stage_state = (
            FIR_coeffs[0] *
            (shift_right(stage_state, 2) + shift_right(stage_state, 1)) +
            FIR_coeffs[1] * shift_right(stage_state, 0) + FIR_coeffs[2] *
            (shift_right(stage_state, -1) + shift_right(stage_state, -2)))
    else:
      raise ValueError('Bad AGC_spatial_n_taps (%d) in CARFAC_Spatial_Smooth' %
                       coeffs.AGC_spatial_n_taps)
  else:
    # use IIR method, back-and-forth first-order smoothers:
    raise NotImplementedError
    # TODO(malcolmslaney) Translate SmoothDoubleExponential()
    # stage_state = SmoothDoubleExponential(stage_state,
    #   coeffs.AGC_polez1[stage], coeffs.AGC_polez2[stage])

  return stage_state


def SmoothDoubleExponential(signal_vecs, polez1, polez2, fast_matlab_way):
  # Not sure why this was forgotten in the first conversion of Matlab to np..
  # but not needed for now.
  raise NotImplementedError


def CARFAC_AGC_Recurse(coeffs: List[AGC_coeffs_struct], AGC_in, stage: int,
                       state: List[agc_state_struct]):
  """Compute the AGC output for one stage, doing the recursion.

  Args:
    coeffs: the details of the AGC design
    AGC_in: the input data for this stage, a vector of channel values
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
  if len(state[stage].AGC_memory) != coeffs[stage].n_ch:
    raise ValueError('Width of AGC_memory (%d) and n_ch (%d) do not agree.' %
                     (len(state[stage].AGC_memory), coeffs[stage].n_ch))
  if len(AGC_in) != coeffs[stage].n_ch:
    raise ValueError('Width of AGC (%d) and n_ch (%d) do not agree.' %
                     (len(AGC_in), coeffs[stage].n_ch))
  assert len(AGC_in) == coeffs[stage].n_ch

  # decim factor for this stage, relative to input or prev. stage:
  decim = coeffs[stage].decimation

  #  decim phase of this stage (do work on phase 0 only):
  def mod(a: int, b: int):
    return a % b

  decim_phase = mod(state[stage].decim_phase + 1, decim)
  state[stage].decim_phase = decim_phase

  # accumulate input for this stage from detect or previous stage:
  state[stage].input_accum = state[stage].input_accum + AGC_in

  # nothing else to do if it's not the right decim_phase
  if decim_phase == 0:
    # Do lots of work, at decimated rate.
    # decimated inputs for this stage, and to be decimated more for next:
    AGC_in = state[stage].input_accum / decim
    state[stage].input_accum[:] = 0  # reset accumulator

    if stage < coeffs[0].n_AGC_stages - 1:
      state, updated = CARFAC_AGC_Recurse(coeffs, AGC_in, stage + 1, state)
      # and add its output to this stage input, whether it updated or not:
      AGC_in = AGC_in + coeffs[stage].AGC_stage_gain * state[stage +
                                                             1].AGC_memory

    AGC_stage_state = state[stage].AGC_memory
    # first-order recursive smoothing filter update, in time:
    AGC_stage_state = AGC_stage_state + coeffs[stage].AGC_epsilon * (
        AGC_in - AGC_stage_state)
    # spatial smooth:
    AGC_stage_state = CARFAC_Spatial_Smooth(coeffs[stage], AGC_stage_state)
    # and store the state back (in C++, do it all in place?)
    state[stage].AGC_memory = AGC_stage_state

    updated = 1  # bool to say we have new state
  else:
    updated = 0

  return state, updated


def CARFAC_AGC_Step(detects, coeffs: List[AGC_coeffs_struct],
                    state: List[agc_state_struct]):
  #  function [state, updated] = CARFAC_AGC_Step(detects, coeffs, state)
  #
  # one time step of the AGC state update; decimates internally

  stage = 0
  AGC_in = coeffs[0].detect_scale * detects
  return CARFAC_AGC_Recurse(coeffs, AGC_in, stage, state)


def CARFAC_Close_AGC_Loop(CF: CF_struct):
  """Fastest decimated rate determines interp needed."""
  decim1 = CF.AGC_params.decimation[0]

  for ear in range(CF.n_ears):
    undamping = 1 - CF.ears[ear].AGC_state[0].AGC_memory  # stage 1 result
    # Update the target stage gain for the new damping:
    new_g = CARFAC_Stage_g(CF.ears[ear].CAR_coeffs, undamping)
    # set the deltas needed to get to the new damping:
    CF.ears[ear].CAR_state.dzB_memory = (
        (CF.ears[ear].CAR_coeffs.zr_coeffs * undamping -
         CF.ears[ear].CAR_state.zB_memory) / decim1)
    CF.ears[ear].CAR_state.dg_memory = (
        new_g - CF.ears[ear].CAR_state.g_memory) / decim1


def CARFAC_Run_Segment(CF: CF_struct,
                       input_waves: np.ndarray,
                       open_loop=0,
                       linear_car=False) -> Tuple[np.ndarray, CF_struct,
                                                  np.ndarray, np.ndarray,
                                                  np.ndarray]:
  """This function runs the entire CARFAC model.

  That is, filters a 1 or more channel
  sound input segment to make one or more neural activity patterns (naps)
  it can be called multiple times for successive segments of any length,
  as long as the returned CF with modified state is passed back in each
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
    CF: a structure that descirbes everything we know about this CARFAC.
    input_waves: the audio input
    open_loop: whether to run CARFAC without the feedback.
    linear_car (new over Matlab): use CAR filters without OHC effects.

  Returns:
    naps: neural activity pattern
    CF: the new structure describing the state of the CARFAC
    BM: The basilar membrane motion
    seg_ohc & seg_agc are optional extra outputs useful for seeing what the
      ohc nonlinearity and agc are doing; both in terms of extra damping.
  """

  do_BM = 1

  if len(input_waves.shape) < 2:
    input_waves = np.reshape(input_waves, (-1, 1))
  [n_samp, n_ears] = input_waves.shape

  if n_ears != CF.n_ears:
    raise ValueError(
        'Bad number of input_waves channels (%d vs %d) passed to CARFAC_Run' %
        (n_ears, CF.n_ears))

  n_ch = CF.n_ch
  naps = np.zeros((n_samp, n_ch, n_ears))  # allocate space for result
  if do_BM:
    BM = np.zeros((n_samp, n_ch, n_ears))
    seg_ohc = np.zeros((n_samp, n_ch, n_ears))
    seg_agc = np.zeros((n_samp, n_ch, n_ears))

  # A 2022 addition to make open-loop running behave:
  if open_loop:
    # zero the deltas:
    for ear in range(CF.n_ears):
      CF.ears[ear].CAR_state.dzB_memory *= 0
      CF.ears[ear].CAR_state.dg_memory *= 0

  for k in range(n_samp):
    # at each time step, possibly handle multiple channels
    for ear in range(n_ears):

      # This would be cleaner if we could just get and use a reference to
      # CF.ears(ear), but Matlab doesn't work that way...
      [car_out, _] = CARFAC_CAR_Step(
          input_waves[k, ear],
          CF.ears[ear].CAR_coeffs,
          CF.ears[ear].CAR_state,
          linear=linear_car)

      # update IHC state & output on every time step, too
      [ihc_out, _] = CARFAC_IHC_Step(car_out, CF.ears[ear].IHC_coeffs,
                                     CF.ears[ear].IHC_state)

      # run the AGC update step, decimating internally,
      [_, updated] = CARFAC_AGC_Step(ihc_out, CF.ears[ear].AGC_coeffs,
                                     CF.ears[ear].AGC_state)

      # save some output data:
      naps[k, :, ear] = ihc_out  # output to neural activity pattern
      if do_BM:
        BM[k, :, ear] = car_out
        state = CF.ears[ear].CAR_state
        seg_ohc[k, :, ear] = state.zA_memory
        seg_agc[k, :, ear] = state.zB_memory

    #  connect the feedback from AGC_state to CAR_state when it updates;
    #  all ears together here due to mixing across them:
    if updated:
      if n_ears > 1:
        #  do multi-aural cross-coupling:
        raise NotImplementedError
        # TODO(malcolmslaney) Translate CARFAC_Cross_Couple()
        # CF.ears = CARFAC_Cross_Couple(CF.ears)

    if not open_loop:
      CARFAC_Close_AGC_Loop(CF)
  return naps, CF, BM, seg_ohc, seg_agc
