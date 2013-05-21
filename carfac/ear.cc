//
//  ear.cc
//  CARFAC Open Source C++ Library
//
//  Created by Alex Brandmeyer on 5/10/13.
//
// This C++ file is part of an implementation of Lyon's cochlear model:
// "Cascade of Asymmetric Resonators with Fast-Acting Compression"
// to supplement Lyon's upcoming book "Human and Machine Hearing"
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ear.h"

// The 'InitEar' function takes a set of model parameters and initializes the
// design coefficients and model state variables needed for running the model
// on a single audio channel. 
void Ear::InitEar(int n_ch, int32_t fs, FloatArray pole_freqs,
                  CARParams car_p, IHCParams ihc_p, AGCParams agc_p) {
  // The first section of code determines the number of channels that will be
  // used in the model on the basis of the sample rate and the CAR parameters
  // that have been passed to this function.
  n_ch_ = n_ch;
  // These functions use the parameters that have been passed to design the
  // coefficients for the three stages of the model.
  DesignFilters(car_p, fs, pole_freqs);
  DesignIHC(ihc_p, fs);
  DesignAGC(agc_p, fs);
  // Once the coefficients have been determined, we can initialize the state
  // variables that will be used during runtime.
  InitCARState();
  InitIHCState();
  InitAGCState();
}

void Ear::DesignFilters(CARParams car_params, int32_t fs,
                              FloatArray pole_freqs) {
  car_coeffs_.velocity_scale_ = car_params.velocity_scale_;
  car_coeffs_.v_offset_ = car_params.v_offset_;
  car_coeffs_.r1_coeffs_.resize(n_ch_);
  car_coeffs_.a0_coeffs_.resize(n_ch_);
  car_coeffs_.c0_coeffs_.resize(n_ch_);
  car_coeffs_.h_coeffs_.resize(n_ch_);
  car_coeffs_.g0_coeffs_.resize(n_ch_);
  FPType f = car_params.zero_ratio_ * car_params.zero_ratio_ - 1;
  FloatArray theta = pole_freqs * ((2 * PI) / fs);
  car_coeffs_.c0_coeffs_ = sin(theta);
  car_coeffs_.a0_coeffs_ = cos(theta);
  FPType ff = car_params.high_f_damping_compression_;
  FloatArray x = theta / PI;
  car_coeffs_.zr_coeffs_ = PI * (x - (ff * (x * x * x)));
  FPType max_zeta = car_params.max_zeta_;
  FPType min_zeta = car_params.min_zeta_;
  car_coeffs_.r1_coeffs_ = (1 - (car_coeffs_.zr_coeffs_ * max_zeta));
  FPType curfreq;
  FloatArray erb_freqs(n_ch_);
  for (int ch=0; ch < n_ch_; ch++) {
    curfreq = pole_freqs(ch);
    erb_freqs(ch) = ERBHz(curfreq, car_params.erb_break_freq_,
                          car_params.erb_q_);
  }
  FloatArray min_zetas = min_zeta + (0.25 * ((erb_freqs / pole_freqs) - min_zeta));
  car_coeffs_.zr_coeffs_ *= max_zeta - min_zetas;
  car_coeffs_.h_coeffs_ = car_coeffs_.c0_coeffs_ * f;
  FloatArray relative_undamping = FloatArray::Ones(n_ch_);
  FloatArray r = car_coeffs_.r1_coeffs_ +
                 (car_coeffs_.zr_coeffs_ * relative_undamping);
  car_coeffs_.g0_coeffs_ = (1 - (2 * r * car_coeffs_.a0_coeffs_) + (r * r)) /
                          (1 - (2 * r * car_coeffs_.a0_coeffs_) +
                          (car_coeffs_.h_coeffs_ * r * car_coeffs_.c0_coeffs_) +
                          (r * r));
}


void Ear::DesignAGC(AGCParams agc_params, int32_t fs) {
  // These data members could probably be initialized locally within the design
  // function.
  agc_coeffs_.n_agc_stages_ = agc_params.n_stages_;
  agc_coeffs_.agc_stage_gain_ = agc_params.agc_stage_gain_;
  FloatArray agc1_scales = agc_params.agc1_scales_;
  FloatArray agc2_scales = agc_params.agc2_scales_;
  FloatArray time_constants = agc_params.time_constants_;
  agc_coeffs_.agc_epsilon_.resize(agc_coeffs_.n_agc_stages_);
  agc_coeffs_.agc_pole_z1_.resize(agc_coeffs_.n_agc_stages_);
  agc_coeffs_.agc_pole_z2_.resize(agc_coeffs_.n_agc_stages_);
  agc_coeffs_.agc_spatial_iterations_.resize(agc_coeffs_.n_agc_stages_);
  agc_coeffs_.agc_spatial_n_taps_.resize(agc_coeffs_.n_agc_stages_);
  agc_coeffs_.agc_spatial_fir_.resize(3,agc_coeffs_.n_agc_stages_);
  agc_coeffs_.agc_mix_coeffs_.resize(agc_coeffs_.n_agc_stages_);
  FPType mix_coeff = agc_params.agc_mix_coeff_;
  int decim = 1;
  agc_coeffs_.decimation_ = agc_params.decimation_;
  FPType total_dc_gain = 0.0;
  // Here we loop through each of the stages of the AGC.
  for (int stage=0; stage < agc_coeffs_.n_agc_stages_; stage++) {
    FPType tau = time_constants(stage);
    decim *= agc_coeffs_.decimation_.at(stage);
    agc_coeffs_.agc_epsilon_(stage) = 1 - exp((-1 * decim) / (tau * fs));
    FPType n_times = tau * (fs / decim);
    FPType delay = (agc2_scales(stage) - agc1_scales(stage)) / n_times;
    FPType spread_sq = (pow(agc1_scales(stage), 2) +
                        pow(agc2_scales(stage), 2)) / n_times;
    FPType u = 1 + (1 / spread_sq);
    FPType p = u - sqrt(pow(u, 2) - 1);
    FPType dp = delay * (1 - (2 * p) + (p * p)) / 2;
    FPType pole_z1 = p - dp;
    FPType pole_z2 = p + dp;
    agc_coeffs_.agc_pole_z1_(stage) = pole_z1;
    agc_coeffs_.agc_pole_z2_(stage) = pole_z2;
    int n_taps = 0;
    bool fir_ok = false;
    int n_iterations = 1;
    // This section initializes the FIR coeffs settings at each stage.
    FloatArray fir(3);
    while (! fir_ok) {
      switch (n_taps) {
        case 0:
          n_taps = 3;
          break;
        case 3:
          n_taps = 5;
          break;
        case 5:
          n_iterations ++;
          if (n_iterations > 16){
            // This implies too many iterations, so we shoud indicate and error.
            // TODO alexbrandmeyer, proper Google error handling.
          }
          break;
        default:
          // This means a bad n_taps has been provided, so there should again be
          // an error.
          // TODO alexbrandmeyer, proper Google error handling.
          break;
      }
      // Now we can design the FIR coefficients.
      FPType var = spread_sq / n_iterations;
      FPType mn = delay / n_iterations;
      FPType a, b;
      switch (n_taps) {
        case 3:
          a = (var + pow(mn, 2) - mn) / 2;
          b = (var + pow(mn, 2) + mn) / 2;
          fir(0) = a;
          fir(1) = 1 - a - b;
          fir(2) = b;
          fir_ok = (fir(2) >= 0.2) ? true : false;
          break;
        case 5:
          a = (((var + pow(mn, 2)) * 2/5) - (mn * 2/3)) / 2;
          b = (((var + pow(mn, 2)) * 2/5) + (mn * 2/3)) / 2;
          fir(0) = a / 2;
          fir(1) = 1 - a - b;
          fir(2) = b / 2;
          fir_ok = (fir(2) >= 0.1) ? true : false;
          break;
        default:
          break;  // Again, we've arrived at a bad n_taps in the design.
          // TODO alexbrandmeyer. Add proper Google error handling.
      }
    }
    // Once we have the FIR design for this stage we can assign it to the
    // appropriate data members.
    agc_coeffs_.agc_spatial_iterations_(stage) = n_iterations;
    agc_coeffs_.agc_spatial_n_taps_(stage) = n_taps;
    // TODO alexbrandmeyer replace using Eigen block method
    agc_coeffs_.agc_spatial_fir_(0, stage) = fir(0);
    agc_coeffs_.agc_spatial_fir_(1, stage) = fir(1);
    agc_coeffs_.agc_spatial_fir_(2, stage) = fir(2);
    total_dc_gain += pow(agc_coeffs_.agc_stage_gain_, (stage));
    agc_coeffs_.agc_mix_coeffs_(stage) =
      (stage == 0) ? 0 : mix_coeff / (tau * (fs /decim));
  }
  agc_coeffs_.agc_gain_ = total_dc_gain;
  agc_coeffs_.detect_scale_ = 1 / total_dc_gain;
}

void Ear::DesignIHC(IHCParams ihc_params, int32_t fs) {
  if (ihc_params.just_hwr_) {
    ihc_coeffs_.just_hwr_ = ihc_params.just_hwr_;
  } else {
    // This section calculates conductance values using two pre-defined scalars.
    FloatArray x(1);
    FPType conduct_at_10, conduct_at_0;
    x(0) = 10.0;
    x = CARFACDetect(x);
    conduct_at_10 = x(0);
    x(0) = 0.0;
    x = CARFACDetect(x);
    conduct_at_0 = x(0);
    if (ihc_params.one_cap_) {
      FPType ro = 1 / conduct_at_10 ;
      FPType c = ihc_params.tau1_out_ / ro;
      FPType ri = ihc_params.tau1_in_ / c;
      FPType saturation_output = 1 / ((2 * ro) + ri);
      FPType r0 = 1 / conduct_at_0;
      FPType current = 1 / (ri + r0);
      ihc_coeffs_.cap1_voltage_ = 1 - (current * ri);
      ihc_coeffs_.just_hwr_ = false;
      ihc_coeffs_.lpf_coeff_ = 1 - exp( -1 / (ihc_params.tau_lpf_ * fs));
      ihc_coeffs_.out1_rate_ = ro / (ihc_params.tau1_out_ * fs);
      ihc_coeffs_.in1_rate_ = 1 / (ihc_params.tau1_in_ * fs);
      ihc_coeffs_.one_cap_ = ihc_params.one_cap_;
      ihc_coeffs_.output_gain_ = 1 / (saturation_output - current);
      ihc_coeffs_.rest_output_ = current / (saturation_output - current);
      ihc_coeffs_.rest_cap1_ = ihc_coeffs_.cap1_voltage_;
    } else {
      FPType ro = 1 / conduct_at_10;
      FPType c2 = ihc_params.tau2_out_ / ro;
      FPType r2 = ihc_params.tau2_in_ / c2;
      FPType c1 = ihc_params.tau1_out_ / r2;
      FPType r1 = ihc_params.tau1_in_ / c1;
      FPType saturation_output = 1 / (2 * ro + r2 + r1);
      FPType r0 = 1 / conduct_at_0;
      FPType current = 1 / (r1 + r2 + r0);
      ihc_coeffs_.cap1_voltage_ = 1 - (current * r1);
      ihc_coeffs_.cap2_voltage_ = ihc_coeffs_.cap1_voltage_ - (current * r2);
      ihc_coeffs_.just_hwr_ = false;
      ihc_coeffs_.lpf_coeff_ = 1 - exp(-1 / (ihc_params.tau_lpf_ * fs));
      ihc_coeffs_.out1_rate_ = 1 / (ihc_params.tau1_out_ * fs);
      ihc_coeffs_.in1_rate_ = 1 / (ihc_params.tau1_in_ * fs);
      ihc_coeffs_.out2_rate_ = ro / (ihc_params.tau2_out_ * fs);
      ihc_coeffs_.in2_rate_ = 1 / (ihc_params.tau2_in_ * fs);
      ihc_coeffs_.one_cap_ = ihc_params.one_cap_;
      ihc_coeffs_.output_gain_ = 1 / (saturation_output - current);
      ihc_coeffs_.rest_output_ = current / (saturation_output - current);
      ihc_coeffs_.rest_cap1_ = ihc_coeffs_.cap1_voltage_;
      ihc_coeffs_.rest_cap2_ = ihc_coeffs_.cap2_voltage_;
    }
  }
  ihc_coeffs_.ac_coeff_ = 2 * PI * ihc_params.ac_corner_hz_ / fs;
}

void Ear::InitCARState() {
  car_state_.z1_memory_ = FloatArray::Zero(n_ch_);
  car_state_.z2_memory_ = FloatArray::Zero(n_ch_);
  car_state_.za_memory_ = FloatArray::Zero(n_ch_);
  car_state_.zb_memory_ = car_coeffs_.zr_coeffs_;
  car_state_.dzb_memory_ = FloatArray::Zero(n_ch_);
  car_state_.zy_memory_ = FloatArray::Zero(n_ch_);
  car_state_.g_memory_ = car_coeffs_.g0_coeffs_;
  car_state_.dg_memory_ = FloatArray::Zero(n_ch_);
}

void Ear::InitIHCState() {
  ihc_state_.ihc_accum_ = FloatArray::Zero(n_ch_);
  if (! ihc_coeffs_.just_hwr_) {
    ihc_state_.ac_coupler_ = FloatArray::Zero(n_ch_);
    ihc_state_.lpf1_state_ = ihc_coeffs_.rest_output_ * FloatArray::Ones(n_ch_);
    ihc_state_.lpf2_state_ = ihc_coeffs_.rest_output_ * FloatArray::Ones(n_ch_);
    if (ihc_coeffs_.one_cap_) {
      ihc_state_.cap1_voltage_ = ihc_coeffs_.rest_cap1_ *
                                 FloatArray::Ones(n_ch_);
    } else {
      ihc_state_.cap1_voltage_ = ihc_coeffs_.rest_cap1_ *
                                 FloatArray::Ones(n_ch_);
      ihc_state_.cap2_voltage_ = ihc_coeffs_.rest_cap2_ *
                                 FloatArray::Ones(n_ch_);
    }
  }
}

void Ear::InitAGCState() {
  agc_state_.n_agc_stages_ = agc_coeffs_.n_agc_stages_;
  agc_state_.agc_memory_.resize(agc_state_.n_agc_stages_);
  agc_state_.input_accum_.resize(agc_state_.n_agc_stages_);
  agc_state_.decim_phase_.resize(agc_state_.n_agc_stages_);
  for (int i = 0; i < agc_state_.n_agc_stages_; i++) {
    agc_state_.decim_phase_.at(i) = 0;
    agc_state_.agc_memory_.at(i) = FloatArray::Zero(n_ch_);
    agc_state_.input_accum_.at(i) = FloatArray::Zero(n_ch_);
  }
}

FloatArray Ear::CARStep(FPType input) {
  FloatArray g(n_ch_);
  FloatArray zb(n_ch_);
  FloatArray za(n_ch_);
  FloatArray v(n_ch_);
  FloatArray nlf(n_ch_);
  FloatArray r(n_ch_);
  FloatArray z1(n_ch_);
  FloatArray z2(n_ch_);
  FloatArray zy(n_ch_);
  FPType in_out;
  // This performs the DOHC stuff.
  g = car_state_.g_memory_ + car_state_.dg_memory_;  // This interpolates g.
  // This calculates the AGC interpolation state.
  zb = car_state_.zb_memory_ + car_state_.dzb_memory_;
  // This updates the nonlinear function of 'velocity' along with zA, which is
  // a delay of z2.
  za = car_state_.za_memory_;
  v = car_state_.z2_memory_ - za;
  nlf = OHC_NLF(v);
  // Here, zb * nfl is "undamping" delta r.
  r = car_coeffs_.r1_coeffs_ + (zb * nlf);
  za = car_state_.z2_memory_;
  // Here we reduce the CAR state by r and rotate with the fixed cos/sin coeffs.
  z1 = r * ((car_coeffs_.a0_coeffs_ * car_state_.z1_memory_) -
            (car_coeffs_.c0_coeffs_ * car_state_.z2_memory_));
  z2 = r * ((car_coeffs_.c0_coeffs_ * car_state_.z1_memory_) +
            (car_coeffs_.a0_coeffs_ * car_state_.z2_memory_));
  zy = car_coeffs_.h_coeffs_ * z2;
  // This section ripples the input-output path, to avoid delay...
  // It's the only part that doesn't get computed "in parallel":
  in_out = input;
  for (int ch = 0; ch < n_ch_; ch++) {
    z1(ch) = z1(ch) + in_out;
    // This performs the ripple, and saves the final channel outputs in zy.
    in_out = g(ch) * (in_out + zy(ch));
    zy(ch) = in_out;
  }
  car_state_.z1_memory_ = z1;
  car_state_.z2_memory_ = z2;
  car_state_.za_memory_ = za;
  car_state_.zb_memory_ = zb;
  car_state_.zy_memory_ = zy;
  car_state_.g_memory_ = g;
  // The output of the CAR is equal to the zy state.
  return zy;
}

// We start with a quadratic nonlinear function, and limit it via a
// rational function. This makes the result go to zero at high
// absolute velocities, so it will do nothing there.
FloatArray Ear::OHC_NLF(FloatArray velocities) {
  return (1 + ((velocities * car_coeffs_.velocity_scale_)
               + car_coeffs_.v_offset_).square()).inverse();
}

// This step is a one sample-time update of the inner-hair-cell (IHC) model,
// including the detection nonlinearity and either one or two capacitor state
// variables.
FloatArray Ear::IHCStep(FloatArray car_out) {
  FloatArray ihc_out(n_ch_);
  FloatArray ac_diff(n_ch_);
  FloatArray conductance(n_ch_);
  ac_diff = car_out - ihc_state_.ac_coupler_;
  ihc_state_.ac_coupler_ = ihc_state_.ac_coupler_ +
                           (ihc_coeffs_.ac_coeff_ * ac_diff);
  if (ihc_coeffs_.just_hwr_) {
    for (int ch = 0; ch < n_ch_; ch++) {
      FPType a;
      a = (ac_diff(ch) > 0.0) ? ac_diff(ch) : 0.0;
      ihc_out(ch) = (a < 2) ? a : 2;
    }
  } else {
    conductance = CARFACDetect(ac_diff);
    if (ihc_coeffs_.one_cap_) {
      ihc_out = conductance * ihc_state_.cap1_voltage_;
      ihc_state_.cap1_voltage_ = ihc_state_.cap1_voltage_ -
                                 (ihc_out * ihc_coeffs_.out1_rate_) +
                                 ((1 - ihc_state_.cap1_voltage_)
                                  * ihc_coeffs_.in1_rate_);
    } else {
      ihc_out = conductance * ihc_state_.cap2_voltage_;
      ihc_state_.cap1_voltage_ = ihc_state_.cap1_voltage_ -
                      ((ihc_state_.cap1_voltage_ - ihc_state_.cap2_voltage_)
                      * ihc_coeffs_.out1_rate_) +
                      ((1 - ihc_state_.cap1_voltage_) * ihc_coeffs_.in1_rate_);
      ihc_state_.cap2_voltage_ = ihc_state_.cap2_voltage_ -
                          (ihc_out * ihc_coeffs_.out2_rate_) +
                          ((ihc_state_.cap1_voltage_ - ihc_state_.cap2_voltage_)
                          * ihc_coeffs_.in2_rate_);
    }
    // Here we smooth the output twice using a LPF.
    ihc_out *= ihc_coeffs_.output_gain_;
    ihc_state_.lpf1_state_ = ihc_state_.lpf1_state_ +
                  (ihc_coeffs_.lpf_coeff_ * (ihc_out - ihc_state_.lpf1_state_));
    ihc_state_.lpf2_state_ = ihc_state_.lpf2_state_ +
                  (ihc_coeffs_.lpf_coeff_ *
                   (ihc_state_.lpf1_state_ - ihc_state_.lpf2_state_));
    ihc_out = ihc_state_.lpf2_state_ - ihc_coeffs_.rest_output_;
  }
  ihc_state_.ihc_accum_ += ihc_out;
  return ihc_out;
}

bool Ear::AGCStep(FloatArray ihc_out) {
  int stage = 0;
  FloatArray agc_in(n_ch_);
  agc_in = agc_coeffs_.detect_scale_ * ihc_out;
  bool updated = AGCRecurse(stage, agc_in);
  return updated;
}

bool Ear::AGCRecurse(int stage, FloatArray agc_in) {
  bool updated = true;
  // This is the decim factor for this stage, relative to input or prev. stage:
  int decim = agc_coeffs_.decimation_.at(stage);
  // This is the decim phase of this stage (do work on phase 0 only):
  int decim_phase = agc_state_.decim_phase_.at(stage);
  decim_phase = decim_phase % decim;
  agc_state_.decim_phase_.at(stage) = decim_phase;
  // Here we accumulate input for this stage from the previous stage:
  agc_state_.input_accum_.at(stage) += agc_in;
  // We don't do anything if it's not the right decim_phase.
  if (decim_phase == 0) {
    // Now we do lots of work, at the decimated rate.
    // These are the decimated inputs for this stage, which will be further
    // decimated at the next stage.
    agc_in = agc_state_.input_accum_.at(stage) / decim;
    // This resets the accumulator.
    agc_state_.input_accum_.at(stage) = FloatArray::Zero(n_ch_);
    if (stage < (agc_coeffs_.decimation_.size() - 1)) {
      // Now we recurse to evaluate the next stage(s).
      updated = AGCRecurse(stage + 1, agc_in);
      // Afterwards we add its output to this stage input, whether it updated or
      // not.
      agc_in += (agc_coeffs_.agc_stage_gain_ *
                         agc_state_.agc_memory_.at(stage + 1));
    }
    FloatArray agc_stage_state = agc_state_.agc_memory_.at(stage);
    // This performs a first-order recursive smoothing filter update, in time.
    agc_stage_state = agc_stage_state + (agc_coeffs_.agc_epsilon_(stage) *
                                         (agc_in - agc_stage_state));
    agc_stage_state = AGCSpatialSmooth(stage, agc_stage_state);
    agc_state_.agc_memory_.at(stage) = agc_stage_state;
  } else {
    updated = false;
  }
  return updated;
}

FloatArray Ear::AGCSpatialSmooth(int stage, FloatArray stage_state) {
  int n_iterations = agc_coeffs_.agc_spatial_iterations_(stage);
  bool use_fir;
  use_fir = (n_iterations < 4) ? true : false;
  if (use_fir) {
    FloatArray fir_coeffs = agc_coeffs_.agc_spatial_fir_.block(0, stage, 3, 1);
    FloatArray ss_tap1(n_ch_);
    FloatArray ss_tap2(n_ch_);
    FloatArray ss_tap3(n_ch_);
    FloatArray ss_tap4(n_ch_);
    int n_taps = agc_coeffs_.agc_spatial_n_taps_(stage);
    // This initializes the first two taps of stage state, which are used for
    // both possible cases.
    ss_tap1(0) = stage_state(0);
    ss_tap1.block(1, 0, n_ch_ - 1, 1) = stage_state.block(0, 0, n_ch_ - 1, 1);
    ss_tap2(n_ch_ - 1) = stage_state(n_ch_ - 1);
    ss_tap2.block(0, 0, n_ch_ - 1, 1) = stage_state.block(1, 0, n_ch_ - 1, 1);
    switch (n_taps) {
      case 3:
        stage_state = (fir_coeffs(0) * ss_tap1) +
                      (fir_coeffs(1) * stage_state) +
                      (fir_coeffs(2) * ss_tap2);
        break;
      case 5:
        // Now we initialize last two taps of stage state, which are only used
        // for the 5-tap case.
        ss_tap3(0) = stage_state(0);
        ss_tap3(1) = stage_state(1);
        ss_tap3.block(2, 0, n_ch_ - 2, 1) =
          stage_state.block(0, 0, n_ch_ - 2, 1);
        ss_tap4(n_ch_ - 2) = stage_state(n_ch_ - 1);
        ss_tap4(n_ch_ - 1) = stage_state(n_ch_ - 2);
        ss_tap4.block(0, 0, n_ch_ - 2, 1) =
          stage_state.block(2, 0, n_ch_ - 2, 1);
        stage_state = (fir_coeffs(0) * (ss_tap3 + ss_tap1)) +
                      (fir_coeffs(1) * stage_state) +
                      (fir_coeffs(2) * (ss_tap2 + ss_tap4));
        break;
      default:
        break;
        // TODO alexbrandmeyer: determine proper error handling implementation.
    }
  } else {
    stage_state = AGCSmoothDoubleExponential(stage_state,
                                             agc_coeffs_.agc_pole_z1_(stage),
                                             agc_coeffs_.agc_pole_z2_(stage));
  }
  return stage_state;
}

FloatArray Ear::AGCSmoothDoubleExponential(FloatArray stage_state,
                                           FPType pole_z1, FPType pole_z2) {
  int32_t n_pts = stage_state.size();
  FPType input;
  FPType state = 0.0;
  // TODO alexbrandmeyer: I'm assuming one dimensional input for now, but this
  // should be verified with Dick for the final version
  for (int i = n_pts - 11; i < n_pts; i ++){
    input = stage_state(i);
    state = state + (1 - pole_z1) * (input - state);
  }
  for (int i = n_pts - 1; i > -1; i --){
    input = stage_state(i);
    state = state + (1 - pole_z2) * (input - state);
  }
  for (int i = 0; i < n_pts; i ++){
    input = stage_state(i);
    state = state + (1 - pole_z1) * (input - state);
    stage_state(i) = state;
  }
  return stage_state;
}

FloatArray Ear::ReturnZAMemory() {
  return car_state_.za_memory_;
}

FloatArray Ear::ReturnZBMemory() {
  return car_state_.zb_memory_;
}

FloatArray Ear::ReturnGMemory() {
  return car_state_.g_memory_;
};

FloatArray Ear::ReturnZRCoeffs() {
  return car_coeffs_.zr_coeffs_;
};

int Ear::ReturnAGCNStages() {
  return agc_coeffs_.n_agc_stages_;
}

int Ear::ReturnAGCStateDecimPhase(int stage) {
  return agc_state_.decim_phase_.at(stage);
}

FPType Ear::ReturnAGCMixCoeff(int stage) {
  return agc_coeffs_.agc_mix_coeffs_(stage);
}

FloatArray Ear::ReturnAGCStateMemory(int stage) {
  return agc_state_.agc_memory_.at(stage);
}

int Ear::ReturnAGCDecimation(int stage) {
  return agc_coeffs_.decimation_.at(stage);
}

void Ear::SetAGCStateMemory(int stage, FloatArray new_values) {
  agc_state_.agc_memory_.at(stage) = new_values;
}

void Ear::SetCARStateDZBMemory(FloatArray new_values) {
  car_state_.dzb_memory_ = new_values;
}

void Ear::SetCARStateDGMemory(FloatArray new_values) {
  car_state_.dg_memory_ = new_values;
}

FloatArray Ear::StageGValue(FloatArray undamping) {
  FloatArray r1 = car_coeffs_.r1_coeffs_;
  FloatArray a0 = car_coeffs_.a0_coeffs_;
  FloatArray c0 = car_coeffs_.c0_coeffs_;
  FloatArray h = car_coeffs_.h_coeffs_;
  FloatArray zr = car_coeffs_.zr_coeffs_;
  FloatArray r = r1 + zr * undamping;
  return (1 - 2 * r * a0 + (r * r)) / (1 - 2 * r * a0 + h * r * c0 + (r * r));
}
