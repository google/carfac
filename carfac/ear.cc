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

#include <assert.h>
#include "ear.h"

// The 'InitEar' function takes a set of model parameters and initializes the
// design coefficients and model state variables needed for running the model
// on a single audio channel.
void Ear::InitEar(const int n_ch, const FPType fs, const CARCoeffs& car_coeffs,
                  const IHCCoeffs& ihc_coeffs,
                  const std::vector<AGCCoeffs>& agc_coeffs) {
  // The first section of code determines the number of channels that will be
  // used in the model on the basis of the sample rate and the CAR parameters
  // that have been passed to this function.
  n_ch_ = n_ch;
  car_coeffs_ = car_coeffs;
  ihc_coeffs_ = ihc_coeffs;
  agc_coeffs_ = agc_coeffs;
  // Once the coefficients have been determined, we can initialize the state
  // variables that will be used during runtime.
  InitCARState();
  InitIHCState();
  InitAGCState();
}

void Ear::InitCARState() {
  car_state_.z1_memory_.setZero(n_ch_);
  car_state_.z2_memory_.setZero(n_ch_);
  car_state_.za_memory_.setZero(n_ch_);
  car_state_.zb_memory_ = car_coeffs_.zr_coeffs_;
  car_state_.dzb_memory_.setZero(n_ch_);
  car_state_.zy_memory_.setZero(n_ch_);
  car_state_.g_memory_ = car_coeffs_.g0_coeffs_;
  car_state_.dg_memory_.setZero(n_ch_);
}

void Ear::InitIHCState() {
  ihc_state_.ihc_accum_ = FloatArray::Zero(n_ch_);
  if (! ihc_coeffs_.just_half_wave_rectify_) {
    ihc_state_.ac_coupler_.setZero(n_ch_);
    ihc_state_.lpf1_state_.setConstant(n_ch_, ihc_coeffs_.rest_output_);
    ihc_state_.lpf2_state_.setConstant(n_ch_, ihc_coeffs_.rest_output_);
    if (ihc_coeffs_.one_capacitor_) {
      ihc_state_.cap1_voltage_.setConstant(n_ch_, ihc_coeffs_.rest_cap1_);
    } else {
      ihc_state_.cap1_voltage_.setConstant(n_ch_, ihc_coeffs_.rest_cap1_);
      ihc_state_.cap2_voltage_.setConstant(n_ch_, ihc_coeffs_.rest_cap2_);
    }
  }
}

void Ear::InitAGCState() {
  int n_agc_stages = agc_coeffs_.size();
  agc_state_.resize(n_agc_stages);
  for (auto& stage_state : agc_state_) {
    stage_state.decim_phase_ = 0;
    stage_state.agc_memory_.setZero(n_ch_);
    stage_state.input_accum_.setZero(n_ch_);
  }
}

void Ear::CARStep(const FPType input) {
  // This interpolates g.
  car_state_.g_memory_ = car_state_.g_memory_ + car_state_.dg_memory_;
  // This calculates the AGC interpolation state.
  car_state_.zb_memory_ = car_state_.zb_memory_ + car_state_.dzb_memory_;
  // This updates the nonlinear function of 'velocity' along with zA, which is
  // a delay of z2.
  FloatArray nonlinear_fun(n_ch_);
  FloatArray velocities = car_state_.z2_memory_ - car_state_.za_memory_;
  OHCNonlinearFunction(velocities, &nonlinear_fun);
  // Here, zb_memory_ * nonlinear_fun is "undamping" delta r.
  FloatArray r = car_coeffs_.r1_coeffs_ + (car_state_.zb_memory_ *
                                           nonlinear_fun);
  car_state_.za_memory_ = car_state_.z2_memory_;
  // Here we reduce the CAR state by r and rotate with the fixed cos/sin coeffs.
  FloatArray z1  = r * ((car_coeffs_.a0_coeffs_ * car_state_.z1_memory_) -
                        (car_coeffs_.c0_coeffs_ * car_state_.z2_memory_));
  car_state_.z2_memory_ = r *
    ((car_coeffs_.c0_coeffs_ * car_state_.z1_memory_) +
     (car_coeffs_.a0_coeffs_ * car_state_.z2_memory_));
  car_state_.zy_memory_ = car_coeffs_.h_coeffs_ * car_state_.z2_memory_;
  // This section ripples the input-output path, to avoid delay...
  // It's the only part that doesn't get computed "in parallel":
  FPType in_out = input;
  for (int ch = 0; ch < n_ch_; ch++) {
    z1(ch) = z1(ch) + in_out;
    // This performs the ripple, and saves the final channel outputs in zy.
    in_out = car_state_.g_memory_(ch) * (in_out + car_state_.zy_memory_(ch));
    car_state_.zy_memory_(ch) = in_out;
  }
  car_state_.z1_memory_ = z1;
}

// We start with a quadratic nonlinear function, and limit it via a
// rational function. This makes the result go to zero at high
// absolute velocities, so it will do nothing there.
void Ear::OHCNonlinearFunction(const FloatArray& velocities,
                               FloatArray* nonlinear_fun) {
  *nonlinear_fun = (1 + ((velocities * car_coeffs_.velocity_scale_) +
                         car_coeffs_.v_offset_).square()).inverse();
}

// This step is a one sample-time update of the inner-hair-cell (IHC) model,
// including the detection nonlinearity and either one or two capacitor state
// variables.
void Ear::IHCStep(const FloatArray& car_out) {
  FloatArray ac_diff = car_out - ihc_state_.ac_coupler_;
  ihc_state_.ac_coupler_ = ihc_state_.ac_coupler_ +
    (ihc_coeffs_.ac_coeff_ * ac_diff);
  if (ihc_coeffs_.just_half_wave_rectify_) {
    FloatArray output(n_ch_);
    for (int ch = 0; ch < n_ch_; ++ch) {
      FPType a = (ac_diff(ch) > 0.0) ? ac_diff(ch) : 0.0;
      output(ch) = (a < 2) ? a : 2;
    }
    ihc_state_.ihc_out_ = output;
  } else {
    FloatArray conductance = CARFACDetect(ac_diff);
    if (ihc_coeffs_.one_capacitor_) {
      ihc_state_.ihc_out_ = conductance * ihc_state_.cap1_voltage_;
      ihc_state_.cap1_voltage_ = ihc_state_.cap1_voltage_ -
        (ihc_state_.ihc_out_ * ihc_coeffs_.out1_rate_) +
        ((1 - ihc_state_.cap1_voltage_) * ihc_coeffs_.in1_rate_);
    } else {
      ihc_state_.ihc_out_ = conductance * ihc_state_.cap2_voltage_;
      ihc_state_.cap1_voltage_ = ihc_state_.cap1_voltage_ -
        ((ihc_state_.cap1_voltage_ - ihc_state_.cap2_voltage_)
         * ihc_coeffs_.out1_rate_) + ((1 - ihc_state_.cap1_voltage_) *
                                      ihc_coeffs_.in1_rate_);
      ihc_state_.cap2_voltage_ = ihc_state_.cap2_voltage_ -
        (ihc_state_.ihc_out_ * ihc_coeffs_.out2_rate_) +
        ((ihc_state_.cap1_voltage_ - ihc_state_.cap2_voltage_)
         * ihc_coeffs_.in2_rate_);
    }
    // Here we smooth the output twice using a LPF.
    ihc_state_.ihc_out_ *= ihc_coeffs_.output_gain_;
    ihc_state_.lpf1_state_ += ihc_coeffs_.lpf_coeff_ *
      (ihc_state_.ihc_out_ - ihc_state_.lpf1_state_);
    ihc_state_.lpf2_state_ += ihc_coeffs_.lpf_coeff_ *
      (ihc_state_.lpf1_state_ - ihc_state_.lpf2_state_);
    ihc_state_.ihc_out_ = ihc_state_.lpf2_state_ - ihc_coeffs_.rest_output_;
  }
  ihc_state_.ihc_accum_ += ihc_state_.ihc_out_;
}

bool Ear::AGCStep(const FloatArray& ihc_out) {
  int stage = 0;
  int n_stages = agc_coeffs_[0].n_agc_stages_;
  FPType detect_scale = agc_coeffs_[n_stages - 1].detect_scale_;
  bool updated = AGCRecurse(stage, detect_scale * ihc_out);
  return updated;
}

bool Ear::AGCRecurse(const int stage, FloatArray agc_in) {
  bool updated = true;
  const auto& agc_coeffs = agc_coeffs_[stage];
  auto& agc_state = agc_state_[stage];
  // This is the decim factor for this stage, relative to input or prev. stage:
  int decim = agc_coeffs.decimation_;
  // This is the decim phase of this stage (do work on phase 0 only):
  int decim_phase = agc_state.decim_phase_ + 1;
  decim_phase = decim_phase % decim;
  agc_state.decim_phase_ = decim_phase;
  // Here we accumulate input for this stage from the previous stage:
  agc_state.input_accum_ += agc_in;
  // We don't do anything if it's not the right decim_phase.
  if (decim_phase == 0) {
    // Now we do lots of work, at the decimated rate.
    // These are the decimated inputs for this stage, which will be further
    // decimated at the next stage.
    agc_in = agc_state.input_accum_ / decim;
    // This resets the accumulator.
    agc_state.input_accum_ = FloatArray::Zero(n_ch_);
    if (stage < (agc_coeffs_.size() - 1)) {
      // Now we recurse to evaluate the next stage(s).
      updated = AGCRecurse(stage + 1, agc_in);
      // Afterwards we add its output to this stage input, whether it updated or
      // not.
      agc_in += agc_coeffs.agc_stage_gain_ *
        agc_state_[stage + 1].agc_memory_;
    }
    // This performs a first-order recursive smoothing filter update, in time.
    agc_state.agc_memory_ += agc_coeffs.agc_epsilon_ *
      (agc_in - agc_state.agc_memory_);
    AGCSpatialSmooth(agc_coeffs_[stage], &agc_state.agc_memory_);
    updated = true;
  } else {
    updated = false;
  }
  return updated;
}

void Ear::AGCSpatialSmooth(const AGCCoeffs& agc_coeffs,
                           FloatArray* stage_state) {
  int n_iterations = agc_coeffs.agc_spatial_iterations_;
  bool use_fir;
  use_fir = (n_iterations < 4) ? true : false;
  if (use_fir) {
    FPType fir_coeffs_left = agc_coeffs.agc_spatial_fir_left_;
    FPType fir_coeffs_mid = agc_coeffs.agc_spatial_fir_mid_;
    FPType fir_coeffs_right = agc_coeffs.agc_spatial_fir_right_;
    FloatArray ss_tap1(n_ch_);
    FloatArray ss_tap2(n_ch_);
    FloatArray ss_tap3(n_ch_);
    FloatArray ss_tap4(n_ch_);
    int n_taps = agc_coeffs.agc_spatial_n_taps_;
    // This initializes the first two taps of stage state, which are used for
    // both possible cases.
    ss_tap1(0) = (*stage_state)(0);
    ss_tap1.block(1, 0, n_ch_ - 1, 1) = stage_state->block(0, 0, n_ch_ - 1, 1);
    ss_tap2(n_ch_ - 1) = (*stage_state)(n_ch_ - 1);
    ss_tap2.block(0, 0, n_ch_ - 1, 1) = stage_state->block(1, 0, n_ch_ - 1, 1);
    switch (n_taps) {
      case 3:
        *stage_state = (fir_coeffs_left * ss_tap1) +
          (fir_coeffs_mid * *stage_state) + (fir_coeffs_right * ss_tap2);
        break;
      case 5:
        // Now we initialize last two taps of stage state, which are only used
        // for the 5-tap case.
        ss_tap3(0) = (*stage_state)(0);
        ss_tap3(1) = (*stage_state)(1);
        ss_tap3.block(2, 0, n_ch_ - 2, 1) =
          stage_state->block(0, 0, n_ch_ - 2, 1);
        ss_tap4(n_ch_ - 2) = (*stage_state)(n_ch_ - 1);
        ss_tap4(n_ch_ - 1) = (*stage_state)(n_ch_ - 2);
        ss_tap4.block(0, 0, n_ch_ - 2, 1) =
        stage_state->block(2, 0, n_ch_ - 2, 1);
        *stage_state = (fir_coeffs_left * (ss_tap3 + ss_tap1)) +
          (fir_coeffs_mid * *stage_state) +
          (fir_coeffs_right * (ss_tap2 + ss_tap4));
        break;
      default:
        assert(true && "Bad n_taps in AGCSpatialSmooth; should be 3 or 5.");
        break;
    }
  } else {
    AGCSmoothDoubleExponential(agc_coeffs.agc_pole_z1_,
                               agc_coeffs.agc_pole_z2_, stage_state);
  }
}

void Ear::AGCSmoothDoubleExponential(const FPType pole_z1,  const FPType pole_z2,
                                FloatArray* stage_state) {
  int32_t n_pts = stage_state->size();
  FPType input;
  FPType state = 0.0;
  // TODO (alexbrandmeyer): I'm assuming one dimensional input for now, but this
  // should be verified with Dick for the final version
  for (int i = n_pts - 11; i < n_pts; ++i) {
    input = (*stage_state)(i);
    state = state + (1 - pole_z1) * (input - state);
  }
  for (int i = n_pts - 1; i > -1; --i) {
    input = (*stage_state)(i);
    state = state + (1 - pole_z2) * (input - state);
  }
  for (int i = 0; i < n_pts; ++i) {
    input = (*stage_state)(i);
    state = state + (1 - pole_z1) * (input - state);
    (*stage_state)(i) = state;
  }
}

FloatArray Ear::StageGValue(const FloatArray& undamping) {
  FloatArray r = car_coeffs_.r1_coeffs_ + car_coeffs_.zr_coeffs_ * undamping;
  return (1 - 2 * r * car_coeffs_.a0_coeffs_ + (r * r)) /
    (1 - 2 * r * car_coeffs_.a0_coeffs_ + car_coeffs_.h_coeffs_ * r *
     car_coeffs_.c0_coeffs_ + (r * r));
}