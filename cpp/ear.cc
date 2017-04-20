// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Alex Brandmeyer
//
// This file is part of an implementation of Lyon's cochlear model:
// "Cascade of Asymmetric Resonators with Fast-Acting Compression"
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

#include "carfac_util.h"

Ear::Ear(int num_channels,
         const CARCoeffs& car_coeffs,
         const IHCCoeffs& ihc_coeffs,
         const std::vector<AGCCoeffs>& agc_coeffs) {
  Redesign(num_channels, car_coeffs, ihc_coeffs, agc_coeffs);
}

void Ear::Redesign(int num_channels,
                   const CARCoeffs& car_coeffs,
                   const IHCCoeffs& ihc_coeffs,
                   const std::vector<AGCCoeffs>& agc_coeffs) {
  num_channels_ = num_channels;
  car_coeffs_ = car_coeffs;
  CARFAC_ASSERT(car_coeffs.r1_coeffs.size() == num_channels &&
                car_coeffs.a0_coeffs.size() == num_channels &&
                car_coeffs.c0_coeffs.size() == num_channels &&
                car_coeffs.h_coeffs.size() == num_channels &&
                car_coeffs.g0_coeffs.size() == num_channels &&
                car_coeffs.zr_coeffs.size() == num_channels &&
                "car_coeffs should be size num_channels.");
  ihc_coeffs_ = ihc_coeffs;
  agc_coeffs_ = agc_coeffs;
  Reset();
}

void Ear::Reset() {
  InitCARState();
  InitIHCState();
  InitAGCState();
  tmp1_.resize(num_channels_);
  tmp2_.resize(num_channels_);
}

void Ear::InitCARState() {
  car_state_.z1_memory.setZero(num_channels_);
  car_state_.z2_memory.setZero(num_channels_);
  car_state_.za_memory.setZero(num_channels_);
  car_state_.zb_memory = car_coeffs_.zr_coeffs;
  car_state_.dzb_memory.setZero(num_channels_);
  car_state_.zy_memory.setZero(num_channels_);
  car_state_.g_memory = car_coeffs_.g0_coeffs;
  car_state_.dg_memory.setZero(num_channels_);
}

void Ear::InitIHCState() {
  ihc_state_.ac_coupler.setZero(num_channels_);
  ihc_state_.ihc_out.setZero(num_channels_);
  if (!ihc_coeffs_.just_half_wave_rectify) {
    ihc_state_.lpf1_state.setConstant(num_channels_, ihc_coeffs_.rest_output);
    ihc_state_.lpf2_state.setConstant(num_channels_, ihc_coeffs_.rest_output);
    if (ihc_coeffs_.one_capacitor) {
      ihc_state_.cap1_voltage.setConstant(num_channels_, ihc_coeffs_.rest_cap1);
    } else {
      ihc_state_.cap1_voltage.setConstant(num_channels_, ihc_coeffs_.rest_cap1);
      ihc_state_.cap2_voltage.setConstant(num_channels_, ihc_coeffs_.rest_cap2);
    }
  }
}

void Ear::InitAGCState() {
  int n_agc_stages = agc_coeffs_.size();
  agc_state_.resize(n_agc_stages);
  for (AGCState& stage_state : agc_state_) {
    stage_state.decim_phase = 0;
    stage_state.agc_memory.setZero(num_channels_);
    stage_state.input_accum.setZero(num_channels_);
  }
}

void Ear::CARStep(FPType input) {
  // Interpolates g.
  car_state_.g_memory = car_state_.g_memory + car_state_.dg_memory;
  // Calculates the AGC interpolation state.
  car_state_.zb_memory = car_state_.zb_memory + car_state_.dzb_memory;
  // This updates the nonlinear function of 'velocity' along with zA, which is
  // a delay of z2.
  ArrayX& r(tmp1_);  // Alias the radius factor r to tmp1_.
  r = car_coeffs_.r1_coeffs +
      // This product is the "undamping" delta r.
      (car_state_.zb_memory *
       // OHC nonlinear function.
       // We start with a quadratic nonlinear function, and limit it via a
       // rational function. This makes the result go to zero at high
       // absolute velocities, so it will do nothing there.
       (1 + ((car_coeffs_.velocity_scale *
              (car_state_.z2_memory - car_state_.za_memory)) +  // velocities.
             car_coeffs_.v_offset).square()).inverse());
  car_state_.za_memory = car_state_.z2_memory;
  // Here we reduce the CAR state by r and then rotate with the fixed cos/sin
  // coeffs, using both temp arrays, ending the scope of tmp1_ as r.
  tmp2_ = r * car_state_.z2_memory;
  // But first stop using tmp1_ for r, and use it for r * z1 instead.
  tmp1_ = r * car_state_.z1_memory;
  car_state_.z1_memory =  // This still needs stage inputs to be added.
      car_coeffs_.a0_coeffs * tmp1_ - car_coeffs_.c0_coeffs * tmp2_;
  car_state_.z2_memory =
      car_coeffs_.c0_coeffs * tmp1_ + car_coeffs_.a0_coeffs * tmp2_;
  car_state_.zy_memory = car_coeffs_.h_coeffs * car_state_.z2_memory;
  // This section ripples the input-output path, to avoid added delays...
  // It's the only part that doesn't get computed "in parallel".
  // Add inputs to z1_memory while looping, since the loop can't run
  // very fast anyway, and adding shifted zy_memory is more awkward.
  FPType in_out = input;  // This is the scalar input waveform sample.
  for (int channel = 0; channel < num_channels_; channel++) {
    car_state_.z1_memory(channel) += in_out;
    // This performs the ripple, and saves the final channel outputs in zy.
    in_out = car_state_.g_memory(channel) *
        (in_out + car_state_.zy_memory(channel));
    car_state_.zy_memory(channel) = in_out;
  }
}

// This step is a one sample-time update of the inner-hair-cell (IHC) model,
// including the detection nonlinearity and either one or two capacitor state
// variables.
void Ear::IHCStep(const ArrayX& car_out) {
  ArrayX& ac_diff = ihc_state_.ihc_out;
  ac_diff = car_out - ihc_state_.ac_coupler;
  ihc_state_.ac_coupler = ihc_state_.ac_coupler +
      (ihc_coeffs_.ac_coeff * ac_diff);
  if (ihc_coeffs_.just_half_wave_rectify) {
    ihc_state_.ihc_out = ac_diff.max(ArrayX::Zero(num_channels_))
        .min(ArrayX::Constant(num_channels_, 2));
  } else {
    CARFACDetect(&ac_diff);
    ArrayX& conductance = ac_diff;
    if (ihc_coeffs_.one_capacitor) {
      ihc_state_.ihc_out = conductance * ihc_state_.cap1_voltage;
      ihc_state_.cap1_voltage = ihc_state_.cap1_voltage -
          (ihc_state_.ihc_out * ihc_coeffs_.out1_rate) +
          ((1 - ihc_state_.cap1_voltage) * ihc_coeffs_.in1_rate);
    } else {
      ihc_state_.ihc_out = conductance * ihc_state_.cap2_voltage;
      ihc_state_.cap1_voltage = ihc_state_.cap1_voltage -
          ((ihc_state_.cap1_voltage - ihc_state_.cap2_voltage)
           * ihc_coeffs_.out1_rate) + ((1 - ihc_state_.cap1_voltage) *
                                       ihc_coeffs_.in1_rate);
      ihc_state_.cap2_voltage = ihc_state_.cap2_voltage -
          (ihc_state_.ihc_out * ihc_coeffs_.out2_rate) +
          ((ihc_state_.cap1_voltage - ihc_state_.cap2_voltage)
           * ihc_coeffs_.in2_rate);
    }
    // Smooth the output twice using an LPF.
    ihc_state_.lpf1_state += ihc_coeffs_.lpf_coeff *
        (ihc_state_.ihc_out * ihc_coeffs_.output_gain -
         ihc_state_.lpf1_state);
    ihc_state_.lpf2_state += ihc_coeffs_.lpf_coeff *
        (ihc_state_.lpf1_state - ihc_state_.lpf2_state);
    ihc_state_.ihc_out = ihc_state_.lpf2_state - ihc_coeffs_.rest_output;
  }
}

bool Ear::AGCStep(const ArrayX& ihc_out) {
  bool updated = false;
  const int num_stages = agc_coeffs_.size();
  if (num_stages > 0) {  // AGC is enabled.
    int stage = 0;
    FPType detect_scale = agc_coeffs_[num_stages - 1].detect_scale;
    ArrayX& agc_in(tmp1_);
    agc_in = detect_scale * ihc_out;
    updated = AGCRecurse(stage, &agc_in);
  }
  return updated;
}

bool Ear::AGCRecurse(int stage, ArrayX* agc_in_out) {
  bool updated = false;
  const AGCCoeffs& agc_coeffs = agc_coeffs_[stage];
  AGCState& agc_state = agc_state_[stage];
  // Unconditionally accumulate input for this stage from the previous stage:
  agc_state.input_accum += *agc_in_out;
  // This is the decim factor for this stage, relative to input or prev. stage:
  int decim = agc_coeffs.decimation;
  // Increment the decimation phase of this stage (do work on phase 0 only):
  if (++agc_state.decim_phase >= decim) {
    agc_state.decim_phase = 0;
    // Now do lots of time and space filtering work, at the decimated rate.
    // These are the decimated inputs for this stage, which will be further
    // decimated at the next stage.
    *agc_in_out = agc_state.input_accum * FPType(1.0 / decim);
    // If more stage(s), recurse to evaluate the next stage.
    if (stage < (agc_coeffs_.size() - 1)) {
      // Use agc_state.input_accum as input to next stage; it will be used
      // as temp agc_in_out space there until we clear it afterward to start
      // over using it again as input accumulator for this stage.
      agc_state.input_accum = *agc_in_out;
      AGCRecurse(stage + 1, &agc_state.input_accum);
      // Finally add next stage's output to this stage input, whether
      // the next stage updated or not (ignoring bool result of AGCRecurse).
      *agc_in_out += agc_coeffs.agc_stage_gain *
          agc_state_[stage + 1].agc_memory;
    }
    // This resets the accumulator.
    agc_state.input_accum = ArrayX::Zero(num_channels_);
    // This performs a first-order recursive smoothing filter update, in time,
    // at this stage's update rate.
    agc_state.agc_memory += agc_coeffs.agc_epsilon *
        (*agc_in_out - agc_state.agc_memory);
    // Smooth the agc_memory across space or frequency.
    AGCSpatialSmooth(agc_coeffs, &agc_state.agc_memory);
    updated = true;
  }
  return updated;
}

void Ear::AGCSpatialSmooth(const AGCCoeffs& agc_coeffs,
                           ArrayX* stage_state) {
  const int num_iterations = agc_coeffs.agc_spatial_iterations;
  const bool use_fir = num_iterations >= 0;
  if (use_fir) {
    const FPType fir_coeffs_left = agc_coeffs.agc_spatial_fir_left;
    const FPType fir_coeffs_mid = agc_coeffs.agc_spatial_fir_mid;
    const FPType fir_coeffs_right = agc_coeffs.agc_spatial_fir_right;
    ArrayX& smoothed_state(tmp2_);  // While tmp1_ is in use as agc_in.
    switch (agc_coeffs.agc_spatial_n_taps) {
      case 3:
        for (int count = 0; count < num_iterations; ++count) {
          // First filter most points, with vector parallelism.
          smoothed_state.segment(1, num_channels_ - 2) =
              fir_coeffs_mid * ((*stage_state).segment(1, num_channels_ - 2)) +
              fir_coeffs_left * ((*stage_state).segment(0, num_channels_ - 2)) +
              fir_coeffs_right * ((*stage_state).segment(2, num_channels_ - 2));
          // Then patch up one point on each end, with clamped edge condition.
          smoothed_state(0) = fir_coeffs_mid * (*stage_state)(0) +
                              fir_coeffs_left * (*stage_state)(0) +
                              fir_coeffs_right * (*stage_state)(1);
          smoothed_state(num_channels_ - 1) =
              fir_coeffs_mid * (*stage_state)(num_channels_ - 1) +
              fir_coeffs_left * (*stage_state)(num_channels_ - 2) +
              fir_coeffs_right * (*stage_state)(num_channels_ - 1);
          // Copy smoothed state back from temp.
          *stage_state = smoothed_state;
        }
        break;
      case 5:
        for (int count = 0; count < num_iterations; ++count) {
          // First filter most points, with vector parallelism.
          smoothed_state.segment(2, num_channels_ - 4) =
              fir_coeffs_mid * ((*stage_state).segment(2, num_channels_ - 4)) +
              fir_coeffs_left * ((*stage_state).segment(0, num_channels_ - 4) +
                                 (*stage_state).segment(1, num_channels_ - 4)) +
              fir_coeffs_right * ((*stage_state).segment(3, num_channels_ - 4) +
                                  (*stage_state).segment(4, num_channels_ - 4));
          // Then patch up 2 points on each end.
          smoothed_state(0) =
              fir_coeffs_mid * (*stage_state)(0) +
              fir_coeffs_left * ((*stage_state)(0) + (*stage_state)(1)) +
              fir_coeffs_right * ((*stage_state)(1) + (*stage_state)(2));
          smoothed_state(1) =
              fir_coeffs_mid * (*stage_state)(1) +
              fir_coeffs_left * ((*stage_state)(0) + (*stage_state)(0)) +
              fir_coeffs_right * ((*stage_state)(2) + (*stage_state)(3));
          smoothed_state(num_channels_ - 1) =
              fir_coeffs_mid * (*stage_state)(num_channels_ - 1) +
              fir_coeffs_left * ((*stage_state)(num_channels_ - 2) +
                                 (*stage_state)(num_channels_ - 3)) +
              fir_coeffs_right * ((*stage_state)(num_channels_ - 1) +
                                  (*stage_state)(num_channels_ - 1));
          smoothed_state(num_channels_ - 2) =
              fir_coeffs_mid * (*stage_state)(num_channels_ - 2) +
              fir_coeffs_left * ((*stage_state)(num_channels_ - 3) +
                                 (*stage_state)(num_channels_ - 4)) +
              fir_coeffs_right * ((*stage_state)(num_channels_ - 1) +
                                  (*stage_state)(num_channels_ - 1));
          // Copy smoothed state back from temp.
          *stage_state = smoothed_state;
        }
        break;
      default:
        CARFAC_ASSERT(false &&
                      "Bad n_taps in AGCSpatialSmooth; should be 3 or 5.");
        break;
    }
  } else {
    // Fall back on IIR smoothing.
    AGCSmoothDoubleExponential(agc_coeffs.agc_pole_z1, agc_coeffs.agc_pole_z2,
                               stage_state);
  }
}

void Ear::AGCSmoothDoubleExponential(FPType pole_z1,
                                     FPType pole_z2,
                                     ArrayX* stage_state) const {
  int32_t num_points = stage_state->size();
  FPType input;
  FPType state = 0.0;
  for (int i = num_points - 11; i < num_points; ++i) {
    input = (*stage_state)(i);
    state = state + (1 - pole_z1) * (input - state);
  }
  for (int i = num_points - 1; i > -1; --i) {
    input = (*stage_state)(i);
    state += (1 - pole_z2) * (input - state);
  }
  for (int i = 0; i < num_points; ++i) {
    input = (*stage_state)(i);
    state += (1 - pole_z1) * (input - state);
    (*stage_state)(i) = state;
  }
}

// This is called when AGC state changes, every 8th sample by default.
// Uses tmp1_ and tmp2_ arrays.
void Ear::CloseAGCLoop(bool open_loop) {
  if (open_loop) {
    // Zero the deltas to make the parameters not keep changing.
    car_state_.dzb_memory.setZero(num_channels_);
    car_state_.dg_memory.setZero(num_channels_);
  } else {
    // Scale factor to get the deltas to update in this many steps.
    FPType scaling = 1.0 / agc_decimation(0);
    ArrayX& undamping(tmp1_);
    undamping = 1 - agc_memory(0);
    // This sets the delta for the damping zb.
    car_state_.dzb_memory = (zr_coeffs() * undamping - zb_memory()) * scaling;
    // Find new stage gains to go with new dampings.
    ArrayX& g_values(tmp2_);
    auto r = car_coeffs_.r1_coeffs + car_coeffs_.zr_coeffs * undamping;
    g_values = (1 - 2 * r * car_coeffs_.a0_coeffs + (r * r)) /
               (1 - 2 * r * car_coeffs_.a0_coeffs +
                car_coeffs_.h_coeffs * r * car_coeffs_.c0_coeffs + (r * r));
    // This updates the target stage gain.
    car_state_.dg_memory = (g_values - g_memory()) * scaling;
  }
}

void Ear::CrossCouple(const ArrayX& mean_state, int stage) {
  ArrayX& stage_state = agc_state_[stage].agc_memory;
  stage_state += agc_coeffs_[stage].agc_mix_coeffs * (mean_state - stage_state);
}
