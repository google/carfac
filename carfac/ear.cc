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

void Ear::InitEar(long fs, CARParams car_p, IHCParams ihc_p, AGCParams agc_p){
  car_params_ = car_p;
  ihc_params_ = ihc_p;
  agc_params_ = agc_p;
  n_ch_ = 0;
  FPType pole_hz = car_params_.first_pole_theta_ * fs / (2 * PI);
  while (pole_hz > car_params_.min_pole_hz_) {
    n_ch_++;
    pole_hz = pole_hz - car_params_.erb_per_step_ *
    ERBHz(pole_hz, car_params_.erb_break_freq_, car_params_.erb_q_);
  }
  FloatArray pole_freqs(n_ch_);
  pole_hz = car_params_.first_pole_theta_ * fs / (2 * PI);
  for(int ch=0;ch < n_ch_; ch++){
    pole_freqs(ch) = pole_hz;
    pole_hz = pole_hz - car_params_.erb_per_step_ *
    ERBHz(pole_hz, car_params_.erb_break_freq_, car_params_.erb_q_);
  }
  max_channels_per_octave_ = log(2) / log(pole_freqs(0) / pole_freqs(1));
  car_coeffs_.DesignFilters(car_params_, fs, &pole_freqs);
  agc_coeffs_.DesignAGC(agc_params_, fs, n_ch_);
  ihc_coeffs_.DesignIHC(ihc_params_, fs, n_ch_);
  car_state_.InitCARState(car_coeffs_);
  agc_state_.InitAGCState(agc_coeffs_);
  ihc_state_.InitIHCState(ihc_coeffs_);

}

FloatArray Ear::CARStep(FPType input){
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
  
  // do the DOHC stuff:
  g = car_state_.g_memory_ + car_state_.dg_memory_; //interp g
  zb = car_state_.zb_memory_ + car_state_.dzb_memory_; //AGC interpolation state
  // update the nonlinear function of "velocity", and zA (delay of z2):
  za = car_state_.za_memory_;
  v = car_state_.z2_memory_ - za;
  nlf = OHC_NLF(v);
  r = car_coeffs_.r1_coeffs_ + (zb * nlf); // zB * nfl is "undamping" delta r
  za = car_state_.z2_memory_;
  // now reduce state by r and rotate with the fixed cos/sin coeffs:
  z1 = r * ((car_coeffs_.a0_coeffs_ * car_state_.z1_memory_) -
            (car_coeffs_.c0_coeffs_ * car_state_.z2_memory_));
  z2 = r * ((car_coeffs_.c0_coeffs_ * car_state_.z1_memory_) +
            (car_coeffs_.a0_coeffs_ * car_state_.z2_memory_));
  zy = car_coeffs_.h_coeffs_ * z2;
  // Ripple input-output path, instead of parallel, to avoid delay...
  // this is the only part that doesn't get computed "in parallel":
  in_out = input;
  for (int ch = 0; ch < n_ch_; ch++){
    z1(ch) = z1(ch) + in_out;
    // ripple, saving final channel outputs in zY
    in_out = g(ch) * (in_out + zy(ch));
    zy(ch) = in_out;
  }
  car_state_.z1_memory_ = z1;
  car_state_.z2_memory_ = z2;
  car_state_.za_memory_ = za;
  car_state_.zb_memory_ = zb;
  car_state_.zy_memory_ = zy;
  car_state_.g_memory_ = g;
  // car_out is equal to zy state;
  return zy;
}

// start with a quadratic nonlinear function, and limit it via a
// rational function; make the result go to zero at high
// absolute velocities, so it will do nothing there.
FloatArray Ear::OHC_NLF(FloatArray velocities){
  FloatArray nlf(n_ch_);
  nlf = 1 / ((velocities * car_coeffs_.velocity_scale_) +
             (car_coeffs_.v_offset_ * car_coeffs_.v_offset_));
  return nlf;
}

// One sample-time update of inner-hair-cell (IHC) model, including the
// detection nonlinearity and one or two capacitor state variables.
FloatArray Ear::IHCStep(FloatArray car_out){
  FloatArray ihc_out(n_ch_);
  FloatArray ac_diff(n_ch_);
  FloatArray conductance(n_ch_);
  ac_diff = car_out - ihc_state_.ac_coupler_;
  ihc_state_.ac_coupler_ = ihc_state_.ac_coupler_ +
                           (ihc_coeffs_.ac_coeff_ * ac_diff);
  if (ihc_coeffs_.just_hwr_) {
    //TODO Figure out best implementation with Eigen max/min methods
    for (int ch = 0; ch < n_ch_; ch++){
      FPType a;
      if (ac_diff(ch) > 0){
        a = ac_diff(ch);
      } else {
        a = 0;
      }
      if (a < 2){
        ihc_out(ch) = a;
      } else {
        ihc_out(ch) = 2;
      }
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
    // smooth it twice with LPF:
    ihc_out = ihc_out * ihc_coeffs_.output_gain_;
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

bool Ear::AGCStep(FloatArray ihc_out){
  int stage = 0;
  FloatArray agc_in(n_ch_);
  agc_in = agc_coeffs_.detect_scale_ * ihc_out;
  bool updated = AGCRecurse(stage, agc_in);
  return updated;
}

bool Ear::AGCRecurse(int stage, FloatArray agc_in){
  bool updated = true;
  // decim factor for this stage, relative to input or prev. stage:
  int decim = agc_coeffs_.decimation_(stage);
  // decim phase of this stage (do work on phase 0 only):
  //TODO FIX MODULO
  
  int decim_phase = agc_state_.decim_phase_(stage);
  decim_phase = decim_phase % decim;
  agc_state_.decim_phase_(stage) = decim_phase;
  // accumulate input for this stage from detect or previous stage:
  agc_state_.input_accum_.block(0,stage,n_ch_,1) =
                        agc_state_.input_accum_.block(0,stage,n_ch_,1) + agc_in;
  
  // nothing else to do if it's not the right decim_phase
  if (decim_phase == 0){
    // do lots of work, at decimated rate.
    // decimated inputs for this stage, and to be decimated more for next:
    agc_in = agc_state_.input_accum_.block(0,stage,n_ch_,1) / decim;
    // reset accumulator:
    agc_state_.input_accum_.block(0,stage,n_ch_,1) = FloatArray::Zero(n_ch_);
    
    if (stage < (agc_coeffs_.decimation_.size() - 1)){
      // recurse to evaluate next stage(s)
      updated = AGCRecurse(stage+1, agc_in);
      // and add its output to this stage input, whether it updated or not:
      agc_in = agc_in + (agc_coeffs_.agc_stage_gain_ *
                         agc_state_.agc_memory_.block(0,stage+1,n_ch_,1));
    }
    FloatArray agc_stage_state = agc_state_.agc_memory_.block(0,stage,n_ch_,1);
    // first-order recursive smoothing filter update, in time:
    agc_stage_state = agc_stage_state + (agc_coeffs_.agc_epsilon_(stage) *
                                         (agc_in - agc_stage_state));
    agc_stage_state = AGCSpatialSmooth(stage, agc_stage_state);
    agc_state_.agc_memory_.block(0,stage,n_ch_,1) = agc_stage_state;
  } else {
    updated = false;
  }
  return updated;
}

FloatArray Ear::AGCSpatialSmooth(int stage, FloatArray stage_state){
  int n_iterations = agc_coeffs_.agc_spatial_iterations_(stage);
  bool use_fir;
  if (n_iterations < 4){
    use_fir = true;
  } else {
    use_fir = false;
  }
  
  if (use_fir) {
    FloatArray fir_coeffs = agc_coeffs_.agc_spatial_fir_.block(0,stage,3,1);
    FloatArray ss_tap1(n_ch_);
    FloatArray ss_tap2(n_ch_);
    FloatArray ss_tap3(n_ch_);
    FloatArray ss_tap4(n_ch_);
    int n_taps = agc_coeffs_.agc_spatial_n_taps_(stage);
    //Initialize first two taps of stage state, used for both cases
    ss_tap1(0) = stage_state(0);
    ss_tap1.block(1,0,n_ch_-1,1) = stage_state.block(0,0,n_ch_-1,1);
    ss_tap2(n_ch_-1) = stage_state(n_ch_-1);
    ss_tap2.block(0,0,n_ch_-1,1) = stage_state.block(1,0,n_ch_-1,1);
    switch (n_taps) {
      case 3:
        stage_state = (fir_coeffs(0) * ss_tap1) +
                      (fir_coeffs(1) * stage_state) +
                      (fir_coeffs(2) * ss_tap2);
        break;
      case 5:
        //Initialize last two taps of stage state, used for 5-tap case
        ss_tap3(0) = stage_state(0);
        ss_tap3(1) = stage_state(1);
        ss_tap3.block(2,0,n_ch_-2,1) = stage_state.block(0,0,n_ch_-2,1);
        ss_tap4(n_ch_-2) = stage_state(n_ch_-1);
        ss_tap4(n_ch_-1) = stage_state(n_ch_-2);
        ss_tap4.block(0,0,n_ch_-2,1) = stage_state.block(2,0,n_ch_-2,1);
        
        stage_state = (fir_coeffs(0) * (ss_tap3 + ss_tap1)) +
                      (fir_coeffs(1) * stage_state) +
                      (fir_coeffs(2) * (ss_tap2 + ss_tap4));
        break;
      default:
        //TODO Throw Error
        std::cout << "Error: bad n-taps in AGCSpatialSmooth" << std::endl;
    }
    
  } else {
    stage_state = AGCSmoothDoubleExponential(stage_state);
  }
  return stage_state;
}

FloatArray Ear::AGCSmoothDoubleExponential(FloatArray stage_state){
  return stage_state;
}
