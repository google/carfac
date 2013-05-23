//
//  agc_coeffs.cc
//  CARFAC Open Source C++ Library
//
//  Created by Alex Brandmeyer on 5/18/13.
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

#include "agc_coeffs.h"

void AGCCoeffs::Design(const AGCParams& agc_params, const int stage,
                       const FPType fs, const FPType previous_stage_gain,
                       FPType decim) {
  n_agc_stages_ = agc_params.n_stages_;
  agc_stage_gain_ = agc_params.agc_stage_gain_;
  std::vector<FPType> agc1_scales = agc_params.agc1_scales_;
  std::vector<FPType> agc2_scales = agc_params.agc2_scales_;
  std::vector<FPType> time_constants = agc_params.time_constants_;
  FPType mix_coeff = agc_params.agc_mix_coeff_;
  decimation_ = agc_params.decimation_[stage];
  FPType total_dc_gain = previous_stage_gain;
  // Here we calculate the parameters for the current stage.
  FPType tau = time_constants[stage];
  decim_ = decim;
  decim_ *= decimation_;
  agc_epsilon_ = 1 - exp((-1 * decim_) / (tau * fs));
  FPType n_times = tau * (fs / decim_);
  FPType delay = (agc2_scales[stage] - agc1_scales[stage]) / n_times;
  FPType spread_sq = (pow(agc1_scales[stage], 2) +
                      pow(agc2_scales[stage], 2)) / n_times;
  FPType u = 1 + (1 / spread_sq);
  FPType p = u - sqrt(pow(u, 2) - 1);
  FPType dp = delay * (1 - (2 * p) + (p * p)) / 2;
  agc_pole_z1_ = p - dp;
  agc_pole_z2_ = p + dp;
  int n_taps = 0;
  bool fir_ok = false;
  int n_iterations = 1;
  // This section initializes the FIR coeffs settings at each stage.
  agc_spatial_fir_.resize(3);
  std::vector<FPType> fir(3);
  while (! fir_ok) {
    switch (n_taps) {
      case 0:
        n_taps = 3;
        break;
      case 3:
        n_taps = 5;
        break;
      case 5:
        n_iterations++;
        assert(n_iterations < 16 &&
               "Too many iterations needed in AGC spatial smoothing.");
        break;
      default:
        assert(true && "Bad n_taps; should be 3 or 5.");
        break;
    }
    // The smoothing function is a space-domain smoothing, but it considered
    // here by analogy to time-domain smoothing, which is why its potential
    // off-centeredness is called a delay.  Since it's a smoothing filter, it
    // is also analogous to a discrete probability distribution (a p.m.f.),
    // with mean corresponding to delay and variance corresponding to squared
    // spatial spread (in samples, or channels, and the square thereof,
    // respecitively). Here we design a filter implementation's coefficient
    // via the method of moment matching, so we get the intended delay and
    // spread, and don't worry too much about the shape of the distribution,
    // which will be some kind of blob not too far from Gaussian if we run
    // several FIR iterations.
    FPType delay_variance = spread_sq / n_iterations;
    FPType mean_delay = delay / n_iterations;
    FPType a, b;
    switch (n_taps) {
      case 3:
        a = (delay_variance + (mean_delay*mean_delay) - mean_delay) / 2.0;
        b = (delay_variance + (mean_delay*mean_delay) + mean_delay) / 2.0;
        fir[0] = a;
        fir[1] = 1 - a - b;
        fir[2] = b;
        fir_ok = fir[1] >= 0.2 ? true : false;
        break;
      case 5:
        a = (((delay_variance + (mean_delay*mean_delay)) * 2.0/5.0) -
             (mean_delay * 2.0/3.0)) / 2.0;
        b = (((delay_variance + (mean_delay*mean_delay)) * 2.0/5.0) +
             (mean_delay * 2.0/3.0)) / 2.0;
        fir[0] = a / 2.0;
        fir[1] = 1 - a - b;
        fir[2] = b / 2.0;
        fir_ok = fir[1] >= 0.1 ? true : false;
        break;
      default:
        break;  // Again, we've arrived at a bad n_taps in the design.
        CHECK_EQ(5, n_taps) << "Bad n_taps; should be 3 or 5.";
    }
  }
  // Once we have the FIR design for this stage we can assign it to the
  // appropriate data members.
  agc_spatial_iterations_ = n_iterations;
  agc_spatial_n_taps_ = n_taps;
  agc_spatial_fir_[0] = fir[0];
  agc_spatial_fir_[1] = fir[1];
  agc_spatial_fir_[2] = fir[2];
  total_dc_gain += pow(agc_stage_gain_, stage);
  agc_mix_coeffs_ = stage == 0 ? 0 : mix_coeff / (tau * (fs /decim_));
  agc_gain_ = total_dc_gain;
  detect_scale_ = 1 / total_dc_gain;
}
