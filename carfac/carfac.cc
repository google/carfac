//
//  carfac.cc
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
#include "carfac.h"
using std::vector;

void CARFAC::Design(const int n_ears, const FPType fs,
                    const CARParams& car_params, const IHCParams& ihc_params,
                    const AGCParams& agc_params) {
  n_ears_ = n_ears;
  fs_ = fs;
  ears_.resize(n_ears_);
  n_ch_ = 0;
  FPType pole_hz = car_params.first_pole_theta_ * fs / (2 * kPi);
  while (pole_hz > car_params.min_pole_hz_) {
    ++n_ch_;
    pole_hz = pole_hz - car_params.erb_per_step_ *
    ERBHz(pole_hz, car_params.erb_break_freq_, car_params.erb_q_);
  }
  pole_freqs_.resize(n_ch_);
  pole_hz = car_params.first_pole_theta_ * fs / (2 * kPi);
  for (int ch = 0; ch < n_ch_; ++ch) {
    pole_freqs_(ch) = pole_hz;
    pole_hz = pole_hz - car_params.erb_per_step_ *
    ERBHz(pole_hz, car_params.erb_break_freq_, car_params.erb_q_);
  }
  max_channels_per_octave_ = log(2) / log(pole_freqs_(0) / pole_freqs_(1));
  CARCoeffs car_coeffs;
  IHCCoeffs ihc_coeffs;
  std::vector<AGCCoeffs> agc_coeffs;
  DesignCARCoeffs(car_params, fs, pole_freqs_, &car_coeffs);
  DesignIHCCoeffs(ihc_params, fs, &ihc_coeffs);
  // This code initializes the coefficients for each of the AGC stages.
  DesignAGCCoeffs(agc_params, fs, &agc_coeffs);
  // Once we have the coefficient structure we can design the ears.
  for (auto& ear : ears_) {
    ear.InitEar(n_ch_, fs_, car_coeffs, ihc_coeffs,
                agc_coeffs);
  }
}

void CARFAC::Run(const vector<vector<float>>& sound_data,
                 CARFACOutput* output) {
  int n_audio_channels = sound_data.size();
  int32_t seg_len = 441;  // We use a fixed segment length for now.
  int32_t n_timepoints = sound_data[0].size();
  int32_t n_segs = ceil((n_timepoints * 1.0) / seg_len);
  // These values store the start and endpoints for each segment
  int32_t start;
  int32_t length = seg_len;
  // This section loops over the individual audio segments.
  for (int32_t i = 0; i < n_segs; ++i) {
    // For each segment we calculate the start point and the segment length.
    start = i * seg_len;
    if (i == n_segs - 1) {
      // The last segment can be shorter than the rest.
      length = n_timepoints - start;
    }
    // Once we've determined the start point and segment length, we run the
    // CARFAC model on the current segment.
    RunSegment(sound_data, start,
               length, false, output);
  }
}

void CARFAC::RunSegment(const vector<vector<float>>& sound_data,
                        const int32_t start, const int32_t length,
                        const bool open_loop, CARFACOutput* seg_output) {
  // A nested loop structure is used to iterate through the individual samples
  // for each ear (audio channel).
  bool updated;  // This variable is used by the AGC stage.
  for (int32_t i = 0; i < length; ++i) {
    for (int j = 0; j < n_ears_; ++j) {
      // First we create a reference to the current Ear object.
      Ear& ear = ears_[j];
      // This stores the audio sample currently being processed.
      FPType input = sound_data[j][start+i];
      // Now we apply the three stages of the model in sequence to the current
      // audio sample.
      ear.CARStep(input);
      ear.IHCStep(ear.car_out());
      updated = ear.AGCStep(ear.ihc_out());
    }
    seg_output->StoreOutput(ears_);
    if (updated) {
      if (n_ears_ > 1) {
        CrossCouple();
      }
      if (! open_loop) {
        CloseAGCLoop();
      }
    }
  }
}

void CARFAC::CrossCouple() {
  for (int stage = 0; stage < ears_[0].agc_nstages(); ++stage) {
    if (ears_[0].agc_decim_phase(stage) > 0) {
      break;
    } else {
      FPType mix_coeff = ears_[0].agc_mix_coeff(stage);
      if (mix_coeff > 0) {
        FloatArray stage_state;
        FloatArray this_stage_values = FloatArray::Zero(n_ch_);
        for (auto& ear : ears_) {
          stage_state = ear.agc_memory(stage);
          this_stage_values += stage_state;
        }
        this_stage_values /= n_ears_;
        for (auto& ear : ears_) {
          stage_state = ear.agc_memory(stage);
          ear.set_agc_memory(stage, stage_state + mix_coeff *
                             (this_stage_values - stage_state));
        }
      }
    }
  }
}

void CARFAC::CloseAGCLoop() {
  for (auto& ear: ears_) {
    FloatArray undamping = 1 - ear.agc_memory(0);
    // This updates the target stage gain for the new damping.
    ear.set_dzb_memory((ear.zr_coeffs() * undamping - ear.zb_memory()) /
                       ear.agc_decimation(0));
    ear.set_dg_memory((ear.StageGValue(undamping) - ear.g_memory()) /
                      ear.agc_decimation(0));
  }
}

void CARFAC::DesignCARCoeffs(const CARParams& car_params, const FPType fs,
                             const FloatArray& pole_freqs,
                             CARCoeffs* car_coeffs) {
  n_ch_ = pole_freqs.size();
  car_coeffs->velocity_scale_ = car_params.velocity_scale_;
  car_coeffs->v_offset_ = car_params.v_offset_;
  car_coeffs->r1_coeffs_.resize(n_ch_);
  car_coeffs->a0_coeffs_.resize(n_ch_);
  car_coeffs->c0_coeffs_.resize(n_ch_);
  car_coeffs->h_coeffs_.resize(n_ch_);
  car_coeffs->g0_coeffs_.resize(n_ch_);
  FPType f = car_params.zero_ratio_ * car_params.zero_ratio_ - 1.0;
  FloatArray theta = pole_freqs * ((2.0 * kPi) / fs);
  car_coeffs->c0_coeffs_ = theta.sin();
  car_coeffs->a0_coeffs_ = theta.cos();
  FPType ff = car_params.high_f_damping_compression_;
  FloatArray x = theta / kPi;
  car_coeffs->zr_coeffs_ = kPi * (x - (ff * (x*x*x)));
  FPType max_zeta = car_params.max_zeta_;
  FPType min_zeta = car_params.min_zeta_;
  car_coeffs->r1_coeffs_ = (1.0 - (car_coeffs->zr_coeffs_ * max_zeta));
  FloatArray erb_freqs(n_ch_);
  for (int ch=0; ch < n_ch_; ++ch) {
    erb_freqs(ch) = ERBHz(pole_freqs(ch), car_params.erb_break_freq_,
                          car_params.erb_q_);
  }
  FloatArray min_zetas = min_zeta + (0.25 * ((erb_freqs / pole_freqs) -
                                             min_zeta));
  car_coeffs->zr_coeffs_ *= max_zeta - min_zetas;
  car_coeffs->h_coeffs_ = car_coeffs->c0_coeffs_ * f;
  FloatArray relative_undamping = FloatArray::Ones(n_ch_);
  FloatArray r = car_coeffs->r1_coeffs_ + (car_coeffs->zr_coeffs_ *
                                           relative_undamping);
  car_coeffs->g0_coeffs_ = (1.0 - (2.0 * r * car_coeffs->a0_coeffs_) + (r*r)) /
    (1 - (2 * r * car_coeffs->a0_coeffs_) +
    (car_coeffs->h_coeffs_ * r * car_coeffs->c0_coeffs_) + (r*r));
}

void CARFAC::DesignIHCCoeffs(const IHCParams& ihc_params, const FPType fs,
                             IHCCoeffs* ihc_coeffs) {
  if (ihc_params.just_half_wave_rectify_) {
    ihc_coeffs->just_half_wave_rectify_ = ihc_params.just_half_wave_rectify_;
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
    if (ihc_params.one_capacitor_) {
      FPType ro = 1 / conduct_at_10 ;
      FPType c = ihc_params.tau1_out_ / ro;
      FPType ri = ihc_params.tau1_in_ / c;
      FPType saturation_output = 1 / ((2 * ro) + ri);
      FPType r0 = 1 / conduct_at_0;
      FPType current = 1 / (ri + r0);
      ihc_coeffs->cap1_voltage_ = 1 - (current * ri);
      ihc_coeffs->just_half_wave_rectify_ = false;
      ihc_coeffs->lpf_coeff_ = 1 - exp( -1 / (ihc_params.tau_lpf_ * fs));
      ihc_coeffs->out1_rate_ = ro / (ihc_params.tau1_out_ * fs);
      ihc_coeffs->in1_rate_ = 1 / (ihc_params.tau1_in_ * fs);
      ihc_coeffs->one_capacitor_ = ihc_params.one_capacitor_;
      ihc_coeffs->output_gain_ = 1 / (saturation_output - current);
      ihc_coeffs->rest_output_ = current / (saturation_output - current);
      ihc_coeffs->rest_cap1_ = ihc_coeffs->cap1_voltage_;
    } else {
      FPType ro = 1 / conduct_at_10;
      FPType c2 = ihc_params.tau2_out_ / ro;
      FPType r2 = ihc_params.tau2_in_ / c2;
      FPType c1 = ihc_params.tau1_out_ / r2;
      FPType r1 = ihc_params.tau1_in_ / c1;
      FPType saturation_output = 1 / (2 * ro + r2 + r1);
      FPType r0 = 1 / conduct_at_0;
      FPType current = 1 / (r1 + r2 + r0);
      ihc_coeffs->cap1_voltage_ = 1 - (current * r1);
      ihc_coeffs->cap2_voltage_ = ihc_coeffs->cap1_voltage_ - (current * r2);
      ihc_coeffs->just_half_wave_rectify_ = false;
      ihc_coeffs->lpf_coeff_ = 1 - exp(-1 / (ihc_params.tau_lpf_ * fs));
      ihc_coeffs->out1_rate_ = 1 / (ihc_params.tau1_out_ * fs);
      ihc_coeffs->in1_rate_ = 1 / (ihc_params.tau1_in_ * fs);
      ihc_coeffs->out2_rate_ = ro / (ihc_params.tau2_out_ * fs);
      ihc_coeffs->in2_rate_ = 1 / (ihc_params.tau2_in_ * fs);
      ihc_coeffs->one_capacitor_ = ihc_params.one_capacitor_;
      ihc_coeffs->output_gain_ = 1 / (saturation_output - current);
      ihc_coeffs->rest_output_ = current / (saturation_output - current);
      ihc_coeffs->rest_cap1_ = ihc_coeffs->cap1_voltage_;
      ihc_coeffs->rest_cap2_ = ihc_coeffs->cap2_voltage_;
    }
  }
  ihc_coeffs->ac_coeff_ = 2 * kPi * ihc_params.ac_corner_hz_ / fs;
}

void CARFAC::DesignAGCCoeffs(const AGCParams& agc_params, const FPType fs,
                             vector<AGCCoeffs>* agc_coeffs) {
  agc_coeffs->resize(agc_params.n_stages_);
  FPType previous_stage_gain = 0.0;
  FPType decim = 1.0;
  for (int stage = 0; stage < agc_params.n_stages_; ++stage) {
    AGCCoeffs& agc_coeff = agc_coeffs->at(stage);
    agc_coeff.n_agc_stages_ = agc_params.n_stages_;
    agc_coeff.agc_stage_gain_ = agc_params.agc_stage_gain_;
    vector<FPType> agc1_scales = agc_params.agc1_scales_;
    vector<FPType> agc2_scales = agc_params.agc2_scales_;
    vector<FPType> time_constants = agc_params.time_constants_;
    FPType mix_coeff = agc_params.agc_mix_coeff_;
    agc_coeff.decimation_ = agc_params.decimation_[stage];
    FPType total_dc_gain = previous_stage_gain;
    // Here we calculate the parameters for the current stage.
    FPType tau = time_constants[stage];
    agc_coeff.decim_ = decim;
    agc_coeff.decim_ *= agc_coeff.decimation_;
    agc_coeff.agc_epsilon_ = 1 - exp((-1 * agc_coeff.decim_) / (tau * fs));
    FPType n_times = tau * (fs / agc_coeff.decim_);
    FPType delay = (agc2_scales[stage] - agc1_scales[stage]) / n_times;
    FPType spread_sq = (pow(agc1_scales[stage], 2) +
                        pow(agc2_scales[stage], 2)) / n_times;
    FPType u = 1 + (1 / spread_sq);
    FPType p = u - sqrt(pow(u, 2) - 1);
    FPType dp = delay * (1 - (2 * p) + (p*p)) / 2;
    agc_coeff.agc_pole_z1_ = p - dp;
    agc_coeff.agc_pole_z2_ = p + dp;
    int n_taps = 0;
    bool fir_ok = false;
    int n_iterations = 1;
    // This section initializes the FIR coeffs settings at each stage.
    FPType fir_left, fir_mid, fir_right;
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
          fir_left = a;
          fir_mid = 1 - a - b;
          fir_right = b;
          fir_ok = fir_mid >= 0.2 ? true : false;
          break;
        case 5:
          a = (((delay_variance + (mean_delay*mean_delay)) * 2.0/5.0) -
               (mean_delay * 2.0/3.0)) / 2.0;
          b = (((delay_variance + (mean_delay*mean_delay)) * 2.0/5.0) +
               (mean_delay * 2.0/3.0)) / 2.0;
          fir_left = a / 2.0;
          fir_mid = 1 - a - b;
          fir_right = b / 2.0;
          fir_ok = fir_mid >= 0.1 ? true : false;
          break;
        default:
          assert(true && "Bad n_taps; should be 3 or 5.");
          break;
      }
    }
    // Once we have the FIR design for this stage we can assign it to the
    // appropriate data members.
    agc_coeff.agc_spatial_iterations_ = n_iterations;
    agc_coeff.agc_spatial_n_taps_ = n_taps;
    agc_coeff.agc_spatial_fir_left_ = fir_left;
    agc_coeff.agc_spatial_fir_mid_ = fir_mid;
    agc_coeff.agc_spatial_fir_right_ = fir_right;
    total_dc_gain += pow(agc_coeff.agc_stage_gain_, stage);
    agc_coeff.agc_mix_coeffs_ = stage == 0 ? 0 : mix_coeff /
    (tau * (fs / agc_coeff.decim_));
    agc_coeff.agc_gain_ = total_dc_gain;
    agc_coeff.detect_scale_ = 1 / total_dc_gain;
    previous_stage_gain = agc_coeff.agc_gain_;
    decim = agc_coeff.decim_;
  }
}
