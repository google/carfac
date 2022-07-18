// Copyright 2013, 2015, 2017, 2022 The CARFAC Authors. All Rights Reserved.
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

#include "carfac.h"

#include <cmath>

#include "carfac_util.h"
#include "ear.h"

CARFAC::CARFAC(int num_ears, FPType sample_rate, const CARParams& car_params,
               const IHCParams& ihc_params, const AGCParams& agc_params) {
  Redesign(num_ears, sample_rate, car_params, ihc_params, agc_params);
}

CARFAC::~CARFAC() {
  for (Ear* ear : ears_) {
    delete ear;
  }
}

void CARFAC::Redesign(int num_ears, FPType sample_rate,
                      const CARParams& car_params, const IHCParams& ihc_params,
                      const AGCParams& agc_params) {
  num_ears_ = num_ears;
  sample_rate_ = sample_rate;
  car_params_ = car_params;
  ihc_params_ = ihc_params;
  agc_params_ = agc_params;

  pole_freqs_ = CARPoleFrequencies(sample_rate, car_params);
  num_channels_ = pole_freqs_.size();

  max_channels_per_octave_ = M_LN2 / std::log(pole_freqs_(0) / pole_freqs_(1));
  CARCoeffs car_coeffs;
  IHCCoeffs ihc_coeffs;
  std::vector<AGCCoeffs> agc_coeffs;
  DesignCARCoeffs(car_params_, sample_rate_, pole_freqs_, &car_coeffs);
  DesignIHCCoeffs(ihc_params_, sample_rate_, &ihc_coeffs);
  DesignAGCCoeffs(agc_params_, sample_rate_, &agc_coeffs);
  // Once we have the coefficient structure we can design the ears.
  ears_.reserve(num_ears_);
  for (int i = 0; i < num_ears_; ++i) {
    if (ears_.size() > i && ears_[i] != NULL) {
      // Reinitialize any existing ears.
      ears_[i]->Redesign(num_channels_, car_coeffs, ihc_coeffs, agc_coeffs);
    } else {
      ears_.push_back(
          new Ear(num_channels_, car_coeffs, ihc_coeffs, agc_coeffs));
    }
  }
  accumulator_.setZero(num_channels_);
}

void CARFAC::Reset() {
  for (Ear* ear : ears_) {
    ear->Reset();
  }
}

void CARFAC::RunSegment(const ArrayXX& sound_data, bool open_loop,
                        CARFACOutput* output) {
  CARFAC_ASSERT(sound_data.rows() == num_ears_);
  output->Resize(num_ears_, num_channels_, sound_data.cols());

  if (open_loop) {
    // If called to run open-loop, this ensures that the deltas are zeroed, to
    // freeze the damping, since it may have been running closed-loop last time.
    CloseAGCLoop(open_loop);
  }
  // A nested loop structure is used to iterate through the individual samples
  // for each ear (audio channel).
  bool agc_memory_updated = false;
  for (int32_t timepoint = 0; timepoint < sound_data.cols(); ++timepoint) {
    for (int audio_channel = 0; audio_channel < num_ears_; ++audio_channel) {
      FPType input_sample = sound_data(audio_channel, timepoint);

      Ear* ear = ears_[audio_channel];
      // Apply the three stages of the model in sequence to the current sample.
      ear->CARStep(input_sample);
      ear->IHCStep(ear->car_out());
      // The AGC work can be skipped if running open loop, since it will not
      // affect the output.  I had kept it running, in the Matlab version, as
      // a way to get at what the AGC filter is doing, for visualization.
      if (!open_loop) {
        agc_memory_updated = ear->AGCStep(ear->ihc_out());
      }
    }
    output->AssignFromEars(ears_, timepoint);
    if (agc_memory_updated) {
      if (num_ears_ > 1) {
        CrossCouple();
      }
      CloseAGCLoop(open_loop);
    }
  }
}

void CARFAC::CrossCouple() {
  for (int stage = 0; stage < ears_[0]->agc_num_stages(); ++stage) {
    if (ears_[0]->agc_decim_phase(stage) > 0) {
      break;
    } else {
      FPType mix_coeff = ears_[0]->agc_mix_coeff(stage);
      if (mix_coeff > 0) {
        accumulator_.setZero(num_channels_);
        for (Ear* ear : ears_) {
          accumulator_ += ear->agc_memory(stage);
        }
        accumulator_ *= FPType(1.0) / num_ears_;  // Ears' mean AGC state.
        // Mix the mean into all.
        for (Ear* ear : ears_) {
          ear->CrossCouple(accumulator_, stage);
        }
      }
    }
  }
}

void CARFAC::CloseAGCLoop(bool open_loop) {
  for (Ear* ear : ears_) {
    // This updates the target damping and stage gain, or just sets the
    // deltas to zero in the open-loop case.
    ear->CloseAGCLoop(open_loop);
  }
}

void CARFAC::DesignCARCoeffs(const CARParams& car_params,
                             FPType sample_rate,
                             const ArrayX& pole_freqs,
                             CARCoeffs* car_coeffs) {
  int num_channels = pole_freqs.size();
  car_coeffs->velocity_scale = car_params.velocity_scale;
  car_coeffs->v_offset = car_params.v_offset;
  car_coeffs->r1_coeffs.resize(num_channels);
  car_coeffs->a0_coeffs.resize(num_channels);
  car_coeffs->c0_coeffs.resize(num_channels);
  car_coeffs->h_coeffs.resize(num_channels);
  car_coeffs->g0_coeffs.resize(num_channels);
  FPType f = car_params.zero_ratio * car_params.zero_ratio - 1.0;
  ArrayX theta = pole_freqs * (FPType(2.0 * M_PI) / sample_rate);
  car_coeffs->c0_coeffs = theta.sin();
  car_coeffs->a0_coeffs = theta.cos();
  FPType ff = car_params.high_f_damping_compression;
  ArrayX x = theta / M_PI;
  car_coeffs->zr_coeffs = M_PI * (x - (ff * (x*x*x)));
  FPType max_zeta = car_params.max_zeta;
  FPType min_zeta = car_params.min_zeta;
  car_coeffs->r1_coeffs = (1.0 - (car_coeffs->zr_coeffs * max_zeta));
  ArrayX erb_freqs(num_channels);
  for (int channel = 0; channel < num_channels; ++channel) {
    erb_freqs(channel) = ERBHz(pole_freqs(channel), car_params.erb_break_freq,
                               car_params.erb_q);
  }
  ArrayX min_zetas = min_zeta + (0.25 * ((erb_freqs / pole_freqs) - min_zeta));
  // Let zeta be smaller where we compress zeros toward poles.
  // Multiply by 1 if high_f_damping_compression is zero, less otherwise.
  bool reduce_high_f_dampings = false;  // TODO(dicklyon) parameterize this.
  if (reduce_high_f_dampings) {
    min_zetas *= car_coeffs->zr_coeffs / theta;
  }
  car_coeffs->zr_coeffs *= max_zeta - min_zetas;
  car_coeffs->h_coeffs = car_coeffs->c0_coeffs * f;
  ArrayX relative_undamping = ArrayX::Ones(num_channels);
  ArrayX r =
      car_coeffs->r1_coeffs + (car_coeffs->zr_coeffs * relative_undamping);
  car_coeffs->g0_coeffs = (1.0 - (2.0 * r * car_coeffs->a0_coeffs) + (r*r)) /
      (1 - (2 * r * car_coeffs->a0_coeffs) +
       (car_coeffs->h_coeffs * r * car_coeffs->c0_coeffs) + (r*r));
}

void CARFAC::DesignIHCCoeffs(const IHCParams& ihc_params, FPType sample_rate,
                             IHCCoeffs* ihc_coeffs) {
  if (ihc_params.just_half_wave_rectify) {
    ihc_coeffs->just_half_wave_rectify = ihc_params.just_half_wave_rectify;
  } else {
    // This section calculates conductance values using two pre-defined scalars.
    ArrayX x(1);
    FPType conduct_at_10, conduct_at_0;
    x(0) = 10.0;
    CARFACDetect(&x);
    conduct_at_10 = x(0);
    x(0) = 0.0;
    CARFACDetect(&x);
    conduct_at_0 = x(0);
    if (ihc_params.one_capacitor) {
      FPType ro = 1 / conduct_at_10;
      FPType c = ihc_params.tau1_out / ro;
      FPType ri = ihc_params.tau1_in / c;
      FPType saturation_output = 1 / ((2 * ro) + ri);
      FPType r0 = 1 / conduct_at_0;
      FPType current = 1 / (ri + r0);
      ihc_coeffs->cap1_voltage = 1 - (current * ri);
      ihc_coeffs->just_half_wave_rectify = false;
      ihc_coeffs->lpf_coeff =
          1 - std::exp(-1 / (ihc_params.tau_lpf * sample_rate));
      ihc_coeffs->out1_rate = ro / (ihc_params.tau1_out * sample_rate);
      ihc_coeffs->in1_rate = 1 / (ihc_params.tau1_in * sample_rate);
      ihc_coeffs->one_capacitor = ihc_params.one_capacitor;
      ihc_coeffs->output_gain = 1 / (saturation_output - current);
      ihc_coeffs->rest_output = current / (saturation_output - current);
      ihc_coeffs->rest_cap1 = ihc_coeffs->cap1_voltage;
    } else {
      FPType ro = 1 / conduct_at_10;
      FPType c2 = ihc_params.tau2_out / ro;
      FPType r2 = ihc_params.tau2_in / c2;
      FPType c1 = ihc_params.tau1_out / r2;
      FPType r1 = ihc_params.tau1_in / c1;
      FPType saturation_output = 1 / (2 * ro + r2 + r1);
      FPType r0 = 1 / conduct_at_0;
      FPType current = 1 / (r1 + r2 + r0);
      ihc_coeffs->cap1_voltage = 1 - (current * r1);
      ihc_coeffs->cap2_voltage = ihc_coeffs->cap1_voltage - (current * r2);
      ihc_coeffs->just_half_wave_rectify = false;
      ihc_coeffs->lpf_coeff =
          1 - std::exp(-1 / (ihc_params.tau_lpf * sample_rate));
      ihc_coeffs->out1_rate = 1 / (ihc_params.tau1_out * sample_rate);
      ihc_coeffs->in1_rate = 1 / (ihc_params.tau1_in * sample_rate);
      ihc_coeffs->out2_rate = ro / (ihc_params.tau2_out * sample_rate);
      ihc_coeffs->in2_rate = 1 / (ihc_params.tau2_in * sample_rate);
      ihc_coeffs->one_capacitor = ihc_params.one_capacitor;
      ihc_coeffs->output_gain = 1 / (saturation_output - current);
      ihc_coeffs->rest_output = current / (saturation_output - current);
      ihc_coeffs->rest_cap1 = ihc_coeffs->cap1_voltage;
      ihc_coeffs->rest_cap2 = ihc_coeffs->cap2_voltage;
    }
  }
  ihc_coeffs->ac_coeff = 2 * M_PI * ihc_params.ac_corner_hz / sample_rate;
}

void CARFAC::DesignAGCCoeffs(const AGCParams& agc_params, FPType sample_rate,
                             std::vector<AGCCoeffs>* agc_coeffs) {
  agc_coeffs->resize(agc_params.num_stages);
  FPType previous_stage_gain = 0.0;
  FPType decim = 1.0;
  for (int stage = 0; stage < agc_params.num_stages; ++stage) {
    AGCCoeffs& agc_coeff = agc_coeffs->at(stage);
    agc_coeff.agc_stage_gain = agc_params.agc_stage_gain;
    std::vector<FPType> agc1_scales = agc_params.agc1_scales;
    std::vector<FPType> agc2_scales = agc_params.agc2_scales;
    std::vector<FPType> time_constants = agc_params.time_constants;
    FPType mix_coeff = agc_params.agc_mix_coeff;
    agc_coeff.decimation = agc_params.decimation[stage];
    FPType total_dc_gain = previous_stage_gain;
    // Calculate the parameters for the current stage.
    FPType tau = time_constants[stage];
    agc_coeff.decim = decim;
    agc_coeff.decim *= agc_coeff.decimation;
    agc_coeff.agc_epsilon =
        1 - std::exp((-1 * agc_coeff.decim) / (tau * sample_rate));
    FPType n_times = tau * (sample_rate / agc_coeff.decim);
    FPType delay = (agc2_scales[stage] - agc1_scales[stage]) / n_times;
    FPType spread_sq =
        (std::pow(agc1_scales[stage], 2) + std::pow(agc2_scales[stage], 2)) /
        n_times;
    FPType u = 1 + (1 / spread_sq);
    FPType p = u - std::sqrt(std::pow(u, 2) - 1);
    FPType dp = delay * (1 - (2 * p) + (p*p)) / 2;
    agc_coeff.agc_pole_z1 = p - dp;
    agc_coeff.agc_pole_z2 = p + dp;
    int n_taps = 0;
    bool done = false;
    int n_iterations = 1;  // The typical case in practice.
    // Initialize the FIR coefficient settings at each stage.
    FPType fir_left = 0;
    FPType fir_mid = 1;
    FPType fir_right = 0;
    if (spread_sq == 0) {
      // Special case for no spatial spreading.
      n_iterations = 0;
      n_taps = 3;  // Use a valid n_taps even if we're doing 0 iterations.
      done = true;
    }
    while (!done) {
      switch (n_taps) {
        case 0:
          // First time through, try to use the simple 3-point smoother.
          n_taps = 3;
          break;
        case 3:
          // Second time, increase n_taps before increasing n_iterations.
          n_taps = 5;
          break;
        case 5:
          // Subsequent times, try more iterations of 5-point FIR.
          n_iterations++;
          // TODO(dicklyon): parameterize the 4.
          if (n_iterations > 4) {
            // Even with SIMD ops, FIR smoothing takes time, so more than a
            // few iterations makes it probably slower than the two-pass IIR
            // smoother.
            n_iterations = -1;  // Signal to use IIR instead.
            done = true;
          }
          break;
        default:
          CARFAC_ASSERT(true && "Bad n_taps; should be 3 or 5.");
          break;
      }
      if (done) break;
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
          done = fir_mid >= 0.25;
          break;
        case 5:
          a = (((delay_variance + (mean_delay*mean_delay)) * 2.0/5.0) -
               (mean_delay * 2.0/3.0)) / 2.0;
          b = (((delay_variance + (mean_delay*mean_delay)) * 2.0/5.0) +
               (mean_delay * 2.0/3.0)) / 2.0;
          fir_left = a / 2.0;
          fir_mid = 1 - a - b;
          fir_right = b / 2.0;
          done = fir_mid >= 0.15;
          break;
        default:
          CARFAC_ASSERT(true && "Bad n_taps; should be 3 or 5.");
          break;
      }
    }
    // Once we have the FIR design for this stage we can assign it to the
    // appropriate data members.
    agc_coeff.agc_spatial_iterations = n_iterations;
    agc_coeff.agc_spatial_n_taps = n_taps;
    agc_coeff.agc_spatial_fir_left = fir_left;
    agc_coeff.agc_spatial_fir_mid = fir_mid;
    agc_coeff.agc_spatial_fir_right = fir_right;
    total_dc_gain += std::pow(agc_coeff.agc_stage_gain, stage);
    agc_coeff.agc_mix_coeffs = stage == 0 ? 0 : mix_coeff /
        (tau * (sample_rate / agc_coeff.decim));
    agc_coeff.agc_gain = total_dc_gain;
    agc_coeff.detect_scale = 1 / total_dc_gain;
    previous_stage_gain = agc_coeff.agc_gain;
    decim = agc_coeff.decim;
  }
}

CARFACOutput::CARFACOutput(bool store_nap, bool store_bm, bool store_ohc,
                           bool store_agc) {
  store_nap_ = store_nap;
  store_bm_ = store_bm;
  store_ohc_ = store_ohc;
  store_agc_ = store_agc;
}

namespace {
void ResizeContainer(int num_ears, int num_channels, int num_samples,
                     std::vector<ArrayXX>* container) {
  container->resize(num_ears);
  for (ArrayXX& matrix : *container) {
    matrix.resize(num_channels, num_samples);
  }
}
}  // namespace

void CARFACOutput::Resize(int num_ears, int num_channels, int num_samples) {
  if (store_nap_) {
    ResizeContainer(num_ears, num_channels, num_samples, &nap_);
  }
  if (store_bm_) {
    ResizeContainer(num_ears, num_channels, num_samples, &bm_);
  }
  if (store_ohc_) {
    ResizeContainer(num_ears, num_channels, num_samples, &ohc_);
  }
  if (store_agc_) {
    ResizeContainer(num_ears, num_channels, num_samples, &agc_);
  }
}

void CARFACOutput::AssignFromEars(const std::vector<Ear*>& ears,
                                  int sample_index) {
  for (int i = 0; i < ears.size(); ++i) {
    const Ear* ear = ears[i];
    if (store_nap_) {
      nap_[i].col(sample_index) = ear->ihc_out();
    }
    if (store_bm_) {
      bm_[i].col(sample_index) = ear->zy_memory();
    }
    if (store_ohc_) {
      ohc_[i].col(sample_index) = ear->za_memory();
    }
    if (store_agc_) {
      agc_[i].col(sample_index) = ear->zb_memory();
    }
  }
}
