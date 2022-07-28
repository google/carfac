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

#include "car.h"

FPType ERBHz(FPType center_frequency_hz, FPType erb_break_freq, FPType erb_q) {
  return (erb_break_freq + center_frequency_hz) / erb_q;
}

ArrayX CARPoleFrequencies(FPType sample_rate_hz, const CARParams& car_params) {
  int num_channels = 0;
  FPType pole_hz = car_params.first_pole_theta * sample_rate_hz /
                   static_cast<FPType>(2.0 * M_PI);
  while (pole_hz > car_params.min_pole_hz) {
    ++num_channels;
    pole_hz -= car_params.erb_per_step *
        ERBHz(pole_hz, car_params.erb_break_freq, car_params.erb_q);
  }

  ArrayX pole_freqs(num_channels);
  pole_hz = car_params.first_pole_theta * sample_rate_hz / (2 * M_PI);
  for (int channel = 0; channel < num_channels; ++channel) {
    pole_freqs[channel] = pole_hz;
    pole_hz -= car_params.erb_per_step *
        ERBHz(pole_hz, car_params.erb_break_freq, car_params.erb_q);
  }

  return pole_freqs;
}

FPType CARChannelIndexToFrequency(FPType sample_rate_hz,
                                  const CARParams& car_params,
                                  FPType channel_index) {
  const FPType pole0_hz = car_params.first_pole_theta * sample_rate_hz /
                          static_cast<FPType>(2.0 * M_PI);
  const FPType break_freq = car_params.erb_break_freq;
  const FPType ratio = 1 - car_params.erb_per_step / car_params.erb_q;
  return (pole0_hz + break_freq) * std::pow(ratio, channel_index) - break_freq;
}

FPType CARFrequencyToChannelIndex(FPType sample_rate_hz,
                                  const CARParams& car_params,
                                  FPType pole_freq) {
  const FPType pole0_hz = car_params.first_pole_theta * sample_rate_hz /
                          static_cast<FPType>(2.0 * M_PI);
  const FPType break_freq = car_params.erb_break_freq;
  const FPType ratio = 1 - car_params.erb_per_step / car_params.erb_q;
  pole_freq = std::min(std::max(pole_freq, car_params.min_pole_hz), pole0_hz);
  return std::log((pole_freq + break_freq) / (pole0_hz + break_freq)) /
         std::log(ratio);
}
