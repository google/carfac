// Copyright 2013, 2022 The CARFAC Authors. All Rights Reserved.
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

#ifndef CARFAC_CAR_H
#define CARFAC_CAR_H

#include <cmath>

#include "common.h"

// A CARParams structure stores the necessary information needed by a CARFAC
// object to design the set of CARCoeffs implementing 'The Cascade of
// Asymmetric Resonators'.
struct CARParams {
  CARParams() {
    velocity_scale = 0.1;
    v_offset = 0.04;
    min_zeta = 0.1;
    max_zeta = 0.35;
    first_pole_theta = 0.85 * M_PI;
    zero_ratio = sqrt(2.0);
    high_f_damping_compression = 0.5;
    erb_per_step = 0.5;
    min_pole_hz = 30;
    erb_break_freq = 165.3;  // The Greenwood map's break frequency in Hertz.
    // Glassberg and Moore's high-cf ratio.
    erb_q = 1000 / (24.7 * 4.37);
  }

  FPType velocity_scale;  // Used for the velocity nonlinearity.
  FPType v_offset;  // The offset gives us quadratic part.
  FPType min_zeta;  // The minimum damping factor in mid-freq channels.
  FPType max_zeta;  // The maximum damping factor in mid-freq channels.
  FPType first_pole_theta;
  FPType zero_ratio;  // This is how far zero is above the pole.
  FPType high_f_damping_compression;  // A range from 0 to 1 to compress theta.
  FPType erb_per_step;
  FPType min_pole_hz;
  FPType erb_break_freq;
  FPType erb_q;
};

// CAR filter coefficients, which are derived from a set of CARParams.
struct CARCoeffs {
  FPType velocity_scale;
  FPType v_offset;
  ArrayX r1_coeffs;
  ArrayX a0_coeffs;
  ArrayX c0_coeffs;
  ArrayX h_coeffs;
  ArrayX g0_coeffs;
  ArrayX zr_coeffs;
};


// CAR filter state.
struct CARState {
  ArrayX z1_memory;
  ArrayX z2_memory;
  ArrayX za_memory;
  ArrayX zb_memory;
  ArrayX dzb_memory;
  ArrayX zy_memory;
  ArrayX g_memory;
  ArrayX dg_memory;
};

// Computes CAR pole frequency in Hz for each channel.
ArrayX CARPoleFrequencies(FPType sample_rate_hz, const CARParams& car_params);

// Computes CAR pole frequency in Hz as a continuous function of channel index.
FPType CARChannelIndexToFrequency(FPType sample_rate_hz,
                                  const CARParams& car_params,
                                  FPType channel_index);

// Computes CAR channel index as a continuous function of pole frequency.
FPType CARFrequencyToChannelIndex(FPType sample_rate_hz,
                                  const CARParams& car_params,
                                  FPType pole_freq);

// Computes the nominal Equivalent Rectangular Bandwidth (ERB) of an auditory
// filter at the given center frequency.
// Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
FPType ERBHz(FPType center_frequency_hz, FPType erb_break_freq, FPType erb_q);

#endif  // CARFAC_CAR_H
