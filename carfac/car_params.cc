//
//  car_params.cc
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

#include "car_params.h"

CARParams::CARParams() {
  velocity_scale_ = 0.1;
  v_offset_ = 0.04;
  min_zeta_ = 0.1;
  max_zeta_ = 0.35;
  first_pole_theta_ = 0.85 * PI;
  zero_ratio_ = 1.4142;
  high_f_damping_compression_ = 0.5;
  erb_per_step_ = 0.5;
  min_pole_hz_ = 30;
  erb_break_freq_ = 165.3;
  erb_q_ = 1000/(24.7*4.37);
};

CARParams::CARParams(FPType vs, FPType voff, FPType min_z, FPType max_z,
                          FPType fpt, FPType zr, FPType hfdc, FPType erbps,
                          FPType mph, FPType erbbf, FPType erbq) {  
  velocity_scale_ = vs;
  v_offset_ = voff;
  min_zeta_ = min_z;
  max_zeta_ = max_z;
  first_pole_theta_ = fpt;
  zero_ratio_ = zr;
  high_f_damping_compression_ = hfdc;
  erb_per_step_ = erbps;
  min_pole_hz_ = mph;
  erb_break_freq_ = erbbf;
  erb_q_ = erbq;
};