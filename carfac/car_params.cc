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

CARParams::CARParams(){
  velocity_scale_ = 0.1; //for the velocity nonlinearity
  v_offset_ = 0.04; //offset gives quadratic part
  min_zeta_ = 0.1; //minimum damping factor in mid-freq channels
  max_zeta_ = 0.35; //maximum damping factor in mid-freq channels
  first_pole_theta_ = 0.85 * PI;
  zero_ratio_ = 1.4142; //how far zero is above pole
  high_f_damping_compression_ = 0.5; //0 to 1 to compress theta
  erb_per_step_ = 0.5; //assume G&M's ERB formula
  min_pole_hz_ = 30;
  erb_break_freq_ = 165.3; //Greenwood map's break frequency
  erb_q_ = 1000/(24.7*4.37);
};

//This method has been created for debugging purposes and depends on <iostream>.
//Could possibly be removed in the final version to reduce dependencies.
void CARParams::OutputParams(){
  std::cout << "CARParams Values" << std::endl;
  std::cout << "****************" << std::endl;
  std::cout << "velocity_scale_ = " << velocity_scale_ << std::endl;
  std::cout << "v_offset_ = " << v_offset_ << std::endl;
  std::cout << "min_zeta_ = " << min_zeta_ << std::endl;
  std::cout << "max_zeta_ = " << max_zeta_ << std::endl;
  std::cout << "first_pole_theta_ = " << first_pole_theta_ << std::endl;
  std::cout << "zero_ratio_ = " << zero_ratio_ << std::endl;
  std::cout << "high_f_damping_compression_ = " << high_f_damping_compression_
            << std::endl;
  std::cout << "erb_per_step_ = " << erb_per_step_ << std::endl;
  std::cout << "min_pole_hz_ = " << min_pole_hz_ << std::endl;
  std::cout << "erb_break_freq_ = " << erb_break_freq_ << std::endl;
  std::cout << "erb_q_ = " << erb_q_ << std::endl << std::endl;
}

void CARParams::SetParams(FPType vs, FPType voff, FPType min_z, FPType max_z,
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