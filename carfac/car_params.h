//
//  car_params.h
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

#ifndef CARFAC_Open_Source_C__Library_CARParams_h
#define CARFAC_Open_Source_C__Library_CARParams_h

#include "carfac_common.h"

class CARParams {
public:
  FPType velocity_scale_; //for the velocity nonlinearity
  FPType v_offset_; //offset gives quadratic part
  FPType min_zeta_; //minimum damping factor in mid-freq channels
  FPType max_zeta_; //maximum damping factor in mid-freq channels
  FPType first_pole_theta_;
  FPType zero_ratio_; //how far zero is above pole
  FPType high_f_damping_compression_; //0 to 1 to compress theta
  FPType erb_per_step_; //assume G&M's ERB formula
  FPType min_pole_hz_;
  FPType erb_break_freq_; //Greenwood map's break frequency
  FPType erb_q_; //G&M's high-cf ratio
  
  CARParams(); //Constructor initializes default parameter values
  void OutputParams(); //Output current parameter values using std::cout
  void SetParams(FPType vs, FPType voff, FPType min_z, FPType max_z, FPType fpt,
                 FPType zr, FPType hfdc, FPType erbps, FPType mph, FPType erbbf,
                 FPType erbq); //Method to set non-default params
};


#endif
