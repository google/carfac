//
//  agc_params.cc
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

#include "agc_params.h"

AGCParams::AGCParams(){
  n_stages_ = 4;
  agc_stage_gain_ = 2;
  FPType agc1f = 1.0;
  FPType agc2f = 1.65;
  time_constants_.resize(n_stages_);
  time_constants_ << 1*0.002, 4*0.002, 16*0.002, 64*0.002;
  decimation_.resize(n_stages_);
  decimation_ << 8, 2, 2, 2;
  agc1_scales_.resize(n_stages_);
  agc1_scales_ << 1.0 * agc1f, 1.4 * agc1f, 2.0 * agc1f, 2.8 * agc1f;
  agc2_scales_.resize(n_stages_);
  agc2_scales_ << 1.0 * agc2f, 1.4 * agc2f, 2.0 * agc2f, 2.8 * agc2f;
  agc_mix_coeff_ = 0.5;
}

void AGCParams::SetParams(int ns, FPType agcsg, FPType agcmc, FloatArray tc,
                          FloatArray dec, FloatArray agc1sc, FloatArray agc2sc){
  n_stages_ = ns;
  agc_stage_gain_ = agcsg;
  agc_mix_coeff_ = agcmc;
  time_constants_ = tc;
  decimation_ = dec;
  agc1_scales_ = agc1sc;
  agc2_scales_ = agc2sc;
}

void AGCParams::OutputParams(){
  std::cout << "AGCParams Values" << std::endl;
  std::cout << "****************" << std::endl;
  std::cout << "n_stages_ = " << n_stages_ << std::endl;
  std::cout << "agc_stage_gain_ = " << agc_stage_gain_ << std::endl;
  std::cout << "agc_mix_coeff_ = " << agc_mix_coeff_ << std::endl;
  std::cout << "time_constants_ = " << time_constants_ << std::endl;
  std::cout << "decimation_ = " << decimation_ << std::endl;
  std::cout << "agc1_scales_ = " << agc1_scales_ << std::endl;
  std::cout << "agc2_scales_ = " << agc2_scales_ << std::endl << std::endl;
}