//
//  carfac_common.cc
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

#include "carfac_common.h"

//Auditory filter nominal Equivalent Rectangular Bandwidth
//Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
FPType ERBHz (FPType cf_hz, FPType erb_break_freq, FPType erb_q) {
  
  FPType erb;
  erb = (erb_break_freq + cf_hz) / erb_q;
  return erb;
}

//An IHC-like sigmoidal detection nonlinearity for the CARFAC.
//Resulting conductance is in about [0...1.3405]
FPType CARFACDetect (FPType x) {
  
  FPType conductance, z;
  FPType a = 0.175;
  //offset of low-end tail into neg x territory
  //this parameter is adjusted for the book, to make the 20% DC response
  //threshold at 0.1
  z  = x + a;
  conductance = pow(z,3) / (pow(z,3) + pow(z,2) + 0.1);
  //zero is the final answer for many points:
  return conductance;
}

FloatArray CARFACDetect (FloatArray x) {
  
  FloatArray conductance, z;
  FPType a = 0.175;
  //offset of low-end tail into neg x territory
  //this parameter is adjusted for the book, to make the 20% DC response
  //threshold at 0.1
  z  = x + a;
  conductance = (z * z * z) / ((z * z * z) + (z * z) + 0.1);
  //zero is the final answer for many points:
  return conductance;
}