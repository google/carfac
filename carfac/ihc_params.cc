//
//  ihc_params.cc
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

#include "ihc_params.h"


//Default constructor for IHCParams initializes with the settings from Lyon's
//book 'Human and Machine Hearing'
IHCParams::IHCParams(){
  just_hwr_ = false; //not just a simple HWR
  one_cap_ = true; //uses the Allen model, as in Lyon's book
  tau_lpf_ = 0.000080; //80 microseconds smoothing twice
  tau1_out_ = 0.0005; //depletion tau is pretty fast
  tau1_in_ = 0.010; //recovery tau is slower
  tau2_out_ = 0.0025;
  tau2_in_ = 0.005;
  ac_corner_hz_ = 20;
}

//OutputParams method uses <iostream> for debugging purposes, could go in the
//final version
void IHCParams::OutputParams(){
  std::cout << "IHCParams Values" << std::endl;
  std::cout << "****************" << std::endl;
  std::cout << "just_hwr_ = " << just_hwr_ << std::endl;
  std::cout << "one_cap_ = " << one_cap_ << std::endl;
  std::cout << "tau_lpf_ = " << tau_lpf_ << std::endl;
  std::cout << "tau1_out_ = " << tau1_out_ << std::endl;
  std::cout << "tau1_in_ = " << tau1_in_ << std::endl;
  std::cout << "tau2_out_ = " << tau2_out_ << std::endl;
  std::cout << "tau2_in_ = " << tau2_in_ << std::endl;
  std::cout << "ac_corner_hz_ = " << ac_corner_hz_ << std::endl << std::endl;
}

//SetParams method allows for use of different inner hair cell parameters
void IHCParams::SetParams(bool jh, bool oc, FPType tlpf, FPType t1out,
                          FPType t1in, FPType t2out, FPType t2in, FPType acchz){
  just_hwr_ = jh;
  one_cap_ = oc;
  tau_lpf_ = tlpf;
  tau1_out_ = t1out;
  tau1_in_ = t1in;
  tau2_out_ = t2out;
  tau2_in_ = t2in;
  ac_corner_hz_ = acchz;  
}