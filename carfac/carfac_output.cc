//
//  carfac_output.cc
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

#include "carfac_output.h"

void CARFACOutput::InitOutput(int n_ears, int n_ch, long n_tp) {
  n_ears_ = n_ears;
  ears_ = new EarOutput[n_ears_];
  for (int i = 0; i < n_ears_; i++) {
    ears_[i].InitOutput(n_ch, n_tp);
  }
}

void CARFACOutput::MergeOutput(CARFACOutput output, long start, long length) {
  for (int i = 0; i < n_ears_; i++){
    ears_[i].MergeOutput(output.ears_[i], start, length);
  }
}