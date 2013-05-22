//
//  ear_output.cc
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

#include "ear_output.h"

void EarOutput::InitOutput(int n_ch, int32_t n_timepoints) {
  n_ch_ = n_ch;
  n_timepoints_ = n_timepoints;
  nap_.resize(n_timepoints_);
  bm_.resize(n_timepoints_);
  ohc_.resize(n_timepoints_);
  agc_.resize(n_timepoints_);
  for (int32_t i = 0; i < n_timepoints; ++i) {
    nap_[i].resize(n_ch);
    bm_[i].resize(n_ch);
    ohc_[i].resize(n_ch);
    agc_[i].resize(n_ch);
  }
}

void EarOutput::StoreNAPOutput(const int32_t timepoint, const FloatArray& nap) {
  nap_[timepoint] = nap;
}

void EarOutput::StoreBMOutput(const int32_t timepoint, const FloatArray& bm) {
  bm_[timepoint] = bm;
}

void EarOutput::StoreOHCOutput(const int32_t timepoint, const FloatArray& ohc) {
  ohc_[timepoint] = ohc;
}

void EarOutput::StoreAGCOutput(const int32_t timepoint, const FloatArray& agc) {
  agc_[timepoint] = agc;
}