//
//  ear_output.h
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

#ifndef CARFAC_Open_Source_C___Library_ear_output_h
#define CARFAC_Open_Source_C___Library_ear_output_h

#include "carfac_common.h"

class EarOutput {
 public:
  void InitOutput(int n_ch, int32_t n_timepoints);
  void StoreNAPOutput(const int32_t timepoint, const FloatArray& nap);
  void StoreBMOutput(const int32_t timepoint, const FloatArray& bm);
  void StoreOHCOutput(const int32_t timepoint, const FloatArray& ohc);
  void StoreAGCOutput(const int32_t timepoint, const FloatArray& agc);
  
 private:
  int n_ch_;
  int32_t n_timepoints_;
  std::vector<FloatArray> nap_;
  std::vector<FloatArray> nap_decim_;  // TODO (alexbrandmeyer): store nap_decim output.
  std::vector<FloatArray> ohc_;
  std::vector<FloatArray> agc_;
  std::vector<FloatArray> bm_;
};

#endif