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
  void InitOutput(int n_ch, int32_t n_tp);
  void MergeOutput(EarOutput output, int32_t start, int32_t length);
  void StoreNAPOutput(int32_t timepoint, int n_ch, FloatArray nap);
  void StoreBMOutput(int32_t timepoint, int n_ch, FloatArray bm);
  void StoreOHCOutput(int32_t timepoint, int n_ch, FloatArray ohc);
  void StoreAGCOutput(int32_t timepoint, int n_ch, FloatArray agc);
 private:
  int n_ch_;
  int32_t n_timepoints_;
  FloatArray2d nap_;
  FloatArray2d nap_decim_;
  FloatArray2d ohc_;
  FloatArray2d agc_;
  FloatArray2d bm_;
};

#endif