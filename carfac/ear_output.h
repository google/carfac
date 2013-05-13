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
  int n_ch_;
  long n_timepoints_;
  FloatArray2d nap_;
  FloatArray2d nap_decim_;
  FloatArray2d ohc_;
  FloatArray2d agc_;
  FloatArray2d bm_;
  void InitOutput(int n_ch, long n_tp);
  void MergeOutput(EarOutput output, long start, long length);
};

#endif
