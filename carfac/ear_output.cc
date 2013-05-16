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

void EarOutput::InitOutput(int n_ch, long n_tp) {
  n_ch_ = n_ch;
  n_timepoints_ = n_tp;
  nap_.resize(n_ch_, n_timepoints_);
  bm_.resize(n_ch_, n_timepoints_);
  ohc_.resize(n_ch_, n_timepoints_);
  agc_.resize(n_ch_, n_timepoints_);
}

void EarOutput::MergeOutput(EarOutput ear_output, long start, long length) {
  nap_.block(0, start, n_ch_, length) = ear_output.nap_.block(0, 0, n_ch_,
                                                              length);
  bm_.block(0, start, n_ch_, length) = ear_output.bm_.block(0, 0, n_ch_,
                                                            length);
  ohc_.block(0, start, n_ch_, length) = ear_output.ohc_.block(0, 0, n_ch_,
                                                              length);
  agc_.block(0, start, n_ch_, length) = ear_output.agc_.block(0, 0, n_ch_,
                                                              length);
}