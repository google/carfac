// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Kevin Wilson <kwwilson@google.com>
//
// This file is part of an implementation of Lyon's cochlear model:
// "Cascade of Asymmetric Resonators with Fast-Acting Compression"
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

#include "binaural_sai.h"

BinauralSAI::BinauralSAI(const SAIParams& params) : SAIBase(params) {
  // BinauralSAI::Reset() must be called here.  It may seem like Reset() is
  // already being called in the SAIBase constructor, but that's
  // actually a call to SAIBase::Reset(), which is not sufficient.
  Reset();
}

void BinauralSAI::Reset() {
  input_buffer_.resize(2);
  output_buffer_.resize(2);

  for (int i = 0; i < 2; ++i) {
    input_buffer_[i].setZero(params().num_channels, buffer_width());
    output_buffer_[i].setZero(params().num_channels, params().sai_width);
  }
}

void BinauralSAI::RunSegment(const std::vector<ArrayXX>& input_segment,
                             std::vector<ArrayXX>* output_frame) {
  CARFAC_ASSERT(input_segment.size() == 2);
  for (int i = 0; i < 2; ++i) {
    ShiftAndAppendInput(input_segment[i], &input_buffer_[i]);
  }

  StabilizeSegment(input_buffer_[0], input_buffer_[1], &output_buffer_[0]);
  StabilizeSegment(input_buffer_[1], input_buffer_[0], &output_buffer_[1]);
  *output_frame = output_buffer_;
}
