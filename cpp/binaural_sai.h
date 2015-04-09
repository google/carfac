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

#ifndef THIRD_PARTY_CARFAC_CPP_BINAURAL_SAI_H_
#define THIRD_PARTY_CARFAC_CPP_BINAURAL_SAI_H_

#include <vector>

#include "common.h"
#include "sai.h"

// Top-level class implementing the binaural Stabilized Auditory Image.
//
// Repeated calls to the RunSegment menthod compute a sort of running
// cross-correlation of a pair of multi-channel input signals,
// typically segments of the neural activity pattern (NAP) outputs of
// CARFAC filterbanks.
class BinauralSAI : public sai_internal::SAIBase {
 public:
  explicit BinauralSAI(const SAIParams& params);

  // Resets the internal state.
  void Reset() override;

  // Fills each of two entries in output_frame with a
  // params().num_channels by params().width SAI frame computed from the
  // given input segments.  The first element in output_frame vector
  // uses the first element of the input as the trigger and the second
  // element as the scaled waveform.  The second element in
  // output_frame uses the second element of the input as the trigger
  // and the first element as the scaled waveform.
  //
  // The input_segment vector must have length two, with each element
  // of size params().num_channels by params().window_width.  Dies if
  // the size is incorrect.  Callers are responsible for zero-padding
  // as desired.
  //
  // Note that each element of the input is the transpose of the input
  // to the MATLAB implementation in SAI_Run_Segment.m.
  void RunSegment(const std::vector<ArrayXX>& input_segment,
                  std::vector<ArrayXX>* output_frame);

 private:
  // Buffer to store a large enough window of input frames to compute
  // a full SAI frame.  Two elements, each of size
  // params().num_channels by buffer_width().
  std::vector<ArrayXX> input_buffer_;
  // Output frame buffer.  Two elements, each of size
  // params().num_channels by params().width.
  std::vector<ArrayXX> output_buffer_;

  DISALLOW_COPY_AND_ASSIGN(BinauralSAI);
};

#endif  // THIRD_PARTY_CARFAC_CPP_BINAURAL_SAI_H_
