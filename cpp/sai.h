// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Ron Weiss <ronw@google.com>
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

#ifndef CARFAC_SAI_H_
#define CARFAC_SAI_H_

#include "common.h"

// Design parameters for a single SAI.
struct SAIParams {
  // Number of channels (height) of the SAI.
  int num_channels;

  // TODO(ronw): Consider parameterizing this as past_lags and
  // future_lags, with width == past_lags + 1 + future_lags.
  //
  // Total width (i.e. number of lag samples) of the SAI.
  int width;
  // Number of lag samples that should come from the future.
  int future_lags;
  // Number of windows (triggers) to consider during each SAI frame.
  int num_window_pos;

  // TODO(ronw): more carefully define terms "window" and "frame"

  // Size of the window to compute.
  int window_width;

  FPType channel_smoothing_scale;
};

// Top-level class implementing the Stabilized Auditory Image.
//
// Repeated calls to the RunSegment menthod compute a sort of running
// autocorrelation of a multi-channel input signal, typically a segment of the
// neural activity pattern (NAP) outputs of the CARFAC filterbank.
class SAI {
 public:
  explicit SAI(const SAIParams& params);

  // Reinitializes using the specified parameters.
  void Redesign(const SAIParams& params);

  // Resets the internal state.
  void Reset();

  // Fills output_frame with a params_.num_channels by params_.width SAI frame
  // computed from the given input frames.
  //
  // The input should have size of params_.num_channels by params_.num_samples.
  // Inputs containing too few frames are zero-padded.
  // Note that the input is the transpose of the input to SAI_Run_Segment.m.
  void RunSegment(const ArrayXX& input, ArrayXX* output_frame);

 private:
  // Processes successive windows within input_buffer, chooses trigger
  // points, and blends each window into output_buffer.
  void StabilizeSegment(const ArrayXX& input_buffer,
                        ArrayXX* output_buffer) const;

  SAIParams params_;
  // Window function to apply before selecting a trigger point.
  // Size: params_.window_width.
  ArrayX window_;
  // Buffer to store a large enough window of input frames to compute
  // a full SAI frame.  Size: params_.num_channels by params_.buffer_width.
  ArrayXX input_buffer_;
  // Output frame buffer.  Size: params_.num_channels by params_.width.
  ArrayXX output_buffer_;
};

#endif  // CARFAC_SAI_H_
