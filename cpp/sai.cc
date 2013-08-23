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

#include "sai.h"

#include <assert.h>
#include <math.h>

SAI::SAI(const SAIParams& params) : params_(params) {
  Redesign(params);
}

void SAI::Redesign(const SAIParams& params) {
  assert(params_.window_width > params_.width &&
         "SAI window_width must be larger than width.");

  int buffer_width = params_.width +
      static_cast<int>((1 + static_cast<float>(params_.num_window_pos - 1)/2) *
                       params_.window_width);
  input_buffer_.setZero(params_.num_channels, buffer_width);
  output_buffer_.setZero(params_.num_channels, params_.width);

  window_ =
      ArrayX::LinSpaced(params_.window_width, M_PI / params_.window_width, M_PI)
      .sin();
}

void SAI::Reset() {
  input_buffer_.setZero();
  output_buffer_.setZero();
}

void SAI::RunSegment(const ArrayXX& input_segment, ArrayXX* output_frame) {
  assert(input_segment.cols() == params_.window_width &&
         "Unexpected number of input samples.");
  assert(input_segment.rows() == params_.num_channels &&
         "Unexpected number of input channels.");

  // Shift and append new data to the input buffer.
  const int overlap_width = input_buffer_.cols() - params_.window_width;
  input_buffer_.leftCols(overlap_width) =
      input_buffer_.rightCols(overlap_width);//.eval();
  input_buffer_.rightCols(params_.window_width) = input_segment;

  StabilizeSegment(input_buffer_, &output_buffer_);
  *output_frame = output_buffer_;
}

void SAI::StabilizeSegment(const ArrayXX& input_buffer,
                           ArrayXX* output_buffer) const {
  // Windows are always approximately 50% overlapped.
  float window_hop = params_.window_width / 2;
  int window_start = (input_buffer.cols() - params_.window_width) -
      (params_.num_window_pos - 1) * window_hop;
  int window_range_start = window_start - params_.future_lags;
  int offset_range_start = 1 + window_start - params_.width;
  assert(offset_range_start > 0);
  for (int i = 0; i < params_.num_channels; ++i) {
    const ArrayX& nap_wave = input_buffer.row(i);
    // TODO(ronw): Smooth row.

    for (int w = 0; w < params_.num_window_pos; ++w) {
      int current_window_offset = w * window_hop;
      // Choose a trigger point.
      int trigger_time;
      const ArrayX& trigger_window =
          nap_wave.segment(window_range_start + current_window_offset,
                           params_.window_width);
      FPType peak_val = (trigger_window * window_).maxCoeff(&trigger_time);
      if (peak_val <= 0) {
        peak_val = window_.maxCoeff(&trigger_time);
      }
      trigger_time += current_window_offset;

      // Blend the window following the trigger into the output
      // buffer, weighted according to the the trigger strength (0.05
      // to near 1.0).
      FPType alpha = (0.025 + peak_val) / (0.5 + peak_val);
      output_buffer->row(i) *= 1 - alpha;
      output_buffer->row(i) += alpha *
          nap_wave.segment(trigger_time + offset_range_start, params_.width);
    }
  }
}
