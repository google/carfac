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

SAI::SAI(const SAIParams& params) : SAIBase(params) {
  // SAI::Reset() must be called here.  It may seem like Reset() is
  // already being called in the SAIBase constructor, but that's
  // actually a call to SAIBase::Reset(), which is not sufficient.
  Reset();
}

void SAI::Reset() {
  input_buffer_.setZero(params().num_channels, buffer_width());
  output_buffer_.setZero(params().num_channels, params().width);
}

void SAI::RunSegment(const ArrayXX& input_segment, ArrayXX* output_frame) {
  assert(input_segment.cols() == params().window_width &&
         "Unexpected number of input samples.");
  assert(input_segment.rows() == params().num_channels &&
         "Unexpected number of input channels.");

  ShiftAndAppendInput(input_segment, &input_buffer_);

  StabilizeSegment(input_buffer_, input_buffer_, &output_buffer_);
  *output_frame = output_buffer_;
}

namespace sai_internal {

SAIBase::SAIBase(const SAIParams& params) {
  Redesign(params);
}

void SAIBase::Redesign(const SAIParams& params) {
  params_ = params;
  assert(params_.window_width > params_.width &&
         "SAI window_width must be larger than width.");

  window_ =
      ArrayX::LinSpaced(params_.window_width, M_PI / params_.window_width, M_PI)
      .sin();
  Reset();
}

void SAIBase::StabilizeSegment(const ArrayXX& triggering_input_buffer,
                               const ArrayXX& nontriggering_input_buffer,
                               ArrayXX* output_buffer) const {
  assert(triggering_input_buffer.cols() == nontriggering_input_buffer.cols() &&
         "Number of columns must match.");
  assert(triggering_input_buffer.rows() == nontriggering_input_buffer.rows() &&
         "Number of rows must match.");

  // Windows are always approximately 50% overlapped.
  float window_hop = params_.window_width / 2;
  int window_start = (triggering_input_buffer.cols() - params_.window_width) -
      (params_.num_window_pos - 1) * window_hop;
  int window_range_start = window_start - params_.future_lags;
  int offset_range_start = 1 + window_start - params_.width;
  assert(offset_range_start > 0);
  for (int i = 0; i < params_.num_channels; ++i) {
    // TODO(kwwilson): Figure out if operating on rows is a
    // performance bottleneck when elements are noncontiguous.
    const ArrayX& triggering_nap_wave = triggering_input_buffer.row(i);
    const ArrayX& nontriggering_nap_wave = nontriggering_input_buffer.row(i);
    // TODO(ronw): Smooth row.

    for (int w = 0; w < params_.num_window_pos; ++w) {
      int current_window_offset = w * window_hop;
      // Choose a trigger point.
      int trigger_time;
      const ArrayX& trigger_window = triggering_nap_wave.segment(
          window_range_start + current_window_offset, params_.window_width);
      // TODO(kwwilson): If peak-finding is a performance bottleneck,
      // do something more SSE friendly.
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
      output_buffer->row(i) +=
          alpha * nontriggering_nap_wave.segment(
                      trigger_time + offset_range_start, params_.width);
    }
  }
}

}  // namespace sai_internal
