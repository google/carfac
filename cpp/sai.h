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

  // Total width (i.e. number of lag samples) of the SAI.
  int width;
  // Number of lag samples that should come from the future.
  int future_lags;
  // Number of input windows (triggers) to consider when computing each SAI
  // frame.
  int num_window_pos;

  // Size of the input window in samples.
  // TODO(kwwilson/ronw): Decouple input segment size from trigger
  // weighting window size.
  int window_width;

  FPType channel_smoothing_scale;
};

// Internal implementation of some common SAI functionality.
namespace sai_internal {

// Base class from which the monaural and binaural Stabilized Auditory
// Image implementations inherit.
class SAIBase {
 public:
  explicit SAIBase(const SAIParams& params);
  virtual ~SAIBase() {}

  // Reinitializes using the specified parameters.
  //
  // Redesign calls Reset().  Subclasses should do any
  // subclass-specific redesign inside Reset().
  void Redesign(const SAIParams& params);

  // Resets the internal state.
  virtual void Reset() {}

 protected:
  // Chooses trigger points and blends windowed signals into
  // output_buffer.  triggering_input_buffer and
  // nontriggering_input_buffer must be the same shape.
  void StabilizeSegment(const ArrayXX& triggering_input_buffer,
                        const ArrayXX& nontriggering_input_buffer,
                        ArrayXX* output_buffer) const;

  int buffer_width() const {
    return params().width +
        static_cast<int>(
            (1 + static_cast<float>(params().num_window_pos - 1) / 2) *
            params().window_width);
  }

  // Shift and append new data to an input buffer.
  void ShiftAndAppendInput(const ArrayXX& fresh_input_segment,
                           ArrayXX* input_buffer) const {
    CARFAC_ASSERT(fresh_input_segment.cols() == params().window_width &&
                  "Unexpected number of input samples.");
    CARFAC_ASSERT(fresh_input_segment.rows() == params().num_channels &&
                  "Unexpected number of input channels.");

    const int overlap_width = buffer_width() - params().window_width;
    input_buffer->leftCols(overlap_width) =
        input_buffer->rightCols(overlap_width);
    input_buffer->rightCols(params().window_width) = fresh_input_segment;
  }

  const SAIParams& params() const { return params_; }

 private:
  SAIParams params_;
  // Window function to apply before selecting a trigger point.
  // Size: params_.window_width.
  ArrayX window_;

  DISALLOW_COPY_AND_ASSIGN(SAIBase);
};

}  // namespace sai_internal

// Top-level class implementing the Stabilized Auditory Image.
//
// Repeated calls to the RunSegment menthod compute a sort of running
// autocorrelation of a multi-channel input signal, typically a segment of the
// neural activity pattern (NAP) outputs of the CARFAC filterbank.
class SAI : public sai_internal::SAIBase {
 public:
  explicit SAI(const SAIParams& params);

  // Resets the internal state.
  virtual void Reset() override;

  // Fills output_frame with a params().num_channels by params().width SAI frame
  // computed from the given input segment.
  //
  // The input_segment must have size of params_.num_channels by
  // params_.window_width.  Dies if the size is incorrect.  Callers are
  // responsible for zero-padding as desired.
  //
  // Note that the input is the transpose of the input to SAI_Run_Segment.m.
  void RunSegment(const ArrayXX& input_segment, ArrayXX* output_frame);

 private:
  // Buffer to store a large enough window of input frames to compute
  // a full SAI frame.  Size: params().num_channels by buffer_width().
  ArrayXX input_buffer_;
  // Output frame buffer.  Size: params().num_channels by params().width.
  ArrayXX output_buffer_;

  DISALLOW_COPY_AND_ASSIGN(SAI);
};

#endif  // CARFAC_SAI_H_
