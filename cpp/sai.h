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

// Design parameters for an SAI object.
//
// Terminology: Each call to SAI::RunSegment consumes a fixed-length
// input "segment" and outputs a single output SAI "frame".
struct SAIParams {
  // Number of channels (height, or number of rows) of an SAI frame.
  int num_channels;

  // TODO(ronw): Consider parameterizing this as past_lags and
  // future_lags, with width == past_lags + 1 + future_lags.

  // Total width (i.e. number of lag samples, or columns) of an SAI frame.
  int sai_width;
  // Number of lag samples that should come from the future.
  int future_lags;

  // Trigger settings.
  //
  // Each SAI frame computed by a call to SAI::RunSegment blends together
  // several 50% overlapping "trigger windows" identified in the input
  // buffer.  The size of the buffer (i.e. the total number of samples used
  // to generate the SAI) is controlled by the number and size of the
  // trigger windows.  See buffer_width() below for details.

  // Number of trigger windows to consider when computing a single SAI
  // frame during each call to RunSegment.
  int num_triggers_per_frame;

  // Size in samples of the window used when searching for each trigger
  // point.
  //
  // TODO(ronw): Consider parameterizing this in terms of the number of
  // samples consumed per call to SAI::RunSegment ("output_window_width" or
  // "buffer_width"?) and derive trigger_window_width from that parameter
  // and num_triggers_per_frame.
  int trigger_window_width;

  // Expected size in samples of the input segments that will be passed
  // into RunSegment.  This is only used to validate the input size.
  //
  // Since each call to RunSegment generates exactly one output SAI frame,
  // this parameter implicitly controls the output frame rate and the hop
  // size (i.e. number of new input samples consumed) between adjacent SAI
  // frames.  See ShiftAndAppendInput() below for details.
  int input_segment_width;

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
  // Redesign calls Reset().  Subclasses should do any subclass-specific
  // redesign inside Reset().
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
    // The buffer must be large enough to store num_triggers_per_frame
    // 50% overlapping trigger windows, with enough additional overhang
    // to be able to fill a full sai_width frame if a trigger point is
    // found at the edge of a trigger window.
    //
    // TODO(ronw): Doesn't that mean that we should be able to
    // s/.sai_width/.future_lags/ below?  Unfortunately this interacts with
    // the window_range and offset_range variables in StabilizeSegment and
    // I haven't been able to figure it out.  Better understand the
    // stabilization code and resolve this.
    return params().sai_width +
           static_cast<int>(
               (1 +
                static_cast<float>(params().num_triggers_per_frame - 1) / 2) *
               params().trigger_window_width);
  }

  // Shifts and appends new data to an input buffer.
  void ShiftAndAppendInput(const ArrayXX& fresh_input_segment,
                           ArrayXX* input_buffer) const {
    CARFAC_ASSERT(fresh_input_segment.rows() == params().num_channels &&
                  "Unexpected number of input channels.");

    const int input_width = fresh_input_segment.cols();
    const int overlap_width = buffer_width() - input_width;
    if (overlap_width > 0) {
      input_buffer->leftCols(overlap_width) =
          input_buffer->rightCols(overlap_width);
      input_buffer->rightCols(input_width) = fresh_input_segment;
    } else {
      // No overlap with old buffer contents; ignore extra input.
      *input_buffer = fresh_input_segment.rightCols(buffer_width());
    }
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

// Class implementing the Stabilized Auditory Image.
//
// Repeated calls to the RunSegment method compute a sparse approximation
// to the running autocorrelation of a multi-channel input signal,
// typically a segment of the neural activity pattern (NAP) outputs of the
// CARFAC filterbank.
class SAI : public sai_internal::SAIBase {
 public:
  explicit SAI(const SAIParams& params);

  // Resets the internal state.
  void Reset() override;

  // Fills output_frame with a params().num_channels by params().sai_width
  // SAI frame computed from the given input segment and the contents of
  // input_buffer_.
  //
  // The input_segment must have size of params_.num_channels by
  // params_.input_segment_width.  Dies if the size is incorrect.  Callers
  // are responsible for zero-padding as desired.
  //
  // Note that the input is the transpose of the input to SAI_Run_Segment.m.
  void RunSegment(const ArrayXX& input_segment, ArrayXX* output_frame);

  // Alternate interface allows inputs of any width and output at any time.
  void RunInput(const ArrayXX& input_segment);
  void GetOutput(ArrayXX* output_frame);

 private:
  // Buffer to store a large enough window of input frames to compute
  // a full SAI frame.  Size: params().num_channels by buffer_width().
  ArrayXX input_buffer_;
  // Output frame buffer.  Size: params().num_channels by params().sai_width.
  ArrayXX output_buffer_;

  DISALLOW_COPY_AND_ASSIGN(SAI);
};

#endif  // CARFAC_SAI_H_
