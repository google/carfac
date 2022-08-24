// Copyright 2022 The CARFAC Authors. All Rights Reserved.
// Author: Pascal Getreuer
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

#ifndef THIRD_PARTY_CARFAC_CPP_PITCHOGRAM_PIPELINE_H_
#define THIRD_PARTY_CARFAC_CPP_PITCHOGRAM_PIPELINE_H_

#include <cmath>
#include <memory>

#include "agc.h"
#include "car.h"
#include "carfac.h"
#include "common.h"
#include "ihc.h"
#include "image.h"
#include "pitchogram.h"
#include "sai.h"

struct PitchogramPipelineParams {
  // Number of frames plotted in the visualization.
  int num_frames;
  // The hop step, number of additional samples processed per SAI frame.
  int num_samples_per_segment;
  // Highest CARFAC pole frequency in Hz.
  float highest_pole_hz;
  // Longest lag computed in the pitchogram in seconds.
  float max_lag_s;
  // Number of trigger windows to consider when computing a single SAI frame.
  int num_triggers_per_frame;

  PitchogramParams pitchogram_params;

  PitchogramPipelineParams()
    : num_frames(400),
      num_samples_per_segment(256),
      highest_pole_hz(7000.0f),
      max_lag_s(0.05f),
      num_triggers_per_frame(2) {}
};

// Class that runs the full pipeline of CARFAC -> SAI -> Pitchogram computation
// to create a pitchogram visualization given a stream of audio samples.
class PitchogramPipeline {
 public:
  PitchogramPipeline(float sample_rate_hz,
                     const PitchogramPipelineParams& params);

  // Process audio samples in a streaming manner. `num_samples` should match
  // `num_samples_per_segment()`.
  void ProcessSamples(const float* samples, int num_samples);

  // Input audio sample rate in Hz.
  float sample_rate_hz() const { return sample_rate_hz_; }

  // CARFAC pole frequencies for each channel in Hz.
  const ArrayX& pole_frequencies() const { return carfac_->pole_frequencies(); }

  // Number of samples per segment for SAI computation.
  int num_samples_per_segment() const {
    return sai_params_.input_segment_width;
  }

  // CARFAC output for the current frame.
  const CARFACOutput& carfac_output() const { return *carfac_output_buffer_; }

  // Current SAI frame.
  const ArrayXX& sai_output() const { return sai_output_buffer_; }

  // Current pitchogram plot image.
  const Image<uint8_t>& image() const { return image_; }
  int width() const { return image_.width(); }
  int height() const { return image_.height(); }

  using VowelCoords = Pitchogram::VowelCoords;
  // Current vowel embedding coordinate.
  const VowelCoords& vowel_coords() const {
    return pitchogram_->vowel_coords();
  }

 private:
  enum { kNumEars = 1 };  // This class processes only monoaural input.

  float sample_rate_hz_;
  CARParams car_params_;
  IHCParams ihc_params_;
  AGCParams agc_params_;
  SAIParams sai_params_;
  PitchogramParams pitchogram_params_;
  std::unique_ptr<CARFAC> carfac_;
  std::unique_ptr<CARFACOutput> carfac_output_buffer_;
  std::unique_ptr<SAI> sai_;
  ArrayXX sai_output_buffer_;
  std::unique_ptr<Pitchogram> pitchogram_;
  Image<uint8_t> image_;
  Image<uint8_t> image_rightmost_col_;
};

#endif  // THIRD_PARTY_CARFAC_CPP_PITCHOGRAM_PIPELINE_H_
