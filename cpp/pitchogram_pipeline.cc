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

#include "pitchogram_pipeline.h"

#include <algorithm>
#include <cmath>

#include "color.h"
#include "common.h"

PitchogramPipeline::PitchogramPipeline(float sample_rate_hz,
                                       const PitchogramPipelineParams& params) {
  CARFAC_ASSERT(sample_rate_hz > 0.0f && "sample_rate_hz must be positive.");
  sample_rate_hz_ = sample_rate_hz;
  pitchogram_params_ = params.pitchogram_params;

  // Initialize CARFAC.
  car_params_.first_pole_theta =
      2 * M_PI * params.highest_pole_hz / sample_rate_hz_;
  carfac_.reset(new CARFAC(kNumEars, sample_rate_hz_, car_params_, ihc_params_,
                           agc_params_));
  carfac_output_buffer_.reset(new CARFACOutput(true, false, false, false));

  // Initialize SAI computation.
  sai_params_.num_channels = carfac_->num_channels();
  sai_params_.sai_width =
      static_cast<int>(std::round(params.max_lag_s * sample_rate_hz_));
  sai_params_.input_segment_width = params.num_samples_per_segment;
  sai_params_.trigger_window_width = sai_params_.input_segment_width + 1;
  sai_params_.future_lags = sai_params_.sai_width - 1;
  sai_params_.num_triggers_per_frame = params.num_triggers_per_frame;
  sai_.reset(new SAI(sai_params_));
  sai_output_buffer_.resize(sai_params_.num_channels, sai_params_.sai_width);

  // Initialize pitchogram computation.
  pitchogram_.reset(new Pitchogram(sample_rate_hz_, car_params_, sai_params_,
                                   pitchogram_params_));
  image_ = Image<uint8_t>(params.num_frames, pitchogram_->num_lags(), 4);
  image_rightmost_col_ = image_.col(params.num_frames - 1);

  // Initialize image background color according to theme.
  Color<uint8_t> background_rgb = (params.pitchogram_params.light_color_theme)
                                      ? Color<uint8_t>::Gray(255)
                                      : Color<uint8_t>::Gray(0);
  const uint8_t background_rgba[4] = {background_rgb[0], background_rgb[1],
                                      background_rgb[2], 255};
  uint32_t background;
  std::memcpy(&background, background_rgba, 4);
  uint32_t* data = reinterpret_cast<uint32_t*>(image_.data());
  std::fill(data, data + image_.num_pixels(), background);
}

void PitchogramPipeline::ProcessSamples(const float* samples, int num_samples) {
  auto input_map = ArrayXX::Map(samples, kNumEars, num_samples / kNumEars);
  carfac_->RunSegment(input_map, false /* open_loop */,
                      carfac_output_buffer_.get());
  sai_->RunSegment(carfac_output_buffer_->nap()[0], &sai_output_buffer_);

  // Compute the next pitchogram frame and 2D vowel embedding.
  pitchogram_->RunFrame(sai_output_buffer_);
  pitchogram_->VowelEmbedding(carfac_output_buffer_->nap()[0]);

  // Scroll image content one pixel left.
  std::memmove(image_.data(), image_.data() + 4, 4 * (image_.num_pixels() - 1));

  // Write the new frame to the rightmost column.
  pitchogram_->DrawColumn(image_rightmost_col_);
}
