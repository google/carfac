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

#include "pitchogram.h"

#include <cmath>

#include "color.h"

Pitchogram::Pitchogram(FPType sample_rate, const CARParams& car_params,
                       const SAIParams& sai_params,
                       const PitchogramParams& pitchogram_params) {
  Redesign(sample_rate, car_params, sai_params, pitchogram_params);
}

void Pitchogram::Redesign(FPType sample_rate, const CARParams& car_params,
                          const SAIParams& sai_params,
                          const PitchogramParams& pitchogram_params) {
  pitchogram_params_ = pitchogram_params;

  ArrayX pole_frequencies = CARPoleFrequencies(sample_rate, car_params);

  // Create a binary mask to suppress the SAI's zero-lag peak. For the cth row,
  // we get the cth pole frequency and mask lags within half a cycle of zero.
  mask_ = ArrayXX::Ones(sai_params.num_channels, sai_params.sai_width);
  const int center = sai_params.sai_width - sai_params.future_lags;
  for (int c = 0; c < pole_frequencies.size(); ++c) {
    const float half_cycle_samples = 0.5f * sample_rate / pole_frequencies[c];
    const int i_start = std::min(std::max(static_cast<int>(std::floor(
      center - half_cycle_samples)), 0), sai_params.sai_width - 1);
    const int i_end = std::min(std::max(static_cast<int>(std::ceil(
      center + half_cycle_samples)), 0), sai_params.sai_width - 1);
    mask_.row(c).segment(i_start, i_end - i_start + 1).setZero();
  }

  // Create vowel embedding matrix. The rows of the matrix are simple extractors
  // for the F1 and F2 formants, each based on a difference of two Gaussians.
  vowel_matrix_.resize(2, sai_params.num_channels);
  auto kernel = [](FPType center, int c) {
    const FPType z = (c - center) / FPType(3.3);
    return std::exp((z * z) / -2);
  };
  FPType f2_hi = CARFrequencyToChannelIndex(sample_rate, car_params, 2365);
  FPType f2_lo = CARFrequencyToChannelIndex(sample_rate, car_params, 1100);
  FPType f1_hi = CARFrequencyToChannelIndex(sample_rate, car_params, 700);
  FPType f1_lo = CARFrequencyToChannelIndex(sample_rate, car_params, 265);
  for (int c = 0; c < pole_frequencies.size(); ++c) {
    vowel_matrix_(0, c) = kernel(f2_lo, c) - kernel(f2_hi, c);
    vowel_matrix_(1, c) = kernel(f1_lo, c) - kernel(f1_hi, c);
  }
  vowel_matrix_ *= car_params.erb_per_step / 2;
  const FPType frame_rate_hz = sample_rate / sai_params.input_segment_width;
  cgram_smoother_ = 1 - std::exp(
      -1 / (pitchogram_params_.vowel_time_constant_s * frame_rate_hz));
  cgram_.resize(pole_frequencies.size());

  log_lag_cells_.clear();
  if (!pitchogram_params_.log_lag) {
    output_.resize(sai_params.sai_width);
  } else {
    // If log_lag is true, set up ResamplingCells to warp the pitchogram to a
    // log axis. The ith cell covers the interval
    //
    //   (min_lag_s + log_offset_s) * spacing^i - log_offset_s
    //   <= lag
    //   <= (min_lag_s + log_offset_s) * spacing^(i + 1) - log_offset_s
    //
    // with spacing = 2^(1/lags_per_octave) such that
    //
    //   i / lags_per_octave
    //   <= log2(lag + log_offset_s) - log2(min_lag_s + log_offset_s)
    //   <= (i + 1) / lags_per_octave.
    //
    // We iteratively construct cells until we exceed sai_width.
    const double spacing = std::exp2(1.0 / pitchogram_params_.lags_per_octave);
    const double log_offset = sample_rate * pitchogram_params_.log_offset_s;
    double left_edge = sample_rate * pitchogram_params_.min_lag_s;

    while (true) {
      const double right_edge = (left_edge + log_offset) * spacing - log_offset;
      ResamplingCell cell(left_edge, right_edge);
      if (cell.right_index >= sai_params.sai_width) { break; }
      log_lag_cells_.push_back(cell);
      left_edge = right_edge;
    }

    workspace_.resize(sai_params.sai_width);
    output_.resize(log_lag_cells_.size());
  }

  Reset();
}

void Pitchogram::Reset() {
  output_.setZero();
  vowel_coords_.setZero();
  cgram_.setZero();
}

const ArrayX& Pitchogram::RunFrame(const ArrayXX& sai_frame) {
  CARFAC_ASSERT(mask_.rows() == sai_frame.rows() &&
                mask_.cols() == sai_frame.cols() &&
                "SAI frame shape mismatches params.");
  if (!pitchogram_params_.log_lag) {
    output_ = (sai_frame * mask_).colwise().mean();
  } else {
    workspace_ = (sai_frame * mask_).colwise().mean();
    for (int i = 0; i < output_.size(); ++i) {
      output_[i] = log_lag_cells_[i].CellAverage(workspace_);
    }
  }
  return output_;
}

const Pitchogram::VowelCoords& Pitchogram::VowelEmbedding(const ArrayXX& nap) {
  cgram_ += cgram_smoother_ * (nap.rowwise().mean() - cgram_);
  vowel_coords_ = vowel_matrix_ * cgram_.matrix();
  return vowel_coords_;
}

void Pitchogram::DrawColumn(Image<uint8_t> dest) const {
  CARFAC_ASSERT(dest.height() == num_lags() &&
                "Image height mismatches num_lags.");
  CARFAC_ASSERT(dest.channels() >= 3 &&
                "Image must have at least 3 channels.");
  CARFAC_ASSERT(dest.c_stride() == 1 &&
                "Image must be interleaved with c_stride == 1.");
  // Convert the 2D embedding vector to an RGB tint color. The intention is a
  // cylindrical mapping like:
  //  * vowel_coords_ = (0, 0) maps close to gray.
  //  * Saturation increases with the norm of vowel_coords_.
  //  * Hue varies with the angle of vowel_coords_.
  //  * Brightness is approximately constant.
  Color<float> tint(0.5f - 0.6f * vowel_coords_[1],
                    0.5f - 0.6f * vowel_coords_[0],
                    0.35f * (vowel_coords_[0] + vowel_coords_[1]) + 0.4f);

  constexpr float kScale = 0.5f * 255;
  uint8_t* data = dest.data();

  if (pitchogram_params_.light_color_theme) {
    // TODO(dkanevsky): Fine-tune light theme coloring.
    tint *= kScale;  // Scale for 0-255 range.
    for (int y = 0; y < output_.size(); ++y, data += dest.y_stride()) {
      Color<uint8_t>::Map(data) =
          (255 - tint * output_[y]).max(0).min(255).cast<uint8_t>();
    }
  } else {
    tint *= kScale;  // Scale for 0-255 range.
    for (int y = 0; y < output_.size(); ++y, data += dest.y_stride()) {
      Color<uint8_t>::Map(data) =
          (tint * output_[y]).max(0).min(255).cast<uint8_t>();
    }
  }
}

Pitchogram::ResamplingCell::ResamplingCell(float left_edge, float right_edge) {
  float cell_width = right_edge - left_edge;

  if (cell_width < 1.0f) {  // Make the cell width at least one sample period.
    float grow = 0.5f * (1.0f - cell_width);
    left_edge -= grow;
    right_edge += grow;
  }

  left_edge = std::max<float>(0.0f, left_edge);
  right_edge = std::max<float>(0.0f, right_edge);
  cell_width = right_edge - left_edge;

  left_index = static_cast<int>(std::round(left_edge));
  right_index = static_cast<int>(std::round(right_edge));
  if (right_index > left_index && cell_width > 0.999f) {
    left_weight = (0.5f - (left_edge - left_index)) / cell_width;
    interior_weight = 1.0f / cell_width;
    right_weight = (0.5f + (right_edge - right_index)) / cell_width;
  } else {
    left_weight = 1.0f;
    interior_weight = 0.0f;
    right_weight = 0.0f;
  }
}

float Pitchogram::ResamplingCell::CellAverage(const ArrayX& samples) const {
  if (left_index == right_index) { return samples[left_index]; }
  return left_weight * samples[left_index] +
          interior_weight *
              samples.segment(left_index + 1, right_index - left_index - 1)
                  .sum() +
          right_weight * samples[right_index];
}
