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

// Computation of pitchograms from stabilized auditory images.

#ifndef CARFAC_PITCHOGRAM_H_
#define CARFAC_PITCHOGRAM_H_

#include <vector>

#include "car.h"
#include "common.h"
#include "image.h"
#include "sai.h"

struct PitchogramParams {
  // If true, warp pitchogram lag to a log axis.
  // NOTE: When log_lag is true, the SAI must be computed with
  // sai_params.future_lags = sai_params.sai_width - 1.
  bool log_lag;
  // If log_lag is true, the number of lags to sample per octave.
  float lags_per_octave;
  // If log_lag is true, the min lag in seconds to sample. The max lag is
  // determined by the sai_width.
  float min_lag_s;
  // If log_lag is true, stabilizing offset added to the lag before taking log.
  float log_offset_s;

  // Time constant for smoothing the cgram used in vowel embedding.
  float vowel_time_constant_s;
  // If true, use a light color theme for rendering the pitchogram with white
  // background. If false, use a dark color theme.
  bool light_color_theme;

  PitchogramParams()
    : log_lag(true),
      lags_per_octave(36.0f),
      min_lag_s(0.0005f),
      log_offset_s(0.0025f),
      vowel_time_constant_s(0.02f),
      light_color_theme(false) {}
};

class Pitchogram {
 public:
  Pitchogram(FPType sample_rate, const CARParams& car_params,
             const SAIParams& sai_params,
             const PitchogramParams& pitchogram_params);
  Pitchogram(const Pitchogram&) = delete;
  Pitchogram& operator=(const Pitchogram&) = delete;

  // Number of lags in the pitchogram output.
  int num_lags() const { return output_.size(); }

  // Reinitializes using the specified parameters.
  void Redesign(FPType sample_rate, const CARParams& car_params,
                const SAIParams& sai_params,
                const PitchogramParams& pitchogram_params);

  // Resets to initial state.
  void Reset();

  // Runs the pitchogram on the given input SAI frame. Returns the next column
  // of the pitchogram, having num_lags() rows. To create a scrolling pitchogram
  // plot, the caller should stack the columns from successive RunFrame() calls.
  const ArrayX& RunFrame(const ArrayXX& sai_frame);

  using VowelCoords = Eigen::Matrix<FPType, 2, 1>;
  // Map the nap to a 2D coordinate in an embedding space that tends to
  // distinguish monophthong vowels.
  const VowelCoords& VowelEmbedding(const ArrayXX& nap);
  // Access the current vowel embedding coords.
  const VowelCoords& vowel_coords() const { return vowel_coords_; }

  // Draw one column of a scrolling pitchogram visualization with vowel coloring
  // into the x=0 column of image `dest`. The image is expected to have height
  // `num_lags`, at least 3 channels, and c_stride == 1.
  void DrawColumn(Image<uint8_t> dest) const;

 private:
  PitchogramParams pitchogram_params_;
  ArrayXX mask_;
  ArrayX workspace_;
  ArrayX output_;

  Eigen::Matrix<FPType, 2, Eigen::Dynamic> vowel_matrix_;
  VowelCoords vowel_coords_;
  ArrayX cgram_;
  FPType cgram_smoother_;

  // Resampling weights, used to resample the pitchogram when log_lag is true.
  // Conceptually, the linear lag pitchogram is considered as a piecewise
  // constant function "f(x) = samples[round(x)]." Each ResamplingCell
  // represents averaging f(x) over the interval [left_edge, right_edge] by a
  // computation of the form
  //
  //   CellAverage(samples) =
  //       left_weight * samples[left_index]
  //     + interior_weight * (samples[left_index + 1]
  //                          + ... + samples[right_index - 1])
  //     + right_weight * samples[right_index].
  struct ResamplingCell {
    int left_index;
    int right_index;
    float left_weight;
    float interior_weight;
    float right_weight;

    ResamplingCell() = default;
    ResamplingCell(const ResamplingCell&) = default;
    ResamplingCell(float left_edge, float right_edge);

    float CellAverage(const ArrayX& samples) const;
  };
  std::vector<ResamplingCell> log_lag_cells_;
};

#endif  // CARFAC_PITCHOGRAM_H_
