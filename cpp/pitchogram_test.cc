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

#include "gtest/gtest.h"

#include "agc.h"
#include "car.h"
#include "carfac.h"
#include "common.h"
#include "ihc.h"
#include "sai.h"

namespace {

struct TestParams {
  int sai_width;
  bool log_lag;
};

std::string PrintToString(const TestParams& params) {
  return "{sai_width=" + testing::PrintToString(params.sai_width) +
         ", log_lag=" + testing::PrintToString(params.log_lag) + "}";
}

class PitchogramTest : public ::testing::WithParamInterface<TestParams>,
                       public ::testing::Test {};

TEST_P(PitchogramTest, OutputSize) {
  const int sai_width = GetParam().sai_width;
  const bool log_lag = GetParam().log_lag;
  constexpr float kSampleRateHz = 24000.0f;
  constexpr int kSegmentWidth = 256;
  constexpr int kNumEars = 1;

  CARParams car_params;
  IHCParams ihc_params;
  AGCParams agc_params;
  CARFAC carfac(kNumEars, kSampleRateHz, car_params, ihc_params, agc_params);
  CARFACOutput carfac_output_buffer(true, false, false, false);

  SAIParams sai_params;
  sai_params.num_channels = carfac.num_channels();
  sai_params.sai_width = sai_width;
  sai_params.input_segment_width = kSegmentWidth;
  sai_params.trigger_window_width = sai_params.input_segment_width + 1;
  sai_params.future_lags = sai_params.sai_width - 1;
  sai_params.num_triggers_per_frame = 2;
  SAI sai(sai_params);
  ArrayXX sai_output_buffer(sai_params.num_channels, sai_params.sai_width);

  PitchogramParams pitchogram_params;
  pitchogram_params.log_lag = log_lag;
  Pitchogram pitchogram(kSampleRateHz, car_params, sai_params,
                        pitchogram_params);

  if (!log_lag) {
    EXPECT_EQ(pitchogram.num_lags(), sai_params.sai_width);
  }

  ArrayXX samples = ArrayXX::Random(kNumEars, kSegmentWidth);
  carfac.RunSegment(samples, false /* open_loop */, &carfac_output_buffer);
  const auto& nap = carfac_output_buffer.nap()[0];
  sai.RunSegment(nap, &sai_output_buffer);
  pitchogram.VowelEmbedding(nap);
  const ArrayX& pitchogram_frame = pitchogram.RunFrame(sai_output_buffer);

  EXPECT_EQ(pitchogram_frame.size(), pitchogram.num_lags());

  Image<uint8_t> col(1, pitchogram.num_lags(), 3);
  pitchogram.DrawColumn(col);  // Runs without assertions failing.
}

INSTANTIATE_TEST_SUITE_P(Params, PitchogramTest,
                         testing::Values(TestParams{240, false},
                                         TestParams{1024, false},
                                         TestParams{240, true},
                                         TestParams{1024, true}));

}  // namespace
