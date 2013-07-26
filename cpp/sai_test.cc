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

#include <iostream>
#include <string>

#include <Eigen/Core>

#include "gtest/gtest.h"

#include "agc.h"
#include "car.h"
#include "carfac.h"
#include "common.h"
#include "ihc.h"
#include "test_util.h"

using testing::Values;

ArrayXX CreatePulseTrain(int num_channels, int num_samples, int period) {
  ArrayXX segment = ArrayXX::Zero(num_channels, num_samples);
  for (int i = 0; i < num_channels; ++i) {
    // Begin each channel at a different phase.
    const int phase = i % period;
    for (int j = phase; j < num_samples; j += period) {
      segment(i, j) = 1;
    }
  }
  return segment;
}

SAIParams CreateSAIParams(int num_channels, int window_width, int width) {
  SAIParams sai_params;
  sai_params.num_channels = num_channels;
  sai_params.window_width = window_width;
  sai_params.width = width;
  // Half of the SAI should come from the future.
  sai_params.future_lags = sai_params.width / 2;
  sai_params.num_window_pos = 2;
  return sai_params;
}

bool HasPeakAt(const ArrayX& frame, int index) {
  if (index == 0) {
    return frame(index) > frame(index + 1);
  } else if (index == frame.size() - 1) {
    return frame(index) > frame(index - 1);
  }
  return frame(index) > frame(index + 1) && frame(index) > frame(index - 1);
}

class SAIPeriodicInputTest
    : public testing::TestWithParam<std::tr1::tuple<int, int>> {
 protected:
  void SetUp() {
    period_ = std::tr1::get<0>(GetParam());
    num_channels_ = std::tr1::get<1>(GetParam());
  }

  int period_;
  int num_channels_;
};

TEST_P(SAIPeriodicInputTest, MultiChannelPulseTrain) {
  const int kNumSamples = 38;
  ArrayXX segment = CreatePulseTrain(num_channels_, kNumSamples, period_);

  const int kSAIWidth = 15;
  SAIParams sai_params = CreateSAIParams(num_channels_, segment.size(),
                                         kSAIWidth);
  sai_params.future_lags = 0;  // Only compute past lags.

  SAI sai(sai_params);
  ArrayXX sai_frame;
  sai.RunSegment(segment, &sai_frame);

  // The output should have peaks at the same positions, regardless of
  // input phase.
  for (int i = 0; i < num_channels_; ++i) {
    const ArrayX& sai_channel = sai_frame.row(i);
    for (int j = sai_channel.size() - 1; j >= 0; j -= period_) {
      EXPECT_TRUE(HasPeakAt(sai_channel, j));
    }
  }

  std::cout << "Input:" << std::endl << segment << std::endl
            << "Output:" << std::endl << sai_frame << std::endl;
}

INSTANTIATE_TEST_CASE_P(PeriodicInputVariations, SAIPeriodicInputTest,
                        testing::Combine(Values(25, 10, 5, 2),  // periods.
                                         Values(1, 2, 15)));  // num_channels.

TEST(SAITest, InputIsZeroPaddedIfShorterThanWindowWidth) {
  const int kNumChannels = 1;
  const int kNumSamples = 10;
  const int kPeriod = 4;
  ArrayXX segment = CreatePulseTrain(kNumChannels, kNumSamples, kPeriod);

  const int kSAIWidth = 20;
  SAIParams sai_params = CreateSAIParams(kNumChannels, kSAIWidth + 1,
                                         kSAIWidth);
  ASSERT_GT(sai_params.window_width, kNumSamples);
  SAI sai(sai_params);
  ArrayXX sai_frame;
  sai.RunSegment(segment, &sai_frame);

  EXPECT_TRUE((sai_frame != 0).any());
  const int kZeroLagIndex = kSAIWidth / 2 - 1;
  EXPECT_TRUE(HasPeakAt(sai_frame.row(0), kZeroLagIndex));
  EXPECT_TRUE(HasPeakAt(sai_frame.row(0), kZeroLagIndex + kPeriod));
}

TEST(SAITest, MatchesMatlabOnBinauralData) {
  const std::string kTestName = "binaural_test";
  const int kNumSamples = 882;
  const int kNumChannels = 71;
  // The Matlab CARFAC output is transposed compared to the C++.
  ArrayXX input_segment = LoadMatrix(kTestName + "-matlab-nap1.txt",
                                     kNumSamples, kNumChannels).transpose();

  const int kWindowWidth = 2000;
  const int kSAIWidth = 500;
  SAIParams sai_params = CreateSAIParams(kNumChannels, kWindowWidth, kSAIWidth);
  SAI sai(sai_params);
  ArrayXX sai_frame;
  sai.RunSegment(input_segment, &sai_frame);
  ArrayXX expected_sai_frame = LoadMatrix(kTestName + "-matlab-sai1.txt",
                                          kNumChannels, sai_params.width);
  AssertArrayNear(expected_sai_frame, sai_frame, kTestPrecision);

  WriteMatrix(kTestName + "-cpp-sai1.txt", sai_frame);
}

TEST(SAITest, CARFACIntegration) {
  const int kNumEars = 1;
  const int kNumSamples = 300;
  ArrayXX segment(kNumEars, kNumSamples);

  // Sinusoid input.
  const float kFrequency = 10.0;
  segment.row(0) =
      ArrayX::LinSpaced(kNumSamples, 0.0, 2 * kFrequency * kPi).sin();

  CARParams car_params;
  IHCParams ihc_params;
  AGCParams agc_params;
  CARFAC carfac(kNumEars, 800, car_params, ihc_params, agc_params);
  CARFACOutput output(true, false, false, false);
  const bool kOpenLoop = false;
  carfac.RunSegment(segment, kOpenLoop, &output);

  const int kSAIWidth = 20;
  SAIParams sai_params = CreateSAIParams(carfac.num_channels(), kNumSamples,
                                         kSAIWidth);
  SAI sai(sai_params);
  ArrayXX sai_frame;
  sai.RunSegment(output.nap()[0], &sai_frame);

  // TODO(ronw): Test something about the output.
}
