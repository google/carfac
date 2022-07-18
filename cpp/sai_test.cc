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

#include <cmath>
#include <iostream>
#include <string>

#include "gtest/gtest.h"

#include "agc.h"
#include "car.h"
#include "carfac.h"
#include "common.h"
#include "ihc.h"
#include "test_util.h"
#include <Eigen/Core>

using testing::Values;

class SAIPeriodicInputTest
    : public SAITestBase,
      public ::testing::WithParamInterface< ::testing::tuple<int, int> > {
 protected:
  void SetUp() {
    period_ = ::testing::get<0>(GetParam());
    num_channels_ = ::testing::get<1>(GetParam());
  }

  int period_;
  int num_channels_;
};

TEST_P(SAIPeriodicInputTest, MultiChannelPulseTrain) {
  const int kInputSegmentWidth = 38;
  ArrayXX segment =
      CreatePulseTrain(num_channels_, kInputSegmentWidth, period_);

  const int kSAIWidth = 15;
  const int kTriggerWindowWidth = kInputSegmentWidth;
  SAIParams sai_params = CreateSAIParams(num_channels_, kInputSegmentWidth,
                                         kTriggerWindowWidth, kSAIWidth);
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

INSTANTIATE_TEST_SUITE_P(PeriodicInputVariations, SAIPeriodicInputTest,
                         testing::Combine(Values(25, 10, 5, 2),  // periods.
                                          Values(1, 2, 15)));  // num_channels.

class SAITest : public SAITestBase {};

TEST_F(SAITest, DiesIfInputSegmentWidthIsLargerThanBuffer) {
  const int kNumChannels = 2;
  const int kSAIWidth = 10;
  const int kInputSegmentWidth = 200;
  SAIParams valid_params = CreateSAIParams(kNumChannels, kInputSegmentWidth,
                                           kInputSegmentWidth, kSAIWidth);
  SAI sai(valid_params);

  const int kTriggerWindowWidth = kInputSegmentWidth / 10;
  SAIParams params = CreateSAIParams(kNumChannels, kInputSegmentWidth,
                                     kTriggerWindowWidth, kSAIWidth);
  ASSERT_GT(kInputSegmentWidth,
            params.num_triggers_per_frame * params.trigger_window_width);
  ASSERT_DEATH(sai.Redesign(params), "Too few triggers");
}

TEST_F(SAITest, DiesIfInputWidthDoesntMatchInputSegmentWidth) {
  const int kNumChannels = 2;
  const int kInputSegmentWidth = 10;
  const int kPeriod = 4;
  ArrayXX segment = CreatePulseTrain(kNumChannels, kInputSegmentWidth, kPeriod);

  const int kSAIWidth = 20;
  const int kExpectedInputSegmentWidth = kInputSegmentWidth - 1;
  const int kTriggerWindowWidth = kSAIWidth + 1;
  SAIParams sai_params = CreateSAIParams(
      kNumChannels, kExpectedInputSegmentWidth, kTriggerWindowWidth, kSAIWidth);
  ASSERT_NE(sai_params.input_segment_width, kInputSegmentWidth);
  SAI sai(sai_params);
  ArrayXX sai_frame;
  ASSERT_DEATH(sai.RunSegment(segment, &sai_frame), "input samples");
}

TEST_F(SAITest, InputSegmentWidthSmallerThanTriggerWindow) {  // I.e. small hop.
  const int kNumChannels = 1;
  const int kTotalInputSamples = 20;
  const int kPeriod = 5;
  ArrayXX input = CreatePulseTrain(kNumChannels, kTotalInputSamples, kPeriod);

  const int kNumFrames = 4;
  const int kInputSegmentWidth = kTotalInputSamples / kNumFrames;
  const int kTriggerWindowWidth = kTotalInputSamples;
  const int kSAIWidth = 15;
  SAIParams sai_params = CreateSAIParams(kNumChannels, kInputSegmentWidth,
                                         kTriggerWindowWidth, kSAIWidth);
  ASSERT_LT(sai_params.input_segment_width, sai_params.trigger_window_width);
  sai_params.future_lags = 0;  // Only compute past lags.

  ASSERT_GE(kPeriod, kInputSegmentWidth);

  SAI sai(sai_params);
  ArrayXX sai_frame;
  for (int i = 0; i < kNumFrames; ++i) {
    ArrayXX segment =
        input.row(0).segment(i * kInputSegmentWidth, kInputSegmentWidth);
    sai.RunSegment(segment, &sai_frame);

    std::cout << "Frame " << i << std::endl
              << "Input:" << std::endl << segment << std::endl
              << "Output:" << std::endl << sai_frame << std::endl;

    EXPECT_NE(segment.cwiseAbs().sum(), 0);
    // Since the input segment is never all zero, there should always
    // be a peak at zero lag.
    ArrayX sai_channel = sai_frame.row(0);
    EXPECT_TRUE(HasPeakAt(sai_channel, sai_channel.size() - 1));

    if (i == 0) {
      // Since the pulse train period is larger than the input segment
      // size, the first input segment will only see a single impulse,
      // most of the SAI will be zero.
      AssertArrayNear(sai_channel.head(sai_channel.size() - 1),
                      ArrayX::Zero(sai_channel.size() - 1), 1e-9);
    }

    if (i == kNumFrames - 1) {
      // By the last frame, the SAI's internal buffer will have
      // accumulated the full input signal, so the resulting image
      // should contain kPeriod peaks.
      for (int j = sai_channel.size() - 1; j >= 0; j -= kPeriod) {
        EXPECT_TRUE(HasPeakAt(sai_channel, j));
      }
    }
  }
}

TEST_F(SAITest, MatchesMatlabOnBinauralData) {
  const std::string kTestName = "binaural_test";
  const int kInputSegmentWidth = 882;
  const int kNumChannels = 71;
  // The Matlab CARFAC output is transposed compared to the C++.
  ArrayXX input_segment =
      LoadMatrix(kTestName + "-matlab-nap1.txt", kInputSegmentWidth,
                 kNumChannels).transpose();

  const int kTriggerWindowWidth = kInputSegmentWidth;
  const int kSAIWidth = 500;
  SAIParams sai_params = CreateSAIParams(kNumChannels, kInputSegmentWidth,
                                         kTriggerWindowWidth, kSAIWidth);
  SAI sai(sai_params);
  ArrayXX sai_frame;
  sai.RunSegment(input_segment, &sai_frame);
  ArrayXX expected_sai_frame = LoadMatrix(kTestName + "-matlab-sai1.txt",
                                          kNumChannels, sai_params.sai_width);
  AssertArrayNear(expected_sai_frame, sai_frame, kTestPrecision);

  WriteMatrix(kTestName + "-cpp-sai1.txt", sai_frame);
}

TEST_F(SAITest, CARFACIntegration) {
  const int kNumEars = 1;
  const int kInputSegmentWidth = 300;
  ArrayXX segment(kNumEars, kInputSegmentWidth);

  // Sinusoid input.
  const float kFrequency = 10.0;
  segment.row(0) =
      ArrayX::LinSpaced(kInputSegmentWidth, 0.0, 2 * kFrequency * M_PI).sin();

  CARParams car_params;
  IHCParams ihc_params;
  AGCParams agc_params;
  CARFAC carfac(kNumEars, 8000, car_params, ihc_params, agc_params);
  CARFACOutput output(true, false, false, false);
  const bool kOpenLoop = false;
  carfac.RunSegment(segment, kOpenLoop, &output);

  const int kSAIWidth = 20;
  const int kTriggerWindowWidth = kInputSegmentWidth;
  SAIParams sai_params =
      CreateSAIParams(carfac.num_channels(), kInputSegmentWidth,
                      kTriggerWindowWidth, kSAIWidth);
  SAI sai(sai_params);
  ArrayXX sai_frame;
  sai.RunSegment(output.nap()[0], &sai_frame);

  // TODO(ronw): Test something about the output.
}
