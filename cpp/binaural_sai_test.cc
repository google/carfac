// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Kevin Wilson <kwwilson@google.com>
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

#include "binaural_sai.h"

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

class BinauralSAIPeriodicInputTest
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

TEST_P(BinauralSAIPeriodicInputTest, MultiChannelPulseTrain) {
  const int kInputSegmentWidth = 38;
  const int kLeftEarDelay = 0;
  const int kRightEarDelay = 1;
  std::vector<ArrayXX> input_segment;
  // Arbitrarily refer to ear 0 as the left ear.
  input_segment.push_back(CreatePulseTrain(num_channels_, kInputSegmentWidth,
                                           period_, kLeftEarDelay));
  input_segment.push_back(CreatePulseTrain(num_channels_, kInputSegmentWidth,
                                           period_, kRightEarDelay));

  const int kSAIWidth = 15;
  const int kTriggerWindowWidth = kInputSegmentWidth;
  SAIParams sai_params = CreateSAIParams(num_channels_, kInputSegmentWidth,
                                         kTriggerWindowWidth, kSAIWidth);
  sai_params.future_lags = 2;

  BinauralSAI binaural_sai(sai_params);
  std::vector<ArrayXX> output_frame;
  binaural_sai.RunSegment(input_segment, &output_frame);

  // The output should have peaks at the same positions across all
  // frequency channels, regardless of input phase.  Output peaks
  // should differ from one "ear" to the other based on the delay
  // difference between ears.
  for (int ear = 0; ear < 2; ++ear) {
    // The peaks in the two ears should be offset from the zero-lag
    // index by a delay equal in magnitude to the relative delay
    // between the ears, positive offset in one ear, negative in the
    // other.
    int ear_offset = sai_params.future_lags + 1 +
                     (ear == 0 ? 1 : -1) * (kLeftEarDelay - kRightEarDelay);
    const ArrayXX& ear_frame = output_frame[ear];
    for (int i = 0; i < num_channels_; ++i) {
      const ArrayX& sai_channel = ear_frame.row(i);
      for (int j = sai_channel.size() - ear_offset; j >= 0; j -= period_) {
        EXPECT_TRUE(HasPeakAt(sai_channel, j)) << "ear: " << ear << " i: " << i
                                               << " j: " << j;
      }
    }
    std::cout << "Input " << ear << ":" << std::endl << input_segment[ear]
              << std::endl << "Output " << ear << ":" << std::endl
              << output_frame[ear] << std::endl;
  }
}

INSTANTIATE_TEST_SUITE_P(PeriodicInputVariations, BinauralSAIPeriodicInputTest,
                         testing::Combine(Values(25, 10, 5, 2),  // periods.
                                          Values(1, 2, 15)));  // num_channels.

class BinauralSAITest : public SAITestBase {};

TEST_F(BinauralSAITest, DiesIfNotTwoEars) {
  const int kNumChannels = 2;
  const int kInputSegmentWidth = 10;
  const int kPeriod = 4;
  std::vector<ArrayXX> input_segment;
  input_segment.push_back(
      CreatePulseTrain(kNumChannels, kInputSegmentWidth, kPeriod, 0));

  const int kSAIWidth = 20;
  const int kTriggerWindowWidth = kSAIWidth + 1;
  SAIParams sai_params = CreateSAIParams(kNumChannels, kInputSegmentWidth,
                                         kTriggerWindowWidth, kSAIWidth);
  BinauralSAI binaural_sai(sai_params);
  std::vector<ArrayXX> output_frame;
  // Dies on one ear.
  ASSERT_DEATH(binaural_sai.RunSegment(input_segment, &output_frame),
               "input_segment.size()");

  input_segment.push_back(
      CreatePulseTrain(kNumChannels, kInputSegmentWidth, kPeriod, 0));
  input_segment.push_back(
      CreatePulseTrain(kNumChannels, kInputSegmentWidth, kPeriod, 0));
  // Dies on three ears.
  ASSERT_DEATH(binaural_sai.RunSegment(input_segment, &output_frame),
               "input_segment.size()");
}
