// Copyright 2013, Google, Inc.
// Author: Ron Weiss <ronw@google.com>
//
// This C++ file is part of an implementation of Lyon's cochlear model:
// "Cascade of Asymmetric Resonators with Fast-Acting Compression"
// to supplement Lyon's upcoming book "Human and Machine Hearing"
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
#include <vector>

#include "gtest/gtest.h"

using testing::Values;
using std::vector;

vector<FloatArray> CreateZeroSegment(int n_ch, int length) {
  vector<FloatArray> segment;
  for (int i = 0; i < length; ++i) {
    segment.push_back(FloatArray::Zero(n_ch));
  }
  return segment;
}

bool HasPeakAt(const Float2dArray& frame, int index) {
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
    phase_ = std::tr1::get<1>(GetParam());
  }

  int period_;
  int phase_;
};

TEST_P(SAIPeriodicInputTest, SingleChannelPulseTrain) {
  vector<FloatArray> segment = CreateZeroSegment(1, 38);
  for (int i = phase_; i < segment.size(); i += period_) {
    segment[i](0) = 1;
  }

  SAIParams sai_params;
  sai_params.window_width = segment.size();
  sai_params.n_ch = 1;
  sai_params.width = 15;
  // Half of the SAI should come from the future.
  // sai_params.future_lags = sai_params.width / 2;
  sai_params.future_lags = 0;
  sai_params.n_window_pos = 2;

  SAI sai(sai_params);
  Float2dArray sai_frame;
  sai.RunSegment(segment, &sai_frame);

  // The output should have peaks at the same positions, regardless of
  // input phase.
  for (int i = sai_frame.size() - 1; i >= 0 ; i -= period_) {
    EXPECT_TRUE(HasPeakAt(sai_frame, i));
  }

  for (int i = 0; i < segment.size(); ++i) {
    std::cout << segment[i](0) << " ";
  }
  std::cout << "\n";
  for (int i = 0; i < sai_frame.size(); ++i) {
    std::cout << sai_frame(i) << " ";
  }
  std::cout << "\n";
}
INSTANTIATE_TEST_CASE_P(PeriodicInputVariations, SAIPeriodicInputTest,
                        testing::Combine(Values(25, 10, 5, 2),  // periods.
                                         Values(0, 3)));  // phases.
