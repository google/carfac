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

vector<ArrayX> CreateZeroSegment(int n_ch, int length) {
  vector<ArrayX> segment;
  for (int i = 0; i < length; ++i) {
    segment.push_back(ArrayX::Zero(n_ch));
  }
  return segment;
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
    n_ch_ = std::tr1::get<1>(GetParam());
  }

  int period_;
  int n_ch_;
};

TEST_P(SAIPeriodicInputTest, MultiChannelPulseTrain) {
  vector<ArrayX> segment = CreateZeroSegment(n_ch_, 38);
  for (int i = 0; i < n_ch_; ++i) {
    // Begin each channel at a different phase.
    const int phase = i;
    for (int j = phase; j < segment.size(); j += period_) {
      segment[j](i) = 1;
    }
  }

  SAIParams sai_params;
  sai_params.window_width = segment.size();
  sai_params.n_ch = n_ch_;
  sai_params.width = 15;
  // Half of the SAI should come from the future.
  // sai_params.future_lags = sai_params.width / 2;
  sai_params.future_lags = 0;
  sai_params.n_window_pos = 2;

  SAI sai(sai_params);
  ArrayXX sai_frame;
  sai.RunSegment(segment, &sai_frame);

  // The output should have peaks at the same positions, regardless of
  // input phase.
  for (int i = 0; i < n_ch_; ++i) {
    const ArrayX& sai_channel = sai_frame.row(i);
    for (int j = sai_channel.size() - 1; j >= 0; j -= period_) {
      EXPECT_TRUE(HasPeakAt(sai_channel, j));
    }
  }

  std::cout << "Input:\n";
  for (int i = 0; i < n_ch_; ++i) {
    for (int j = 0; j < segment.size(); ++j) {
      std::cout << segment[j](i) << " ";
    }
    std::cout << "\n";
  }

  std::cout << "Output:\n";
  for (int i = 0; i < sai_frame.rows(); ++i) {
    for (int j = 0; j < sai_frame.cols(); ++j) {
      std::cout << sai_frame(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}
INSTANTIATE_TEST_CASE_P(PeriodicInputVariations, SAIPeriodicInputTest,
                        testing::Combine(Values(25, 10, 5, 2),  // periods.
                                         Values(1, 2, 15)));  // n_ch.
