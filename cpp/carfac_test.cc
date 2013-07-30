// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Alex Brandmeyer
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

#include "carfac.h"

#include <math.h>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "gtest/gtest.h"

#include "agc.h"
#include "car.h"
#include "common.h"
#include "ihc.h"
#include "test_util.h"

using std::vector;

// Reads a two dimensional vector of audio data from a text file
// containing the output of the Matlab wavread() function.
ArrayXX LoadAudio(const std::string& filename, int num_samples, int num_ears) {
  // The Matlab audio input is transposed compared to the C++.
  return LoadMatrix(filename, num_samples, num_ears).transpose();
}

// Writes the CARFAC NAP output to a text file.
void WriteNAPOutput(const CARFACOutput& output, const std::string& filename,
                    int ear) {
  WriteMatrix(filename, output.nap()[ear].transpose());
}

class CARFACTest : public testing::Test {
 protected:
  vector<ArrayXX> LoadTestData(const std::string& basename,
                               int num_samples,
                               int num_ears,
                               int num_channels) const {
    vector<ArrayXX> test_data;
    for (int ear = 0; ear < num_ears; ++ear) {
      std::string filename = basename + std::to_string(ear + 1) + ".txt";
      // The Matlab CARFAC output is transposed compared to the C++.
      test_data.push_back(
          LoadMatrix(filename, num_samples, num_channels).transpose());
    }
    return test_data;
  }

  void AssertCARFACOutputNear(const vector<ArrayXX>& expected,
                              const vector<ArrayXX>& actual) const {
    ASSERT_EQ(expected.size(), actual.size());
    for (int ear = 0; ear < expected.size(); ++ear) {
      AssertArrayNear(expected[ear], actual[ear], kTestPrecision);
    }
  }

  void RunCARFACAndCompareWithMatlab(const std::string& test_name,
                                     int num_samples,
                                     int num_ears,
                                     int num_channels,
                                     FPType sample_rate) const {
    ArrayXX sound_data =
        LoadAudio(test_name + "-audio.txt", num_samples, num_ears);

    CARParams car_params;
    IHCParams ihc_params;
    AGCParams agc_params;
    CARFAC carfac(num_ears, sample_rate, car_params, ihc_params, agc_params);
    CARFACOutput output(true, true, false, false);
    const bool kOpenLoop = false;
    carfac.RunSegment(sound_data, kOpenLoop, &output);

    vector<ArrayXX> expected_nap = LoadTestData(
        test_name + "-matlab-nap", num_samples, num_ears, num_channels);
    AssertCARFACOutputNear(expected_nap, output.nap());
    vector<ArrayXX> expected_bm = LoadTestData(
        test_name + "-matlab-bm", num_samples, num_ears, num_channels);
    AssertCARFACOutputNear(expected_bm, output.bm());

    // TODO(ronw): Don't unconditionally overwrite files that are
    // checked in to the repository on every test run.
    WriteNAPOutput(output, test_name + "-cpp-nap1.txt", 0);
    WriteNAPOutput(output, test_name + "-cpp-nap2.txt", 1);
  }
};

TEST_F(CARFACTest, MatchesMatlabOnBinauralData) {
  const int kNumSamples = 882;
  const int kNumEars = 2;
  const int kNumChannels = 71;
  const FPType kSampleRate = 22050.0;
  RunCARFACAndCompareWithMatlab(
      "binaural_test", kNumSamples, kNumEars, kNumChannels, kSampleRate);
}

TEST_F(CARFACTest, MatchesMatlabOnLongBinauralData) {
  const int kNumSamples = 2000;
  const int kNumEars = 2;
  const int kNumChannels = 83;
  const FPType kSampleRate = 44100.0;
  RunCARFACAndCompareWithMatlab(
      "long_test", kNumSamples, kNumEars, kNumChannels, kSampleRate);
}

TEST_F(CARFACTest, CanDisableAGC) {
  const FPType kSampleRate = 8000.0;  // Hz.
  const int kNumEars = 1;
  const int kNumSamples = static_cast<int>(kSampleRate);

  // Sinusoid input.
  const float kFrequency = 10.0;  // Hz.
  ArrayXX sound_data(kNumEars, kNumSamples);
  sound_data.row(0) =
      ArrayX::LinSpaced(kNumSamples, 0.0, 2 * kFrequency * M_PI).sin();

  CARParams car_params;
  IHCParams ihc_params;
  AGCParams agc_params;
  CARFACOutput output(false, false, false, true);  // store_agc

  // AGC enabled.
  CARFAC carfac(kNumEars, kSampleRate, car_params, ihc_params, agc_params);
  const bool kOpenLoop = false;
  carfac.RunSegment(sound_data, kOpenLoop, &output);
  ArrayXX agc_enabled_output = output.agc()[0];

  // AGC disabled.
  agc_params.num_stages = 0;
  carfac.Redesign(kNumEars, kSampleRate, car_params, ihc_params, agc_params);
  carfac.RunSegment(sound_data, kOpenLoop, &output);
  ArrayXX agc_disabled_output = output.agc()[0];

  ASSERT_EQ(agc_enabled_output.size(), agc_disabled_output.size());
  EXPECT_TRUE((agc_enabled_output !=  agc_disabled_output.size()).any());
  for (int c = 0; c < agc_disabled_output.rows(); ++c) {
    // With the AGC disabled, the agc output for a given channel
    // should be identical at all times.
    EXPECT_TRUE(
        (agc_disabled_output.row(c) == agc_disabled_output(c, 0)).all());
    EXPECT_FALSE(
        (agc_enabled_output.row(c) == agc_enabled_output(c, 0)).all());
  }
}
