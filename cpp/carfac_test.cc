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

#include <cmath>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "agc.h"
#include "car.h"
#include "common.h"
#include "ear.h"
#include "ihc.h"
#include "test_util.h"
#include <Eigen/Core>

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

TEST(CarTest, PoleFrequencies) {
  constexpr FPType SampleRateHz = 16000;
  const CARParams params;
  ArrayX poles = CARPoleFrequencies(SampleRateHz, params);
  for (int pole_index = 0; pole_index < poles.size(); ++pole_index) {
    SCOPED_TRACE(std::string("pole_index: ") +
                 testing::PrintToString(pole_index));
    const FPType pole = poles[pole_index];
    ASSERT_NEAR(pole,
                CARChannelIndexToFrequency(SampleRateHz, params, pole_index),
                1e-5f * pole);
    ASSERT_NEAR(pole_index,
                CARFrequencyToChannelIndex(SampleRateHz, params, pole),
                1e-5f * pole_index);
  }
}

class CARFACTest : public testing::Test {
 protected:
  CARFACTest() : open_loop_(false) {}

  std::vector<ArrayXX> LoadTestData(const std::string& basename,
                                    int num_samples,
                                    int num_ears,
                                    int num_channels) const {
    std::vector<ArrayXX> test_data;
    for (int ear = 0; ear < num_ears; ++ear) {
      std::string filename = basename + std::to_string(ear + 1) + ".txt";
      // The Matlab CARFAC output is transposed compared to the C++.
      test_data.push_back(
          LoadMatrix(filename, num_samples, num_channels).transpose());
    }
    return test_data;
  }

  void AssertCARFACOutputNear(const std::vector<ArrayXX>& expected,
                              const std::vector<ArrayXX>& actual) const {
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

    CARFAC carfac(num_ears, sample_rate, car_params_, ihc_params_, agc_params_);
    CARFACOutput output(true, true, false, false);
    carfac.RunSegment(sound_data, open_loop_, &output);

    std::vector<ArrayXX> expected_nap = LoadTestData(
        test_name + "-matlab-nap", num_samples, num_ears, num_channels);
    AssertCARFACOutputNear(expected_nap, output.nap());
    std::vector<ArrayXX> expected_bm = LoadTestData(
        test_name + "-matlab-bm", num_samples, num_ears, num_channels);
    AssertCARFACOutputNear(expected_bm, output.bm());

    // TODO(dicklyon): Conditionally overwrite test output files.
    // WriteNAPOutput(output, test_name + "-cpp-nap1.txt", 0);
    // WriteNAPOutput(output, test_name + "-cpp-nap2.txt", 1);
  }

  CARParams car_params_;
  IHCParams ihc_params_;
  AGCParams agc_params_;
  bool open_loop_;
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

  CARFACOutput output(false, false, false, true);  // store_agc

  // AGC enabled.
  CARFAC carfac(kNumEars, kSampleRate, car_params_, ihc_params_, agc_params_);
  const bool kOpenLoop = false;
  carfac.RunSegment(sound_data, kOpenLoop, &output);
  ArrayXX agc_enabled_output = output.agc()[0];

  // AGC disabled.
  agc_params_.num_stages = 0;
  carfac.Redesign(kNumEars, kSampleRate, car_params_, ihc_params_, agc_params_);
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

TEST_F(CARFACTest, PoleFrequenciesAreDecreasing) {
  const FPType kSampleRate = 8000.0;  // Hz.
  const int kNumEars = 1;

  CARFAC carfac(kNumEars, kSampleRate, car_params_, ihc_params_, agc_params_);
  ArrayX pole_frequencies = carfac.pole_frequencies();
  EXPECT_EQ(carfac.num_channels(), pole_frequencies.size());
  for (int i = 1; i < pole_frequencies.size(); ++i) {
    EXPECT_LT(pole_frequencies(i), pole_frequencies(i - 1));
  }
}

TEST_F(CARFACTest, MatchesMatlabWithAGCOff) {
  const int kNumSamples = 2000;
  const int kNumEars = 2;
  const int kNumChannels = 83;
  const FPType kSampleRate = 44100.0;
  open_loop_ = true;  // Turns off the AGC feedback.
  RunCARFACAndCompareWithMatlab(
      "agc_test", kNumSamples, kNumEars, kNumChannels, kSampleRate);
}

TEST_F(CARFACTest, MatchesMatlabWithIHCJustHalfWaveRectifyOn) {
  const int kNumSamples = 2000;
  const int kNumEars = 2;
  const int kNumChannels = 83;
  const FPType kSampleRate = 44100.0;
  ihc_params_.just_half_wave_rectify = true;
  RunCARFACAndCompareWithMatlab(
      "ihc_just_hwr_test", kNumSamples, kNumEars, kNumChannels, kSampleRate);
}

TEST_F(CARFACTest, AGCDesignBehavesSensibly) {
  const int kNumEars = 2;
  const FPType kSampleRate = 22050.0;
  const std::vector<FPType> spreads = {0.0, 1.0, 1.4, 2.0, 2.8, 3.5, 4.0};
  for (int i = 0; i < spreads.size(); ++i) {
    FPType spread = spreads[i];
    EXPECT_GE(spread, 0);
    // Test zero and default and larger AGC spread.
    agc_params_.agc1_scales[0] = 1.0 * spread;
    agc_params_.agc2_scales[0] = 1.65 * spread;
    for (int i = 1; i < agc_params_.agc1_scales.size(); ++i) {
      agc_params_.agc1_scales[i] = agc_params_.agc1_scales[i - 1] * sqrt(2.0);
      agc_params_.agc2_scales[i] = agc_params_.agc2_scales[i - 1] * sqrt(2.0);
    }
    CARFAC carfac(kNumEars, kSampleRate, car_params_, ihc_params_, agc_params_);
    for (int stage = 0; stage < 4; ++stage) {
      // Stages all should be alike, as their time constants quadruple, sample
      // rates halve, and variances double at each stage, so equivalent filters.
      const AGCCoeffs& coeffs = carfac.get_ear(0).get_agc_coeffs(stage);
      switch (i) {
        case 0:
          EXPECT_EQ(3, coeffs.agc_spatial_n_taps);
          EXPECT_EQ(0, coeffs.agc_spatial_iterations);
          break;
        case 1:
          EXPECT_EQ(3, coeffs.agc_spatial_n_taps);
          EXPECT_EQ(1, coeffs.agc_spatial_iterations);
          break;
        case 2:
          EXPECT_EQ(5, coeffs.agc_spatial_n_taps);
          EXPECT_EQ(1, coeffs.agc_spatial_iterations);
          break;
        case 3:
          EXPECT_EQ(5, coeffs.agc_spatial_n_taps);
          EXPECT_EQ(2, coeffs.agc_spatial_iterations);
          break;
        case 4:
          EXPECT_EQ(5, coeffs.agc_spatial_n_taps);
          EXPECT_EQ(3, coeffs.agc_spatial_iterations);
          break;
        case 5:
          EXPECT_EQ(5, coeffs.agc_spatial_n_taps);
          EXPECT_EQ(4, coeffs.agc_spatial_iterations);
          break;
        case 6:
          EXPECT_EQ(-1, coeffs.agc_spatial_iterations);
          break;
      }
    }
  }
}

TEST_F(CARFACTest, AGCDesignAtLowSampleRate) {
  const int kNumEars = 2;
  const FPType kSampleRate = 8000.0;
  const std::vector<FPType> spreads = {0.0, 1.0, 1.4, 2.0, 2.8};
  for (int i = 0; i < spreads.size(); ++i) {
    FPType spread = spreads[i];
    EXPECT_GE(spread, 0);
    // Test zero and default and larger AGC spread.

    agc_params_.agc1_scales[0] = 1.0 * spread;
    agc_params_.agc2_scales[0] = 1.65 * spread;
    for (int i = 1; i < agc_params_.agc1_scales.size(); ++i) {
      agc_params_.agc1_scales[i] = agc_params_.agc1_scales[i - 1] * sqrt(2.0);
      agc_params_.agc2_scales[i] = agc_params_.agc2_scales[i - 1] * sqrt(2.0);
    }
    CARFAC carfac(kNumEars, kSampleRate, car_params_, ihc_params_, agc_params_);
    const AGCCoeffs& coeffs = carfac.get_ear(0).get_agc_coeffs(0);
    switch (i) {
      case 0:
        EXPECT_EQ(3, coeffs.agc_spatial_n_taps);
        EXPECT_EQ(0, coeffs.agc_spatial_iterations);
        break;
      case 1:
        EXPECT_EQ(5, coeffs.agc_spatial_n_taps);
        EXPECT_EQ(1, coeffs.agc_spatial_iterations);
        break;
      case 2:
        EXPECT_EQ(5, coeffs.agc_spatial_n_taps);
        EXPECT_EQ(2, coeffs.agc_spatial_iterations);
        break;
      case 3:
        EXPECT_EQ(5, coeffs.agc_spatial_n_taps);
        EXPECT_EQ(4, coeffs.agc_spatial_iterations);
        break;
      case 4:
        EXPECT_EQ(-1, coeffs.agc_spatial_iterations);
        break;
    }
  }
}

void BM_CarfacSegment(benchmark::State& state) {
  const auto segment_length_samples = state.range(0);
  const int num_ears = 1;
  const FPType sample_rate = 22050.0;
  CARParams car_params;
  IHCParams ihc_params;
  AGCParams agc_params;
  CARFAC carfac(num_ears, sample_rate, car_params, ihc_params, agc_params);
  // Sinusoid input.
  const float kFrequency = 500.0;  // Hz.
  ArrayXX sound_data(num_ears, segment_length_samples);
  sound_data.row(0) =
      ArrayX::LinSpaced(segment_length_samples, 0.0, 2 * kFrequency * M_PI)
          .sin();

  const bool open_loop = false;
  for (auto s : state) {
    // store everything for the benchmarks.
    CARFACOutput output(true, true, true, true);
    carfac.RunSegment(sound_data, open_loop, &output);
  }
}

BENCHMARK(BM_CarfacSegment)->ThreadRange(1, 128)
    ->Arg(220)
    ->Arg(2205)
    ->Arg(22050)
    ->Arg(44100)
    ->Arg(220500);
