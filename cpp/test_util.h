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

// Shared test utilities.

#ifndef CARFAC_TEST_UTIL_H
#define CARFAC_TEST_UTIL_H

#include <fstream>
#include <string>

#include "gtest/gtest.h"

#include "common.h"
#include "sai.h"
#include <Eigen/Core>

// Location of the text files produced by 'CARFAC_GenerateTestData.m' for
// comparing the ouput of the Matlab implementation with the C++ one.
static const char* kTestDataDir = "../test_data/";

inline std::string GetTestDataPath(const std::string& filename) {
  return kTestDataDir + filename;
}

// Reads a size rows by columns Eigen matrix from a text file written
// using the Matlab dlmwrite function.
ArrayXX LoadMatrix(const std::string& filename, int rows, int columns) {
  const std::string fullfile = GetTestDataPath(filename);
  std::ifstream file(fullfile.c_str());
  ArrayXX output(rows, columns);
  CARFAC_ASSERT(file.is_open());
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      file >> output(i, j);
    }
  }
  file.close();
  return output;
}

void WriteMatrix(const std::string& filename, const ArrayXX& matrix) {
  std::string fullfile = kTestDataDir + filename;
  std::ofstream ofile(fullfile.c_str());
  const int kPrecision = 9;
  ofile.precision(kPrecision);
  if (ofile.is_open()) {
    Eigen::IOFormat ioformat(kPrecision, Eigen::DontAlignCols);
    ofile << matrix.format(ioformat) << std::endl;
  }
  ofile.close();
}

void AssertArrayNear(const ArrayXX& expected, const ArrayXX& actual,
                     double precision) {
  ArrayXX abs_difference = (expected - actual).cwiseAbs();
  ASSERT_TRUE(abs_difference.maxCoeff() <= precision)
      << "expected differs from actual by more than " << precision
      << "\n  max(abs(expected - actual)) = " << abs_difference.maxCoeff()
      << "\n  max(abs(expected)) = " << expected.cwiseAbs().maxCoeff()
      << "\n  max(abs(actual)) = " << actual.cwiseAbs().maxCoeff();
}

// The level of precision with which the C++ library's outputs match
// those of Matlab (which always uses double precision) depends on the
// precision of FPType.  This is a hack that uses template
// specialization to conditionally set the precision at compile-time
// based on the definition of FPType.
template <typename T>
constexpr double GetTestPrecision();
template <> constexpr double GetTestPrecision<double>() { return 1e-7; }
template <> constexpr double GetTestPrecision<float>() { return 7e-3; }
static constexpr double kTestPrecision = GetTestPrecision<FPType>();

// Base class for SAI unit tests that provides helper functions for
// constructing test signals and analyzing results.
class SAITestBase : public ::testing::Test {
 protected:
  static ArrayXX CreatePulseTrain(int num_channels, int num_samples, int period,
                                  int leading_zeros) {
    ArrayXX segment = ArrayXX::Zero(num_channels, num_samples);
    for (int i = 0; i < num_channels; ++i) {
      // Begin each channel at a different phase.
      const int phase = (i + leading_zeros) % period;
      for (int j = phase; j < num_samples; j += period) {
        segment(i, j) = 1;
      }
    }
    return segment;
  }

  static ArrayXX CreatePulseTrain(int num_channels, int num_samples,
                                  int period) {
    return CreatePulseTrain(num_channels, num_samples, period, 0);
  }

  static SAIParams CreateSAIParams(int num_channels, int input_segment_width,
                                   int trigger_window_width, int sai_width) {
    SAIParams sai_params;
    sai_params.num_channels = num_channels;
    sai_params.input_segment_width = input_segment_width;
    sai_params.trigger_window_width = trigger_window_width;
    sai_params.sai_width = sai_width;
    // Half of the SAI should come from the future.
    sai_params.future_lags = sai_params.sai_width / 2;
    sai_params.num_triggers_per_frame = 2;
    return sai_params;
  }

  static bool HasPeakAt(const ArrayX& frame, int index) {
    if (index == 0) {
      return frame(index) > frame(index + 1);
    } else if (index == frame.size() - 1) {
      return frame(index) > frame(index - 1);
    }
    return frame(index) > frame(index + 1) && frame(index) > frame(index - 1);
  }
};

#endif  // CARFAC_TEST_UTIL_H
