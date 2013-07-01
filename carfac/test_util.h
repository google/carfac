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

// Shared test utilities.

#ifndef CARFAC_TEST_UTIL_H
#define CARFAC_TEST_UTIL_H

#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "common.h"

// Location of the text files produced by 'CARFAC_GenerateTestData.m' for
// comparing the ouput of the Matlab implementation with the C++ one.
static const char* kTestDataDir = "./test_data/";

// Reads a matrix (size rows vector of size columns Container objects)
// from a text file written using the Matlab dlmwrite function.
template <typename Container = ArrayX, bool ColMajor = true>
std::vector<Container> LoadMatrix(const std::string& filename, int rows,
                                  int columns) {
  std::string fullfile = kTestDataDir + filename;
  std::ifstream file(fullfile.c_str());
  std::vector<Container> output;
  if (ColMajor) {
    output.assign(rows, Container(columns));
  } else {
    output.assign(columns, Container(rows));
  }
  if (file.is_open()) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < columns; ++j) {
        if (ColMajor) {
          file >> output[i][j];
        } else {
          file >> output[j][i];
        }
      }
    }
  }
  file.close();
  return output;
}

bool ArraysNear(const ArrayX& expected, const ArrayX& actual,
                double precision) {
  return (expected - actual).cwiseAbs().maxCoeff() <= precision;
}

#endif  // CARFAC_TEST_UTIL_H
