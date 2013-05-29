//
//  carfac_test.cc
//  CARFAC Open Source C++ Library
//
//  Created by Alex Brandmeyer on 5/22/13.
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

#include <string>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "car_params.h"
#include "ihc_params.h"
#include "agc_params.h"
#include "carfac.h"

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;

// This is the 'test_data' subdirectory of aimc/carfac that specifies where to
// locate the text files produced by 'CARFAC_GenerateTestData.m' for comparing
// the ouput of the Matlab version of CARFAC with this C++ version.
static const char* kTestSourceDir= "./test_data/";
// Here we specify the level to which the output should match (7 decimals).
static const float kPrecisionLevel = 1.0e-7;

// Three helper functions are defined here for loading the test data generated
// by the Matlab version of CARFAC.
// This loads one-dimensional FloatArrays from single-column text files.
void WriteNAPOutput(CARFACOutput output, const string filename, int ear) {
  string fullfile = kTestSourceDir + filename;
  ofstream ofile(fullfile.c_str());
  int32_t n_timepoints = output.nap().size();
  int channels = output.nap()[0][0].size();
  if (ofile.is_open()) {
    for (int32_t i = 0; i < n_timepoints; ++i) {
      for (int j = 0; j < channels; ++j) {
        ofile << output.nap()[i][ear](j);
        if ( j < channels - 1) {
          ofile << " ";
        }
      }
      ofile << "\n";
    }
  }
  ofile.close();
}

FloatArray LoadTestData(const string filename, const int number_points) {
  string fullfile = kTestSourceDir + filename;
  ifstream file(fullfile.c_str());
  FPType myarray[number_points];
  FloatArray output(number_points);
  if (file.is_open()) {
    for (int i = 0; i < number_points; ++i) {
      file >> myarray[i];
      output(i) = myarray[i];
    }
  }
  file.close();
  return output;
}

// This loads a vector of FloatArrays from multi-column text files.
vector<FloatArray> Load2dTestData(const string filename, const int rows,
                            const int columns) {
  string fullfile = kTestSourceDir + filename;
  ifstream file(fullfile.c_str());
  FPType myarray[rows][columns];
  vector<FloatArray> output;
  output.resize(rows);
  for (auto& timepoint : output) {
    timepoint.resize(columns);
  }
  if (file.is_open()) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < columns; ++j) {
        file >> myarray[i][j];
        output[i](j) = myarray[i][j];
      }
    }
  }
  file.close();
  return output;
}

// This loads two dimensional vectors of audio data using data generated in
// Matlab using the wavread() function.
vector<vector<float>> Load2dAudioVector(string filename, int timepoints,
                                        int channels) {
  string fullfile = kTestSourceDir + filename;
  ifstream file(fullfile.c_str());
  vector<vector<float>> output;
  output.resize(channels);
  for (auto& channel : output) {
    channel.resize(timepoints);
  }
  if (file.is_open()) {
    for (int i = 0; i < timepoints; ++i) {
      for (int j = 0; j < channels; ++j) {
        file >> output[j][i];
      }
    }
  }
  file.close();
  return output;
}

TEST(CARFACTest, Binaural_Output_test) {
  int n_timepoints = 882;
  int n_channels = 71;
  int n_ears = 2;
  string filename = "binaural_test_nap1.txt";
  vector<FloatArray> nap1 = Load2dTestData(filename, n_timepoints, n_channels);
  filename = "binaural_test_bm1.txt";
  vector<FloatArray> bm1 = Load2dTestData(filename, n_timepoints, n_channels);
  filename = "binaural_test_nap2.txt";
  vector<FloatArray> nap2 = Load2dTestData(filename, n_timepoints, n_channels);
  filename = "binaural_test_bm2.txt";
  vector<FloatArray> bm2 = Load2dTestData(filename, n_timepoints, n_channels);
  filename = "file_signal_binaural_test.txt";
  vector<vector<float>> sound_data = Load2dAudioVector(filename, n_timepoints,
                                                       n_ears);
  CARParams car_params;
  IHCParams ihc_params;
  AGCParams agc_params;
  CARFAC mycf;
  mycf.Design(n_ears, 22050, car_params, ihc_params,
              agc_params);
  CARFACOutput my_output;
  my_output.Init(n_ears, true, false, true, false, false);
  mycf.Run(sound_data, &my_output);
  filename = "cpp_nap_output_1_binaural_test.txt";
  WriteNAPOutput(my_output, filename, 0);
  filename = "cpp_nap_output_2_binaural_test.txt";
  WriteNAPOutput(my_output, filename, 1);
  int ear = 0;
  int n_ch = 71;
  for (int timepoint = 0; timepoint < n_timepoints; ++timepoint) {
    for (int channel = 0; channel < n_ch; ++channel) {
      FPType cplusplus = my_output.nap()[timepoint][ear](channel);
      FPType matlab = nap1[timepoint](channel);
      ASSERT_NEAR(cplusplus, matlab, kPrecisionLevel);
      cplusplus = my_output.bm()[timepoint][ear](channel);
      matlab = bm1[timepoint](channel);
      ASSERT_NEAR(cplusplus, matlab, kPrecisionLevel);
    }
  }
  ear = 1;
  for (int timepoint = 0; timepoint < n_timepoints; ++timepoint) {
    for (int channel = 0; channel < n_ch; ++channel) {
      FPType cplusplus = my_output.nap()[timepoint][ear](channel);
      FPType matlab = nap2[timepoint](channel);
      ASSERT_NEAR(cplusplus, matlab, kPrecisionLevel);
      cplusplus = my_output.bm()[timepoint][ear](channel);
      matlab = bm2[timepoint](channel);
      ASSERT_NEAR(cplusplus, matlab, kPrecisionLevel);
    }
  }
}

TEST(CARFACTest, Long_Output_test) {
  int n_timepoints = 2000;
  int n_channels = 83;
  int n_ears = 2;
  string filename = "long_test_nap1.txt";
  vector<FloatArray> nap1 = Load2dTestData(filename, n_timepoints, n_channels);
  filename = "long_test_bm1.txt";
  vector<FloatArray> bm1 = Load2dTestData(filename, n_timepoints, n_channels);
  filename = "long_test_nap2.txt";
  vector<FloatArray> nap2 = Load2dTestData(filename, n_timepoints, n_channels);
  filename = "long_test_bm2.txt";
  vector<FloatArray> bm2 = Load2dTestData(filename, n_timepoints, n_channels);
  filename = "file_signal_long_test.txt";
  vector<vector<float>> sound_data = Load2dAudioVector(filename, n_timepoints,
                                                       n_ears);
  CARParams car_params;
  IHCParams ihc_params;
  AGCParams agc_params;
  CARFAC mycf;
  mycf.Design(n_ears, 44100, car_params, ihc_params,
              agc_params);
  CARFACOutput my_output;
  my_output.Init(n_ears, true, false, true, false, false);
  mycf.Run(sound_data, &my_output);
  filename = "cpp_nap_output_1_long_test.txt";
  WriteNAPOutput(my_output, filename, 0);
  filename = "cpp_nap_output_2_long_test.txt";
  WriteNAPOutput(my_output, filename, 1);
  int ear = 0;
  for (int timepoint = 0; timepoint < n_timepoints; ++timepoint) {
    for (int channel = 0; channel < n_channels; ++channel) {
      FPType cplusplus = my_output.nap()[timepoint][ear](channel);
      FPType matlab = nap1[timepoint](channel);
      ASSERT_NEAR(cplusplus, matlab, kPrecisionLevel);
      cplusplus = my_output.bm()[timepoint][ear](channel);
      matlab = bm1[timepoint](channel);
      ASSERT_NEAR(cplusplus, matlab, kPrecisionLevel);
    }
  }
  ear = 1;
  for (int timepoint = 0; timepoint < n_timepoints; ++timepoint) {
    for (int channel = 0; channel < n_channels; ++channel) {
      FPType cplusplus = my_output.nap()[timepoint][ear](channel);
      FPType matlab = nap2[timepoint](channel);
      ASSERT_NEAR(cplusplus, matlab, kPrecisionLevel);
      cplusplus = my_output.bm()[timepoint][ear](channel);
      matlab = bm2[timepoint](channel);
      ASSERT_NEAR(cplusplus, matlab, kPrecisionLevel);
    }
  }
}