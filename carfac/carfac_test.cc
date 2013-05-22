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

#include "carfac_test.h"
// Three helper functions are defined here for loading the test data generated
// by the Matlab version of CARFAC.
// This loads one-dimensional FloatArrays from single-column text files.
FloatArray LoadTestData(const std::string filename, const int number_points) {
  std::string fullfile = TEST_SRC_DIR + filename;
  std::ifstream file(fullfile.c_str());
  FPType myarray[number_points];
  FloatArray output(number_points);
  if (file.is_open()) {
    for (int i = 0; i < number_points; ++i) {
      file >> myarray[i];
      output(i) = myarray[i];
    }
  }
  return output;
}

// This loads two-dimensional FloatArrays from multi-column text files.
std::vector<FloatArray> Load2dTestData(const std::string filename, const int rows,
                            const int columns) {
  std::string fullfile = TEST_SRC_DIR + filename;
  std::ifstream file(fullfile.c_str());
  FPType myarray[rows][columns];
  std::vector<FloatArray> output;
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
  return output;
}

// This loads two dimensional vectors of audio data using data generated in
// Matlab using the wavread() function.
std::vector<std::vector<float>> Load2dAudioVector(std::string filename,
                                                  int timepoints,
                                                  int channels) {
  std::string fullfile = TEST_SRC_DIR + filename;
  std::ifstream file(fullfile.c_str());
  std::vector<std::vector<float>> output;
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
  return output;
}

// The first test verifies that the resulting CAR coefficients are the same as
// in Matlab when using the default CAR parameter set.
TEST(CARFACTest, CARCoeffs_Test){
  // These initialze the CAR Params and Coeffs objects needed for this test.
  CARParams car_params;
  CARCoeffs car_coeffs;
  FPType fs = 22050.0;
  // We calculate the pole frequencies and number of channels in the same way
  // as in the CARFAC 'Design' method.
  int n_ch = 0;
  FPType pole_hz = car_params.first_pole_theta_ * fs / (2 * PI);
  while (pole_hz > car_params.min_pole_hz_) {
    n_ch++;
    pole_hz = pole_hz - car_params.erb_per_step_ *
    ERBHz(pole_hz, car_params.erb_break_freq_, car_params.erb_q_);
  }
  FloatArray pole_freqs(n_ch);
  pole_hz = car_params.first_pole_theta_ * fs / (2 * PI);
  for (int ch = 0; ch < n_ch; ++ch) {
    pole_freqs(ch) = pole_hz;
    pole_hz = pole_hz - car_params.erb_per_step_ *
    ERBHz(pole_hz, car_params.erb_break_freq_, car_params.erb_q_);
  }
  // This initializes the CAR coeffecients object and runs the design method.
  car_coeffs.Design(car_params, 22050, pole_freqs);
  // Now we go through each set of coefficients to verify that the values are
  // the same as in MATLAB.
  std::string filename;
  FloatArray output;
  
  ASSERT_EQ(car_coeffs.v_offset_, 0.04);
  ASSERT_EQ(car_coeffs.velocity_scale_, 0.1);
  
  filename = "r1_coeffs.txt";
  output = LoadTestData(filename, n_ch);
  for (int i = 0; i < n_ch; ++i) {
    ASSERT_NEAR(output(i), car_coeffs.r1_coeffs_(i), PRECISION_LEVEL);
  }
  
  filename = "a0_coeffs.txt";
  output = LoadTestData(filename, n_ch);
  for (int i = 0; i < n_ch; ++i) {
    ASSERT_NEAR(output(i), car_coeffs.a0_coeffs_(i), PRECISION_LEVEL);
  }
  
  filename = "c0_coeffs.txt";
  output = LoadTestData(filename, n_ch);
  for (int i = 0; i < n_ch; ++i) {
    ASSERT_NEAR(output(i), car_coeffs.c0_coeffs_(i), PRECISION_LEVEL);
  }
  
  filename = "zr_coeffs.txt";
  output = LoadTestData(filename, n_ch);
  for (int i = 0; i < n_ch; ++i) {
    ASSERT_NEAR(output(i), car_coeffs.zr_coeffs_(i), PRECISION_LEVEL);
  }
  
  filename = "h_coeffs.txt";
  output = LoadTestData(filename, n_ch);
  for (int i = 0; i < n_ch; ++i) {
    ASSERT_NEAR(output(i), car_coeffs.h_coeffs_(i), PRECISION_LEVEL);
  }
  
  filename = "g0_coeffs.txt";
  output = LoadTestData(filename, n_ch);
  for (int i = 0; i < n_ch; ++i) {
    ASSERT_NEAR(output(i), car_coeffs.g0_coeffs_(i), PRECISION_LEVEL);
  }
}

// The second test verifies that the IHC coefficient calculations result in the
// same set of values as in the Matlab version of the CARFAC.
TEST(CARFACTest, IHCCoeffs_Test){
  IHCParams ihc_params;
  IHCCoeffs ihc_coeffs;
  FPType fs = 22050.0;
  ihc_coeffs.Design(ihc_params, fs);
  
  std::string filename = "ihc_coeffs.txt";
  FloatArray output = LoadTestData(filename, 9);
  
  // The sequence of the individual coefficients is determined using the
  // CARFAC_GenerateTestData() function in the Matlab version, with all of the
  // parameters placed in a single output file for convenience.
  bool just_hwr = output(0);
  FPType lpf_coeff = output(1);
  FPType out_rate = output(2);
  FPType in_rate = output(3);
  bool one_cap = output(4);
  FPType output_gain = output(5);
  FPType rest_output = output(6);
  FPType rest_cap = output(7);
  FPType ac_coeff = output(8);
  
  // Once we have the Matlab values initialized, we can compare them to the
  // output of the IHCCoeffs 'Design' method.
  ASSERT_EQ(just_hwr, ihc_coeffs.just_hwr_);
  ASSERT_NEAR(lpf_coeff, ihc_coeffs.lpf_coeff_, PRECISION_LEVEL);
  ASSERT_NEAR(out_rate, ihc_coeffs.out1_rate_, PRECISION_LEVEL);
  ASSERT_NEAR(in_rate, ihc_coeffs.in1_rate_, PRECISION_LEVEL);
  ASSERT_EQ(one_cap, ihc_coeffs.one_cap_);
  ASSERT_NEAR(output_gain, ihc_coeffs.output_gain_, PRECISION_LEVEL);
  ASSERT_NEAR(rest_output, ihc_coeffs.rest_output_, PRECISION_LEVEL);
  ASSERT_NEAR(rest_cap, ihc_coeffs.rest_cap1_, PRECISION_LEVEL);
  ASSERT_NEAR(ac_coeff, ihc_coeffs.ac_coeff_, PRECISION_LEVEL);
}


TEST(CARFACTest, AGCCoeffs_Test) {
  AGCParams agc_params;
  std::vector<AGCCoeffs> agc_coeffs;
  std::vector<FloatArray> output;
  output.resize(agc_params.n_stages_);
  std::string filename = "agc_coeffs_1.txt";
  output[0] = LoadTestData(filename, 14);
  filename = "agc_coeffs_2.txt";
  output[1] = LoadTestData(filename, 14);
  filename = "agc_coeffs_3.txt";
  output[2] = LoadTestData(filename, 14);
  filename = "agc_coeffs_4.txt";
  output[3] = LoadTestData(filename, 14);
  agc_coeffs.resize(agc_params.n_stages_);
  // We initialize the AGC stages in the same was as in Ear::Init.
  FPType fs = 22050.0;
  FPType previous_stage_gain = 0.0;
  FPType decim = 1.0;
  for (int stage = 0; stage < agc_params.n_stages_; ++stage) {
    agc_coeffs[stage].Design(agc_params, stage, fs, previous_stage_gain, decim);
    previous_stage_gain = agc_coeffs[stage].agc_gain_;
    decim = agc_coeffs[stage].decim_;
  }
  // Now we run through the individual coefficients and verify that they're the
  // same as in Matlab.
  for (int stage = 0; stage < agc_params.n_stages_; ++stage) {
    int n_agc_stages = output[stage](1);
    FPType agc_stage_gain = output[stage](2);
    int decimation = output[stage](3);
    FPType agc_epsilon = output[stage](4);
    FPType agc_polez1 = output[stage](5);
    FPType agc_polez2 = output[stage](6);
    int agc_spatial_iterations = output[stage](7);
    FPType agc_spatial_fir_1 = output[stage](8);
    FPType agc_spatial_fir_2 = output[stage](9);
    FPType agc_spatial_fir_3 = output[stage](10);
    int agc_spatial_n_taps = output[stage](11);
    FPType agc_mix_coeffs = output[stage](12);
    FPType detect_scale = output[stage](13);
    
    ASSERT_EQ(n_agc_stages, agc_coeffs[stage].n_agc_stages_);
    ASSERT_NEAR(agc_stage_gain, agc_coeffs[stage].agc_stage_gain_,
                PRECISION_LEVEL);
    ASSERT_EQ(decimation, agc_coeffs[stage].decimation_);
    ASSERT_NEAR(agc_epsilon, agc_coeffs[stage].agc_epsilon_, PRECISION_LEVEL);
    ASSERT_NEAR(agc_polez1, agc_coeffs[stage].agc_pole_z1_, PRECISION_LEVEL);
    ASSERT_NEAR(agc_polez2, agc_coeffs[stage].agc_pole_z2_, PRECISION_LEVEL);
    ASSERT_EQ(agc_spatial_iterations,
              agc_coeffs[stage].agc_spatial_iterations_);
    ASSERT_NEAR(agc_spatial_fir_1, agc_coeffs[stage].agc_spatial_fir_[0],
                PRECISION_LEVEL);
    ASSERT_NEAR(agc_spatial_fir_2, agc_coeffs[stage].agc_spatial_fir_[1],
                PRECISION_LEVEL);
    ASSERT_EQ(agc_spatial_n_taps,
              agc_coeffs[stage].agc_spatial_n_taps_);
    ASSERT_NEAR(agc_spatial_fir_3, agc_coeffs[stage].agc_spatial_fir_[2],
                PRECISION_LEVEL);
    ASSERT_NEAR(agc_mix_coeffs, agc_coeffs[stage].agc_mix_coeffs_,
                PRECISION_LEVEL);
    
    // The last stage will have the correct detect_scale_ value on the basis of
    // the total gain accumlated over the stages.
    if (stage == agc_params.n_stages_ - 1) {
      ASSERT_NEAR(detect_scale, agc_coeffs[stage].detect_scale_,
                  PRECISION_LEVEL);
    }
  }
}

// This test verifies the output of the C++ code relative to that of the Matlab
// version using a single segment (441 samples) of audio from the "plan.wav"
// file. The single-channel audio data and different output matrices from Matlab
// are stored in text files and then read into 2d Eigen arrays (for now, this
// should be changed to a vector of FloatArrays... TODO (alexbrandmeyer)). For
// reference, see the CARFAC_GenerateTestData() function in the Matlab branch
// of the repository.
//
// A single Ear object is used along with the code from CARFAC.RunSegment() to
// evaluate the output of the CAR and IHC steps on a sample by sample basis
// relative to the output read in from Matlab. The test passes with 11 degrees
// of precision, with the Matlab data stored using 12 decimals.
//
// TODO (alexbrandmeyer): A subseqent version of this test will operate directly
// on the CARFACOutput structure and will evaluate binaural data.
TEST(CARFACTest, Monaural_Output_Test) {
  std::string filename = "monaural_test_nap.txt";
  std::vector<FloatArray> nap = Load2dTestData(filename, 441, 71);
  filename = "monaural_test_bm.txt";
  std::vector<FloatArray> bm = Load2dTestData(filename, 441, 71);
  filename = "monaural_test_ohc.txt";
  std::vector<FloatArray> ohc = Load2dTestData(filename, 441, 71);
  filename = "monaural_test_agc.txt";
  std::vector<FloatArray> agc = Load2dTestData(filename, 441, 71);
  filename = "file_signal_monaural_test.txt";
  std::vector<std::vector<float>> sound_data = Load2dAudioVector(filename, 441,
                                                                 1);
  // The number of timepoints is determined from the length of the audio
  // segment.
  int32_t n_timepoints = sound_data[0].size();
  
  CARParams car_params;
  IHCParams ihc_params;
  AGCParams agc_params;
  FPType fs = 22050.0;
  int n_ch = 0;
  FPType pole_hz = car_params.first_pole_theta_ * fs / (2 * PI);
  while (pole_hz > car_params.min_pole_hz_) {
    n_ch++;
    pole_hz = pole_hz - car_params.erb_per_step_ *
    ERBHz(pole_hz, car_params.erb_break_freq_, car_params.erb_q_);
  }
  FloatArray pole_freqs(n_ch);
  pole_hz = car_params.first_pole_theta_ * fs / (2 * PI);
  for (int ch = 0; ch < n_ch; ++ch) {
    pole_freqs(ch) = pole_hz;
    pole_hz = pole_hz - car_params.erb_per_step_ *
    ERBHz(pole_hz, car_params.erb_break_freq_, car_params.erb_q_);
  }
  
  // This initializes the CARFAC object and runs the design method.
  Ear ear;
  ear.InitEar(n_ch, fs, pole_freqs, car_params, ihc_params, agc_params);
  
  CARFACOutput seg_output;
  seg_output.InitOutput(1, n_ch, n_timepoints);
  
  // A nested loop structure is used to iterate through the individual samples
  // for each ear (audio channel).
  FloatArray car_out(n_ch);
  FloatArray ihc_out(n_ch);
  FloatArray matlab_car_out(n_ch);
  FloatArray matlab_ihc_out(n_ch);
  bool updated;  // This variable is used by the AGC stage.
  for (int32_t i = 0; i < n_timepoints; ++i) {
    int j = 0;
    // First we create a reference to the current Ear object.
    // This stores the audio sample currently being processed.
    FPType input = sound_data[j][i];
    // Now we apply the three stages of the model in sequence to the current
    // audio sample.
    ear.CARStep(input, &car_out);
    matlab_car_out = bm[i];
    // This step verifies that the ouput of the CAR step is the same at each
    // timepoint and channel as that of the Matlab version.
    for (int channel = 0; channel < n_ch; ++channel) {
      FPType a = matlab_car_out(channel);
      FPType b = car_out(channel);
      ASSERT_NEAR(a, b, PRECISION_LEVEL);
    }
    ear.IHCStep(car_out, &ihc_out);
    matlab_ihc_out = nap[i];
    // This step verifies that the ouput of the IHC step is the same at each
    // timepoint and channel as that of the Matlab version.
    for (int channel = 0; channel < n_ch; ++channel) {
      FPType a = matlab_ihc_out(channel);
      FPType b = ihc_out(channel);
      ASSERT_NEAR(a, b, PRECISION_LEVEL);
    }
    
    updated = ear.AGCStep(ihc_out);
    // These lines assign the output of the model for the current sample
    // to the appropriate data members of the current ear in the output
    // object.
    seg_output.StoreNAPOutput(i, j, ihc_out);
    seg_output.StoreBMOutput(i, j, car_out);
    seg_output.StoreOHCOutput(i, j, ear.za_memory());
    seg_output.StoreAGCOutput(i, j, ear.zb_memory());
    if (updated) {
      FloatArray undamping = 1 - ear.agc_memory(0);
      // This updates the target stage gain for the new damping.
      ear.set_dzb_memory((ear.zr_coeffs() * undamping - ear.zb_memory()) /
                         ear.agc_decimation(0));
      ear.set_dg_memory((ear.StageGValue(undamping) - ear.g_memory()) /
                        ear.agc_decimation(0));
    }
  }
}