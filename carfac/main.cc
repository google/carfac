//
//  main.cc
//  CARFAC Open Source C++ Library
//
//  Created by Alex Brandmeyer on 5/10/13.
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
//
// *****************************************************************************
// main.cc
// *****************************************************************************
// This 'main' file is not currently intended as part of the CARFAC distribution
// but serves as a testbed for debugging and implementing various aspects of the
// library. It currently includes the "libsndfile" API for loading soundfiles:
//
//    http://www.mega-nerd.com/libsndfile/
//
// I've currently tested the code on Mac OS X 10.8 using XCode and dynamically
// linking to the libsndfile.dylib file. Two helper functions 'ReadSound' and
// 'ReadSoundInfo' are used to obtain header information (needed for the CARFAC
// design stage) and sound data (for running the model).
#include <sndfile.h>
#include "carfac.h"
//GoogleTest is now included for running unit tests
#include <gtest/gtest.h>

// ReadSound takes a character array (filename string) as an argument and
// returns a two dimensional (samples x channels) FloatArray (Eigen ArrayXX)
// containing the sound data
FloatArray2d ReadSound(const char * filename) {
  // This output variable stores the sound data in an Eigen Array:
  FloatArray2d mysnd;
  // These two structures from the libsndfile library are used to store sound
  // file information and load the data.
  SNDFILE *sf; 
  SF_INFO info;
  // Several scalars are used to store relevant information needed for
  // transfering data from the WAV file to an Eigen Array. 
  int32_t num, num_items;
  double *buf;
  int32_t f, sr, c;
  // This opens the sound file and prints an error message if the file can't
  // be found.
  sf = sf_open(filename,SFM_READ,&info);
  if (sf == NULL)
  {
    return mysnd;
  }
  // Here we store relevant header information in our scalars and use them to
  // read into our data buffer.
  f = info.frames;
  sr = info.samplerate;
  c = info.channels;
  num_items = f * c;
  buf = new double[num_items];
  num = sf_read_double(sf, buf, num_items);
  // We resize our Eigen array to hold the correct number of samples and
  // audio channels.
  mysnd.resize(f, c);
  // Now we move through the data buffer and store the stereo audio channels
  // in the two columns of the Eigen Array
  // TODO alexbrandmeyer: this assumes stereo and will give problems otherwise.
  int j = 0;
  for(int i = 0; i < num_items; i = i + 2) {
    mysnd(j, 0) = buf[i];
    mysnd(j, 1) = buf[i + 1];
    j++;
  }
  // Now it's time to close the audio file and return the Eigen Array as output.
  sf_close(sf);
  return mysnd;
};

// ReadSoundInfo takes a character array (filename string) as an argument and
// returns an SF_INFO structure containing the sample rate and channel info
// needed during the call to CARFAC::Design
SF_INFO ReadSoundInfo(const char * filename) {
  SNDFILE *sf;
  SF_INFO info;
  sf = sf_open(filename,SFM_READ,&info);
  if (sf == NULL) {
    return info;
  }
  return info;
};

// This 'main' function serves as the primary testbed for this C++ CARFAC
// implementation. It currently uses a hardcoded filename to obtain sound file
// info and sound data, and designs a CARFAC on the basis of the header data
// using the default parameters.
int main(int argc, char **argv) {
  // This initializes the GoogleTest unit testing framework.
  ::testing::InitGoogleTest(&argc, argv);
  // Here we specify a path to a test file.
  const char * filename = "/Users/alexbrandmeyer/aimc/carfac/test_signal.wav";
  // This loads the header info and sound data.
  SF_INFO info = ReadSoundInfo(filename);
  FloatArray2d mysnd = ReadSound(filename);
  // These initialze the default parameter objects needed for the CARFAC design.
  CARParams *car_params = new CARParams();
  IHCParams *ihc_params = new IHCParams();
  AGCParams *agc_params = new AGCParams();
  // This initializes the CARFAC object and runs the design method.
  CARFAC *mycf = new CARFAC();
  mycf->Design(info.channels, info.samplerate, *car_params, *ihc_params,
               *agc_params);
  // This clears the parameters which are no longer needed.
  delete car_params;
  delete ihc_params;
  delete agc_params;
  // Now we run the model on the test data (using a closed loop for now).
  CARFACOutput output = mycf->Run(mysnd, false, true, true, true);
  // Finally we clear the CARFAC object when we're done.
  delete mycf;
  return 0;
}
