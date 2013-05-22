//
//  carfac_test.h
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

#ifndef __CARFAC_Open_Source_C___Library__carfac_test__
#define __CARFAC_Open_Source_C___Library__carfac_test__

// <fstream> is currently used for reading text data in. This can be replaced by
// another file reader.
#include <fstream>
// GoogleTest is now included for running unit tests
#include <gtest/gtest.h>
#include "carfac.h"

// This variable defines the location of the test data used to compare this
// C++ version's output with that of the Matlab version. It should be changed
// when trying to build on a different system.
#define TEST_SRC_DIR "/Users/alexbrandmeyer/aimc/carfac/test_data/"
// Here we specify the level to which the output should match (10 decimals).
#define PRECISION_LEVEL 10.0e-11

#endif /* defined(__CARFAC_Open_Source_C___Library__carfac_test__) */
