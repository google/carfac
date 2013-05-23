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
// library. I've currently tested the code on Mac OS X 10.8 using XCode.

#include "carfac_test.h"

// This 'main' function serves as the primary testbed for this C++ CARFAC
// implementation. The tests defined above are excuted by the 'RUN_ALL_TESTS()'
// function of the Google unit testing framework.
int main(int argc, char **argv) {
  // This initializes the GoogleTest unit testing framework.
  ::testing::InitGoogleTest(&argc, argv);
  // This runs all of the tests that we've defined above.
  return RUN_ALL_TESTS();
}
