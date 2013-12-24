// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Ron Weiss
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

#include <memory>
#include <emscripten/bind.h>

#include "carfac.h"

using namespace emscripten;

// A more restricted interface to CARFAC that is easier to interact
// with from Javascript (i.e. uses vector<float> instead of Eigen
// arrays) when compiled using emscripten.
class EmscriptenCARFAC {
 public:
  explicit EmscriptenCARFAC(float sample_rate) {
    constexpr int kNumEars = 1;
    CARParams car_params;
    IHCParams ihc_params;
    AGCParams agc_params;
    carfac_.reset(new CARFAC(kNumEars, sample_rate, car_params, ihc_params,
                             agc_params));
  }

  void Reset() { carfac_->Reset(); }
  int num_channels() { return carfac_->num_channels(); }

  // The output buffer is a flattened vector containing the NAP
  // CARFACOutput in column-major order.
  std::vector<float> RunSegment(const std::vector<float>& input_samples) {
    constexpr int kNumInputEars = 1;
    auto input_map = ArrayXX::Map(&input_samples.front(), kNumInputEars,
                                  input_samples.size());
    // Only store NAP outputs.
    CARFACOutput output(true, false, false, false);
    carfac_->RunSegment(input_map, false /* open_loop */, &output);

    std::vector<float> output_vector(
        input_samples.size() * carfac_->num_channels());
    auto output_map = ArrayXX::Map(&output_vector.front(),
                                   carfac_->num_channels(),
                                   input_samples.size());
    output_map = output.nap()[0];
    return output_vector;
  }

 private:
  std::unique_ptr<CARFAC> carfac_;
};

EMSCRIPTEN_BINDINGS(EmscriptenCARFAC) {
  register_vector<float>("FloatVector");
  class_<EmscriptenCARFAC>("EmscriptenCARFAC")
    .constructor<float>()
    .function("Reset", &EmscriptenCARFAC::Reset)
    .function("num_channels", &EmscriptenCARFAC::num_channels)
    .function("RunSegment", &EmscriptenCARFAC::RunSegment, allow_raw_pointers())
    ;
}
