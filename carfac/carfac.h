//
//  carfac.h
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

#ifndef CARFAC_CARFAC_H
#define CARFAC_CARFAC_H

#include <vector>

#include "common.h"
#include "carfac_util.h"
#include "car.h"
#include "ihc.h"
#include "agc.h"
#include "ear.h"
#include "carfac_output.h"

// This is the top-level class implementing the CAR-FAC C++ model. See the
// chapter entitled 'The CAR-FAC Digital Cochlear Model' in Lyon's book "Human
// and Machine Hearing" for an overview.
//
// A CARFAC object knows how to design its details from a modest set of
// parameters, and knows how to process sound signals to produce "neural
// activity patterns" (NAPs) which are stored in a CARFACOutput object during
// the call to CARFAC::RunSegment.
class CARFAC {
 public:
  // The 'Design' method takes a set of CAR, IHC and AGC parameters along with
  // arguments specifying the number of 'ears' (audio file channels) and sample
  // rate. This initializes a vector of 'Ear' objects -- one for mono, two for
  // stereo, or more.
  CARFAC(const int num_ears, const FPType sample_rate,
         const CARParams& car_params, const IHCParams& ihc_params,
         const AGCParams& agc_params);

  void Reset(const int num_ears, const FPType sample_rate,
             const CARParams& car_params, const IHCParams& ihc_params,
             const AGCParams& agc_params);

  // The 'RunSegment' method processes individual sound segments and stores the
  // output of the model in a CARFACOutput object.
  void RunSegment(const std::vector<std::vector<float>>& sound_data,
                  const int32_t start, const int32_t length,
                  const bool open_loop, CARFACOutput* seg_output);

 private:
  void DesignCARCoeffs(const CARParams& car_params, const FPType sample_rate,
                       const ArrayX& pole_freqs, CARCoeffs* car_coeffs);
  void DesignIHCCoeffs(const IHCParams& ihc_params, const FPType sample_rate,
                       IHCCoeffs* ihc_coeffs);
  void DesignAGCCoeffs(const AGCParams& agc_params, const FPType sample_rate,
                       std::vector<AGCCoeffs>* agc_coeffs);
  void CrossCouple();
  void CloseAGCLoop();

  // Function: ERBHz
  // Auditory filter nominal Equivalent Rectangular Bandwidth
  // Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
  // See also the section 'Auditory Frequency Scales' of the chapter 'Acoustic
  // Approaches and Auditory Influence' in "Human and Machine Hearing".
  FPType ERBHz(const FPType center_frequency_hz, const FPType erb_break_freq,
               const FPType erb_q);

  CARParams car_params_;
  IHCParams ihc_params_;
  AGCParams agc_params_;
  int num_ears_;
  FPType sample_rate_;
  int num_channels_;
  FPType max_channels_per_octave_;

  // We store a vector of Ear objects for mono/stereo/multichannel processing:
  std::vector<Ear*> ears_;
  ArrayX pole_freqs_;

  DISALLOW_COPY_AND_ASSIGN(CARFAC);
};

#endif  // CARFAC_CARFAC_H
