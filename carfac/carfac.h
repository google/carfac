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

#include "agc.h"
#include "car.h"
#include "common.h"
#include "ihc.h"

class CARFACOutput;
class Ear;

// Top-level class implementing the CAR-FAC C++ model. See the chapter entitled
// 'The CAR-FAC Digital Cochlear Model' in Lyon's book "Human and Machine
// Hearing" for an overview.
//
// A CARFAC object knows how to design its details from a modest set of
// parameters, and knows how to process sound signals to produce "neural
// activity patterns" (NAPs) using CARFAC::RunSegment.
class CARFAC {
 public:
  // Constructs a vector of Ear objects, one for each input audio channel,
  // using the given CAR, IHC and AGC parameters.
  CARFAC(const int num_ears, const FPType sample_rate,
         const CARParams& car_params, const IHCParams& ihc_params,
         const AGCParams& agc_params);
  ~CARFAC();

  // Reinitialize using the specified parameters.
  void Redesign(const int num_ears, const FPType sample_rate,
                const CARParams& car_params, const IHCParams& ihc_params,
                const AGCParams& agc_params);

  // Reset the internal state so that subsequent calls to RunSegment are
  // independent of previous calls.  Does not modify the filterbank design.
  void Reset();

  // Processes an individual sound segment and copies the model output to
  // seg_output.
  //
  // The input sound_data should contain a vector of audio samples for each
  // ear.
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

  // Computes the nominal Equivalent Rectangular Bandwidth (ERB) of an auditory
  // filter at the given center frequency.
  // Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
  // See also the section 'Auditory Frequency Scales' of the chapter 'Acoustic
  // Approaches and Auditory Influence' in "Human and Machine Hearing".
  FPType ERBHz(const FPType center_frequency_hz, const FPType erb_break_freq,
               const FPType erb_q) const;

  CARParams car_params_;
  IHCParams ihc_params_;
  AGCParams agc_params_;
  int num_ears_;
  FPType sample_rate_;
  int num_channels_;
  FPType max_channels_per_octave_;

  // One Ear per input audio channel.
  std::vector<Ear*> ears_;
  ArrayX pole_freqs_;

  DISALLOW_COPY_AND_ASSIGN(CARFAC);
};

#endif  // CARFAC_CARFAC_H
