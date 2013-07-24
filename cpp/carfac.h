// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Alex Brandmeyer
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

// This header declares the main interface for the CARFAC cochlear model.

#ifndef CARFAC_CARFAC_H
#define CARFAC_CARFAC_H

#include <vector>

#include "agc.h"
#include "car.h"
#include "common.h"
#include "ihc.h"

class CARFACOutput;
class Ear;

// Top-level class implementing the Cascade of Asymmetric Resonators
// with Fast-Acting Compression (CARFAC) cochlear model.
//
// A CARFAC object knows how to design its details from a modest set of
// parameters, and knows how to process sound signals to produce "neural
// activity patterns" (NAPs) using the RunSegment method.
class CARFAC {
 public:
  // Constructs a vector of Ear objects, one for each input audio channel,
  // using the given CAR, IHC and AGC parameters.
  CARFAC(int num_ears, FPType sample_rate, const CARParams& car_params,
         const IHCParams& ihc_params, const AGCParams& agc_params);
  ~CARFAC();

  // Reinitializes using the specified parameters.
  void Redesign(int num_ears, FPType sample_rate, const CARParams& car_params,
                const IHCParams& ihc_params, const AGCParams& agc_params);

  // Resets the internal state so that subsequent calls to RunSegment are
  // independent of previous calls.  Does not modify the filterbank design.
  void Reset();

  // Processes an individual sound segment and stores the model output in
  // output.  Consumes the entire input signal.
  //
  // The input sound_data should contain a vector of audio samples for
  // each ear, i.e. the outer vector should have size num_ears, and
  // the inner vector should have size num_samples.
  void RunSegment(const std::vector<std::vector<float>>& sound_data,
                  bool open_loop, CARFACOutput* output);

  int num_channels() const { return num_channels_; }

 private:
  static void DesignCARCoeffs(const CARParams& car_params, FPType sample_rate,
                              const ArrayX& pole_freqs, CARCoeffs* car_coeffs);
  static void DesignIHCCoeffs(const IHCParams& ihc_params, FPType sample_rate,
                              IHCCoeffs* ihc_coeffs);
  static void DesignAGCCoeffs(const AGCParams& agc_params, FPType sample_rate,
                              std::vector<AGCCoeffs>* agc_coeffs);
  void CrossCouple();
  void CloseAGCLoop();

  // Computes the nominal Equivalent Rectangular Bandwidth (ERB) of an auditory
  // filter at the given center frequency.
  // Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
  static FPType ERBHz(FPType center_frequency_hz, FPType erb_break_freq,
                      FPType erb_q);

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
