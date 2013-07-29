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
  //
  // If agc_params.num_stages == 0 the AGC will be disabled.
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
  // output.  Consumes the entire input signal and overwrites the contents of
  // output.
  //
  // The input sound_data should have size num_ears by num_samples.  Note that
  // this is the transpose of the input to CARFAC_Run_Segment.m
  void RunSegment(const ArrayXX& sound_data, bool open_loop,
                  CARFACOutput* output);

  int num_channels() const { return num_channels_; }
  // Returns an array of pole/center frequencies in Hertz for each output
  // channel.
  const ArrayX& pole_frequencies() const { return pole_freqs_; }

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

// Container for the different types of output from a CARFAC model.
class CARFACOutput {
 public:
  // The boolean argument indicate which portions of the CARFAC model's output
  // should be stored by subsequent calls to AppendOutput.
  //
  // TODO(ronw): Consider removing store_nap, unless there is a reasonable use
  // case for setting it to false?
  CARFACOutput(bool store_nap, bool store_bm, bool store_ohc, bool store_agc);

  // Data are stored in nested containers with dimensions:
  // num_ears by num_channels by num_samples.
  const std::vector<ArrayXX>& nap() const { return nap_; }
  std::vector<ArrayXX>* mutable_nap() { return &nap_; }
  const std::vector<ArrayXX>& bm() const { return bm_; }
  std::vector<ArrayXX>* mutable_bm() { return &bm_; }
  const std::vector<ArrayXX>& ohc() const { return ohc_; }
  std::vector<ArrayXX>* mutable_ohc() { return &ohc_; }
  const std::vector<ArrayXX>& agc() const { return agc_; }
  std::vector<ArrayXX>* mutable_agc() { return &agc_; }

 private:
  friend class CARFAC;

  // Resizes the internal containers for each output type, destroying the
  // previous contents.  Must be called before AssignFromEars.
  void Resize(int num_ears, int num_channels, int num_samples);

  // For each ear, assigns a single frame of state at time sample_index to the
  // the individual data members selected for storage.  sample_index must be
  // less than num_samples specified in the last call to Resize.
  //
  // This is called on a sample by sample basis by CARFAC::RunSegment.
  void AssignFromEars(const std::vector<Ear*>& ears, int sample_index);

  bool store_nap_;
  bool store_bm_;
  bool store_ohc_;
  bool store_agc_;

  // Neural activity pattern rates.
  std::vector<ArrayXX> nap_;
  // Basilar membrane displacement.
  std::vector<ArrayXX> bm_;
  // Outer hair cell state.
  std::vector<ArrayXX> ohc_;
  // Automatic gain control state.
  std::vector<ArrayXX> agc_;

  DISALLOW_COPY_AND_ASSIGN(CARFACOutput);
};

#endif  // CARFAC_CARFAC_H
