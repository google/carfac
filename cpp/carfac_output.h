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

#ifndef CARFAC_CARFAC_OUTPUT_H
#define CARFAC_CARFAC_OUTPUT_H

#include <vector>

#include "common.h"

class Ear;

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

#endif  // CARFAC_CARFAC_OUTPUT_H
