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

#include <deque>
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

  // Appends a single frame of n_ears x n_channels data to the end of the
  // individual data members selected for storage.  This is called on a sample
  // by sample basis by CARFAC::RunSegment.
  void AppendOutput(const std::vector<Ear*>& ears);

  const std::deque<std::vector<ArrayX>>& nap() const { return nap_; }
  const std::deque<std::vector<ArrayX>>& bm() const { return bm_; }
  const std::deque<std::vector<ArrayX>>& ohc() const { return ohc_; }
  const std::deque<std::vector<ArrayX>>& agc() const { return agc_; }

 private:
  bool store_nap_;
  bool store_bm_;
  bool store_ohc_;
  bool store_agc_;

  // CARFAC outputs are stored in nested containers with dimensions:
  // n_frames x n_ears x n_channels.

  // Neural activity pattern rates.
  std::deque<std::vector<ArrayX>> nap_;
  // Basilar membrane displacement.
  std::deque<std::vector<ArrayX>> bm_;
  // Outer hair cell state.
  std::deque<std::vector<ArrayX>> ohc_;
  // Automatic gain control state.
  std::deque<std::vector<ArrayX>> agc_;

  DISALLOW_COPY_AND_ASSIGN(CARFACOutput);
};

#endif  // CARFAC_CARFAC_OUTPUT_H
