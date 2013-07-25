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

#include "carfac_output.h"
#include "ear.h"

using std::vector;

CARFACOutput::CARFACOutput(bool store_nap, bool store_bm, bool store_ohc,
                           bool store_agc) {
  store_nap_ = store_nap;
  store_bm_ = store_bm;
  store_ohc_ = store_ohc;
  store_agc_ = store_agc;
}

namespace {
void ResizeContainer(int num_ears, int num_channels, int num_samples,
                     vector<ArrayXX>* container) {
  container->resize(num_ears);
  for (ArrayXX& matrix : *container) {
    matrix.resize(num_channels, num_samples);
  }
}
}  // anonymous namespace

void CARFACOutput::Resize(int num_ears, int num_channels, int num_samples) {
  if (store_nap_) {
    ResizeContainer(num_ears, num_channels, num_samples, &nap_);
  }
  if (store_ohc_) {
    ResizeContainer(num_ears, num_channels, num_samples, &ohc_);
  }
  if (store_agc_) {
    ResizeContainer(num_ears, num_channels, num_samples, &agc_);
  }
  if (store_bm_) {
    ResizeContainer(num_ears, num_channels, num_samples, &bm_);
  }
}

void CARFACOutput::AssignFromEars(const vector<Ear*>& ears, int sample_index) {
  for (int i = 0; i < ears.size(); ++i) {
    const Ear* ear = ears[i];
    if (store_nap_) {
      nap_[i].col(sample_index) = ear->ihc_out();
    }
    if (store_ohc_) {
      ohc_[i].col(sample_index) = ear->za_memory();
    }
    if (store_agc_) {
      agc_[i].col(sample_index) = ear->zb_memory();
    }
    if (store_bm_) {
      bm_[i].col(sample_index) = ear->zy_memory();
    }
  }
}
