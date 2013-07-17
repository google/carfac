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

CARFACOutput::CARFACOutput(const bool store_nap, const bool store_bm,
                           const bool store_ohc, const bool store_agc) {
  store_nap_ = store_nap;
  store_bm_ = store_bm;
  store_ohc_ = store_ohc;
  store_agc_ = store_agc;
}

void CARFACOutput::AppendOutput(const vector<Ear*>& ears) {
  if (store_nap_) {
    nap_.push_back(vector<ArrayX>());
    for (Ear* ear : ears) {
      nap_.back().push_back(ear->ihc_out());
    }
  }
  if (store_ohc_) {
    ohc_.push_back(vector<ArrayX>());
    for (Ear* ear : ears) {
      ohc_.back().push_back(ear->za_memory());
    }
  }
  if (store_agc_) {
    agc_.push_back(vector<ArrayX>());
    for (Ear* ear : ears) {
      agc_.back().push_back(ear->zb_memory());
    }
  }
  if (store_bm_) {
    bm_.push_back(vector<ArrayX>());
    for (Ear* ear : ears) {
      bm_.back().push_back(ear->zy_memory());
    }
  }
}
