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

// Shared CARFAC utility functions.

#ifndef CARFAC_CARFAC_UTIL_H
#define CARFAC_CARFAC_UTIL_H

#include "common.h"

// Computes the IHC detection nonlinearity function of the filter output
// values.  This is here because it is called both in design and run phases.
inline void CARFACDetect(ArrayX* input_output) {
  constexpr FPType a = 0.175;
  constexpr FPType b = 0.1;
  // This offsets the low-end tail into negative x territory.
  // The parameter a is adjusted for the book, to make the 20% DC response
  // threshold at 0.1.
  ArrayX& c = *input_output;
  c = c.cwiseMax(-a) + a;
  // Zero is the final answer for many points.
  *input_output = c.cube() / (c.cube() + c.square() + b);
}

#endif  // CARFAC_CARFAC_UTIL_H
