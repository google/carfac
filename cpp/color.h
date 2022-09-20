// Copyright 2022 The CARFAC Authors. All Rights Reserved.
// Author: Pascal Getreuer
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

// Utilities for working with colors.

#ifndef CARFAC_COLOR_H_
#define CARFAC_COLOR_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "common.h"

// Templated class representing a color. The `Scalar` template arg may be either
// uint8_t or float. Color<float> inherits from Eigen::Array3f (and similarly
// for Color<uint8_t>) so that the color may be manipulated with Eigen APIs.
template <typename _Scalar /* uint8_t or float */>
class Color : public Eigen::Array<_Scalar, 3, 1> {
 public:
  using Scalar = _Scalar;
  using Array = Eigen::Array<Scalar, 3, 1>;

  Color(): Array() {}
  template<typename OtherDerived>
  Color(const Eigen::ArrayBase<OtherDerived>& other): Array(other) {}
  template<typename OtherDerived>
  Color& operator=(const Eigen::ArrayBase<OtherDerived>& other) {
    this->Array::operator=(other);
    return *this;
  }

  // Constructs from 3 Scalars.
  Color(Scalar r, Scalar g, Scalar b): Array(r, g, b) {}

  // Creates a grayscale Color (value, value, value).
  static Color Gray(Scalar value) { return Color(value, value, value); }
};

// Converts a Color from sRGB to grayscale using the Rec. 601 coefficients.
uint8_t RgbToGray(const Color<uint8_t>& color);
float RgbToGray(const Color<float>& color);

// Struct representing a 256-entry colormap / color ramp.
struct Colormap {
  // Nearest neighbor sampling of the colormap color at 0 <= x <= 1. If x is
  // outside this range, it is saturated.
  Color<uint8_t> operator()(float x) const {
    const int index = static_cast<int>(std::min<float>(
        std::max<float>(std::round((kLevels - 1) * x), 0.0f), kLevels - 1));
    return Color<uint8_t>::Map(lut[index]);
  }

  enum { kLevels = 256 };  // Number of colors in the colormap.
  // Lookup table of colors where each entry is a (red, green, blue, padding)
  // 4-byte tuple. The padding byte is such that the ith color is an efficient
  // power-of-two offset into the table, `lut[i]` = `&lut[0][0] + 4 * i`.
  uint8_t lut[kLevels][4];
};

// Matplotlib's "magma" colormap, a perceptually-linear colormap that fades
// through black -> purple -> yellow -> white.
extern const Colormap kMagmaColormap;

#endif  // CARFAC_COLOR_H_
