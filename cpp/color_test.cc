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

#include "color.h"

#include "gtest/gtest.h"

using ::testing::PrintToString;

// Definitions so that GUnit can print Colors.
void PrintTo(const Color<uint8_t>& color, std::ostream* os) {
  *os << '(' << static_cast<int>(color[0]) << ", "
             << static_cast<int>(color[1]) << ", "
             << static_cast<int>(color[2]) << ')';
}
void PrintTo(const Color<float>& color, std::ostream* os) {
  *os << '(' << color[0] << ", " << color[1] << ", " << color[2] << ')';
}

// Asserts that two Colors are close, |actual[c] - expected[c]| <= tol.
template <typename Scalar>
testing::AssertionResult ColorIsClose(const Color<Scalar>& actual,
                                      const Color<Scalar>& expected,
                                      double tol) {
  // Compute the max absolute difference, casting to double first to prevent
  // overflow if the Scalar type is uint8_t.
  const double diff = (actual.template cast<double>() -
                       expected.template cast<double>())
                          .matrix().template lpNorm<Eigen::Infinity>();
  if (diff <= tol) {
    return testing::AssertionSuccess();
  } else {
    auto failure = testing::AssertionFailure()
                   << PrintToString(actual)
                   << (tol == 0.0 ? " is not equal to " : " differs from ")
                   << PrintToString(expected);
    if (tol != 0.0) { failure << " by more than " << tol; }
    return failure;
  }
}

// Asserts that two Colors are equal.
template <typename Scalar>
testing::AssertionResult ColorIsEqual(const Color<Scalar>& actual,
                                      const Color<Scalar>& expected) {
  return ColorIsClose(actual, expected, 0.0);
}

namespace {

TEST(ColorTest, BasicUint8) {
  Color<uint8_t> color(120, 71, 5);
  ASSERT_EQ(color[0], 120);
  ASSERT_EQ(color[1], 71);
  ASSERT_EQ(color[2], 5);

  Color<uint8_t> other;
  other = color;
  ASSERT_TRUE(ColorIsEqual(other, {120, 71, 5}));
}

TEST(ColorTest, BasicFloat) {
  Color<float> color(0.1f, 0.2f, 0.3f);
  ASSERT_TRUE(ColorIsEqual(color, {0.1f, 0.2f, 0.3f}));

  Color<float> other;
  other = color;
  ASSERT_TRUE(ColorIsEqual(other, {0.1f, 0.2f, 0.3f}));
}

TEST(ColorTest, InheritsEigenMethods) {
  Color<uint8_t> color;
  color.setConstant(40);
  ASSERT_TRUE(ColorIsEqual(color, Color<uint8_t>::Gray(40)));
}

TEST(ColorTest, CopyToFrom) {
  Color<uint8_t> color(120, 71, 5);

  uint8_t buffer[3];
  Color<uint8_t>::Map(buffer) = color;

  Color<uint8_t> other = Color<uint8_t>::Map(buffer);
  ASSERT_TRUE(ColorIsEqual(other, {120, 71, 5}));
}

TEST(ColorTest, ToGrayUint8) {
  ASSERT_EQ(RgbToGray(Color<uint8_t>(120, 71, 5)), 78);
  ASSERT_EQ(RgbToGray(Color<uint8_t>(30, 190, 255)), 149);

  for (int value = 0; value < 256; ++value) {
    ASSERT_EQ(RgbToGray(Color<uint8_t>::Gray(value)), value);
  }
}

TEST(ColorTest, ToGrayFloat) {
  ASSERT_NEAR(RgbToGray(Color<float>(0.471f, 0.278f, 0.020f)), 0.306f, 5e-4f);

  for (int i = 0; i < 256; ++i) {
    const float value = i / 255.0f;
    ASSERT_NEAR(RgbToGray(Color<float>::Gray(value)), value, 1e-6f);
  }
}

TEST(ColorTest, Colormap) {
  Colormap colormap;
  for (int i = 0; i < Colormap::kLevels; ++i) {
    colormap.lut[i][0] = i;
    colormap.lut[i][1] = i / 2;
    colormap.lut[i][2] = 255 - i;
    colormap.lut[i][3] = 255;
  }

  constexpr int kTol = 2;
  ASSERT_TRUE(ColorIsClose(colormap(0.1f), {26, 13, 229}, kTol));
  ASSERT_TRUE(ColorIsClose(colormap(0.2f), {51, 25, 204}, kTol));
  ASSERT_TRUE(ColorIsClose(colormap(0.7f), {179, 89, 76}, kTol));
  ASSERT_TRUE(ColorIsClose(colormap(0.9f), {229, 114, 26}, kTol));
  // Out of range input is saturated.
  ASSERT_TRUE(ColorIsEqual(colormap(-10.0f), {0, 0, 255}));
  ASSERT_TRUE(ColorIsEqual(colormap(10.0f), {255, 127, 0}));
}

TEST(ColorTest, MagmaColormap) {
  ASSERT_TRUE(ColorIsEqual(kMagmaColormap(8.0f / 255), {3, 3, 18}));
  ASSERT_TRUE(ColorIsEqual(kMagmaColormap(1.0f), {252, 253, 191}));
}

}  // namespace
