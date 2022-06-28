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

#include "image.h"

#include <cstdlib>
#include <cstdio>

#include "gtest/gtest.h"

namespace {

// Test basic Image operations.
TEST(ImageTest, Basic) {
  // Default constructor makes an empty image.
  Image<float> u;
  ASSERT_EQ(u.width(), 0);
  ASSERT_EQ(u.height(), 0);
  ASSERT_EQ(u.channels(), 0);
  ASSERT_EQ(u.data(), nullptr);

  // Allocate a 30x25 single-channel image.
  Image<float> f(30, 25);
  ASSERT_EQ(f.width(), 30);
  ASSERT_EQ(f.height(), 25);
  ASSERT_EQ(f.channels(), 1);
  // Pixel access.
  f(15, 10) = 3.14f;
  ASSERT_EQ(f(15, 10), 3.14f);

  // Image assignment is "shallow", `u` is now a view of `f`.
  u = f;
  ASSERT_EQ(u.data(), f.data());
  u(15, 10) = 123.0f;
  ASSERT_EQ(f(15, 10), 123.0f);

  // Create a 20x21x3 multichannel image.
  u = Image<float>(20, 21, 3);
  ASSERT_EQ(u.width(), 20);
  ASSERT_EQ(u.height(), 21);
  ASSERT_EQ(u.channels(), 3);
  ASSERT_EQ(u.x_stride(), 3);
  ASSERT_EQ(u.y_stride(), 3 * 20);
  ASSERT_EQ(u.c_stride(), 1);
  ASSERT_EQ(u.data() + 3 * (8 + 20 * 9) + 2, &u(8, 9, 2));
  u(8, 9, 2) = 1.25f;
  ASSERT_EQ(u(8, 9, 2), 1.25f);
}

// Test that images properly handle shared memory allocations.
TEST(ImageTest, MemorySharing) {
  Image<float> a;

  {
    Image<float> b(10, 8);
    ASSERT_EQ(b.use_count(), 1);
    Image<float> c(b);
    ASSERT_EQ(b.use_count(), 2);  // b and c refer to the same memory.
    a = c;
    ASSERT_EQ(a.use_count(), 3);  // a, b, and c refer to the same memory.
    ASSERT_EQ(b.use_count(), 3);
  }
  ASSERT_EQ(a.use_count(), 1);  // Only one remaining reference to the memory.

  std::vector<Image<float>> v;
  // Make vector reallocate and move its contents several times as it grows.
  v.reserve(2);
  for (int i = 0; i < 16; ++i) { v.push_back(a); }
  ASSERT_EQ(a.use_count(), 17);
  a = Image<float>();
  ASSERT_EQ(v[0].use_count(), 16);
}

// Test mapping an existing memory buffer.
TEST(ImageTest, MapExistingMemory) {
  constexpr int kWidth = 24;
  constexpr int kHeight = 20;
  constexpr int kChannels = 3;

  Image<float> u;
  std::vector<float> buffer(kWidth * kHeight * kChannels);

  {
    // Map buffer as an image with planar memory layout (instead of Image's
    // usual interleaved memory layout).
    Image<float> map(buffer.data(),
                     kWidth, 1,
                     kHeight, kWidth,
                     kChannels, kWidth * kHeight);
    ASSERT_EQ(map.data(), buffer.data());
    ASSERT_EQ(map.width(), kWidth);
    ASSERT_EQ(map.height(), kHeight);
    ASSERT_EQ(map.channels(), kChannels);
    ASSERT_EQ(map.x_stride(), 1);
    ASSERT_EQ(map.y_stride(), kWidth);
    ASSERT_EQ(map.c_stride(), kWidth * kHeight);
    ASSERT_EQ(map.use_count(), 0);
    u = map;
  }

  ASSERT_EQ(u.data(), buffer.data());
  ASSERT_EQ(u.use_count(), 0);
  u(10, 18, 1) = 42.0f;
  ASSERT_EQ(buffer[10 + kWidth * (18 + kHeight * 1)], 42.0f);
}

// Test behavior of Image<const float> read-only images.
TEST(ImageTest, ReadOnlyImage) {
  auto f = Image<float>(7, 12).Fill([](int x, int y) { return 0.0f; });

  Image<const float> read_only_view = f;
  ASSERT_EQ(f.use_count(), 2);
  f(3, 4) = 6.18f;
  ASSERT_EQ(read_only_view(3, 4), 6.18f);  // Reading is allowed.

  // Writing to read_only_view or assigning it to Image<float> is not allowed.
  // read_only_view(3, 4) = 100.0f;  // Fails to compile.
  // Image<float> u = read_only_view;  // Fails to compile.
}

void CheckFileBytes(const char* file_name, const uint8_t* expected_bytes,
                    size_t num_bytes) {
  uint8_t* bytes = ABSL_DIE_IF_NULL((uint8_t *) std::malloc(num_bytes + 1));
  std::FILE* f = ABSL_DIE_IF_NULL(std::fopen(file_name, "rb"));
  ASSERT_EQ(std::fread(bytes, 1, num_bytes + 1, f), num_bytes);
  std::fclose(f);
  ASSERT_EQ(std::memcmp(bytes, expected_bytes, num_bytes), 0);
  std::free(bytes);
}

TEST(ImageTest, WritePnmGrayscale) {
  const char* pnm_file_name = nullptr;
  pnm_file_name = ABSL_DIE_IF_NULL(std::tmpnam(nullptr));
  auto u = Image<uint8_t>(4, 3).Fill([](int x, int y) { return 10 * x + y; });

  ASSERT_TRUE(WritePnm(pnm_file_name, u));

  const uint8_t kExpected[] = {
    'P', '5', ' ', '4', ' ', '3', ' ', '2', '5', '5', '\n',
    0, 10, 20, 30,
    1, 11, 21, 31,
    2, 12, 22, 32,
  };

  CheckFileBytes(pnm_file_name, kExpected, sizeof(kExpected));
  std::remove(pnm_file_name);
}

TEST(ImageTest, WritePnmRgb) {
  const char* pnm_file_name = nullptr;
  pnm_file_name = ABSL_DIE_IF_NULL(std::tmpnam(nullptr));
  auto u = Image<uint8_t>(2, 3, 3).Fill(
      [](int x, int y, int c) { return 100 * x + 10 * y + c; });

  ASSERT_TRUE(WritePnm(pnm_file_name, u));

  const uint8_t kExpected[] = {
    'P', '6', ' ', '2', ' ', '3', ' ', '2', '5', '5', '\n',
    0, 1, 2, 100, 101, 102,
    10, 11, 12, 110, 111, 112,
    20, 21, 22, 120, 121, 122,
  };

  CheckFileBytes(pnm_file_name, kExpected, sizeof(kExpected));
  std::remove(pnm_file_name);
}

}  // namespace
