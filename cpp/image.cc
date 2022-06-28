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

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

bool WritePnm(const std::string& filename, const Image<const uint8_t>& image) {
  const int channels = image.channels();
  if (channels != 1 && channels != 3) {
    std::fprintf(stderr, "Unable to write image with %d channels.\n", channels);
    return false;
  }

  std::FILE* f = std::fopen(filename.c_str(), "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "Failed to write \"%s\": %s\n", filename.c_str(),
                 std::strerror(errno));
    return false;
  }

  std::fprintf(f, "P%d %d %d 255\n", (channels == 1) ? 5 : 6,
               image.width(), image.height());

  std::vector<uint8_t> row(image.width() * channels);
  for (int y = 0; y < image.height(); ++y) {
    if (channels == 1) {
      for (int x = 0; x < image.width(); ++x) {
        row[x] = image(x, y);
      }
    } else {
      uint8_t* dest = row.data();
      for (int x = 0; x < image.width(); ++x, dest += 3) {
        for (int c = 0; c < 3; ++c) {
          dest[c] = image(x, y, c);
        }
      }
    }
    std::fwrite(row.data(), 1, row.size(), f);
  }

  std::fclose(f);
  return true;
}
