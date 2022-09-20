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

#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

bool WritePnm(const std::string& filename, const Image<const uint8_t>& image) {
  const int channels = std::min<int>(image.channels(), 3);
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

namespace {
// Reads the next char from `f`, skipping over '#' comments.
int PnmNextChar(std::FILE* f) {
  int c;
  while ((c = std::getc(f)) == '#') {
    do {
      c = std::getc(f);
    } while (c != EOF && c != '\n');
  }
  return c;
}
}  // namespace

bool ReadPnm(const std::string& filename, Image<uint8_t>* image) {
  std::FILE* f = std::fopen(filename.c_str(), "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "Failed to read \"%s\": %s\n", filename.c_str(),
                 std::strerror(errno));
    return false;
  }

  // Read the PNM identifier. This should be e.g. "P6" for binary PixMap.
  const int magic = PnmNextChar(f);
  const int pnm_format = std::getc(f);
  const int channels = (pnm_format == '5') ? 1 : 3;

  if (magic != 'P' || !('1' <= pnm_format && pnm_format <= '6') ||
      !std::isspace(PnmNextChar(f))) {
    std::fprintf(stderr, "Not a PNM image. ");
    goto fail;
  } else if (pnm_format != '5' && pnm_format != '6') {
    std::fprintf(stderr, "Unsupported PNM format (P%d). ", pnm_format);
    goto fail;
  }

  int width;
  int height;

  // Parse PNM header fields.
  for (int field = 0; field < 3; ++field) {
    // Read the next non-space character.
    int c = PnmNextChar(f);
    while (std::isspace(c)) {
      c = PnmNextChar(f);
    }

    // Parse a nonnegative integer value.
    int value = 0;
    bool found_digit = false;

    while ('0' <= c && c <= '9') {
      found_digit = true;
      if (value > (INT_MAX - 9) / 10) {
        break;
      }  // Prevent overflow.
      value = 10 * value + (c - '0');
      c = PnmNextChar(f);
    }

    if (!found_digit || !std::isspace(c)) {
      std::fprintf(stderr, "Bad header. ");
      goto fail;
    }

    switch (field) {
      case 0:
        width = value;
        break;
      case 1:
        height = value;
        break;
      case 2:
        if (value > 255) {
          std::fprintf(stderr, "Only 8-bit PNM is supported. %d ", value);
          goto fail;
        }
    }
  }

  *image = Image<uint8_t>(width, height, channels);
  if (!std::fread(image->data(), image->size_in_bytes(), 1, f)) {
    if (std::feof(f)) {
      std::fprintf(stderr, "File ended unexpectedly. ");
    } else {
      std::fprintf(stderr, "I/O error. ");
    }
    goto fail;
  }

  std::fclose(f);
  return true;

fail:
  std::fprintf(stderr, "Error reading PNM \"%s\".\n", filename.c_str());
  std::fclose(f);
  return false;
}
