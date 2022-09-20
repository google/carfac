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

// This test compares PitchogramPipeline output with golden images. To create
// updated goldens, run this test with flag `--write_goldens=<directory>`.


#include "pitchogram_pipeline.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "image.h"
#include "test_util.h"
#include "testing/base/public/gunit.h"

char* goldens_dir = NULL;

namespace {

// Reads an int16 in little endian byte order from `f`.
int16_t ReadInt16Le(std::FILE* f) {
  uint16_t sample_u16 = static_cast<uint16_t>(std::getc(f));
  sample_u16 |= static_cast<uint16_t>(std::getc(f)) << 8;
  int16_t sample_i16;
  std::memcpy(&sample_i16, &sample_u16, sizeof(sample_i16));
  return sample_i16;
}

// Views a 4-channel RGBA image as a 3-channel RGB image.
Image<const uint8_t> RgbaToRgbImage(const Image<const uint8_t>& image_rgba) {
  return Image<const uint8_t>(image_rgba.data(), image_rgba.width(),
                              image_rgba.x_stride(), image_rgba.height(),
                              image_rgba.y_stride(), 3, image_rgba.c_stride());
}

// Returns true if string `s` starts with `prefix`.
bool StartsWith(const char* s, const char* prefix) {
  while (*prefix) {
    if (*s++ != *prefix++) { return false; }
  }
  return true;
}

struct TestParams {
  bool light_color_theme;

  std::string name() const {
    return std::string("pitchogram_pipeline_test-") +
          (light_color_theme ? "light" : "dark") + "_theme";
  }
};

std::string PrintToString(const TestParams& params) { return params.name(); }

class PitchogramPipelineTest : public ::testing::WithParamInterface<TestParams>,
                       public ::testing::Test {};

TEST_P(PitchogramPipelineTest, CheckGolden) {
  constexpr float kSampleRateHz = 44100.0f;
  constexpr int kChunkSize = 512;  // 11 ms.
  constexpr int kNumChunks = 250;

  PitchogramPipelineParams params;
  params.num_frames = kNumChunks + 1;
  params.num_samples_per_segment = kChunkSize;
  params.pitchogram_params.light_color_theme = GetParam().light_color_theme;
  PitchogramPipeline pipeline(kSampleRateHz, params);

  std::FILE* wav = std::fopen(GetTestDataPath("long_test.wav").c_str(), "rb");
  ASSERT_TRUE(wav != nullptr);
  // Seek past 44-byte header and one second into the recording.
  ASSERT_EQ(std::fseek(wav, 44 + 4 * 44100, SEEK_CUR), 0);

  for (int i = 0; i < kNumChunks; ++i) {
    // Read a chunk, converting int16 -> float and mixing stereo down to mono.
    float input[kChunkSize];
    for (int j = 0; j < kChunkSize; ++j) {
      input[j] = (static_cast<float>(ReadInt16Le(wav)) +
                  static_cast<float>(ReadInt16Le(wav))) / 65536.0f;
    }

    pipeline.ProcessSamples(input, kChunkSize);
  }

  std::fclose(wav);

  const std::string golden = GetParam().name() + ".pnm";
  if (goldens_dir) {
    const std::string path = std::string(goldens_dir) + '/' + golden;
    ASSERT_TRUE(WritePnm(path, pipeline.image()));
    printf("Wrote %s\n", path.c_str());
  } else {
    const std::string path = GetTestDataPath(golden);
    Image<uint8_t> expected;
    ASSERT_TRUE(ReadPnm(path, &expected));
    ASSERT_LE(RgbaToRgbImage(pipeline.image()).RootMeanSquareDiff(expected),
              3.0f) << "Pitchogram output mismatches " << path;
  }
}

INSTANTIATE_TEST_SUITE_P(Params, PitchogramPipelineTest,
                         testing::Values(TestParams{false}, TestParams{true}));

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  if (argc > 1 && StartsWith(argv[1], "--write_goldens=")) {
    goldens_dir = std::strchr(argv[1], '=') + 1;
  }

  return RUN_ALL_TESTS();
}
