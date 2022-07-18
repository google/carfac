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

#if !defined(__EMSCRIPTEN__)
#error This file must be built with emscripten
#endif

#include <emscripten.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "SDL2/SDL.h"
#include "agc.h"
#include "car.h"
#include "carfac.h"
#include "color.h"
#include "common.h"
#include "ihc.h"
#include "image.h"
#include "pitchogram.h"
#include "sai.h"

// Width of the pitchogram in frames.
constexpr int kNumFrames = 400;
// Max lag in the pitchogram in units of seconds.
constexpr float kMaxLag = 0.03f;
// Highest pole frequency in Hz.
constexpr float kHighestPoleHz = 7000.0f;

// Dimensions of the SDL surface in pixels used to show the pitchogram.
constexpr int kCanvasWidth = 1024;
constexpr int kCanvasHeight = 768;

template <typename T>
const T& clamp(const T& x, const T& min_value, const T& max_value) {
  return std::min(std::max(x, min_value), max_value);
}

// Create a scrolling plot given successive columns.
class ScrollingPlot {
 public:
  ScrollingPlot() = default;

  ScrollingPlot(int width, int height)
      : plot_(width, height, 3), plot_view_(plot_) {
    Reset();
  }

  void Redesign(int width, int height) {
    plot_ = Image<uint8_t>(width, height, 3);
    plot_view_ = plot_;
    Reset();
  }

  void Reset() {
    std::memset(plot_.data(), 0, plot_.size_in_bytes());
  }

  const Image<const uint8_t>& image() const { return plot_view_; }

  // Scroll left one frame and append `frame` as the rightmost column.
  void PushFrame(const ArrayX& frame, float max_value) {
    CARFAC_ASSERT(plot_.height() == frame.size());

    // Scroll the plot image one frame left.
    std::memmove(plot_.data(), plot_.data() + plot_.channels(),
                 (plot_.num_pixels() - 1) * plot_.channels());

    const Colormap& colormap = kMagmaColormap;
    const float scale = 1.0f / max_value;
    for (int y = 0; y < frame.size(); ++y) {
      Color<uint8_t>::Vector::Map(&plot_(plot_.width() - 1, y)) =
          colormap(scale * frame[y]);
    }
  }

 private:
  Image<uint8_t> plot_;
  Image<const uint8_t> plot_view_;
};

// Create a pitchogram visualization.
class PitchogramPlotter {
 public:
  explicit PitchogramPlotter(float sample_rate, int num_samples_per_segment) {
    sample_rate_ = sample_rate;
    car_params_.first_pole_theta = 2 * M_PI * kHighestPoleHz / sample_rate;
    carfac_.reset(new CARFAC(kNumEars, sample_rate, car_params_, ihc_params_,
                             agc_params_));
    carfac_output_buffer_.reset(new CARFACOutput(true, false, false, false));

    sai_params_.num_channels = carfac_->num_channels();
    sai_params_.sai_width =
          static_cast<int>(std::round(kMaxLag * sample_rate));
    sai_params_.input_segment_width = num_samples_per_segment;
    sai_params_.trigger_window_width = sai_params_.input_segment_width + 1;
    sai_params_.future_lags = sai_params_.sai_width - 1;
    sai_params_.num_triggers_per_frame = 2;
    sai_.reset(new SAI(sai_params_));
    sai_output_buffer_.resize(sai_params_.num_channels, sai_params_.sai_width);

    pitchogram_.reset(new Pitchogram(sample_rate_, car_params_, sai_params_,
                                     pitchogram_params_));
    plot_.Redesign(kNumFrames, pitchogram_->num_lags());
  }

  // Returns the current pitchogram plot image.
  const Image<const uint8_t>& image() const { return plot_.image(); }
  int width() const { return plot_.image().width(); }
  int height() const { return plot_.image().height(); }

  // Process audio samples in a streaming manner.
  void ProcessSamples(const float* samples, int num_samples) {
    auto input_map = ArrayXX::Map(samples, kNumEars, num_samples / kNumEars);
    carfac_->RunSegment(input_map, false /* open_loop */,
                        carfac_output_buffer_.get());
    sai_->RunSegment(carfac_output_buffer_->nap()[0], &sai_output_buffer_);
    const ArrayX& pitchogram_frame = pitchogram_->RunFrame(sai_output_buffer_);
    plot_.PushFrame(pitchogram_frame, 4.0f);
  }

 private:
  enum { kNumEars = 1 };

  float sample_rate_;
  CARParams car_params_;
  IHCParams ihc_params_;
  AGCParams agc_params_;
  SAIParams sai_params_;
  PitchogramParams pitchogram_params_;
  std::unique_ptr<CARFAC> carfac_;
  std::unique_ptr<CARFACOutput> carfac_output_buffer_;
  std::unique_ptr<SAI> sai_;
  ArrayXX sai_output_buffer_;
  std::unique_ptr<Pitchogram> pitchogram_;
  ScrollingPlot plot_;
};

struct {
  SDL_Window* window;
  SDL_Renderer* renderer;
  SDL_Surface* surface;
  std::unique_ptr<PitchogramPlotter> plotter;
  int chunk_size;
} engine;  // NOLINT

static void MainTick();

// Initializes SDL. This gets called immediately after the emscripten runtime
// has initialized.
extern "C" void EMSCRIPTEN_KEEPALIVE OnLoad() {
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::fprintf(stderr, "Error: %s\n", SDL_GetError());
    std::exit(1);
  }

  SDL_EventState(SDL_MOUSEMOTION, SDL_IGNORE);
  // Disable SDL keyboard events. Otherwise, the tab key (to navigate
  // interactive elements) does not work on the web page since SDL captures it.
  SDL_EventState(SDL_TEXTINPUT, SDL_DISABLE);
  SDL_EventState(SDL_KEYDOWN, SDL_DISABLE);
  SDL_EventState(SDL_KEYUP, SDL_DISABLE);

  // Set the event handling loop. This must be set *before* creating the window,
  // otherwise there is an error "Cannot set timing mode for main loop".
  emscripten_set_main_loop(MainTick, 0, false);

  if (!(engine.window = SDL_CreateWindow(
      "", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
      kCanvasWidth, kCanvasHeight, SDL_WINDOW_SHOWN))) {
    std::fprintf(stderr, "Failed to create window: %s\n", SDL_GetError());
    std::exit(1);
  }

  engine.renderer = SDL_CreateRenderer(
      engine.window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (!engine.renderer) {  // Fall back to software renderer.
    engine.renderer = SDL_CreateRenderer(engine.window, -1, 0);
  }

  static SDL_RendererInfo renderer_info = {0};
  if (engine.renderer) {
    SDL_GetRendererInfo(engine.renderer, &renderer_info);
  }
  if (!engine.renderer || renderer_info.num_texture_formats == 0) {
    std::fprintf(stderr, "Failed to create renderer: %s\n", SDL_GetError());
    std::exit(1);
  }
}

// Emscripten will call this function once per frame to do event processing
// (though we ignore all events) and to render the next frame.
static void MainTick() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {}  // Ignore events.

  if (SDL_MUSTLOCK(engine.surface)) { SDL_LockSurface(engine.surface); }
  uint8_t* pixels = reinterpret_cast<uint8_t*>(engine.surface->pixels);
  const auto& image = engine.plotter->image();
  std::memcpy(pixels, image.data(), image.size_in_bytes());
  if (SDL_MUSTLOCK(engine.surface)) { SDL_UnlockSurface(engine.surface); }

  SDL_Texture* texture =
      SDL_CreateTextureFromSurface(engine.renderer, engine.surface);
  SDL_RenderCopy(engine.renderer, texture, nullptr, nullptr);
  SDL_RenderPresent(engine.renderer);
  SDL_DestroyTexture(texture);
}

// Initializes audio processing. This gets called after WebAudio has started.
extern "C" void EMSCRIPTEN_KEEPALIVE DemoInitAudio(
    int sample_rate_hz, int chunk_size) {
  engine.chunk_size = chunk_size;
  engine.plotter.reset(new PitchogramPlotter(sample_rate_hz, chunk_size));
  engine.surface = SDL_CreateRGBSurface(-1, engine.plotter->width(),
                                        engine.plotter->height(), 24,
                                        0xff, 0xff00, 0xff0000, 0);
}

// Processes one chunk of audio data. Called from onaudioprocess.
extern "C" void EMSCRIPTEN_KEEPALIVE DemoProcessAudio(
    intptr_t input_ptr, int chunk_size) {
  engine.plotter->ProcessSamples(
      reinterpret_cast<float*>(input_ptr), chunk_size);
}
