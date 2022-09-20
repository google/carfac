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

#include <cstdint>
#include <vector>
#if !defined(__EMSCRIPTEN__)
#error This file must be built with emscripten
#endif

#include <emscripten.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "SDL2/SDL.h"
#include "pitchogram_pipeline.h"

struct {
  SDL_Window* window;
  SDL_Renderer* renderer;
  SDL_Surface* surface;
  std::unique_ptr<PitchogramPipeline> plotter;
} engine;  // NOLINT

static void MainTick();

// Initializes SDL. This gets called immediately after the emscripten runtime
// has initialized.
extern "C" void EMSCRIPTEN_KEEPALIVE OnLoad(
    int canvas_width, int canvas_height) {
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

  if (!(engine.window = SDL_CreateWindow("", SDL_WINDOWPOS_UNDEFINED,
                                         SDL_WINDOWPOS_UNDEFINED, canvas_width,
                                         canvas_height, SDL_WINDOW_SHOWN))) {
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

  // Use bilinear sampling in the SDL_RenderCopy operation below.
  SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");

  // Initially fill the canvas with gray #444 color.
  SDL_SetRenderDrawColor(engine.renderer, 0x44, 0x44, 0x44, 255);
  SDL_RenderClear(engine.renderer);
}

// Resize the canvas, called from JavaScript when the browser window resizes.
extern "C" void EMSCRIPTEN_KEEPALIVE SetSize(int new_width, int new_height) {
  SDL_SetWindowSize(engine.window, new_width, new_height);
}

// Emscripten will call this function once per frame to do event processing
// (though we ignore all events) and to render the next frame.
static void MainTick() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {}  // Ignore events.

  if (!engine.surface) { return; }

  SDL_Texture* texture =
      SDL_CreateTextureFromSurface(engine.renderer, engine.surface);
  SDL_RenderCopy(engine.renderer, texture, nullptr, nullptr);
  SDL_RenderPresent(engine.renderer);
  SDL_DestroyTexture(texture);
}

// Initializes audio processing. This gets called after WebAudio has started.
extern "C" void EMSCRIPTEN_KEEPALIVE DemoInitAudio(
    int sample_rate_hz, int chunk_size, bool log_lag, bool light_color_theme) {
  PitchogramPipelineParams params;
  params.num_samples_per_segment = chunk_size;
  params.pitchogram_params.log_lag = log_lag;
  params.pitchogram_params.light_color_theme = light_color_theme;
  engine.plotter.reset(new PitchogramPipeline(sample_rate_hz, params));

  const Image<uint8_t>& image = engine.plotter->image();
  constexpr uint32_t r_mask = 0x000000ff;
  constexpr uint32_t g_mask = 0x0000ff00;
  constexpr uint32_t b_mask = 0x00ff0000;
  engine.surface = SDL_CreateRGBSurfaceFrom(
      reinterpret_cast<void*>(image.data()), image.width(), image.height(), 32,
      image.y_stride(), r_mask, g_mask, b_mask, 0);
  if (!engine.surface) {
    std::fprintf(stderr, "Failed to create surface: %s\n", SDL_GetError());
    std::exit(1);
  }
}

// Processes one chunk of audio data. Called from onaudioprocess.
extern "C" void EMSCRIPTEN_KEEPALIVE DemoProcessAudio(
    intptr_t input_ptr, int chunk_size) {
  engine.plotter->ProcessSamples(
      reinterpret_cast<float*>(input_ptr), chunk_size);
}
