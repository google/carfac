// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Ron Weiss
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

#if EMSCRIPTEN
#include <emscripten/bind.h>
#endif
#include <SDL/SDL.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "agc.h"
#include "car.h"
#include "carfac.h"
#include "ihc.h"
#include "sai.h"

using namespace emscripten;

namespace {
constexpr int kNumEars = 1;

inline Uint8 RoundAndSaturate(float x) {
  return std::min(std::max(std::round(x), 0.0f), 255.0f);
}
}  // anonymous namespace

// Helper class to run CARFAC, compute SAI frames, and plot them using
// SDL.  The interface is designed to be easy to interact with from
// Javascript when compiled using emscripten.
class SAIPlotter {
 public:
  explicit SAIPlotter(float sample_rate, int num_samples_per_segment) {
    sample_rate_ = sample_rate;
    carfac_.reset(new CARFAC(kNumEars, sample_rate, car_params_, ihc_params_,
                             agc_params_));
    // Only store NAP outputs.
    carfac_output_buffer_.reset(new CARFACOutput(true, false, false, false));

    sai_params_.num_channels = carfac_->num_channels();
    sai_params_.sai_width = num_samples_per_segment;
    sai_params_.input_segment_width = num_samples_per_segment;
    sai_params_.trigger_window_width = sai_params_.input_segment_width + 1;
    // Half of the SAI should come from the future.
    sai_params_.future_lags = sai_params_.sai_width / 2;
    sai_params_.num_triggers_per_frame = 2;
    sai_.reset(new SAI(sai_params_));

    SDL_Init(SDL_INIT_VIDEO);
    Redesign(car_params_, ihc_params_, agc_params_, sai_params_);
  }

  void Redesign(const CARParams& car_params, const IHCParams& ihc_params,
                const AGCParams& agc_params, const SAIParams& sai_params) {
    car_params_ = car_params;
    ihc_params_ = ihc_params;
    agc_params_ = agc_params;
    carfac_->Redesign(kNumEars, sample_rate_, car_params_, ihc_params_,
                      agc_params_);

    int input_segment_width = sai_params_.input_segment_width;
    sai_params_ = sai_params;
    sai_params_.num_channels = carfac_->num_channels();
    sai_params_.input_segment_width = input_segment_width;
    screen_ = SDL_SetVideoMode(sai_params_.sai_width, carfac_->num_channels(),
                               32 /* bits_per_pixel */,
                               SDL_SWSURFACE | SDL_RESIZABLE);
    sai_output_buffer_.reset(new ArrayXX(sai_params_.num_channels,
                                         sai_params_.sai_width));
    sai_->Redesign(sai_params_);
  }

  void Reset() {
    carfac_->Reset();
    sai_->Reset();
  }

  // Runs the given (single channel) audio samples through the CARFAC
  // filterbank, computes an SAI, and plots the result.
  //
  // The input_samples pointer type is chosen to avoid emscripten
  // compilation errors.
  // TODO(ronw): Figure out if this is the best way to pass a
  // javascript FloatArray to a C++ function using emscripten.
  void ComputeAndPlotSAI(intptr_t input_samples, size_t num_input_samples) {
    CARFAC_ASSERT(num_input_samples == sai_params_.input_segment_width);
    constexpr int kNumInputEars = 1;
    auto input_map = ArrayXX::Map(reinterpret_cast<const float*>(input_samples),
                                  kNumInputEars, num_input_samples);
    carfac_->RunSegment(input_map, false /* open_loop */,
                        carfac_output_buffer_.get());
    sai_->RunSegment(carfac_output_buffer_->nap()[0], sai_output_buffer_.get());
    PlotMatrix(*sai_output_buffer_);
  }

  CARParams car_params() const { return car_params_; }
  IHCParams ihc_params() const { return ihc_params_; }
  AGCParams agc_params() const { return agc_params_; }
  SAIParams sai_params() const { return sai_params_; }

 private:
  // Plots the given matrix.  This assumes that the values of matrix
  // are within (0, 5), and clips values outside of this range.
  // Emscripten renders the output to an HTML5 canvas element.
  void PlotMatrix(const ArrayXX& matrix) {
    float min = matrix.minCoeff();
    float norm = 5.0 - min;
    // Avoid dividing by zero.
    if (norm == 0.0) {
      norm = 1.0;
    }

    SDL_LockSurface(screen_);
    Uint32* pixels = static_cast<Uint32*>(screen_->pixels);
    for (int row = 0; row < matrix.rows(); ++row) {
      for (int col = 0; col < matrix.cols(); ++col) {
        float normalized_value = (matrix(row, col) - min) / norm;
        Uint8 gray_value = RoundAndSaturate(255 * (1.0 - normalized_value));
        pixels[(row * screen_->w) + col] =
          SDL_MapRGB(screen_->format, gray_value, gray_value, gray_value);
      }
    }
    SDL_UnlockSurface(screen_);
    SDL_Flip(screen_);
  }

  CARParams car_params_;
  IHCParams ihc_params_;
  AGCParams agc_params_;
  std::unique_ptr<CARFAC> carfac_;
  std::unique_ptr<CARFACOutput> carfac_output_buffer_;
  SAIParams sai_params_;
  std::unique_ptr<SAI> sai_;
  std::unique_ptr<ArrayXX> sai_output_buffer_;
  float sample_rate_;
  SDL_Surface* screen_;
};

#if EMSCRIPTEN
EMSCRIPTEN_BINDINGS(SAIPlotter) {
  register_vector<float>("FloatVector");

  value_object<AGCParams>("AGCParams")
    .field("num_stages", &AGCParams::num_stages)
    .field("agc_stage_gain", &AGCParams::agc_stage_gain)
    .field("agc_mix_coeff", &AGCParams::agc_mix_coeff);
    // .field("time_constants", &AGCParams::time_constants)
    // .field("decimation", &AGCParams::decimation)
    // .field("agc1_scales", &AGCParams::agc1_scales)
    // .field("agc2_scales", &AGCParams::agc2_scales);

  value_object<CARParams>("CARParams")
    .field("velocity_scale", &CARParams::velocity_scale)
    .field("v_offset", &CARParams::v_offset)
    .field("min_zeta", &CARParams::min_zeta)
    .field("max_zeta", &CARParams::max_zeta)
    .field("first_pole_theta", &CARParams::first_pole_theta)
    .field("zero_ratio", &CARParams::zero_ratio)
    .field("high_f_damping_compression", &CARParams::high_f_damping_compression)
    .field("erb_per_step", &CARParams::erb_per_step)
    .field("min_pole_hz", &CARParams::min_pole_hz)
    .field("erb_break_freq", &CARParams::erb_break_freq)
    .field("erb_q", &CARParams::erb_q);

  value_object<IHCParams>("IHCParams")
    .field("just_half_wave_rectify", &IHCParams::just_half_wave_rectify)
    .field("one_capacitor", &IHCParams::one_capacitor)
    .field("tau_lpf", &IHCParams::tau_lpf)
    .field("tau1_out", &IHCParams::tau1_out)
    .field("tau1_in", &IHCParams::tau1_in)
    .field("tau2_out", &IHCParams::tau2_out)
    .field("tau2_in", &IHCParams::tau2_in)
    .field("ac_corner_hz", &IHCParams::ac_corner_hz);

  value_object<SAIParams>("SAIParams")
    .field("sai_width", &SAIParams::sai_width)
    .field("future_lags", &SAIParams::future_lags)
    .field("num_triggers_per_frame", &SAIParams::num_triggers_per_frame)
    .field("trigger_window_width", &SAIParams::trigger_window_width);

  class_<SAIPlotter>("SAIPlotter")
    .constructor<float, int>()
    .function("Redesign", &SAIPlotter::Redesign)
    .function("Reset", &SAIPlotter::Reset)
    .function("ComputeAndPlotSAI", &SAIPlotter::ComputeAndPlotSAI)
    .function("car_params", &SAIPlotter::car_params)
    .function("ihc_params", &SAIPlotter::ihc_params)
    .function("agc_params", &SAIPlotter::agc_params)
    .function("sai_params", &SAIPlotter::sai_params);
}
#endif
