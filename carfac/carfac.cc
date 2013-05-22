//
//  carfac.cc
//  CARFAC Open Source C++ Library
//
//  Created by Alex Brandmeyer on 5/10/13.
//
// This C++ file is part of an implementation of Lyon's cochlear model:
// "Cascade of Asymmetric Resonators with Fast-Acting Compression"
// to supplement Lyon's upcoming book "Human and Machine Hearing"
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

#include "carfac.h"

void CARFAC::Design(const int n_ears, const FPType fs,
                    const CARParams& car_params, const IHCParams& ihc_params,
                    const AGCParams& agc_params) {
  n_ears_ = n_ears;
  fs_ = fs;
  ears_.resize(n_ears_);
  
  n_ch_ = 0;
  FPType pole_hz = car_params.first_pole_theta_ * fs / (2 * PI);
  while (pole_hz > car_params.min_pole_hz_) {
    ++n_ch_;
    pole_hz = pole_hz - car_params.erb_per_step_ *
    ERBHz(pole_hz, car_params.erb_break_freq_, car_params.erb_q_);
  }
  pole_freqs_.resize(n_ch_);
  pole_hz = car_params.first_pole_theta_ * fs / (2 * PI);
  for (int ch = 0; ch < n_ch_; ++ch) {
    pole_freqs_(ch) = pole_hz;
    pole_hz = pole_hz - car_params.erb_per_step_ *
    ERBHz(pole_hz, car_params.erb_break_freq_, car_params.erb_q_);
  }
  max_channels_per_octave_ = log(2) / log(pole_freqs_(0) / pole_freqs_(1));
  // Once we have the basic information about the pole frequencies and the
  // number of channels, we initialize the ear(s).
  for (auto& ear : ears_) {
    ear.InitEar(n_ch_, fs_, pole_freqs_, car_params, ihc_params,
                agc_params);
  }
}

CARFACOutput CARFAC::Run(const std::vector<std::vector<float>>& sound_data) {
  // We initialize one output object to store the final output.
  CARFACOutput seg_output;
  int n_audio_channels = sound_data.size();
  int32_t seg_len = 441;  // We use a fixed segment length for now.
  int32_t n_timepoints = sound_data[0].size();
  int32_t n_segs = ceil((n_timepoints * 1.0) / seg_len);
  seg_output.InitOutput(n_audio_channels, n_ch_, n_timepoints);
  // These values store the start and endpoints for each segment
  int32_t start;
  int32_t length = seg_len;
  // This section loops over the individual audio segments.
  for (int32_t i = 0; i < n_segs; ++i) {
    // For each segment we calculate the start point and the segment length.
    start = i * seg_len;
    if (i == n_segs - 1) {
      // The last segment can be shorter than the rest.
      length = n_timepoints - start;
    }
    std::vector<std::vector<float>> segment_data;
    segment_data.resize(n_audio_channels);
    for (int channel = 0; channel < n_audio_channels; ++channel) {
      segment_data[channel].resize(length);
      for (int32_t timepoint = 0; timepoint < length; ++timepoint) {
        segment_data[channel][timepoint] =
        sound_data[channel][start + timepoint];
      }
    }
    // Once we've determined the start point and segment length, we run the
    // CARFAC model on the current segment.
    RunSegment(segment_data, start,
               length, &seg_output, true);
    // Afterwards we merge the output for the current segment into the larger
    // output structure for the entire audio file.
  }
  return seg_output;
}

void CARFAC::RunSegment(const std::vector<std::vector<float>>& sound_data,
                        const int32_t start, const int32_t length,
                        CARFACOutput* seg_output, const bool open_loop) {
  // The number of ears is equal to the number of audio channels. This could
  // potentially be removed since we already know the n_ears_ during the design
  // stage.
  int n_ears = sound_data.size();
  // The number of timepoints is determined from the length of the audio
  // segment.
  int32_t n_timepoints = sound_data[0].size();
  // A nested loop structure is used to iterate through the individual samples
  // for each ear (audio channel).
  FloatArray car_out(n_ch_);
  FloatArray ihc_out(n_ch_);
  bool updated;  // This variable is used by the AGC stage.
  for (int32_t i = 0; i < n_timepoints; ++i) {
    for (int j = 0; j < n_ears; ++j) {
      // First we create a reference to the current Ear object.
      Ear& ear = ears_[j];
      // This stores the audio sample currently being processed.
      FPType input = sound_data[j][i];
      // Now we apply the three stages of the model in sequence to the current
      // audio sample.
      ear.CARStep(input, &car_out);
      ear.IHCStep(car_out, &ihc_out);
      updated = ear.AGCStep(ihc_out);
      // These lines assign the output of the model for the current sample
      // to the appropriate data members of the current ear in the output
      // object.
      seg_output->StoreNAPOutput(start + i, j, ihc_out);
      seg_output->StoreBMOutput(start + i, j, car_out);
      seg_output->StoreOHCOutput(start + i, j, ear.za_memory());
      seg_output->StoreAGCOutput(start + i, j, ear.zb_memory());
    }
    if (updated && n_ears > 1) {
      CrossCouple();
    }
    if (! open_loop) {
      CloseAGCLoop();
    }
  }
}

void CARFAC::CrossCouple() {
  for (int stage = 0; stage < ears_[0].agc_nstages(); ++stage) {
    if (ears_[0].agc_decim_phase(stage) > 0) {
      break;
    } else {
      FPType mix_coeff = ears_[0].agc_mix_coeff(stage);
      if (mix_coeff > 0) {
        FloatArray stage_state;
        FloatArray this_stage_values = FloatArray::Zero(n_ch_);
        for (auto& ear : ears_) {
          stage_state = ear.agc_memory(stage);
          this_stage_values += stage_state;
        }
        this_stage_values /= n_ears_;
        for (auto& ear : ears_) {
          stage_state = ear.agc_memory(stage);
          ear.set_agc_memory(stage, stage_state + mix_coeff *
                             (this_stage_values - stage_state));
        }
      }
    }
  }
}

void CARFAC::CloseAGCLoop() {
  for (auto& ear: ears_) {
    FloatArray undamping = 1 - ear.agc_memory(0);
    // This updates the target stage gain for the new damping.
    ear.set_dzb_memory((ear.zr_coeffs() * undamping - ear.zb_memory()) /
                       ear.agc_decimation(0));
    ear.set_dg_memory((ear.StageGValue(undamping) - ear.g_memory()) /
                      ear.agc_decimation(0));
  }
}