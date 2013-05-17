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
void CARFAC::Design(int n_ears, int32_t fs, CARParams car_params,
                    IHCParams ihc_params, AGCParams agc_params) {
  n_ears_ = n_ears;
  fs_ = fs;
  ears_.resize(n_ears_);
  
  n_ch_ = 0;
  FPType pole_hz = car_params.first_pole_theta_ * fs / (2 * PI);
  while (pole_hz > car_params.min_pole_hz_) {
    n_ch_++;
    pole_hz = pole_hz - car_params.erb_per_step_ *
    ERBHz(pole_hz, car_params.erb_break_freq_, car_params.erb_q_);
  }
  FloatArray pole_freqs(n_ch_);
  pole_hz = car_params.first_pole_theta_ * fs / (2 * PI);
  for(int ch=0;ch < n_ch_; ch++) {
    pole_freqs(ch) = pole_hz;
    pole_hz = pole_hz - car_params.erb_per_step_ *
    ERBHz(pole_hz, car_params.erb_break_freq_, car_params.erb_q_);
  }
  max_channels_per_octave_ = log(2) / log(pole_freqs(0) / pole_freqs(1));
  // Once we have the basic information about the pole frequencies and the
  // number of channels, we initialize the ear(s).
  for (int i = 0; i < n_ears_; i++) {
    ears_.at(i).InitEar(n_ch_, fs_, pole_freqs, car_params, ihc_params,
                     agc_params);
  }
}

CARFACOutput CARFAC::Run(FloatArray2d sound_data, bool open_loop,
                         bool store_bm, bool store_ohc, bool store_agc) {
  // We initialize one output object to store the final output.
  CARFACOutput *output = new CARFACOutput();
  // A second object is used to store the output for the individual segments.
  CARFACOutput *seg_output = new CARFACOutput();
  int n_audio_channels = int(sound_data.cols());
  int32_t seg_len = 441;  // We use a fixed segment length for now.
  int32_t n_timepoints = sound_data.rows();
  int32_t n_segs = ceil((n_timepoints * 1.0) / seg_len);
  output->InitOutput(n_audio_channels, n_ch_, n_timepoints);
  seg_output->InitOutput(n_audio_channels, n_ch_, seg_len);
  // These values store the start and endpoints for each segment
  int32_t start;
  int32_t length = seg_len;
  // This section loops over the individual audio segments.
  for (int32_t i = 0; i < n_segs; i++) {
    // For each segment we calculate the start point and the segment length.
    start = i * seg_len;
    if (i == n_segs - 1) {
      // The last segment can be shorter than the rest.
      length = n_timepoints - start;
    }
    // Once we've determined the start point and segment length, we run the
    // CARFAC model on the current segment.
    RunSegment(sound_data.block(start, 0, length, n_audio_channels),
               seg_output, open_loop, store_bm, store_ohc,
               store_agc);
    // Afterwards we merge the output for the current segment into the larger
    // output structure for the entire audio file.
    output->MergeOutput(*seg_output, start, length);
  }
  return *output;
}

void CARFAC::RunSegment(FloatArray2d sound_data, CARFACOutput *seg_output,
                        bool open_loop, bool store_bm, bool store_ohc,
                        bool store_agc) {
  // The number of timepoints is determined from the length of the audio
  // segment.
  int32_t n_timepoints = sound_data.rows();
  // The number of ears is equal to the number of audio channels. This could
  // potentially be removed since we already know the n_ears_ during the design
  // stage. 
  int n_ears = int(sound_data.cols());
  // A nested loop structure is used to iterate through the individual samples
  // for each ear (audio channel).
  bool updated;  // This variable is used by the AGC stage.
  FloatArray car_out(n_ch_);
  FloatArray ihc_out(n_ch_);
  for (int32_t i = 0; i < n_timepoints; i++) {
    for (int j = 0; j < n_ears; j++) {
      // This stores the audio sample currently being processed.
      FPType input = sound_data(i, j);
      // Now we apply the three stages of the model in sequence to the current
      // audio sample.
      car_out = ears_.at(j).CARStep(input);
      ihc_out = ears_.at(j).IHCStep(car_out);
      updated = ears_.at(j).AGCStep(ihc_out);
      // These lines assign the output of the model for the current sample
      // to the appropriate data members of the current ear in the output
      // object.
      seg_output->StoreNAPOutput(i, j, n_ch_, ihc_out);
      // TODO alexbrandmeyer: Check with Dick to determine the C++ strategy for
      // storing optional output structures. Note for revision 271: added flags
      // to the 'Run' and 'RunSegment' methods to allow selective storage of
      // the different model output stages.
      if (store_bm) { seg_output->StoreBMOutput(i, j, n_ch_, car_out); }
      if (store_ohc) {
        seg_output->StoreOHCOutput(i, j, n_ch_, ears_.at(j).ReturnZAMemory()); }
      if (store_agc) {
        seg_output->StoreAGCOutput(i, j, n_ch_, ears_.at(j).ReturnZBMemory()); }
    }
    if (updated && n_ears > 1) { CrossCouple(); }
    if (! open_loop) { CloseAGCLoop(); }
  }
}

void CARFAC::CrossCouple() {
  for (int stage = 0; stage < ears_[0].ReturnAGCNStages(); stage++) {
    if (ears_[0].ReturnAGCStateDecimPhase(stage) > 0) {
      break;
    } else {
      FPType mix_coeff = ears_[0].ReturnAGCMixCoeff(stage);
      if (mix_coeff > 0) {
        FloatArray stage_state;
        FloatArray this_stage_values = FloatArray::Zero(n_ch_);
        for (int ear = 0; ear < n_ears_; ear++) {
          stage_state = ears_.at(ear).ReturnAGCStateMemory(stage);
          this_stage_values += stage_state;
        }
        this_stage_values /= n_ears_;
        for (int ear = 0; ear < n_ears_; ear++) {
          stage_state = ears_.at(ear).ReturnAGCStateMemory(stage);
          ears_.at(ear).SetAGCStateMemory(stage,
                                         stage_state + mix_coeff *
                                         (this_stage_values - stage_state));
        }
      }
    }
  }
}

void CARFAC::CloseAGCLoop() {
  for (int ear = 0; ear < n_ears_; ear++) {
    FloatArray undamping = 1 - ears_[ear].ReturnAGCStateMemory(1);
    // This updates the target stage gain for the new damping.
    ears_.at(ear).SetCARStateDZBMemory(ears_.at(ear).ReturnZRCoeffs() *
                                       undamping -
                                       ears_.at(ear).ReturnZBMemory() /
                                       ears_.at(ear).ReturnAGCDecimation(1));
    ears_.at(ear).SetCARStateDGMemory((ears_.at(ear).StageGValue(undamping) -
                                       ears_.at(ear).ReturnGMemory()) /
                                       ears_.at(ear).ReturnAGCDecimation(1));
  }
}
