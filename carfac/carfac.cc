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
void CARFAC::Design(int n_ears, long fs, CARParams car_params,
                    IHCParams ihc_params, AGCParams agc_params) {
  n_ears_ = n_ears;
  fs_ = fs;
  ears_ = new Ear[n_ears_];
  
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
    ears_[i].InitEar(n_ch_, fs_, pole_freqs, car_params, ihc_params, agc_params);
  }
}

CARFACOutput CARFAC::Run(FloatArray2d sound_data, bool open_loop) {
  // We initialize one output object to store the final output.
  CARFACOutput *output = new CARFACOutput();
  // A second object is used to store the output for the individual segments.
  CARFACOutput *seg_output = new CARFACOutput();
  int n_audio_channels = int(sound_data.cols());
  long seg_len = 441;  // We use a fixed segment length for now.
  long n_timepoints = sound_data.rows();
  long n_segs = ceil((n_timepoints * 1.0) / seg_len);
  output->InitOutput(n_audio_channels, n_ch_, n_timepoints);
  seg_output->InitOutput(n_audio_channels, n_ch_, seg_len);
  // These values store the start and endpoints for each segment
  long start;
  long length = seg_len;
  // This section loops over the individual audio segments.
  for (long i = 0; i < n_segs; i++) {
    // For each segment we calculate the start point and the segment length.
    start = i * seg_len;
    if (i == n_segs - 1) {
      // The last segment can be shorter than the rest.
      length = n_timepoints - start;
    }
    // Once we've determined the start point and segment length, we run the
    // CARFAC model on the current segment.
    RunSegment(sound_data.block(start, 0, length, n_audio_channels),
               seg_output, open_loop);
    // Afterwards we merge the output for the current segment into the larger
    // output structure for the entire audio file.
    output->MergeOutput(*seg_output, start, length);
  }
  return *output;
}

void CARFAC::RunSegment(FloatArray2d sound_data, CARFACOutput *seg_output,
                        bool open_loop) {
  // The number of timepoints is determined from the length of the audio
  // segment.
  long n_timepoints = sound_data.rows();
  // The number of ears is equal to the number of audio channels. This could
  // potentially be removed since we already know the n_ears_ during the design
  // stage. 
  int n_ears = int(sound_data.cols());
  // A nested loop structure is used to iterate through the individual samples
  // for each ear (audio channel).
  bool updated;  // This variable is used by the AGC stage.
  for (long i = 0; i < n_timepoints; i++) {
    for (int j = 0; j < n_ears; j++) {
      // This stores the audio sample currently being processed.
      FPType input = sound_data(i, j);
      // Now we apply the three stages of the model in sequence to the current
      // audio sample.
      FloatArray car_out = ears_[j].CARStep(input);
      FloatArray ihc_out = ears_[j].IHCStep(car_out);
      updated = ears_[j].AGCStep(ihc_out);
      // These lines assign the output of the model for the current sample
      // to the appropriate data members of the current ear in the output
      // object.
      seg_output->ears_[j].nap_.block(0, i, n_ch_, 1) = ihc_out;
      // TODO alexbrandmeyer: Check with Dick to determine the C++ strategy for
      // storing optional output structures.
      seg_output->ears_[j].bm_.block(0, i, n_ch_, 1) = car_out;
      seg_output->ears_[j].ohc_.block(0, 1, n_ch_, 1) =
        ears_[j].ReturnZAMemory();
      seg_output->ears_[j].agc_.block(0, i, n_ch_, 1) =
        ears_[j].ReturnZBMemory();
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
          stage_state = ears_[ear].ReturnAGCStateMemory(stage);
          this_stage_values += stage_state;
        }
        this_stage_values /= n_ears_;
        for (int ear = 0; ear < n_ears_; ear++) {
          stage_state = ears_[ear].ReturnAGCStateMemory(stage);
          ears_[ear].SetAGCStateMemory(stage,
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
    ears_[ear].SetCARStateDZBMemory(ears_[ear].ReturnZRCoeffs() * undamping -
                                    ears_[ear].ReturnZBMemory() /
                                    ears_[ear].ReturnAGCDecimation(1));
    ears_[ear].SetCARStateDGMemory((ears_[ear].StageGValue(undamping) -
                                   ears_[ear].ReturnGMemory()) /
                                   ears_[ear].ReturnAGCDecimation(1));
  }
}
