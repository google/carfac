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
                    IHCParams ihc_params, AGCParams agc_params){
  n_ears_ = n_ears;
  fs_ = fs;
  ears_ = new Ear[n_ears_];
  for (int i = 0; i < n_ears_; i++){
    ears_[i].InitEar(fs_, car_params,ihc_params,agc_params);
  }
  n_ch_ = ears_[0].car_coeffs_.n_ch_;
}

CARFACOutput CARFAC::Run(FloatArray2d sound_data){
  //to store the final output
  CARFACOutput *output = new CARFACOutput();
  //to store the output of the individual segments
  CARFACOutput *seg_output = new CARFACOutput();
  
  int n_audio_channels = int(sound_data.cols());
  long seg_len = 441; //Fixed segment length for now
  long n_timepoints = sound_data.rows();
  double n_segs = ceil(double(n_timepoints)/double(seg_len));
  output->InitOutput(n_audio_channels, n_ch_, n_timepoints);
  seg_output->InitOutput(n_audio_channels, n_ch_, seg_len);
  //loop over individual audio segments
  long start, length; //to store the start and endpoints for each segment
  for (long i = 0; i < long(n_segs); i++){
    //determine start and end points
    start = (i * seg_len);
    if (i < n_segs - 1){
      length = seg_len;
    } else {
      length = n_timepoints - start; //the last segment can be shorter than the rest
    }
    RunSegment(sound_data.block(start,0,length,n_audio_channels),seg_output);
    output->MergeOutput(*seg_output, start, length);
  }
  
  return *output;
}

void CARFAC::RunSegment(FloatArray2d sound_data, CARFACOutput *seg_output){
  long n_timepoints = sound_data.rows();
  int n_ears = int(sound_data.cols());
  for (long i = 0; i < n_timepoints; i++){
    for (int j = 0; j < n_ears; j++){
      FPType input = sound_data(i,j);
      FloatArray car_out = ears_[j].CARStep(input);
      FloatArray ihc_out = ears_[j].IHCStep(car_out);
      bool updated = ears_[j].AGCStep(ihc_out);
      seg_output->ears_[j].nap_.block(0,i,n_ch_,1) = ihc_out;
      seg_output->ears_[j].bm_.block(0,i,n_ch_,1) = car_out;
      seg_output->ears_[j].ohc_.block(0,1,n_ch_,1) =
        ears_[j].car_state_.za_memory_;
      seg_output->ears_[j].agc_.block(0,i,n_ch_,1) =
        ears_[j].car_state_.zb_memory_;
    }
  }
  
}
