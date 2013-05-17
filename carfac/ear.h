//
//  ear.h
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

#ifndef CARFAC_Open_Source_C__Library_Ear_h
#define CARFAC_Open_Source_C__Library_Ear_h

#include "car_state.h"
#include "ihc_state.h"
#include "agc_state.h"

class Ear {
 public:
  // This is the primary initialization function that is called for each
  // Ear object in the CARFAC 'Design' method.
  void InitEar(int n_ch, int32_t fs, FloatArray pole_freqs, CARParams car_p,
               IHCParams ihc_p, AGCParams agc_p);
  // These three methods apply the different stages of the model in sequence
  // to individual audio samples.
  FloatArray CARStep(FPType input);
  FloatArray IHCStep(FloatArray car_out);
  bool AGCStep(FloatArray ihc_out);
  // These helper functions return portions of the CAR state for storage in the
  // CAROutput structures.
  FloatArray ReturnZAMemory();
  FloatArray ReturnZBMemory();
  FloatArray ReturnGMemory();
  FloatArray ReturnZRCoeffs();
  // These helper functions return portions of the AGC state during the cross
  // coupling of the ears.
  int ReturnAGCNStages();
  int ReturnAGCStateDecimPhase(int stage);
  FPType ReturnAGCMixCoeff(int stage);
  int ReturnAGCDecimation(int stage);
  FloatArray ReturnAGCStateMemory(int stage);
  // This returns the stage G value during the closing of the AGC loop.
  FloatArray StageGValue(FloatArray undamping);
  // This function sets the AGC memory during the cross coupling stage.
  void SetAGCStateMemory(int stage, FloatArray new_values);
  // These are two functions to set the CARState dzB and dG memories when
  // closing the AGC loop
  void SetCARStateDZBMemory(FloatArray new_values);
  void SetCARStateDGMemory(FloatArray new_values);
 private:
  // These methods carry out the design of the coefficient sets for each of the
  // three model stages.
  void DesignFilters(CARParams car_params, int32_t fs, FloatArray pole_freqs);
  void DesignIHC(IHCParams ihc_params, int32_t fs);
  void DesignAGC(AGCParams agc_params, int32_t fs);
  // These are the corresponding methods that initialize the model state
  // variables before runtime using the model coefficients.
  void InitIHCState();
  void InitAGCState();
  void InitCARState();
  // These are the various helper functions called during the model runtime.
  FloatArray OHC_NLF(FloatArray velocities);
  bool AGCRecurse(int stage, FloatArray agc_in);
  FloatArray AGCSpatialSmooth(int stage, FloatArray stage_state);
  FloatArray AGCSmoothDoubleExponential(FloatArray stage_state, FPType pole_z1,
                                        FPType pole_z2);
  // These are the private data members that store the state and coefficient
  // information.
  CARCoeffs car_coeffs_;
  IHCCoeffs ihc_coeffs_;
  AGCCoeffs agc_coeffs_;
  CARState car_state_;
  IHCState ihc_state_;
  AGCState agc_state_;
  int n_ch_;
};

#endif