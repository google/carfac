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

#ifndef CARFAC_EAR_H
#define CARFAC_EAR_H

#include <vector>
#include "carfac_common.h"
#include "car_coeffs.h"
#include "ihc_coeffs.h"
#include "agc_coeffs.h"
#include "car_state.h"
#include "ihc_state.h"
#include "agc_state.h"

class Ear {
 public:
  // This is the primary initialization function that is called for each
  // Ear object in the CARFAC 'Design' method.
  void InitEar(const int n_ch, const FPType fs,
               const CARCoeffs& car_coeffs, const IHCCoeffs& ihc_coeffs,
               const std::vector<AGCCoeffs>& agc_coeffs);
  // These three methods apply the different stages of the model in sequence
  // to individual audio samples.
  void CARStep(const FPType input);
  void IHCStep(const FloatArray& car_out);
  bool AGCStep(const FloatArray& ihc_out);
  // These accessor functions return portions of the CAR state for storage in
  // the CAROutput structures.
  const FloatArray& za_memory() { return car_state_.za_memory; }
  const FloatArray& zb_memory() { return car_state_.zb_memory; }
  // The zy_memory_ of the CARState is equivalent to the CAR output. A second
  // accessor function is included for documentation purposes.
  const FloatArray& zy_memory() { return car_state_.zy_memory; }
  const FloatArray& car_out() { return car_state_.zy_memory; }
  const FloatArray& g_memory() { return car_state_.g_memory; }
  // This returns the IHC output for storage.
  const FloatArray& ihc_out() { return ihc_state_.ihc_out; }
  const FloatArray& dzb_memory() { return car_state_.dzb_memory; }
  // These accessor functions return CAR coefficients.
  const FloatArray& zr_coeffs() { return car_coeffs_.zr_coeffs; }
  // These accessor functions return portions of the AGC state during the cross
  // coupling of the ears.
  const int agc_nstages() { return agc_coeffs_.size(); }
  const int agc_decim_phase(const int stage) {
    return agc_state_[stage].decim_phase; }
  const FPType agc_mix_coeff(const int stage) {
    return agc_coeffs_[stage].agc_mix_coeffs; }
  const FloatArray& agc_memory(const int stage) {
    return agc_state_[stage].agc_memory; }
  const int agc_decimation(const int stage) {
    return agc_coeffs_[stage].decimation; }
  // This returns the stage G value during the closing of the AGC loop.
  FloatArray StageGValue(const FloatArray& undamping);
  // This function sets the AGC memory during the cross coupling stage.
  void set_agc_memory(const int stage, const FloatArray& new_values) {
    agc_state_[stage].agc_memory = new_values; }
  // These are the setter functions for the CAR memory states.
  void set_dzb_memory(const FloatArray& new_values) {
    car_state_.dzb_memory = new_values; }
  void set_dg_memory(const FloatArray& new_values) {
    car_state_.dg_memory = new_values; }

 private:
  // These are the corresponding methods that initialize the model state
  // variables before runtime using the model coefficients.
  void InitIHCState();
  void InitAGCState();
  void InitCARState();
  // These are the various helper functions called during the model runtime.
  void OHCNonlinearFunction(const FloatArray& velocities,
                            FloatArray* nonlinear_fun);
  bool AGCRecurse(const int stage, FloatArray agc_in);
  void AGCSpatialSmooth(const AGCCoeffs& agc_coeffs , FloatArray* stage_state);
  void AGCSmoothDoubleExponential(const FPType pole_z1, const FPType pole_z2,
                                  FloatArray* stage_state);
  // These are the private data members that store the state and coefficient
  // information.
  CARCoeffs car_coeffs_;
  IHCCoeffs ihc_coeffs_;
  std::vector<AGCCoeffs> agc_coeffs_;
  CARState car_state_;
  IHCState ihc_state_;
  std::vector<AGCState> agc_state_;
  int n_ch_;
};

#endif  // CARFAC_EAR_H