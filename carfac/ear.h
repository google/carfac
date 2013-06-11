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

#include "agc.h"
#include "car.h"
#include "common.h"
#include "ihc.h"

// The Ear object carries out the three steps of the CARFAC model on a single
// channel of audio data, and stores information about the CAR, IHC and AGC
// filter coefficients and states.
class Ear {
 public:
  Ear(const int num_channels, const CARCoeffs& car_coeffs,
      const IHCCoeffs& ihc_coeffs,
      const std::vector<AGCCoeffs>& agc_coeffs);

  void Reset(const int num_channels, const CARCoeffs& car_coeffs,
             const IHCCoeffs& ihc_coeffs,
             const std::vector<AGCCoeffs>& agc_coeffs);

  // These three methods apply the different steps of the model in sequence
  // to individual audio samples during the call to CARFAC::RunSegment.
  void CARStep(const FPType input);
  // TODO(ronw): Consider changing the interface for the following two
  // methods to access the internal state members directly instead of
  // requiring callers to confusingly have to call an accessor method
  // just to pass internal data back into the same object as in:
  //   ear.IHCStep(ear.car_out());
  void IHCStep(const ArrayX& car_out);
  bool AGCStep(const ArrayX& ihc_out);

  // These accessor functions return portions of the CAR state for storage in
  // the CAROutput structures.
  const ArrayX& za_memory() const { return car_state_.za_memory; }
  const ArrayX& zb_memory() const { return car_state_.zb_memory; }

  // The zy_memory_ of the CARState is equivalent to the CAR output. A second
  // accessor function is included for documentation purposes.
  const ArrayX& zy_memory() const { return car_state_.zy_memory; }
  const ArrayX& car_out() const { return car_state_.zy_memory; }
  const ArrayX& g_memory() const { return car_state_.g_memory; }
  const ArrayX& ihc_out() const { return ihc_state_.ihc_out; }
  const ArrayX& dzb_memory() const { return car_state_.dzb_memory; }
  const ArrayX& zr_coeffs() const { return car_coeffs_.zr_coeffs; }

  // These accessor functions return portions of the AGC state during the cross
  // coupling of the ears.
  const int agc_num_stages() const { return agc_coeffs_.size(); }
  const int agc_decim_phase(const int stage) const {
    return agc_state_[stage].decim_phase;
  }
  const FPType agc_mix_coeff(const int stage) const {
    return agc_coeffs_[stage].agc_mix_coeffs;
  }
  const ArrayX& agc_memory(const int stage) const {
    return agc_state_[stage].agc_memory;
  }
  const int agc_decimation(const int stage) const {
    return agc_coeffs_[stage].decimation;
  }

  // Returns the stage G value during the closing of the AGC loop.
  ArrayX StageGValue(const ArrayX& undamping) const;

  // Sets the AGC memory during the cross coupling stage.
  void set_agc_memory(const int stage, const ArrayX& new_values) {
    agc_state_[stage].agc_memory = new_values;
  }

  // Setter functions for the CAR memory states.
  void set_dzb_memory(const ArrayX& new_values) {
    car_state_.dzb_memory = new_values;
  }
  void set_dg_memory(const ArrayX& new_values) {
    car_state_.dg_memory = new_values;
  }

 private:
  // Initializes the model state variables prior to runtime.
  void ResetIHCState();
  void ResetAGCState();
  void ResetCARState();

  // Helper sub-functions called during the model runtime.
  void OHCNonlinearFunction(const ArrayX& velocities,
                            ArrayX* nonlinear_fun) const;
  bool AGCRecurse(const int stage, ArrayX agc_in);
  void AGCSpatialSmooth(const AGCCoeffs& agc_coeffs, ArrayX* stage_state) const;
  void AGCSmoothDoubleExponential(const FPType pole_z1, const FPType pole_z2,
                                  ArrayX* stage_state) const;

  CARCoeffs car_coeffs_;
  CARState car_state_;
  IHCCoeffs ihc_coeffs_;
  IHCState ihc_state_;

  // The AGC coefficient and state variables are both stored in vectors
  // containing one element for each stage (default = 4).
  std::vector<AGCCoeffs> agc_coeffs_;
  std::vector<AGCState> agc_state_;
  int num_channels_;

  DISALLOW_COPY_AND_ASSIGN(Ear);
};

#endif  // CARFAC_EAR_H
