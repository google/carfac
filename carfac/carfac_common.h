//
//  carfac_common.h
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
//
// *****************************************************************************
// carfac_common.h
// *****************************************************************************
// This file contains the base level definitions and includes used within the
// CARFAC C++ library. It also defines some low level functions which are used
// during the calculation of the various coefficient sets required by the model.
//
// The current implementation of the library is dependent on the use of the
// Eigen C++ library for linear algebra. Specifically, Eigen Arrays are used
// extensively for coefficient wise operations during both the design and run
// stages of the model.
//
// The 'FPType' typedef is specified in this file in order to enable quick
// switching between precision levels (i.e. float vs. double) throughout the
// library. The remainder of the code uses this type for specifying floating
// point scalars.
//
// Two additional typedefs are defined for one and two dimensional arrays:
// FloatArray and FloatArray2d. These in turn make use of FPType so that the
// precision level across floating point data is consistent.
//
// The functions 'ERBHz' and 'CARFACDetect' are defined here, and are used
// during the design stage of a CARFAC model. 

#ifndef CARFAC_Open_Source_C__Library_CARFACCommon_h
#define CARFAC_Open_Source_C__Library_CARFACCommon_h

#include <iostream> //Used for debugging output, could go in final version
#include <math.h> //Used during coefficient calculations
#include <Eigen/Dense> //Used for 1d and 2d floating point arrays
using namespace Eigen;

#define PI 3.141592
typedef float FPType; //Used to enable easy switching in precision level

//typedefs for Eigen floating point arrays
typedef Eigen::Array<FPType,Dynamic,1> FloatArray; //1d floating point array
typedef Eigen::Array<FPType,Dynamic,Dynamic> FloatArray2d; //2d fpoint array

// Function: ERBHz
// Auditory filter nominal Equivalent Rectangular Bandwidth
// Ref: Glasberg and Moore: Hearing Research, 47 (1990), 103-138
FPType ERBHz(FPType cf_hz, FPType erb_break_freq, FPType erb_q);

// Function CARFACDetect
// TODO explain a bit more
FPType CARFACDetect (FPType x);
FloatArray CARFACDetect (FloatArray x);
#endif
