// Copyright 2013 The CARFAC Authors. All Rights Reserved.
// Author: Alex Brandmeyer
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

#ifndef CARFAC_COMMON_H
#define CARFAC_COMMON_H

#include <cassert>

// The Eigen library is used extensively for floating point arrays.
// For more information, see: http://eigen.tuxfamily.org
#include <Eigen/Core>

// This typedef is used to enable easy switching in precision level
// for the Eigen containers used throughout this library.
typedef float FPType;
typedef Eigen::Array<FPType, Eigen::Dynamic, 1> ArrayX;
typedef Eigen::Array<FPType, Eigen::Dynamic, Eigen::Dynamic> ArrayXX;

// This macro disallows the copy constructor and operator= functions.
// This should be used in the private: declarations for a class.
#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&)
#endif

// This abstraction makes it easy to redefine all assertions used in
// this library if the basic assert macro is insufficient.
#define CARFAC_ASSERT(expression) assert(expression);

#endif  // CARFAC_COMMON_H
