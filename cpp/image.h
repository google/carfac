// Copyright 2022 The CARFAC Authors. All Rights Reserved.
// Author: Pascal Getreuer
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

// Image class.
//
// A templated image class with elements of type `Scalar`. Images may be
// single-channel or multichannel, having any number of channels. By default, an
// Image owns and manages the underlying memory. Alternatively, an Image can map
// (without taking ownership) an existing memory buffer.

#ifndef CARFAC_IMAGE_H_
#define CARFAC_IMAGE_H_

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>

#include "common.h"

template <typename _Scalar>
class Image {
 public:
  // Image element type.
  // `Scalar` may be either `unsigned char` or `float`, or for a read-only
  // image view, it may be `const unsigned char` or `const float`.
  using Scalar = _Scalar;

  // Constructs an empty image.
  Image();
  // Allocates a single-channel image of size `width` by `height` in pixels.
  // Only allowed for nonconst Scalar. Memory is uninitialized.
  Image(int width, int height);
  // Same as above, but allocates a multichannel image. The memory layout is
  // interleaved (channel stride = 1).
  Image(int width, int height, int channels);

  // Construct a single-channel image as a view of existing memory.
  // Memory is mapped as
  //
  //   `image(x, y) = *(data + x * x_stride + y * y_stride)`
  //
  // for 0 <= x < width, 0 <= y < height. Does not take ownership of data.
  // Strides `x_stride` and `y_stride` are in units of `Scalar` elements.
  // Negative strides are allowed, for example, to represent a flipped view of
  // the image.
  Image(Scalar* data, int width, int x_stride, int height, int y_stride);
  // Same as above, but for a multichannel image view.
  // Memory is mapped as
  //
  //   `image(x, y, c) = *(data + x * x_stride + y * y_stride + c * c_stride).`
  Image(Scalar* data, int width, int x_stride,
        int height, int y_stride, int channels, int c_stride);

  // Copy constructor, creates a shallow view of rhs image.
  template <typename RhsScalar> Image(const Image<RhsScalar>& rhs);  // NOLINT
  // Assignment, creates a shallow view of rhs image.
  template <typename RhsScalar> Image& operator=(const Image<RhsScalar>& rhs);

  // Basic accessors.

  int width() const { return width_; }  // Image width in pixels.
  int height() const { return height_; }  // Image height in pixels.
  int channels() const { return channels_; }  // Number of image channels.
  Scalar* data() const { return data_; }  // Pointer to element (0, 0, 0).
  int x_stride() const { return x_stride_; }  // Stride between columns.
  int y_stride() const { return y_stride_; }  // Stride between rows.
  int c_stride() const { return c_stride_; }  // Stride between channels.
  // True if image dimensions are zero.
  bool empty() const { return width_ <= 0 || height_ <= 0 || channels_ <= 0; }
  // Number of pixels.
  int num_pixels() const { return width_ * height_; }
  // Number of bytes.
  int size_in_bytes() const {
    return num_pixels() * channels_ * sizeof(Scalar);
  }

  // True if Image the maps unowned memory.
  bool maps_unowned_memory() const { return data_ != nullptr && !alloc_; }
  // Reference count for shared image memory.
  int use_count() const { return alloc_.use_count(); }

  // Accessors for image pixels.

  // Access element (x, y). Coordinates must satisfy 0 <= x < width, 0 <= y <
  // height. No bounds checking is done.
  Scalar& operator()(int x, int y) const {
    return *(data_ + x * x_stride_ + y * y_stride_);
  }
  // Access element (x, y, c). Coordinates must satisfy 0 <= x < width, 0 <= y
  // height, 0 <= c <= channels. No bounds checking is done.
  Scalar& operator()(int x, int y, int c) const {
    return *(data_ + x * x_stride_ + y * y_stride_ + c * c_stride_);
  }

  // Cropping and slicing.

  // Returns a crop with upper left corner (x0, y0) and size width by height.
  // Args are in units of pixels.
  Image<const Scalar> crop(int x0, int y0, int width, int height) const {
    return Image(data_ + x0 * x_stride_ + y0 * y_stride_, width, x_stride_,
                 height, y_stride_, channels_, c_stride_);
  }
  Image<Scalar> crop(int x0, int y0, int width, int height) {
    return Image(data_ + x0 * x_stride_ + y0 * y_stride_, width, x_stride_,
                 height, y_stride_, channels_, c_stride_);
  }

  // Get the xth column as a 1-pixel-wide image.
  Image<const Scalar> col(int x) const { return crop(x, 0, 1, height_); }
  Image<Scalar> col(int x) { return crop(x, 0, 1, height_); }

  // Computes the root mean square difference vs. `reference`, provided
  // the images have the same shape. Returns infinity if the shapes differ.
  float RootMeanSquareDiff(const Image<const Scalar>& reference) const;

 private:
  template <typename Fun>
  struct NumArgs : public NumArgs<decltype(&Fun::operator())> {};
  template <typename ClassType, typename ReturnType, typename... Args>
  struct NumArgs<ReturnType (ClassType::*)(Args...) const>
      : public std::integral_constant<int, sizeof...(Args)> {};

  void Allocate(int width, int height, int channels);
  template <typename RhsScalar>
  void AssertCanAssignFrom() const;

  typedef typename std::remove_const<Scalar>::type NonConstScalar;
  std::shared_ptr<NonConstScalar> alloc_;
  Scalar* data_;
  int width_;
  int height_;
  int channels_;
  int x_stride_;
  int y_stride_;
  int c_stride_;

  template <typename> friend class Image;
};

// Writes `image` to `filename` in Portable Anymap (PNM) format in binary mode,
// a simple format which can be read for instance by GIMP. This is useful to
// dump an image without depending on libpng or another library for image I/O.
//
// The image must have 1 or 3 channels. Returns true on success.
bool WritePnm(const std::string& filename, const Image<const uint8_t>& image);

// Reads a Portable Anymap (PNM), Portable PixMap (PPM), or Portable GrayMap
// (PGM) image file in 8-bit binary format. Returns true on success.
bool ReadPnm(const std::string& filename, Image<uint8_t>* image);

// Implementation details only below this line. --------------------------------

template <typename Scalar>
void Image<Scalar>::Allocate(int width, int height, int channels) {
  CARFAC_ASSERT(width >= 0);
  CARFAC_ASSERT(height >= 0);
  CARFAC_ASSERT(channels >= 0);
  data_ = new NonConstScalar[width * height * channels];
  alloc_.reset(data_, +[](NonConstScalar* p) { delete [] p; });
  width_ = width;
  height_ = height;
  channels_ = channels;
  x_stride_ = channels;
  y_stride_ = channels * width;
  c_stride_ = 1;
}

template <typename Scalar>
Image<Scalar>::Image(): data_(nullptr), width_(0), height_(0), channels_(0) {}

template <typename Scalar>
Image<Scalar>::Image(Scalar* data, int width, int x_stride,
                     int height, int y_stride)
    : Image(data, width, x_stride, height, y_stride, 1, 0) {}

template <typename Scalar>
Image<Scalar>::Image(Scalar* data, int width, int x_stride,
                     int height, int y_stride, int channels, int c_stride)
    : data_(data),
      width_(width),
      height_(height),
      channels_(channels),
      x_stride_(x_stride),
      y_stride_(y_stride),
      c_stride_(c_stride) {
  if (!empty()) {
    CARFAC_ASSERT(data != nullptr);
  }
}

template <typename Scalar>
Image<Scalar>::Image(int width, int height) {
  static_assert(!std::is_const<Scalar>::value,
                "Image allocation must have nonconst type");
  Allocate(width, height, 1);
}

template <typename Scalar>
Image<Scalar>::Image(int width, int height, int channels) {
  static_assert(!std::is_const<Scalar>::value,
                "Image allocation must have nonconst type");
  Allocate(width, height, channels);
}

template <typename Scalar>
template <typename RhsScalar>
void Image<Scalar>::AssertCanAssignFrom() const {
  static_assert(std::is_same<typename std::remove_const<Scalar>::type,
                             typename std::remove_const<RhsScalar>::type
                             >::value,
                "Invalid assignment from incompatible Image type");
  static_assert(std::is_const<Scalar>::value ||
                !std::is_const<RhsScalar>::value,
                "Invalid assignment from const to nonconst Image");
}

template <typename Scalar>
template <typename RhsScalar>
Image<Scalar>::Image(const Image<RhsScalar>& rhs) {
  AssertCanAssignFrom<RhsScalar>();
  alloc_ = rhs.alloc_;
  data_ = rhs.data_;
  width_ = rhs.width_;
  height_ = rhs.height_;
  channels_ = rhs.channels_;
  x_stride_ = rhs.x_stride_;
  y_stride_ = rhs.y_stride_;
  c_stride_ = rhs.c_stride_;
}

template <typename Scalar>
template <typename RhsScalar>
Image<Scalar>& Image<Scalar>::operator=(const Image<RhsScalar>& rhs) {
  AssertCanAssignFrom<RhsScalar>();
  alloc_ = rhs.alloc_;
  data_ = rhs.data_;
  width_ = rhs.width_;
  height_ = rhs.height_;
  channels_ = rhs.channels_;
  x_stride_ = rhs.x_stride_;
  y_stride_ = rhs.y_stride_;
  c_stride_ = rhs.c_stride_;
  return *this;
}

template <typename Scalar>
float Image<Scalar>::RootMeanSquareDiff(
    const Image<const Scalar>& reference) const {
  if (width_ != reference.width() ||
      height_ != reference.height() ||
      channels_ != reference.channels()) {
    return INFINITY;
  }

  double sum = 0.0;
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      for (int c = 0; c < channels_; ++c) {
        const double diff = static_cast<double>((*this)(x, y, c)) -
                            static_cast<double>(reference(x, y, c));
        sum += diff * diff;
      }
    }
  }

  if (sum == 0.0) { return 0.0f; }

  sum /= width_ * height_ * channels_;
  return std::sqrt(static_cast<float>(sum));
}

#endif  // CARFAC_IMAGE_H_
