# Lint as: python3
#!/usr/bin/env python

# Copyright 2013 The CARFAC Authors. All Rights Reserved.
#
# This file is part of an implementation of Lyon's cochlear model:
# "Cascade of Asymmetric Resonators with Fast-Acting Compression"
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for carfac.python.sai."""

import os
import unittest

import numpy as np

import sai as pysai

_TEST_DATA_DIR = "../test_data"


def LoadMatrix(filename, rows, columns):
  """Reads a matrix with shape (rows, columns) matrix from a text file."""
  matrix = np.loadtxt(os.path.join(_TEST_DATA_DIR, filename))
  assert matrix.shape == (rows, columns)
  return matrix


def WriteMatrix(filename, matrix):
  np.savetxt(os.path.join(_TEST_DATA_DIR, filename), matrix, fmt="%0.12f")


def CreatePulseTrain(num_channels, num_samples, period, leading_zeros=0):
  segment = np.zeros((num_channels, num_samples))
  for i in range(num_channels):
    # Begin each channel at a different phase.
    phase = (i + leading_zeros) % period
    for j in range(phase, num_samples, period):
      segment[i, j] = 1
  return segment


def CreateSAIParams(sai_width, num_triggers_per_frame=2, **kwargs):
  """Fills an SAIParams object using reasonable defaults for some fields."""
  return pysai.SAIParams(sai_width=sai_width,
                         # Half of the SAI should come from the future.
                         future_lags=sai_width // 2,
                         num_triggers_per_frame=num_triggers_per_frame,
                         **kwargs)


def HasPeakAt(frame, index):
  if index == 0:
    return frame[index] > frame[index + 1]
  elif index == len(frame) - 1:
    return frame[index] > frame[index - 1]
  return frame[index] > frame[index + 1] and frame[index] > frame[index - 1]


class PeriodicInputTest(unittest.TestCase):
  def _RunMultiChannelPulseTrainTest(self, period, num_channels):
    input_segment_width = 38
    segment = CreatePulseTrain(num_channels, input_segment_width, period)

    sai_params = CreateSAIParams(num_channels=num_channels,
                                 input_segment_width=input_segment_width,
                                 trigger_window_width=input_segment_width,
                                 sai_width=15)
    sai_params.future_lags = 0  # Only compute past lags.

    sai = pysai.SAI(sai_params)
    sai_frame = sai.RunSegment(segment)

    # The output should have peaks at the same positions, regardless of
    # input phase.
    for i in range(num_channels):
      sai_channel = sai_frame[i, :]
      for j in range(len(sai_channel) - 1, 0, -period):
        print(i, j, sai_channel, HasPeakAt(sai_channel, j))
        self.assertTrue(HasPeakAt(sai_channel, j))

    print("Input\n{}\nOutput\n{}".format(segment, sai_frame))

  def testMultiChannelPulseTrain(self):
    for period in [25, 10, 5, 2]:
      for num_channels in [1, 2, 15]:
        print("Testing period={}, num_channels={}".format(period, num_channels))
        self._RunMultiChannelPulseTrainTest(period, num_channels)


class SAITest(unittest.TestCase):
  def testInputSegmentWidthIsLargerThanBuffer(self):
    params = CreateSAIParams(num_channels=2, sai_width=10,
                             input_segment_width=200,
                             trigger_window_width=200)
    sai = pysai.SAI(params)

    params.trigger_window_width = params.input_segment_width // 10
    self.assertGreater(params.input_segment_width,
                       params.num_triggers_per_frame *
                       params.trigger_window_width)
    self.assertRaises(AssertionError, sai.Redesign, params)

  def testInputWidthDoesntMatchInputSegmentWidth(self):
    num_channels = 2
    input_segment_width = 10
    segment = CreatePulseTrain(num_channels, input_segment_width, period=4)

    sai_width = 20
    expected_input_segment_width = input_segment_width - 1
    sai_params = CreateSAIParams(
        num_channels=num_channels,
        sai_width=sai_width,
        input_segment_width=expected_input_segment_width,
        trigger_window_width=sai_width + 1)
    self.assertNotEqual(sai_params.input_segment_width, input_segment_width)
    sai = pysai.SAI(sai_params)
    self.assertRaises(AssertionError, sai.RunSegment, segment)

  def testInputSegmentWidthSmallerThanTriggerWindow(self):
    """Tests small hop between segments."""
    num_channels = 1
    total_input_samples = 20
    period = 5
    full_input = CreatePulseTrain(num_channels, total_input_samples, period)

    num_frames = 4
    input_segment_width = total_input_samples // num_frames
    sai_params = CreateSAIParams(num_channels=num_channels,
                                 input_segment_width=input_segment_width,
                                 trigger_window_width=total_input_samples,
                                 sai_width=15)
    self.assertLess(sai_params.input_segment_width,
                    sai_params.trigger_window_width)
    sai_params.future_lags = 0  # Only compute past lags.

    self.assertGreaterEqual(period, input_segment_width)

    sai = pysai.SAI(sai_params)
    for i in range(num_frames):
      segment = (
          full_input[:, i * input_segment_width:(i+1) * input_segment_width])
      sai_frame = sai.RunSegment(segment)

      print("Frame {}\nInput\n{}\nOutput\n{}".format(i, segment, sai_frame))

      self.assertNotEqual(np.abs(segment).sum(), 0)
      # Since the input segment is never all zero, there should always
      # be a peak at zero lag.
      sai_channel = sai_frame[0, :]
      self.assertTrue(HasPeakAt(sai_channel, len(sai_channel) - 1))

      if i == 0:
        # Since the pulse train period is larger than the input segment
        # size, the first input segment will only see a single impulse,
        # most of the SAI will be zero.
        np.testing.assert_allclose(sai_channel[:len(sai_channel) - 1],
                                   np.zeros(len(sai_channel) - 1), 1e-9)

      if i == num_frames - 1:
        # By the last frame, the SAI's internal buffer will have
        # accumulated the full input signal, so the resulting image
        # should contain kPeriod peaks.
        for j in range(len(sai_channel) - 1, 0, -period):
          self.assertTrue(HasPeakAt(sai_channel, j))

  def testMatchesMatlabOnBinauralData(self):
    test_name = "binaural_test"
    input_segment_width = 882
    num_channels = 71
    # The Matlab CARFAC output is transposed compared to the C++.
    input_segment = LoadMatrix(test_name + "-matlab-nap1.txt",
                               input_segment_width,
                               num_channels).transpose()

    sai_params = CreateSAIParams(num_channels=num_channels,
                                 input_segment_width=input_segment_width,
                                 trigger_window_width=input_segment_width,
                                 sai_width=500)
    sai = pysai.SAI(sai_params)
    sai_frame = sai.RunSegment(input_segment)
    expected_sai_frame = LoadMatrix(test_name + "-matlab-sai1.txt",
                                    num_channels, sai_params.sai_width)
    np.testing.assert_allclose(expected_sai_frame, sai_frame, rtol=1e-5)

    WriteMatrix(test_name + "-py-sai1.txt", sai_frame)


if __name__ == "__main__":
  unittest.main()
