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

"""Classes to compute Stabilized Auditory Images from filterbank outputs."""

import numpy as np


class SAIParams(object):
  """Design parameters for an SAI object.

  Terminology: Each call to SAI.RunSegment consumes a fixed-length input
  "segment" and outputs a single output SAI "frame".

  Note on trigger settings: Each SAI frame computed by a call to
  SAI.RunSegment blends together several 50% overlapping "trigger windows"
  identified in the input buffer.  The size of the buffer (i.e. the
  total number of samples used to generate the SAI) is controlled by the
  number and size of the trigger windows.  See _SAIBase._buffer_width below
  for details.

  Attributes:
    num_channels: Number of channels (height, or number of rows) of an
        SAI frame.

    sai_width: The total width (i.e. number of lag samples, or columns)
        of an SAI frame.

    future_lags: Number of lag samples that should come from the future.

    num_triggers_per_frame: Number of trigger windows to consider when
        computing a single SAI frame during each call to RunSegment.

    trigger_window_width: Size in samples of the window used when
        searching for each trigger point.

    input_segment_width: Expected size in samples of the input segments
        that will be passed into RunSegment.  This is only used to
        validate the input size.  Since each call to RunSegment
        generates exactly one output SAI frame, this parameter
        implicitly controls the output frame rate and the hop size
        (i.e. number of new input samples consumed) between adjacent SAI
        frames.  See ShiftAndAppendInput() below for details.
  """

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def __repr__(self):
    kwargs_str = ", ".join(
        "{}={}".format(k, v) for k, v in self.__dict__.iteritems())
    return "SAIParams({})".format(kwargs_str)


class _SAIBase(object):
  """Base class for the monaural and binaural SAI implementations."""

  def __init__(self, params):
    self.Redesign(params)

  def Redesign(self, params):
    """Reinitializes using the specified parameters.

    Args:
      params: An SAIParams object.

    Redesign calls Reset().  Subclasses should do any subclass-specific
    redesign inside Reset().
    """
    self.params = params
    self._buffer_width = params.sai_width + int(
        (1 + float(params.num_triggers_per_frame - 1) / 2) *
        params.trigger_window_width)

    assert params.trigger_window_width > params.sai_width
    assert params.num_triggers_per_frame > 0
    assert params.input_segment_width < self._buffer_width

    # Window function to apply before selecting a trigger point.
    self.window = np.sin(np.linspace(num=params.trigger_window_width,
                                     start=(np.pi /
                                            self.params.trigger_window_width),
                                     stop=np.pi))

    self.Reset()

  def _ShiftAndAppendInput(self, fresh_input_segment, input_buffer):
    """Shifts and appends new data to input_buffer."""
    num_rows, num_cols = fresh_input_segment.shape
    assert num_rows == self.params.num_channels
    assert num_cols == self.params.input_segment_width

    overlap_width = self._buffer_width - self.params.input_segment_width
    input_buffer[:, :overlap_width] = (
        input_buffer[:, self._buffer_width - overlap_width:])
    input_buffer[:, overlap_width:] = fresh_input_segment

  @staticmethod
  def _MaxAndIndex(array):
    """Analog to Eigen's maxCoeff method."""
    index = np.argmax(array)
    return array[index], index

  def _StabilizeSegment(self, triggering_input_buffer,
                        nontriggering_input_buffer, output_buffer):
    """Chooses trigger points and blends windowed signals into output_buffer."""
    assert triggering_input_buffer.shape == nontriggering_input_buffer.shape

    # Windows are always approximately 50% overlapped.
    num_samples = triggering_input_buffer.shape[1]
    window_hop = self.params.trigger_window_width / 2
    window_start = ((num_samples - self.params.trigger_window_width) -
                    (self.params.num_triggers_per_frame - 1) * window_hop)
    window_range_start = window_start - self.params.future_lags

    offset_range_start = 1 + window_start - self.params.sai_width
    assert offset_range_start > 0
    for i in xrange(self.params.num_channels):
      triggering_nap_wave = triggering_input_buffer[i, :]
      nontriggering_nap_wave = nontriggering_input_buffer[i, :]
      # TODO(ronw): Smooth triggering signal to be consistent with the
      # Matlab implementation.

      for w in xrange(self.params.num_triggers_per_frame):
        current_window_offset = w * window_hop

        # Choose a trigger point.
        current_window_start = window_range_start + current_window_offset
        trigger_window = triggering_nap_wave[current_window_start:
                                             current_window_start +
                                             self.params.trigger_window_width]
        peak_val, trigger_time = self._MaxAndIndex(trigger_window * self.window)
        if peak_val <= 0:
          peak_val, trigger_time = self._MaxAndIndex(self.window)
        trigger_time += current_window_offset

        # Blend the window following the trigger into the output buffer,
        # weighted according to the the trigger strength (0.05 to near 1.0).
        alpha = (0.025 + peak_val) / (0.5 + peak_val)
        output_buffer[i, :] *= 1 - alpha
        output_buffer[i, :] += alpha * nontriggering_nap_wave[
            trigger_time + offset_range_start:
            trigger_time + offset_range_start + self.params.sai_width]


class SAI(_SAIBase):
  """Class implementing the Stabilized Auditory Image.

  Repeated calls to the RunSegment method compute a sparse approximation
  to the running autocorrelation of a multi-channel input signal,
  typically a segment of the neural activity pattern (NAP) outputs of the
  CARFAC filterbank.
  """

  def __init__(self, params):
    super(SAI, self).__init__(params)

  def Reset(self):
    """Resets the internal state."""
    self._input_buffer = np.zeros((self.params.num_channels,
                                   self._buffer_width))
    self._output_buffer = np.zeros((self.params.num_channels,
                                    self.params.sai_width))

  def RunSegment(self, input_segment):
    """Computes an SAI frame from input_segment and self._input_buffer.

    Args:
      input_segment: A numpy array with shape (params.num_channels,
          params.input_segment_width) containing a segment of filterbank
          output.  Note that the expected input shape is the transpose
          of the shape of the input to SAI_Run_Segment.m.

    Returns:
      A numpy array with shape (params.num_channels, params.sai_width)
      containing an SAI frame.
    """
    self._ShiftAndAppendInput(input_segment, self._input_buffer)
    self._StabilizeSegment(self._input_buffer, self._input_buffer,
                           self._output_buffer)
    return self._output_buffer

  def __repr__(self):
    return "SAI({})".format(self.params)


# TODO(ronw): Port the C++ binaural SAI to Python.
