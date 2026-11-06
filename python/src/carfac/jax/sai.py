"""Stabilized Auditory Image (SAI) computation in JAX.

Copyright 2026 The CARFAC Authors. All Rights Reserved.

This file is part of an implementation of Lyon's cochlear model:
"Cascade of Asymmetric Resonators with Fast-Acting Compression"

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp


@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SaiDesignParameters:
  """Static parameters for monaural or binaural SAI computation.

  Validated on construction and passed as a static argument to JIT-compiled SAI
  functions.

  Attributes:
    num_channels: Number of filterbank channels (rows of an SAI frame).
    sai_width: Number of lag samples (columns) in each output SAI frame.
    future_lags: Number of lag samples taken from after the trigger point.
    num_triggers_per_frame: Number of 50%-overlapped trigger windows blended per
      output frame.
    trigger_window_width: Width in samples of each trigger search window.
    input_segment_width: Number of new samples consumed per `run_segment` call.
    num_ears: 1 for monaural SAI, 2 for cross-ear binaural SAI.
  """

  num_channels: int = 71
  sai_width: int = 400
  future_lags: int = 199
  num_triggers_per_frame: int = 2
  trigger_window_width: int = 401
  input_segment_width: int = 200
  num_ears: int = 1

  def __post_init__(self):
    if self.num_channels <= 0:
      raise ValueError(f'num_channels must be > 0, got {self.num_channels}.')
    if self.sai_width <= 0:
      raise ValueError(f'sai_width must be > 0, got {self.sai_width}.')
    if self.trigger_window_width <= 0:
      raise ValueError(
          f'trigger_window_width must be > 0, got {self.trigger_window_width}.'
      )
    if self.num_triggers_per_frame <= 0:
      raise ValueError(
          'num_triggers_per_frame must be > 0, got'
          f' {self.num_triggers_per_frame}.'
      )
    if self.num_ears not in (1, 2):
      raise ValueError(f'num_ears must be 1 or 2, got {self.num_ears}.')
    if not 0 < self.input_segment_width <= self.buffer_width:
      raise ValueError(
          'input_segment_width must be in (0, buffer_width], got'
          f' {self.input_segment_width} with'
          f' buffer_width={self.buffer_width}.'
      )
    if not 0 <= self.future_lags <= self.window_start:
      raise ValueError(
          f'future_lags must be in [0, {self.window_start}], got'
          f' {self.future_lags}.'
      )

  @property
  def buffer_width(self) -> int:
    """Width in samples of the per-channel input history buffer."""
    return self.sai_width + int(
        (1 + float(self.num_triggers_per_frame - 1) / 2)
        * self.trigger_window_width
    )

  @property
  def window_hop(self) -> int:
    """Hop in samples between consecutive trigger windows."""
    # Integer floor division matches carfac.python.sai.
    return self.trigger_window_width // 2

  @property
  def window_start(self) -> int:
    """Buffer index of the first trigger window when `future_lags == 0`."""
    return (self.buffer_width - self.trigger_window_width) - (
        self.num_triggers_per_frame - 1
    ) * self.window_hop


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SaiWeights:
  """Weights used during SAI trigger selection."""

  window: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SaiState:
  """Mutable buffer state carried across `run_segment` calls."""

  input_buffer: jax.Array
  output_buffer: jax.Array


def design_sai(params: SaiDesignParameters) -> SaiWeights:
  """Computes the trigger selection `SaiWeights` for `params`."""
  n = params.trigger_window_width
  # Matches carfac.python.sai's float64 window exactly; float32 alters triggers.
  start = math.pi / n
  step = (math.pi - start) / max(n - 1, 1)
  window = [math.sin(start + i * step) for i in range(n - 1)]
  window.append(math.sin(math.pi))
  return SaiWeights(window=jnp.asarray(window))


def init_sai(params: SaiDesignParameters) -> SaiState:
  """Initializes zeroed SAI state buffers for `params`."""
  return SaiState(
      input_buffer=jnp.zeros(
          (params.num_ears, params.num_channels, params.buffer_width)
      ),
      output_buffer=jnp.zeros(
          (params.num_ears, params.num_channels, params.sai_width)
      ),
  )


def design_and_init_sai(
    params: SaiDesignParameters,
) -> tuple[SaiWeights, SaiState]:
  """Designs and initializes an SAI instance in one call."""
  return design_sai(params), init_sai(params)


def _check_shape(name: str, array: jax.Array, expected: tuple[int, ...]):
  if array.shape != expected:
    raise ValueError(f'Expected {name} shape {expected}, got {array.shape}.')


def shift_and_append_input(
    fresh_input_segment: jax.Array,
    input_buffer: jax.Array,
    params: SaiDesignParameters,
) -> jax.Array:
  """Shifts `input_buffer` left and appends `fresh_input_segment`."""
  ears_channels = (params.num_ears, params.num_channels)
  _check_shape(
      'input_segment',
      fresh_input_segment,
      (*ears_channels, params.input_segment_width),
  )
  _check_shape(
      'input_buffer', input_buffer, (*ears_channels, params.buffer_width)
  )
  return jnp.concatenate(
      [input_buffer[..., params.input_segment_width :], fresh_input_segment],
      axis=-1,
  )


def stabilize_segment(
    input_buffer: jax.Array,
    output_buffer: jax.Array,
    params: SaiDesignParameters,
    weights: SaiWeights,
) -> jax.Array:
  """Selects trigger peaks per channel and blends aligned lag slices.

  Each ear triggers on its own input and blends lag slices from the opposite
  ear's input (its own input when monaural).

  Args:
    input_buffer: Input history with shape `(num_ears, num_channels,
      buffer_width)`.
    output_buffer: Previous SAI frame with shape `(num_ears, num_channels,
      sai_width)`.
    params: Static SAI parameters.
    weights: SAI weights from `design_sai`.

  Returns:
    The updated SAI frame, shaped like `output_buffer`.
  """
  ears_channels = (params.num_ears, params.num_channels)
  _check_shape(
      'input_buffer', input_buffer, (*ears_channels, params.buffer_width)
  )
  _check_shape(
      'output_buffer', output_buffer, (*ears_channels, params.sai_width)
  )
  _check_shape('weights.window', weights.window, (params.trigger_window_width,))
  window_range_start = params.window_start - params.future_lags
  offset_range_start = 1 + params.window_start - params.sai_width
  fallback_idx = jnp.argmax(weights.window)
  fallback_val = weights.window[fallback_idx]

  def _stabilize_channel(
      trig_wave: jax.Array,
      nontrig_wave: jax.Array,
      out_row: jax.Array,
  ) -> jax.Array:
    for w in range(params.num_triggers_per_frame):
      offset = w * params.window_hop
      start = window_range_start + offset
      windowed = (
          trig_wave[start : start + params.trigger_window_width]
          * weights.window
      )
      peak_idx = jnp.argmax(windowed)
      peak_val = windowed[peak_idx]
      use_fallback = peak_val <= 0
      trigger_time = jnp.where(use_fallback, fallback_idx, peak_idx) + offset
      peak_val = jnp.where(use_fallback, fallback_val, peak_val)
      alpha = (0.025 + peak_val) / (0.5 + peak_val)
      aligned = jax.lax.dynamic_slice_in_dim(
          nontrig_wave, trigger_time + offset_range_start, params.sai_width
      )
      out_row = (1.0 - alpha) * out_row + alpha * aligned
    return out_row

  # Outer vmap maps over ears and inner over channels; [::-1] swaps the ears.
  return jax.vmap(jax.vmap(_stabilize_channel))(
      input_buffer, input_buffer[::-1], output_buffer
  )


def run_segment(
    input_segment: jax.Array,
    params: SaiDesignParameters,
    weights: SaiWeights,
    state: SaiState,
) -> tuple[jax.Array, SaiState]:
  """Computes one SAI frame from `input_segment`.

  Args:
    input_segment: Filterbank segment with shape `(num_ears, num_channels,
      input_segment_width)`.
    params: Static SAI parameters.
    weights: SAI weights from `design_sai`.
    state: Current `SaiState`.

  Returns:
    Tuple `(output_frame, new_state)` where `output_frame` has shape
    `(num_ears, num_channels, sai_width)`.
  """
  input_buffer = shift_and_append_input(
      input_segment, state.input_buffer, params
  )
  output_buffer = stabilize_segment(
      input_buffer, state.output_buffer, params, weights
  )
  new_state = SaiState(input_buffer=input_buffer, output_buffer=output_buffer)
  return output_buffer, new_state


@functools.partial(jax.jit, static_argnames=('params',))
def run_segment_jit(
    input_segment: jax.Array,
    params: SaiDesignParameters,
    weights: SaiWeights,
    state: SaiState,
) -> tuple[jax.Array, SaiState]:
  """JIT-compiled wrapper around `run_segment`."""
  return run_segment(input_segment, params, weights, state)


def _scan_segment_step(
    params: SaiDesignParameters,
    weights: SaiWeights,
    carry: SaiState,
    seg: jax.Array,
) -> tuple[SaiState, jax.Array]:
  frame, next_state = run_segment(seg, params, weights, carry)
  return next_state, frame


def run_sai(
    naps: jax.Array,
    params: SaiDesignParameters,
    weights: SaiWeights,
    state: SaiState | None = None,
) -> tuple[jax.Array, SaiState]:
  """Runs SAI across a multi-segment NAP sequence via `jax.lax.scan`.

  Args:
    naps: NAP array of shape `(num_samples, num_channels, num_ears)`, matching
      the CARFAC JAX output layout. `num_samples` must be a multiple of
      `params.input_segment_width`.
    params: Static SAI parameters.
    weights: SAI weights.
    state: Optional initial `SaiState`; initialized to zeros if omitted.

  Returns:
    Tuple `(sai_frames, final_state)` where `sai_frames` has shape
    `(num_frames, num_ears, num_channels, sai_width)`.
  """
  if naps.ndim != 3 or naps.shape[1:] != (
      params.num_channels,
      params.num_ears,
  ):
    raise ValueError(
        f'Expected naps shape (num_samples, {params.num_channels},'
        f' {params.num_ears}), got {naps.shape}.'
    )
  num_samples = naps.shape[0]
  if num_samples % params.input_segment_width != 0:
    raise ValueError(
        f'num_samples ({num_samples}) must be divisible by'
        f' input_segment_width ({params.input_segment_width}).'
    )
  segments = naps.reshape(
      num_samples // params.input_segment_width,
      params.input_segment_width,
      params.num_channels,
      params.num_ears,
  ).transpose(0, 3, 2, 1)
  init_state = state if state is not None else init_sai(params)
  final_state, frames = jax.lax.scan(
      functools.partial(_scan_segment_step, params, weights),
      init_state,
      segments,
  )
  return frames, final_state


@functools.partial(jax.jit, static_argnames=('params',))
def run_sai_jit(
    naps: jax.Array,
    params: SaiDesignParameters,
    weights: SaiWeights,
    state: SaiState | None = None,
) -> tuple[jax.Array, SaiState]:
  """JIT-compiled wrapper around `run_sai`."""
  return run_sai(naps, params, weights, state)
