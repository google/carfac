"""Tests for Stabilized Auditory Image (SAI) in JAX."""

import importlib.resources

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from jax.tree_util import tree_unflatten
import numpy as np

from carfac import sai as pysai
from carfac.jax import carfac as carfac_jax
from carfac.jax import sai as sai_jax

_TEST_DATA_PACKAGE = 'carfac.test_data'
_TEST_DATA_DIR = importlib.resources.files(_TEST_DATA_PACKAGE)


def _load_matrix(filename: str, rows: int, columns: int) -> jax.Array:
  text = _TEST_DATA_DIR.joinpath(filename).read_text()
  data = [
      [float(val) for val in line.split()]
      for line in text.splitlines()
      if line.strip()
  ]
  matrix = jnp.asarray(data, dtype=jnp.float32)
  assert matrix.shape == (rows, columns)
  return matrix


def _create_pulse_train(
    num_channels: int,
    num_samples: int,
    period: int,
    leading_zeros: int = 0,
) -> jax.Array:
  ch = jnp.arange(num_channels)[:, None]
  t = jnp.arange(num_samples)[None, :]
  phase = (ch + leading_zeros) % period
  return jnp.where((t - phase) % period == 0, 1.0, 0.0).astype(jnp.float32)


def _has_peak_at(frame: jax.Array, index: int) -> bool:
  if index == 0:
    return bool(frame[index] > frame[index + 1])
  if index == len(frame) - 1:
    return bool(frame[index] > frame[index - 1])
  return bool(
      frame[index] > frame[index + 1] and frame[index] > frame[index - 1]
  )


class SaiJaxTest(parameterized.TestCase):

  def _assert_allclose(
      self,
      actual: jax.Array,
      expected: jax.Array,
      rtol: float = 1e-5,
      atol: float = 1e-6,
  ):
    actual_arr = jnp.asarray(actual)
    expected_arr = jnp.asarray(expected, dtype=actual_arr.dtype)
    self.assertEqual(actual_arr.shape, expected_arr.shape)
    self.assertTrue(
        bool(jnp.allclose(actual_arr, expected_arr, rtol=rtol, atol=atol))
    )

  @parameterized.product(
      period=[25, 10, 5, 2],
      num_channels=[1, 2, 15],
  )
  def test_monaural_periodic_pulse_train(self, period: int, num_channels: int):
    input_segment_width = 38
    segment = _create_pulse_train(num_channels, input_segment_width, period)
    params = sai_jax.SaiDesignParameters(
        num_channels=num_channels,
        sai_width=15,
        future_lags=0,
        num_triggers_per_frame=2,
        trigger_window_width=input_segment_width,
        input_segment_width=input_segment_width,
    )
    weights, state = sai_jax.design_and_init_sai(params)
    frame, _ = sai_jax.run_segment_jit(segment[None], params, weights, state)
    self.assertEqual(frame.shape, (1, num_channels, 15))
    for ch in range(num_channels):
      for idx in range(frame.shape[2] - 1, -1, -period):
        self.assertTrue(_has_peak_at(frame[0, ch], idx))

  @parameterized.product(
      period=[25, 10, 5, 2],
      num_channels=[1, 2, 15],
  )
  def test_binaural_periodic_pulse_train(self, period: int, num_channels: int):
    input_segment_width = 38
    left_delay = 0
    right_delay = 1
    seg_left = _create_pulse_train(
        num_channels, input_segment_width, period, left_delay
    )
    seg_right = _create_pulse_train(
        num_channels, input_segment_width, period, right_delay
    )
    segment = jnp.stack([seg_left, seg_right], axis=0)
    params = sai_jax.SaiDesignParameters(
        num_channels=num_channels,
        sai_width=15,
        future_lags=2,
        num_triggers_per_frame=2,
        trigger_window_width=input_segment_width,
        input_segment_width=input_segment_width,
        num_ears=2,
    )
    weights, state = sai_jax.design_and_init_sai(params)
    frame, _ = sai_jax.run_segment_jit(segment, params, weights, state)
    self.assertEqual(frame.shape, (2, num_channels, 15))
    for ear in range(2):
      sign = 1 if ear == 0 else -1
      ear_offset = params.future_lags + 1 + sign * (left_delay - right_delay)
      for ch in range(num_channels):
        for idx in range(frame.shape[2] - ear_offset, -1, -period):
          self.assertTrue(_has_peak_at(frame[ear, ch], idx))

  @parameterized.parameters(
      (16, 40, 20, 2, 41, 20),
      (8, 32, 0, 1, 33, 16),
      (12, 24, 12, 3, 26, 12),
      (5, 15, 7, 4, 20, 10),
      (12, 24, 12, 3, 25, 12),
      (5, 15, 7, 4, 21, 10),
  )
  def test_matches_python_sai_across_frames(
      self,
      num_channels: int,
      sai_width: int,
      future_lags: int,
      num_triggers: int,
      trigger_window_width: int,
      input_segment_width: int,
  ):
    key = jax.random.PRNGKey(42)
    py_params = pysai.SAIParams(
        num_channels=num_channels,
        sai_width=sai_width,
        future_lags=future_lags,
        num_triggers_per_frame=num_triggers,
        trigger_window_width=trigger_window_width,
        input_segment_width=input_segment_width,
    )
    ref_sai = pysai.SAI(py_params)
    jax_params = sai_jax.SaiDesignParameters(
        num_channels=num_channels,
        sai_width=sai_width,
        future_lags=future_lags,
        num_triggers_per_frame=num_triggers,
        trigger_window_width=trigger_window_width,
        input_segment_width=input_segment_width,
    )
    weights, state = sai_jax.design_and_init_sai(jax_params)
    self.assertTrue(
        bool(
            jnp.all(
                weights.window == jnp.asarray(
                    ref_sai.window, weights.window.dtype
                )
            )
        )
    )

    segments = []
    ref_frames = []
    for frame_idx in range(6):
      key, subkey = jax.random.split(key)
      noise = jax.random.normal(
          subkey, (num_channels, input_segment_width), dtype=jnp.float32
      )
      if frame_idx == 2:
        # Exercise non-positive peak fallback path.
        seg = jnp.zeros((num_channels, input_segment_width), dtype=jnp.float32)
      elif frame_idx == 3:
        seg = -jnp.abs(noise)
      else:
        seg = jnp.maximum(0.0, noise)
      segments.append(seg)
      ref_frames.append(jnp.asarray(ref_sai.RunSegment(seg).copy()))
      out_jax, state = sai_jax.run_segment_jit(
          seg[None], jax_params, weights, state
      )
      self._assert_allclose(out_jax[0], ref_frames[-1], rtol=1e-5, atol=1e-6)

    naps = jnp.concatenate(segments, axis=1).T[:, :, None]
    expected_stack = jnp.stack(ref_frames, axis=0)
    scanned_frames, _ = sai_jax.run_sai_jit(naps, jax_params, weights)
    self._assert_allclose(
        scanned_frames[:, 0], expected_stack, rtol=1e-5, atol=1e-6
    )
    scanned_frames_no_static, _ = jax.jit(sai_jax.run_sai)(
        naps, jax_params, weights
    )
    self._assert_allclose(
        scanned_frames_no_static[:, 0],
        expected_stack,
        rtol=1e-5,
        atol=1e-6,
    )

  def test_small_hop_between_segments(self):
    num_channels = 1
    total_input_samples = 20
    period = 5
    full_input = _create_pulse_train(num_channels, total_input_samples, period)
    num_frames = 4
    input_segment_width = total_input_samples // num_frames
    params = sai_jax.SaiDesignParameters(
        num_channels=num_channels,
        sai_width=15,
        future_lags=0,
        num_triggers_per_frame=2,
        trigger_window_width=total_input_samples,
        input_segment_width=input_segment_width,
    )
    weights, state = sai_jax.design_and_init_sai(params)
    for i in range(num_frames):
      seg = full_input[
          None, :, i * input_segment_width : (i + 1) * input_segment_width
      ]
      frame, state = sai_jax.run_segment_jit(seg, params, weights, state)
      sai_channel = frame[0, 0]
      self.assertTrue(_has_peak_at(sai_channel, len(sai_channel) - 1))
      if i == 0:
        self._assert_allclose(
            sai_channel[:-1], jnp.zeros(len(sai_channel) - 1), atol=1e-7
        )
      if i == num_frames - 1:
        for idx in range(len(sai_channel) - 1, -1, -period):
          self.assertTrue(_has_peak_at(sai_channel, idx))

  @parameterized.parameters(
      ('binaural_test', 882, 71),
      ('long_test', 2000, 83),
      ('agc_test', 2000, 83),
      ('ihc_just_hwr_test', 2000, 83),
  )
  def test_matches_matlab_on_binaural_data(
      self, test_name: str, input_segment_width: int, num_channels: int
  ):
    input_segment = _load_matrix(
        f'{test_name}-matlab-nap1.txt', input_segment_width, num_channels
    ).T
    expected_sai_frame = _load_matrix(
        f'{test_name}-matlab-sai1.txt', num_channels, 500
    )
    params = sai_jax.SaiDesignParameters(
        num_channels=num_channels,
        sai_width=500,
        future_lags=250,
        num_triggers_per_frame=2,
        trigger_window_width=input_segment_width,
        input_segment_width=input_segment_width,
    )
    weights, state = sai_jax.design_and_init_sai(params)
    sai_frame, _ = sai_jax.run_segment_jit(
        input_segment[None], params, weights, state
    )
    self._assert_allclose(
        sai_frame[0], expected_sai_frame, rtol=1e-5, atol=1e-6
    )

  def test_supports_wide_sai_pitchogram_settings(self):
    # Matches C++ PitchogramPipeline where sai_width > trigger_window_width.
    params = sai_jax.SaiDesignParameters(
        num_channels=8,
        sai_width=800,
        future_lags=799,
        num_triggers_per_frame=2,
        trigger_window_width=257,
        input_segment_width=256,
    )
    weights, state = sai_jax.design_and_init_sai(params)
    seg = jnp.ones((1, 8, 256), dtype=jnp.float32)
    frame, state = sai_jax.run_segment_jit(seg, params, weights, state)
    self.assertEqual(frame.shape, (1, 8, 800))
    self.assertTrue(bool(jnp.all(jnp.isfinite(frame))))

  def test_binaural_matches_python_cross_ear_stabilization(self):
    kwargs = dict(
        num_channels=4,
        sai_width=10,
        future_lags=4,
        num_triggers_per_frame=2,
        trigger_window_width=11,
        input_segment_width=6,
    )
    ref_sai = pysai.SAI(pysai.SAIParams(**kwargs))
    params = sai_jax.SaiDesignParameters(**kwargs, num_ears=2)
    weights, state = sai_jax.design_and_init_sai(params)
    key = jax.random.PRNGKey(7)
    for _ in range(4):
      key, subkey = jax.random.split(key)
      seg = jnp.maximum(0.0, jax.random.normal(subkey, (2, 4, 6)))
      prev_output = np.array(state.output_buffer)
      frame, state = sai_jax.run_segment(seg, params, weights, state)
      buf = np.asarray(state.input_buffer)
      for ear in range(2):
        # Each ear triggers on its own input and blends the other ear's input.
        expected = prev_output[ear].copy()
        ref_sai._StabilizeSegment(buf[ear], buf[1 - ear], expected)  # pylint: disable=protected-access
        self._assert_allclose(frame[ear], expected)

  def test_carfac_jax_integration_and_pytree_roundtrip(self):
    cf_hypers, cf_weights, cf_state = carfac_jax.design_and_init_carfac(
        carfac_jax.CarfacDesignParameters.with_n_ears(2, fs=16000.0)
    )
    audio = jnp.sin(
        2.0 * jnp.pi * 220.0 * jnp.arange(640, dtype=jnp.float32) / 16000.0
    )
    stereo = jnp.stack([audio, jnp.roll(audio, 2)], axis=-1)
    naps, _, _, _, _, _ = carfac_jax.run_segment_jit(
        stereo, cf_hypers, cf_weights, cf_state
    )
    n_ch = cf_hypers.ears[0].car.n_ch
    sai_params = sai_jax.SaiDesignParameters(
        num_channels=n_ch,
        sai_width=200,
        future_lags=100,
        num_triggers_per_frame=2,
        trigger_window_width=201,
        input_segment_width=160,
        num_ears=2,
    )
    weights, state = sai_jax.design_and_init_sai(sai_params)
    leaves, treedef = tree_flatten((sai_params, weights, state))
    self.assertLen(leaves, 3)
    params_rt, weights_rt, state_rt = tree_unflatten(treedef, leaves)
    self.assertEqual(sai_params, params_rt)
    frames, final_state = sai_jax.run_sai(naps, params_rt, weights_rt, state_rt)
    self.assertEqual(frames.shape, (4, 2, n_ch, 200))
    self.assertEqual(final_state.output_buffer.shape, (2, n_ch, 200))

    mono_params = sai_jax.SaiDesignParameters(
        num_channels=n_ch,
        sai_width=200,
        future_lags=100,
        num_triggers_per_frame=2,
        trigger_window_width=201,
        input_segment_width=160,
        num_ears=1,
    )
    mono_weights = sai_jax.design_sai(mono_params)
    identical_naps = jnp.concatenate([naps[:, :, :1], naps[:, :, :1]], axis=-1)
    identical_frames, _ = sai_jax.run_sai(identical_naps, sai_params, weights)
    mono_frames, _ = sai_jax.run_sai(naps[:, :, :1], mono_params, mono_weights)
    self.assertEqual(mono_frames.shape, (4, 1, n_ch, 200))
    self._assert_allclose(identical_frames[:, :1], mono_frames)
    self._assert_allclose(identical_frames[:, 1:], mono_frames)

  @parameterized.parameters(
      dict(num_channels=0),
      dict(sai_width=0),
      dict(trigger_window_width=0),
      dict(num_triggers_per_frame=0),
      dict(num_ears=3),
      dict(input_segment_width=0),
      dict(
          sai_width=10,
          future_lags=0,
          input_segment_width=200,
          trigger_window_width=20,
      ),
      dict(future_lags=-1),
      dict(sai_width=20, future_lags=25),
  )
  def test_invalid_params_raise(self, **kwargs):
    with self.assertRaises(ValueError):
      sai_jax.SaiDesignParameters(**kwargs)

  def test_invalid_shapes_raise(self):
    params = sai_jax.SaiDesignParameters(
        num_channels=2,
        sai_width=20,
        future_lags=10,
        trigger_window_width=21,
        input_segment_width=9,
    )
    weights, state = sai_jax.design_and_init_sai(params)
    with self.assertRaises(ValueError):
      sai_jax.run_segment(jnp.zeros((1, 2, 10)), params, weights, state)
    with self.assertRaises(ValueError):
      sai_jax.run_segment(jnp.zeros((2, 9)), params, weights, state)
    valid_seg = jnp.zeros((1, 2, 9))
    bad_input_buffer_state = sai_jax.SaiState(
        input_buffer=jnp.zeros((1, 2, params.buffer_width + 1)),
        output_buffer=state.output_buffer,
    )
    with self.assertRaises(ValueError):
      sai_jax.run_segment(valid_seg, params, weights, bad_input_buffer_state)
    bad_output_buffer_state = sai_jax.SaiState(
        input_buffer=state.input_buffer,
        output_buffer=jnp.zeros((1, 2, params.sai_width + 1)),
    )
    with self.assertRaises(ValueError):
      sai_jax.run_segment(valid_seg, params, weights, bad_output_buffer_state)
    bad_weights = sai_jax.SaiWeights(
        window=jnp.zeros((params.trigger_window_width + 1,))
    )
    with self.assertRaises(ValueError):
      sai_jax.run_segment(valid_seg, params, bad_weights, state)
    with self.assertRaises(ValueError):
      sai_jax.run_sai(jnp.zeros((18, 2)), params, weights)

  @parameterized.parameters(1, 2)
  @jax.numpy_dtype_promotion('strict')
  def test_strict_dtype_promotion(self, num_ears: int):
    params = sai_jax.SaiDesignParameters(
        num_channels=4,
        sai_width=10,
        future_lags=2,
        num_triggers_per_frame=2,
        trigger_window_width=12,
        input_segment_width=8,
        num_ears=num_ears,
    )
    weights, state = sai_jax.design_and_init_sai(params)
    seg = jax.random.normal(jax.random.PRNGKey(0), (num_ears, 4, 8))
    frame, _ = sai_jax.run_segment_jit(seg, params, weights, state)
    self.assertTrue(bool(jnp.all(jnp.isfinite(frame))))

    naps = jax.random.normal(jax.random.PRNGKey(1), (16, 4, num_ears))
    frames, _ = sai_jax.run_sai_jit(naps, params, weights)
    self.assertTrue(bool(jnp.all(jnp.isfinite(frames))))

  def test_vmap_and_grad(self):
    batch_size = 3
    num_channels = 2
    sai_width = 10
    trigger_window_width = 11
    input_segment_width = 21
    num_samples = 42
    params = sai_jax.SaiDesignParameters(
        num_channels=num_channels,
        sai_width=sai_width,
        future_lags=0,
        num_triggers_per_frame=1,
        trigger_window_width=trigger_window_width,
        input_segment_width=input_segment_width,
    )
    weights = sai_jax.design_sai(params)

    base_naps = jnp.full((num_samples, num_channels), 0.5, dtype=jnp.float32)
    # Set isolated peaks with margin to avoid argmax boundary transitions.
    base_naps = base_naps.at[15, 0].set(3.0)
    base_naps = base_naps.at[14, 1].set(3.0)
    base_naps = base_naps.at[21 + 16, 0].set(3.0)
    base_naps = base_naps.at[21 + 13, 1].set(3.0)[:, :, None]

    batch_naps = jnp.stack(
        [base_naps * (1.0 + 0.2 * i) for i in range(batch_size)], axis=0
    )
    vmapped_run = jax.vmap(sai_jax.run_sai, in_axes=(0, None, None, None))
    batch_frames, _ = vmapped_run(batch_naps, params, weights, None)
    expected_frames = [
        sai_jax.run_sai(batch_naps[i], params, weights, None)[0]
        for i in range(batch_size)
    ]
    self._assert_allclose(
        batch_frames,
        jnp.stack(expected_frames, axis=0),
        rtol=1e-5,
        atol=1e-6,
    )

    def loss_fn(x, w):
      return jnp.sum(sai_jax.run_sai(x, params, w)[0] ** 2)

    grad_x, grad_w = jax.grad(loss_fn, argnums=(0, 1))(base_naps, weights)
    self.assertTrue(bool(jnp.all(jnp.isfinite(grad_x))))
    self.assertTrue(bool(jnp.any(grad_x != 0.0)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(grad_w.window))))
    self.assertTrue(bool(jnp.any(grad_w.window != 0.0)))

    eps = 1e-3
    w_vec = weights.window
    jitted_loss = jax.jit(loss_fn)
    num_grad = []
    for i in range(len(w_vec)):
      e_i = jnp.zeros_like(w_vec).at[i].set(eps)
      loss_pos = jitted_loss(base_naps, sai_jax.SaiWeights(window=w_vec + e_i))
      loss_neg = jitted_loss(base_naps, sai_jax.SaiWeights(window=w_vec - e_i))
      num_grad.append((loss_pos - loss_neg) / (2.0 * eps))
    self._assert_allclose(
        grad_w.window, jnp.stack(num_grad), rtol=0.02, atol=0.01
    )


if __name__ == '__main__':
  absltest.main()
