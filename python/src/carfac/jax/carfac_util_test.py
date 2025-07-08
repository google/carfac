"""Test suite for carfac_util.

Unit tests for carfac util functions.
"""

from absl.testing import absltest
import jax
import jax.flatten_util
import jax.numpy as jnp

from carfac.jax import carfac
from carfac.jax import carfac_util

import os
os.environ["XLA_FLAGS"] = (
   "--xla_force_host_platform_device_count=2"
)

_NOISE_FACTOR = 1e-2


class CarfacUtilTest(absltest.TestCase):
  default_delta = 1e-7

  def setUp(self):
    super().setUp()
    self.ihc_style = 'two_cap'
    self.random_seed = 17234
    self.open_loop = False
    params_jax = carfac.CarfacDesignParameters()
    params_jax.ears[0].ihc.ihc_style = self.ihc_style
    params_jax.ears[0].car.linear_car = False
    self.random_generator = jax.random.PRNGKey(self.random_seed)
    self.hypers, self.weights, self.init_state = carfac.design_and_init_carfac(
        params_jax
    )
    self.n_ears = 1
    self.n_samp = 22050 * 2  # 2 seconds of audio.
    # make 2 samples, run both in parallel or together, and see if they're the
    # same. One side is _much_ louder than the other.

    self.sample_a = (
        jax.random.normal(self.random_generator, (self.n_samp, self.n_ears))
        * _NOISE_FACTOR
    )
    self.random_generator, sub_key = jax.random.split(self.random_generator)
    self.sample_b = (
        jax.random.normal(sub_key, (self.n_samp, self.n_ears))
        * _NOISE_FACTOR
        * 10
    )
    self.sample_c = jax.random.normal(sub_key, (250, self.n_ears)) * _NOISE_FACTOR

  def test_same_outputs_parallel_for_pmap(self):
    combined_samples = jnp.concatenate(
        [
            self.sample_a.reshape((1, self.n_samp, self.n_ears)),
            self.sample_b.reshape((1, self.n_samp, self.n_ears)),
        ],
        axis=0,
    )
    nap_out_a, nap_fibers_out_a, state_out_a, bm_out_a, ohc_out_a, agc_out_a = (
        carfac.run_segment_jit(
            self.sample_a,
            self.hypers,
            self.weights,
            self.init_state,
            self.open_loop,
        )
    )
    nap_out_b, nap_fibers_out_b, state_out_b, bm_out_b, ohc_out_b, agc_out_b = (
        carfac.run_segment_jit(
            self.sample_b,
            self.hypers,
            self.weights,
            self.init_state,
            self.open_loop,
        )
    )
    combined_output = carfac_util.run_multiple_segment_pmap(
        combined_samples,
        self.hypers,
        self.weights,
        self.init_state,
        self.open_loop,
    )
    self.assertTrue((combined_output[0][0] == nap_out_a).all())
    self.assertTrue((combined_output[1][0] == nap_out_b).all())
    self.assertTrue((combined_output[0][1] == nap_fibers_out_a).all())
    self.assertTrue((combined_output[1][1] == nap_fibers_out_b).all())
    self.assertTrue((combined_output[0][3] == bm_out_a).all())
    self.assertTrue((combined_output[1][3] == bm_out_b).all())
    self.assertTrue((combined_output[0][4] == ohc_out_a).all())
    self.assertTrue((combined_output[1][4] == ohc_out_b).all())
    self.assertTrue((combined_output[0][5] == agc_out_a).all())
    self.assertTrue((combined_output[1][5] == agc_out_b).all())
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree.map(jnp.allclose, state_out_a, combined_output[0][2])
        )
    )
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree.map(jnp.allclose, state_out_b, combined_output[1][2])
        )
    )

  def test_same_outputs_parallel_for_shmap(self):
    combined_samples = jnp.concatenate(
        [
            self.sample_a.reshape((1, self.n_samp, self.n_ears)),
            self.sample_b.reshape((1, self.n_samp, self.n_ears)),
        ],
        axis=0,
    )

    nap_out_a, nap_fibers_out_a, state_out_a, bm_out_a, ohc_out_a, agc_out_a = (
        carfac.run_segment_jit(
            self.sample_a,
            self.hypers,
            self.weights,
            self.init_state,
            self.open_loop,
        )
    )

    # Run sample B twice, so we have a separate "starting" state for the
    # test for shmap.
    _, _, state_out_b_first, _, _, _ = carfac.run_segment_jit(
        self.sample_b,
        self.hypers,
        self.weights,
        self.init_state,
        self.open_loop,
    )

    nap_out_b, nap_fibers_out_b, state_out_b, bm_out_b, ohc_out_b, agc_out_b = (
        carfac.run_segment_jit(
            self.sample_b,
            self.hypers,
            self.weights,
            state_out_b_first,
            self.open_loop,
        )
    )
    combined_output = carfac_util.run_multiple_segment_states_shmap(
        combined_samples,
        self.hypers,
        self.weights,
        [self.init_state, state_out_b_first],
        self.open_loop,
    )
    self.assertTrue((combined_output[0][0] == nap_out_a).all())
    self.assertTrue((combined_output[1][0] == nap_out_b).all())
    self.assertTrue((combined_output[0][1] == nap_fibers_out_a).all())
    self.assertTrue((combined_output[1][1] == nap_fibers_out_b).all())
    self.assertTrue((combined_output[0][3] == bm_out_a).all())
    self.assertTrue((combined_output[1][3] == bm_out_b).all())
    self.assertTrue((combined_output[0][4] == ohc_out_a).all())
    self.assertTrue((combined_output[1][4] == ohc_out_b).all())
    self.assertTrue((combined_output[0][5] == agc_out_a).all())
    self.assertTrue((combined_output[1][5] == agc_out_b).all())
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree.map(jnp.allclose, state_out_a, combined_output[0][2])
        )
    )
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree.map(jnp.allclose, state_out_b, combined_output[1][2])
        )
    )

    # The state is the most unusual one in all the outputs: ensure that
    # equality is complete and double sided.
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree.map(jnp.allclose, combined_output[0][2], state_out_a)
        )
    )
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree.map(jnp.allclose, combined_output[1][2], state_out_b)
        )
    )

def test_noise_run_scan_for_grad_for_naps(self):
    # stim_hypers for chunk_length of entire stimulus i.e. one iteration of lax.scan
    stim_hypers = carfac_util.ScanGradHypers()
    chunk_length = 44100
    stim_hypers.chunk_num = self.n_samp // chunk_length
    # stim_hypers for chunk_length of 100 samples i.e. 441 iterations of lax.scan
    stim_hypers_v2 = carfac_util.ScanGradHypers()
    chunk_length_v2 = 100
    stim_hypers_v2.chunk_num = self.n_samp // chunk_length_v2

    self.assertEqual(stim_hypers.chunk_num * chunk_length, self.n_samp)
    # Confirm chunk_length (44100 samples) * chunk_num is the same length as original stimulus

    self.assertEqual(stim_hypers_v2.chunk_num * chunk_length_v2, self.n_samp)
    # Confirm chunk_length (100 samples) * chunk_num is the same length as original stimulus

    nap_out_a, _, _, _, _, _ = carfac.run_segment(
        self.sample_a,
        self.hypers,
        self.weights,
        self.init_state,
        self.open_loop,
    )

    nap_out_b, stimulus = carfac_util.run_scan_for_grad(
        self.sample_a,
        self.hypers,
        self.weights,
        self.init_state,
        stim_hypers,
    )

    nap_out_b_v2, stimulus_v2 = carfac_util.run_scan_for_grad(
        self.sample_a,
        self.hypers,
        self.weights,
        self.init_state,
        stim_hypers_v2,
    )
    self.assertTrue((stimulus == jnp.squeeze(self.sample_a[:, 1])).all())
    # Confirm no difference between reconstructed stimulus_stacked and original noise
    self.assertTrue((stimulus == stimulus_v2).all())
    # Confirm different chunk_length does not affect reconstructed stimulus
    self.assertTrue((nap_out_b_v2 == nap_out_b).all())
    # Confirm no difference in nap_out for different chunk_lengths
    diff_direct_scanned = nap_out_a - nap_out_b
    self.assertTrue(jnp.max(jnp.abs(diff_direct_scanned)) < 1e-4)
    # Maximum difference between direct and scanned NAPs is less that 1e-4 (for fp32)
    self.assertTrue(jnp.sqrt(jnp.mean(diff_direct_scanned**2)) < 1e-5)
    # RMS difference between direct and scanned NAPs is less that 1e-5 (for fp32)

def test_gradient_direct_vs_scanned(self):
    stim_hypers = carfac_util.ScanGradHypers()
    chunk_length = 25
    stim_hypers.chunk_num = len(self.sample_c) // chunk_length

    def loss_func(
        audio: jnp.ndarray,
        hypers: carfac.CarfacHypers,
        weights: carfac.CarfacWeights,
        state: carfac.CarfacState,
    ):
        nap_output, _, _, _, _, _ = carfac.run_segment(audio, hypers, weights, state)
        return jnp.sum(nap_output), nap_output

    def loss_func_chunked(
        audio: jnp.ndarray,
        hypers: carfac.CarfacHypers,
        weights: carfac.CarfacWeights,
        state: carfac.CarfacState,
        stim_hypers: carfac_util.ScanGradHypers,
    ):
        naps, _ = carfac_util.run_scan_for_grad(
            audio, hypers, weights, state, stim_hypers
        )
        return jnp.sum(naps), naps

    jitted_loss = jax.jit(
        jax.grad(loss_func, argnums=2, has_aux=True),
        static_argnames=["hypers"],
    )
    jitted_chunked_loss = jax.jit(
        jax.grad(loss_func_chunked, argnums=2, has_aux=True),
        static_argnames=["hypers", "stim_hypers"],
    )

    (grad_direct, _) = jitted_loss(
        self.sample_c, self.hypers, self.weights, self.init_state
    )
    (grad_chunked, _) = jitted_chunked_loss(
        self.sample_c,
        self.hypers,
        self.weights,
        self.init_state,
        stim_hypers,
    )

    grad_direct_flat, _ = jax.flatten_util.ravel_pytree(grad_direct)
    grad_chunked_flat, _ = jax.flatten_util.ravel_pytree(grad_chunked)
    self.assertSequenceAlmostEqual(grad_direct_flat, grad_chunked_flat, delta=0.3)
    # Confirm no difference > 0.3 between gradient calculated for direct and scanned CARFAC
    percent_diff = 100 * ((grad_direct_flat - grad_chunked_flat) / grad_direct_flat)
    percent_diff = percent_diff[~jnp.isnan(percent_diff)]
    self.assertFalse(jnp.max(percent_diff) > 1e-4)
    # Confirm no difference between gradient calculated for direct and scanned CARFAC is greater than 0.0001%


if __name__ == "__main__":
  absltest.main()
