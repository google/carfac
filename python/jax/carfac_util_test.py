"""Test suite for carfac_util.

Unit tests for carfac util functions.
"""

from absl.testing import absltest
import jax
import jax.numpy as jnp

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
import carfac as carfac
import carfac_util as carfac_util

_NOISE_FACTOR = 1e-2


class CarfacUtilTest(absltest.TestCase):
  default_delta = 1e-7

  def setUp(self):
    super().setUp()
    self.one_cap = False
    self.random_seed = 17234
    self.open_loop = False
    params_jax = carfac.CarfacDesignParameters()
    params_jax.ears[0].ihc.n_caps = 1 if self.one_cap else 2
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

  def test_same_outputs_parallel_for_pmap(self):
    combined_samples = jnp.concatenate(
        [
            self.sample_a.reshape((1, self.n_samp, self.n_ears)),
            self.sample_b.reshape((1, self.n_samp, self.n_ears)),
        ],
        axis=0,
    )
    nap_out_a, state_out_a, bm_out_a, ohc_out_a, agc_out_a = (
        carfac.run_segment_jit(
            self.sample_a,
            self.hypers,
            self.weights,
            self.init_state,
            self.open_loop,
        )
    )
    nap_out_b, state_out_b, bm_out_b, ohc_out_b, agc_out_b = (
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
    self.assertTrue((combined_output[0][2] == bm_out_a).all())
    self.assertTrue((combined_output[1][2] == bm_out_b).all())
    self.assertTrue((combined_output[0][3] == ohc_out_a).all())
    self.assertTrue((combined_output[1][3] == ohc_out_b).all())
    self.assertTrue((combined_output[0][4] == agc_out_a).all())
    self.assertTrue((combined_output[1][4] == agc_out_b).all())
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_map(jnp.allclose, state_out_a, combined_output[0][1])
        )
    )
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_map(jnp.allclose, state_out_b, combined_output[1][1])
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

    nap_out_a, state_out_a, bm_out_a, ohc_out_a, agc_out_a = (
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
    _, state_out_b_first, _, _, _ = carfac.run_segment_jit(
        self.sample_b,
        self.hypers,
        self.weights,
        self.init_state,
        self.open_loop,
    )

    nap_out_b, state_out_b, bm_out_b, ohc_out_b, agc_out_b = (
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
    self.assertTrue((combined_output[0][2] == bm_out_a).all())
    self.assertTrue((combined_output[1][2] == bm_out_b).all())
    self.assertTrue((combined_output[0][3] == ohc_out_a).all())
    self.assertTrue((combined_output[1][3] == ohc_out_b).all())
    self.assertTrue((combined_output[0][4] == agc_out_a).all())
    self.assertTrue((combined_output[1][4] == agc_out_b).all())
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_map(jnp.allclose, state_out_a, combined_output[0][1])
        )
    )
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_map(jnp.allclose, state_out_b, combined_output[1][1])
        )
    )

    # The state is the most unusual one in all the outputs: ensure that
    # equality is complete and double sided.
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_map(jnp.allclose, combined_output[0][1], state_out_a)
        )
    )
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_map(jnp.allclose, combined_output[1][1], state_out_b)
        )
    )


if __name__ == '__main__':
  absltest.main()
