"""Tests that needs enabling float64 in JAX.

For some tests, we'd better enable `float64` in JAX to obtain a meaningful
bound. These tests are meant to test the "correctness" of the code rather than
the numeric stability of the code. We will have specific tests on numeric
stability separately.
"""
import copy
import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.flatten_util
import jax.numpy as jnp

import sys
sys.path.insert(0, '.')
import carfac as carfac_jax
import utils


class CarfacJaxFloat64Test(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # The default tolerance used in `assertAlmostEqual` etc.
    self.default_delta = 1e-6

  def _assert_almost_equal_pytrees(self, pytree1, pytree2, delta=None):
    # Tests whether 2 pytrees are almost equal.
    delta = delta or self.default_delta
    elements1, _ = jax.flatten_util.ravel_pytree(pytree1)
    elements2, _ = jax.flatten_util.ravel_pytree(pytree2)
    self.assertSequenceAlmostEqual(elements1, elements2, delta=delta)

  @parameterized.product(
      random_seed=[x for x in range(20)],
      ihc_style=['one_cap', 'two_cap', 'two_cap_with_syn'],
      n_ears=[1, 2],
  )
  def test_backward_pass(self, random_seed, ihc_style, n_ears):
    # Tests `jax.grad` can give similar gradients computed by numeric method.
    @functools.partial(jax.jit, static_argnames=('hypers',))
    def loss(weights, input_waves, hypers, state):
      # A loss function for tests. Note that we shouldn't use `run_segment_jit`
      # here because it will donate the `state` which causes unnecessary
      # inconvenience for tests.
      naps_jax, _, state_jax, _, _, _ = carfac_jax.run_segment(
          input_waves, hypers, weights, state, open_loop=False
      )
      # For testing, just fit `naps` to 1.
      return (
          jnp.sum((naps_jax - 1.0) ** 2)
          / jnp.prod(jnp.asarray(naps_jax.shape)),
          state_jax,
      )

    # Generate some random inputs.
    # It shouldn't be too long to avoid unit tests running too long.
    # It shouldn't be too short to ensure all AGC layers are run. That is, it
    # should be bigger than 64 (i.e. `prod(AgcDesignParameters.decimation)`).
    n_samp = 200
    run_seg_input = jax.random.normal(
        jax.random.PRNGKey(random_seed), (n_samp, n_ears)
    )

    # Computes gradients by `jax.grad`.
    gfunc = jax.grad(loss, has_aux=True)
    params_jax = carfac_jax.CarfacDesignParameters(n_ears=n_ears)
    for ear in range(n_ears):
      params_jax.ears[ear].ihc.ihc_style = ihc_style
      params_jax.ears[ear].car.linear_car = False
    hypers_jax, weights_jax, state_jax = carfac_jax.design_and_init_carfac(
        params_jax
    )
    grad_jax, _ = gfunc(weights_jax, run_seg_input, hypers_jax, state_jax)

    # Computes gradients by numeric methods.
    perturbs = [-1e-6, 1e-6]
    grad_numeric = copy.deepcopy(weights_jax)
    contents_numeric, def_numeric = jax.tree_util.tree_flatten(grad_numeric)
    for (pytree_neg, content_idx_neg, array_idx_neg), (
        pytree_pos,
        _,
        _,
    ) in utils.iter_perturbed(weights_jax, perturbs=perturbs):
      f_neg, _ = loss(pytree_neg, run_seg_input, hypers_jax, state_jax)
      f_pos, _ = loss(pytree_pos, run_seg_input, hypers_jax, state_jax)
      g = (f_pos - f_neg) / (perturbs[1] - perturbs[0])
      if array_idx_neg:
        # The leaf is a `jax.Array`
        contents_numeric[content_idx_neg] = (
            contents_numeric[content_idx_neg].at[array_idx_neg].set(g)
        )
      else:
        # The leaf is a `float`.
        contents_numeric[content_idx_neg] = g

    grad_numeric = jax.tree_util.tree_unflatten(def_numeric, contents_numeric)

    # Ensures a loose bound on all the differentiable weights.
    self._assert_almost_equal_pytrees(
        grad_numeric, grad_jax, delta=1e-3  # Low Precision.
    )

    # Ensures a tight bound on the weights we mostly care.
    # Currently, we only care about `ohc_health`.
    for ear_weights_numeric, ear_weights_jax in zip(
        grad_numeric.ears, grad_jax.ears
    ):
      self.assertSequenceAlmostEqual(
          ear_weights_numeric.car.ohc_health,
          ear_weights_jax.car.ohc_health,
          delta=self.default_delta,
      )


if __name__ == '__main__':
  # Needs to enable float64 at startup. Reference:
  # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
  jax.config.update('jax_enable_x64', True)
  absltest.main()
