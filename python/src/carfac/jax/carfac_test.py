import collections
import copy
import numbers

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten
from jax.tree_util import tree_unflatten

from carfac.jax import carfac as carfac_jax
from carfac.np import carfac as carfac_np


class CarfacJaxTest(parameterized.TestCase):
  def setUp(self):
    super().setUp()
    # The default tolerance used in `assertAlmostEqual` etc.
    self.default_delta = 1e-6

  def test_post_init_syn_params(self):
    # We add this test since SynDesignParameters contains a __post_init__ that
    # we want to make sure does not interfere.
    syn = carfac_jax.SynDesignParameters()
    treedef, leaves = tree_flatten(syn)
    reconstituted_syn = tree_unflatten(leaves, treedef)
    self.assertLess(
        jnp.max(abs(syn.healthy_n_fibers - reconstituted_syn.healthy_n_fibers)),
        self.default_delta,
    )

  def test_hypers_hash(self):
    # Tests hyperparameter objects can be hashed. This is needed by `jax.jit`
    # because hyperparameters are tagged `static`.
    hypers = carfac_jax.CarfacHypers()
    hypers.ears = [
        carfac_jax.EarHypers(
            n_ch=0,
            pole_freqs=jnp.array([]),
            max_channels_per_octave=0.0,
            car=carfac_jax.CarHypers(),
            agc=[
                carfac_jax.AgcHypers(n_ch=1, n_agc_stages=2),
                carfac_jax.AgcHypers(n_ch=1, n_agc_stages=2),
            ],
            ihc=carfac_jax.IhcHypers(n_ch=1, ihc_style=1),
            syn=carfac_jax.SynHypers(),
        )
    ]
    h1 = hash(hypers)
    hypers.ears[0].car.n_ch += 1
    h2 = hash(hypers)
    self.assertNotEqual(h1, h2)
    hypers.ears[0].agc[1].reverse_cumulative_decimation += 1
    h3 = hash(hypers)
    self.assertNotEqual(h2, h3)
    hypers.ears[0].ihc.ihc_style = 2
    h4 = hash(hypers)
    self.assertNotEqual(h3, h4)

    hypers_copy1 = copy.deepcopy(hypers)
    hypers.ears[0].car.linear_car = not hypers.ears[0].car.linear_car
    self.assertNotEqual(hypers, hypers_copy1)
    self.assertEqual(hypers, hypers)

  def container_comparison(self, left_side, right_side, exclude_keys=None):
    """Compare all the non private keys of the left side to the right side.

    Note that this doesn't compare all keys on the right, just all those on
    left, excluding those that are listed in exclude keys. By default all
    non private keys are checked.

    The equality assertion is checked using knowledge of the types in sequences
    and elsewhere.

    Args:
      left_side: A carfac jax container with fields to check. All non private
        keys are checked from this side.
      right_side: Second container, only those specified in left are checked for
        equality.
      exclude_keys: Keys to exclude.
    """
    right_items = vars(right_side)
    for right_key in right_items.keys():
      logging.info('key on right is %s', right_key)

    for k, item in vars(left_side).items():
      logging.info('looking at key %s', k)
      if k.startswith('__') or (exclude_keys is not None and k in exclude_keys):
        continue
      if isinstance(item, jnp.ndarray) and item.size == 0:
        # This is a "not initialized array" and we check that it is not
        # present on right hand side.
        self.assertNotIn(k, right_items, 'failed comparison on k %s' % (k))
      elif (
          isinstance(item, jnp.ndarray) and item.size == 1 and item.item() == 0
      ):
        self.assertTrue(
            k not in right_items or right_items[k] == 0,
            'failed comparison on k %s' % (k),
        )
      elif (
          isinstance(item, collections.abc.Sequence)
          or hasattr(item, '__len__')
          and not (isinstance(item, jnp.ndarray) and item.size == 1)
      ):
        self.assertSequenceAlmostEqual(
            item,
            getattr(right_side, k),
            delta=self.default_delta,
            msg='failed comparison on key item %s' % (k),
        )
      elif isinstance(item, numbers.Integral):
        # for Integers, no approximation.
        self.assertEqual(
            item,
            getattr(right_side, k),
            msg='failed comparison on key item %s' % (k),
        )
      else:
        self.assertAlmostEqual(
            item,
            getattr(right_side, k),
            delta=self.default_delta,
            msg='failed comparison on key item %s' % (k),
        )

  @parameterized.parameters(
      ['just_hwr', 'one_cap', 'two_cap', 'two_cap_with_syn']
  )
  def test_equal_design(self, ihc_style):
    # Test: the designs are similar.
    cfp = carfac_np.design_carfac(ihc_style=ihc_style)
    carfac_np.carfac_init(cfp)
    cfp.ears[0].car_coeffs.linear = False

    params_jax = carfac_jax.CarfacDesignParameters()
    params_jax.ears[0].ihc.ihc_style = ihc_style
    params_jax.ears[0].car.linear_car = False
    hypers_jax, weights_jax, state_jax = carfac_jax.design_and_init_carfac(
        params_jax
    )

    self.assertEqual(params_jax.fs, cfp.fs)
    self.assertEqual(params_jax.n_ears, cfp.n_ears)

    for ear_idx, ear_params_np in enumerate(cfp.ears):
      # In Jax, mix coeffs is held as a union, but not in numpy.
      self.container_comparison(
          params_jax.ears[ear_idx].agc,
          cfp.agc_params,
          exclude_keys={'agc_mix_coeffs'},
      )
      self.assertAlmostEqual(
          cfp.agc_params.agc_mix_coeff,
          params_jax.ears[ear_idx].agc.agc_mix_coeffs,
          delta=self.default_delta,
      )
      # In jax world, linear car and delay buffer are in both params and
      # hypers, but we only check in one.
      self.container_comparison(
          params_jax.ears[ear_idx].car,
          cfp.car_params,
          exclude_keys={'use_delay_buffer', 'linear_car'},
      )

      self.container_comparison(
          hypers_jax.ears[ear_idx].car,
          ear_params_np.car_coeffs,
          exclude_keys={'linear_car'},
      )
      # Linear car is named differently in plain numpy,
      self.assertEqual(
          ear_params_np.car_coeffs.linear,
          hypers_jax.ears[ear_idx].car.linear_car
      )
      self.container_comparison(
          state_jax.ears[ear_idx].car, ear_params_np.car_state
      )
      self.container_comparison(
          weights_jax.ears[ear_idx].car, ear_params_np.car_coeffs
      )
      self.container_comparison(
          weights_jax.ears[ear_idx].ihc, ear_params_np.ihc_coeffs
      )
      self.container_comparison(
          hypers_jax.ears[ear_idx].ihc,
          ear_params_np.ihc_coeffs,
          exclude_keys={'ihc_style'},
      )
      self.assertEqual(
          ear_params_np.ihc_coeffs.ihc_style,
          hypers_jax.ears[ear_idx].ihc.ihc_style,
      )

      self.container_comparison(
          state_jax.ears[ear_idx].ihc,
          ear_params_np.ihc_state,
          exclude_keys='lpf2_state',
      )

      if ear_params_np.ihc_coeffs.ihc_style == 1:
        self.assertSequenceAlmostEqual(
            state_jax.ears[ear_idx].ihc.lpf2_state,
            ear_params_np.ihc_state.lpf2_state,
            delta=1e-5,  # Low precision.
        )

      self.assertEqual(cfp.fs, params_jax.fs)

      # Ihc params very specific for one cap, only a subset is used, so for
      # now we only check these one by one. We could add tests for 2 cap
      # similarly.
      if ihc_style == 'two_cap_with_syn':
        # on the numpy side, we keep the IHC as two cap name.
        self.assertEqual('two_cap', cfp.ihc_params.ihc_style)
        self.assertEqual(
            'two_cap_with_syn', params_jax.ears[ear_idx].ihc.ihc_style
        )
      else:
        self.assertEqual(
            cfp.ihc_params.ihc_style, params_jax.ears[ear_idx].ihc.ihc_style
        )

      if ihc_style != 'just_hwr':
        self.assertEqual(
            cfp.ihc_params.tau_in, params_jax.ears[ear_idx].ihc.tau_in
        )
        self.assertEqual(
            cfp.ihc_params.tau_lpf, params_jax.ears[ear_idx].ihc.tau_lpf
        )
        self.assertEqual(
            cfp.ihc_params.tau_out, params_jax.ears[ear_idx].ihc.tau_out
        )
      self.assertAlmostEqual(
          cfp.max_channels_per_octave,
          hypers_jax.ears[ear_idx].max_channels_per_octave,
          delta=1e-5,  # Low precision.
      )
      self.assertEqual(cfp.n_ch, hypers_jax.ears[ear_idx].n_ch)
      self.assertEqual(cfp.n_ears, params_jax.n_ears)
      self.assertSequenceAlmostEqual(
          cfp.pole_freqs,
          hypers_jax.ears[ear_idx].pole_freqs,
          delta=self.default_delta,
      )
      self.container_comparison(
          weights_jax.ears[ear_idx].car, ear_params_np.car_coeffs
      )

      if ihc_style == 'two_cap_with_syn':
        self.container_comparison(
            weights_jax.ears[ear_idx].syn,
            ear_params_np.syn_coeffs,
            exclude_keys=['n_fibers', 'v_widths', 'v_halfs'],
        )
        self.assertLess(
            jnp.max(
                jnp.abs(
                    weights_jax.ears[ear_idx].syn.n_fibers
                    - ear_params_np.syn_coeffs.n_fibers
                )
            ),
            self.default_delta,
        )
        self.assertLess(
            jnp.max(
                jnp.abs(
                    weights_jax.ears[ear_idx].syn.v_widths
                    - ear_params_np.syn_coeffs.v_widths
                )
            ),
            self.default_delta,
        )
        self.assertLess(
            jnp.max(
                jnp.abs(
                    weights_jax.ears[ear_idx].syn.v_halfs
                    - ear_params_np.syn_coeffs.v_halfs
                ),
            ),
            self.default_delta,
        )

      # Test AGC parameters for each stage.
      for stage_idx, ear_agc_coeffs_np in enumerate(ear_params_np.agc_coeffs):
        logging.info('running stage idx %d', stage_idx)
        self.container_comparison(
            weights_jax.ears[ear_idx].agc[stage_idx], ear_agc_coeffs_np
        )

        # reverse_cumulative_decimation and _max_cumulative_decimation are
        # exclusive to the JAX implementation.
        self.container_comparison(
            hypers_jax.ears[ear_idx].agc[stage_idx],
            ear_agc_coeffs_np,
            exclude_keys={
                'reverse_cumulative_decimation',
                'max_cumulative_decimation',
            },
        )
        self.container_comparison(
            state_jax.ears[ear_idx].agc[stage_idx],
            ear_params_np.agc_state[stage_idx],
        )

  @parameterized.product(
      random_seed=[x for x in range(5)],
      ihc_style=['one_cap', 'two_cap', 'two_cap_with_syn'],
  )
  def test_chunked_naps_same_as_jit(self, random_seed, ihc_style):
    """Tests whether `run_segment` produces the same results as np version."""
    # Inits JAX version
    params_jax = carfac_jax.CarfacDesignParameters()
    params_jax.ears[0].ihc.ihc_style = ihc_style
    params_jax.ears[0].car.linear_car = False
    hypers_jax, weights_jax, state_jax = carfac_jax.design_and_init_carfac(
        params_jax
    )

    # Generate some random inputs.
    # It shouldn't be too long to avoid unit tests running too long.
    # It shouldn't be too short to ensure all AGC layers are run. That is, it
    # should be bigger than 64 (i.e. `prod(AgcDesignParameters.decimation)`).
    n_samp = 200
    n_ears = 1
    random_generator = jax.random.PRNGKey(random_seed)
    run_seg_input = jax.random.normal(random_generator, (n_samp, n_ears))

    # Copy the state first.
    state_jax_copied = copy.deepcopy(state_jax)

    # Only tests the JITted version because this is what we will use.
    naps_jax, _, _, bm_jax, ohc_jax, agc_jax = (
        carfac_jax.run_segment_jit(
            run_seg_input, hypers_jax, weights_jax, state_jax, open_loop=False
        )
    )
    (
        naps_jax_chunked,
        _,
        _,
        bm_chunked,
        ohc_chunked,
        agc_chunked,
    ) = carfac_jax.run_segment_jit_in_chunks_notraceable(
        run_seg_input,
        hypers_jax,
        weights_jax,
        state_jax_copied,
        open_loop=False,
    )
    self.assertLess(jnp.max(abs(naps_jax_chunked - naps_jax)), 1e-7)
    self.assertLess(jnp.max(abs(bm_chunked - bm_jax)), 1e-7)
    self.assertLess(jnp.max(abs(ohc_chunked - ohc_jax)), 1e-7)
    self.assertLess(jnp.max(abs(agc_chunked - agc_jax)), 1e-7)

  @parameterized.product(
      random_seed=[x for x in range(20)],
      ihc_style=['just_hwr', 'one_cap', 'two_cap', 'two_cap_with_syn'],
      n_ears=[1, 2],
      delay_buffer=[False, True],
  )
  def test_equal_forward_pass(
      self, random_seed, ihc_style, n_ears, delay_buffer
  ):
    """Tests whether `run_segment` produces the same results as np version."""
    # Inits JAX version
    params_jax = carfac_jax.CarfacDesignParameters(
        n_ears=n_ears, use_delay_buffer=delay_buffer
    )
    params_jax.n_ears = n_ears
    for ear in range(n_ears):
      params_jax.ears[ear].ihc.ihc_style = ihc_style
      params_jax.ears[ear].car.linear_car = False
    hypers_jax, weights_jax, state_jax = carfac_jax.design_and_init_carfac(
        params_jax
    )
    # Inits numpy version
    cfp = carfac_np.design_carfac(
        ihc_style=ihc_style, n_ears=n_ears, use_delay_buffer=delay_buffer
    )

    carfac_np.carfac_init(cfp)
    for i in range(n_ears):
      cfp.ears[i].car_coeffs.linear = False

    # Generate some random inputs.
    # It shouldn't be too long to avoid unit tests running too long.
    # It shouldn't be too short to ensure all AGC layers are run. That is, it
    # should be bigger than 64 (i.e. `prod(AgcDesignParameters.decimation)`).
    n_samp = 200
    random_generator = jax.random.PRNGKey(random_seed)
    run_seg_input = jax.random.normal(random_generator, (n_samp, n_ears))

    # Only tests the JITted version because this is what we will use.
    naps_jax, _, state_jax, bm_jax, seg_ohc_jax, seg_agc_jax = (
        carfac_jax.run_segment_jit(
            run_seg_input, hypers_jax, weights_jax, state_jax, open_loop=False
        )
    )

    naps_np, state_np, bm_np, seg_ohc_np, seg_agc_np = carfac_np.run_segment(
        cfp, run_seg_input, open_loop=False, linear_car=False
    )

    # Tests the generated "naps" are similar.
    self.assertLess(
        jnp.max(abs(naps_jax.block_until_ready() - naps_np)),
        1e-3,  # Low Precision.
    )
    # Tests the generated "bms" are similar.
    self.assertLess(
        jnp.max(abs(bm_jax.block_until_ready() - bm_np)), 1e-3  # Low Precision.
    )
    # Tests the generated "seg_ohcs" are similar.
    self.assertLess(
        jnp.max(abs(seg_ohc_jax.block_until_ready() - seg_ohc_np)),
        2e-3,  # Low Precision
    )
    # Tests the generated "seg_agcs" are similar.
    self.assertLess(
        jnp.max(abs(seg_agc_jax.block_until_ready() - seg_agc_np)),
        self.default_delta,
    )

    # Because different arrays have different `delta`s, we can not easily use
    # `self.container_comparison()`.
    for ear in range(params_jax.n_ears):
      # Compares CAR state.
      self.assertSequenceAlmostEqual(
          state_jax.ears[ear].car.z1_memory,
          state_np.ears[ear].car_state.z1_memory,
          delta=2e-3,  # Low Precision
      )
      self.assertSequenceAlmostEqual(
          state_jax.ears[ear].car.z2_memory,
          state_np.ears[ear].car_state.z2_memory,
          delta=2e-3,  # Low Precision
      )
      self.assertSequenceAlmostEqual(
          state_jax.ears[ear].car.za_memory,
          state_np.ears[ear].car_state.za_memory,
          delta=2e-3,  # Low Precision
      )
      self.assertSequenceAlmostEqual(
          state_jax.ears[ear].car.zb_memory,
          state_np.ears[ear].car_state.zb_memory,
          delta=self.default_delta,
      )
      self.assertSequenceAlmostEqual(
          state_jax.ears[ear].car.dzb_memory,
          state_np.ears[ear].car_state.dzb_memory,
          delta=self.default_delta,
      )
      self.assertSequenceAlmostEqual(
          state_jax.ears[ear].car.zy_memory,
          state_np.ears[ear].car_state.zy_memory,
          delta=1e-4,  # Low Precision
      )
      self.assertSequenceAlmostEqual(
          state_jax.ears[ear].car.g_memory,
          state_np.ears[ear].car_state.g_memory,
          delta=self.default_delta,
      )
      self.assertSequenceAlmostEqual(
          state_jax.ears[ear].car.dg_memory,
          state_np.ears[ear].car_state.dg_memory,
          delta=self.default_delta,
      )
      self.assertSequenceAlmostEqual(
          state_jax.ears[ear].car.ac_coupler,
          state_np.ears[ear].car_state.ac_coupler,
          delta=1e-5,  # Low Precision
      )
      # Compares IHC state.
      self.assertSequenceAlmostEqual(
          state_jax.ears[ear].ihc.ihc_accum,
          state_np.ears[ear].ihc_state.ihc_accum,
          delta=8e-3,  # Low Precision
      )
      if cfp.ears[ear].ihc_coeffs.ihc_style != 0:
        self.assertSequenceAlmostEqual(
            state_jax.ears[ear].ihc.lpf1_state,
            state_np.ears[ear].ihc_state.lpf1_state,
            delta=1e-3,  # Low Precision
        )
      if cfp.ears[ear].ihc_coeffs.ihc_style == 1:
        self.assertSequenceAlmostEqual(
            state_jax.ears[ear].ihc.lpf2_state,
            state_np.ears[ear].ihc_state.lpf2_state,
            delta=5e-4,  # Low Precision
        )
        self.assertSequenceAlmostEqual(
            state_jax.ears[ear].ihc.cap_voltage,
            state_np.ears[ear].ihc_state.cap_voltage,
            delta=2e-5,  # Low Precision
        )
      elif cfp.ears[ear].ihc_coeffs.ihc_style == 2:
        # `state_np` won't have `cap1_voltage` or `cap2_voltage` if
        # `one_cap==True`.
        self.assertSequenceAlmostEqual(
            state_jax.ears[ear].ihc.cap1_voltage,
            state_np.ears[ear].ihc_state.cap1_voltage,
            delta=1e-5,  # Low Precision
        )
        self.assertSequenceAlmostEqual(
            state_jax.ears[ear].ihc.cap2_voltage,
            state_np.ears[ear].ihc_state.cap2_voltage,
            delta=1e-5,  # Low Precision
        )
      elif cfp.ears[ear].ihc_coeffs.ihc_style != 0:
        self.fail('Unsupported IHC style.')
      # Comapares agc state
      for stage in range(hypers_jax.ears[ear].agc[0].n_agc_stages):
        self.assertSequenceAlmostEqual(
            state_jax.ears[ear].agc[stage].agc_memory,
            state_np.ears[ear].agc_state[stage].agc_memory,
            delta=1e-5,  # Low Precision
        )
        self.assertSequenceAlmostEqual(
            state_jax.ears[ear].agc[stage].input_accum,
            state_np.ears[ear].agc_state[stage].input_accum,
            delta=3e-5,  # Low precision
        )
        self.assertEqual(
            state_jax.ears[ear].agc[stage].decim_phase,
            state_np.ears[ear].agc_state[stage].decim_phase,
        )


if __name__ == '__main__':
  absltest.main()
