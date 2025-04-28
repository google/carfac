"""Benchmarking suite for CARFAC on JAX with a comparison with Numpy.

Implements a benchmarking suite for the CARFAC JAX suite, along with a
comparison with the plain CARFAC NumPy version.

Initial benchmarks are to time requirements to run a segment of moderated volume
noise audio through CARFAC, which should be indicative of performance of real
world performance.
"""

import google_benchmark
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
import carfac as carfac_jax
import carfac_util
import np.carfac as carfac_np
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2048"

# Noise factor to mulitply for random noise.
_NOISE_FACTOR = 1e-4


def tree_unstack(tree):
  leaves, treedef = jtu.tree_flatten(tree)
  return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


@google_benchmark.register
@google_benchmark.option.measure_process_cpu_time()
@google_benchmark.option.use_real_time()
@google_benchmark.option.unit(google_benchmark.kMicrosecond)
@google_benchmark.option.arg_names(['segment_sample_length', 'split_count'])
@google_benchmark.option.args([22050, 100])
@google_benchmark.option.args([22050, 10])
def bench_numpy_in_slices(state: google_benchmark.State):
  """Benchmark numpy by slicing a long segment into many very small ones.

    The two arguments are the number of samples and the number of slices to
    split the segment into when executing. For each segment, we run
    slice-by-slice, and collect the outputs.

    This benchmark is intended to simulate a "streaming" experience.

  Args:
    state: the benchmark state for this execution run.
  """
  random_seed = 1
  ihc_style = 'two_cap'
  cfp = carfac_np.design_carfac(ihc_style=ihc_style)

  carfac_np.carfac_init(cfp)
  cfp.ears[0].car_coeffs.linear = False
  n_samp = state.range(0)
  n_ears = 1
  np_random = np.random.default_rng(random_seed)
  split_count = state.range(1)

  while state:
    state.pause_timing()
    naps = []
    bm = []
    ohc = []
    agc = []
    run_seg_input_full = (
        np_random.standard_normal(size=(n_samp, n_ears)) * _NOISE_FACTOR
    )
    run_seg_slices = np.array_split(run_seg_input_full, split_count)
    cfp = carfac_np.design_carfac(ihc_style=ihc_style)
    carfac_np.carfac_init(cfp)
    cfp.ears[0].car_coeffs.linear = False
    state.resume_timing()
    for _, segment in enumerate(run_seg_slices):
      seg_naps, cfp, seg_bm, seg_ohc, seg_agc = carfac_np.run_segment(
          cfp, segment, open_loop=False, linear_car=False
      )
      # We include an append in the loop to ensure a fair comparison, and
      # include the cost of append operations in timing, even though the
      # variables are not used again.
      naps.append(seg_naps)
      bm.append(seg_bm)
      ohc.append(seg_ohc)
      agc.append(seg_agc)


@google_benchmark.register
@google_benchmark.option.measure_process_cpu_time()
@google_benchmark.option.use_real_time()
@google_benchmark.option.unit(google_benchmark.kMicrosecond)
@google_benchmark.option.arg_names([
    'segment_sample_length',
    'ihc_style',
])
@google_benchmark.option.args_product(
    [
        [220, 2205, 22050, 44100, 220500],
        [0, 1],
    ],
)
def bench_numpy(state: google_benchmark.State):
  """Benchmark the numpy version of carfac.

    The argument is the number of samples to generate for the segment to run,
    with 22050 samples per second.

  Args:
    state: the benchmark state for this execution run.
  """
  random_seed = 1
  if state.range(1) == 0:
    ihc_style = 'two_cap'
  elif state.range(1) == 1:
    ihc_style = 'two_cap_with_syn'
  else:
    raise ValueError('Invalid ihc_style')
  cfp = carfac_np.design_carfac(ihc_style=ihc_style)

  carfac_np.carfac_init(cfp)
  cfp.ears[0].car_coeffs.linear = False
  n_samp = state.range(0)
  n_ears = 1
  np_random = np.random.default_rng(random_seed)

  while state:
    state.pause_timing()
    run_seg_input = (
        np_random.standard_normal(size=(n_samp, n_ears)) * _NOISE_FACTOR
    )
    cfp = carfac_np.design_carfac(ihc_style=ihc_style)
    carfac_np.carfac_init(cfp)
    cfp.ears[0].car_coeffs.linear = False
    state.resume_timing()
    carfac_np.run_segment(cfp, run_seg_input, open_loop=False, linear_car=False)


@google_benchmark.register
@google_benchmark.option.measure_process_cpu_time()
@google_benchmark.option.use_real_time()
@google_benchmark.option.unit(google_benchmark.kMicrosecond)
@google_benchmark.option.arg_names(['segment_sample_length', 'ihc_style'])
@google_benchmark.option.args_product(
    [
        [128, 256, 512, 1024, 2048, 4096],
        [0, 1],
    ],
)
def bench_jax_grad(state: google_benchmark.State):
  """Benchmark JAX Value and Gradient function on Carfac.

  Args:
    state: The Benchmark state for this run.
  """
  if state.range(1) == 0:
    ihc_style = 'two_cap'
  elif state.range(1) == 1:
    ihc_style = 'two_cap_with_syn'
  else:
    raise ValueError('Invalid ihc_style')
  random_seed = 1
  params_jax = carfac_jax.CarfacDesignParameters()
  params_jax.ears[0].ihc.ihc_style = ihc_style
  params_jax.ears[0].car.linear_car = False
  random_generator = jax.random.PRNGKey(random_seed)
  n_samp = state.range(0)
  n_ears = 1
  hypers_jax, weights_jax, state_jax = carfac_jax.design_and_init_carfac(
      params_jax
  )

  def loss_func(
      audio: jnp.ndarray,
      hypers: carfac_jax.CarfacHypers,
      weights: carfac_jax.CarfacWeights,
      state: carfac_jax.CarfacState,
  ):
    nap_output, _, _, _, _, _, _ = carfac_jax.run_segment(
        audio, hypers, weights, state
    )
    return jnp.sum(nap_output), nap_output

  jitted_loss = jax.jit(
      jax.value_and_grad(loss_func, argnums=2, has_aux=True),
      static_argnames=['hypers'],
  )
  short_silence = jnp.zeros(shape=(n_samp, n_ears))
  (_, gradient), _ = jitted_loss(
      short_silence, hypers_jax, weights_jax, state_jax
  )
  gradient.block_until_ready()
  while state:
    state.pause_timing()
    _, random_generator = jax.random.split(random_generator)
    run_seg_input = (
        jax.random.normal(random_generator, (n_samp, n_ears)) * _NOISE_FACTOR
    ).block_until_ready()
    state.resume_timing()
    (_, gradient), _ = jitted_loss(
        run_seg_input, hypers_jax, weights_jax, state_jax
    )
    gradient.block_until_ready()


@google_benchmark.register
@google_benchmark.option.measure_process_cpu_time()
@google_benchmark.option.use_real_time()
@google_benchmark.option.unit(google_benchmark.kMicrosecond)
def bench_jit_compile_time(state: google_benchmark.State):
  """Benchmark the amount of time to compile a JIT execution of carfac.

  Do so by iterating over compiling extremely short samples of audio and running
  CARFAC over them. Assumes that the amount of time to run through a segment of
  very short samples is neglible in comparison.

  Args:
    state: The benchmark state to execute over.
  """
  ihc_style = 'two_cap'
  random_seed = 1
  params_jax = carfac_jax.CarfacDesignParameters()
  params_jax.ears[0].ihc.ihc_style = ihc_style
  params_jax.ears[0].car.linear_car = False
  random_generator = jax.random.PRNGKey(random_seed)
  n_samp = 1
  n_ears = 1
  hypers_jax, weights_jax, state_jax = carfac_jax.design_and_init_carfac(
      params_jax
  )

  while state:
    state.pause_timing()
    # reset and make a new chunk of audio for next round.
    run_seg_input = (
        jax.random.normal(random_generator, (n_samp, n_ears)) * _NOISE_FACTOR
    ).block_until_ready()
    # We always run with different sample count to ensure a recompile so
    # that this benchmark is appropriate.
    n_samp += 1
    state.resume_timing()
    naps_jax, _, state_jax, _, _, _, _ = carfac_jax.run_segment_jit(
        run_seg_input, hypers_jax, weights_jax, state_jax, open_loop=False
    )
    naps_jax.block_until_ready()


@google_benchmark.register
@google_benchmark.option.measure_process_cpu_time()
@google_benchmark.option.use_real_time()
@google_benchmark.option.unit(google_benchmark.kMicrosecond)
@google_benchmark.option.arg_names(['segment_sample_length', 'split_count'])
@google_benchmark.option.args([22050, 100])
@google_benchmark.option.args([22050, 10])
def bench_jax_in_slices(state: google_benchmark.State):
  """Benchmark JAX by slicing a long segment into many very small ones.

    The two arguments are the number of samples and the number of slices to
    split the segment into when executing. For each segment, we run
    slice-by-slice.

    This benchmark is intended to simulate a "streaming" experience, and only
    tests the JIT compiled JAX implementation of CARFAC.

  Args:
    state: the benchmark state for this execution run.
  """
  # Inits JAX version
  ihc_style = 'two_cap'
  random_seed = 1
  params_jax = carfac_jax.CarfacDesignParameters()
  params_jax.ears[0].ihc.ihc_style = ihc_style
  params_jax.ears[0].car.linear_car = False

  # Generate some random inputs.
  n_samp = state.range(0)
  split_count = state.range(1)
  n_ears = 1
  random_generator = jax.random.PRNGKey(random_seed)

  # Do a compile
  hypers_jax, weights_jax, state_jax = carfac_jax.design_and_init_carfac(
      params_jax
  )
  silence_slices = jnp.array_split(
      jnp.zeros(shape=(n_samp, n_ears)), split_count
  )
  compiled_shapes = set()
  # array_split can split an array into a number of different shapes. To avoid
  # timing compiles, we compile all the shapes we see for this full segment
  # slice combination.
  for _, segment in enumerate(silence_slices):
    if segment.shape not in compiled_shapes:
      compiled_shapes.add(segment.shape)
      naps_jax, _, _, _, _, _, _ = carfac_jax.run_segment_jit(
          segment, hypers_jax, weights_jax, state_jax, open_loop=False
      )
      naps_jax.block_until_ready()

  while state:
    state.pause_timing()
    naps = []
    bm = []
    ohc = []
    agc = []
    run_seg_input_full = (
        jax.random.normal(random_generator, (n_samp, n_ears)) * _NOISE_FACTOR
    )
    run_seg_slices = jnp.array_split(run_seg_input_full, split_count)
    for _, run_seg_slice in enumerate(run_seg_slices):
      run_seg_slice.block_until_ready()

    jax_loop_state = state_jax
    state.resume_timing()
    for _, segment in enumerate(run_seg_slices):
      seg_naps, seg_naps_fibers, jax_loop_state, seg_bm, seg_receptor_pot, seg_ohc, seg_agc = (
          carfac_jax.run_segment_jit(
              segment, hypers_jax, weights_jax, jax_loop_state, open_loop=False
          )
      )
      naps.append(seg_naps)
      bm.append(seg_bm)
      ohc.append(seg_ohc)
      agc.append(seg_agc)


@google_benchmark.register
@google_benchmark.option.measure_process_cpu_time()
@google_benchmark.option.use_real_time()
@google_benchmark.option.unit(google_benchmark.kMicrosecond)
@google_benchmark.option.arg_names([
    'jax_chunked_uncompiled',
    'segment_sample_length',
    'use_delay_buffer',
    'ihc_style',
])
@google_benchmark.option.args_product(
    [
        [0, 1, 2],
        [220, 2205, 22050, 44100, 220500, 2205000],
        [False, True],
        [0, 1],
    ],
)
def bench_jax(state: google_benchmark.State):
  """Benchmark the JAX version of carfac.

    There are two sets of arguments. The first is either 0 or 1, which is for a
    JIT or non-JIT version, respectively.

    The second is for the number of samples in the segment, where the
    expectation is for the default 22050 samples per second.

    For the JIT version, we run a segment of identical length audio to ensure
    the JIT compiler has a cached version of the JIT code.

  Args:
    state: the benchmark state for this execution run.
  """
  # Inits JAX version
  if state.range(3) == 0:
    ihc_style = 'two_cap'
  elif state.range(3) == 1:
    ihc_style = 'two_cap_with_syn'
  else:
    raise ValueError('Invalid ihc_style')
  random_seed = 1
  params_jax = carfac_jax.CarfacDesignParameters()
  params_jax.ears[0].car.use_delay_buffer = state.range(2)
  params_jax.ears[0].ihc.ihc_style = ihc_style
  params_jax.ears[0].car.linear_car = False

  # Generate some random inputs.
  n_samp = state.range(1)
  n_ears = 1
  random_generator = jax.random.PRNGKey(random_seed)
  run_segment_function = None
  if state.range(0) == 1:
    run_segment_function = carfac_jax.run_segment_jit_in_chunks_notraceable
  elif state.range(0) == 0:
    run_segment_function = carfac_jax.run_segment_jit
  else:
    run_segment_function = carfac_jax.run_segment

  hypers_jax, weights_jax, state_jax = carfac_jax.design_and_init_carfac(
      params_jax
  )
  short_silence = jnp.zeros(shape=(n_samp, n_ears))
  naps_jax, _, state_jax, _, _, _, _ = run_segment_function(
      short_silence, hypers_jax, weights_jax, state_jax, open_loop=False
  )
  # This block ensures calculation.
  # When running the merged version, the returned values are plain numpy.
  if state.range(0) != 1:
    naps_jax.block_until_ready()
  while state:
    state.pause_timing()
    # reset and make a new chunk of audio for next round.
    _, random_generator = jax.random.split(random_generator)
    run_seg_input = (
        jax.random.normal(random_generator, (n_samp, n_ears)) * _NOISE_FACTOR
    ).block_until_ready()
    state.resume_timing()
    naps_jax, _, state_jax, _, _, _, _ = run_segment_function(
        run_seg_input, hypers_jax, weights_jax, state_jax, open_loop=False
    )
    if state.range(0) != 1:
      # When running the merged version, the returned values are plain numpy.
      naps_jax.block_until_ready()


@google_benchmark.register
@google_benchmark.option.arg_names(['num_audio'])
@google_benchmark.option.range_multiplier(2)
@google_benchmark.option.range(1, 1 << 9)
def bench_jax_util_mapped(state: google_benchmark.State):
  """benchmark shard_map carfac-jax.

  Args:
    state: The benchmark state for this execution.
  """
  if jax.device_count() < state.range(0):
    state.skip_with_error(f'requires {state.range(0)} devices')
  random_seed = state.range(0)
  ihc_style = 'two_cap'
  params_jax = carfac_jax.CarfacDesignParameters()
  params_jax.ears[0].ihc.ihc_style = ihc_style
  params_jax.ears[0].car.linear_car = False
  random_generator = jax.random.PRNGKey(random_seed)
  hypers_jax, weights_jax, state_jax = carfac_jax.design_and_init_carfac(
      params_jax
  )
  n_ears = 1
  n_samp = 22050 * 2  # 2 seconds of audio.
  # Precompile with silence.
  short_silence = jnp.zeros(shape=(state.range(0), n_samp, n_ears))
  res = carfac_util.run_multiple_segment_pmap(
      short_silence,
      hypers_jax,
      weights_jax,
      state_jax,
      open_loop=False,
  )
  res[0][0].block_until_ready()

  while state:
    state.pause_timing()
    _, random_generator = jax.random.split(random_generator)
    audios = (
        jax.random.normal(random_generator, (state.range(0), n_samp, n_ears))
        * _NOISE_FACTOR
    )
    audios.block_until_ready()
    state.resume_timing()
    results = carfac_util.run_multiple_segment_pmap(
        audios,
        hypers_jax,
        weights_jax,
        state_jax,
        open_loop=False,
    )
    results[0][0].block_until_ready()


if __name__ == '__main__':
  google_benchmark.main()
