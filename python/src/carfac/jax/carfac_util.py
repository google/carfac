"""Utility functions based on CARFAC.

This adds a utility library of functions that make use of the CARFAC-JAX
library.
"""

from collections.abc import Sequence
import functools
from typing import Tuple

import jax
from jax.experimental import mesh_utils as jmesh_utils
import jax.experimental.shard_map as jshard_map
import jax.numpy as jnp
import jax.sharding as jsharding
import jax.tree_util as jtu

from carfac.jax import carfac as carfac_jax

@jax.tree_util.register_pytree_node_class
class ScanGradHypers:
  exactlen: bool = (
      False  # whether chunk_num*chunk_length should equal stimulus length exactly
  )
  chunk_num: int = 100  # number of chunks in stacked stimulus

  def tree_flatten(self):
    children = (self.exactlen, self.chunk_num)
    aux_data = ("exactlen", "chunk_num")
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, _, children):
    return cls(*children)

  def __hash__(self):
    return hash(
      (
        self.exactlen,
        self.chunk_num,
      )
    )

  def __eq__(self, other):
    return carfac_jax._are_two_equal_hypers(self, other)


def _tree_unstack(tree):
  leaves, treedef = jtu.tree_flatten(tree)
  return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def _tree_stack(trees):
  return jtu.tree_map(lambda *v: jnp.stack(v), *trees)


def run_multiple_segment_states_shmap(
    input_waves_array: jnp.ndarray,
    hypers: carfac_jax.CarfacHypers,
    weights: carfac_jax.CarfacWeights,
    states: Sequence[carfac_jax.CarfacState],
    open_loop: bool = False,
) -> Sequence[
    Tuple[
        jnp.ndarray,
        jnp.ndarray,
        carfac_jax.CarfacState,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
]:
  """Run multiple equal-length, segments in carfac, Jitted, in parallel.

  Args:
    input_waves_array: Stacked ndarray of equal length audio. First dimension is
      number of inputs (N).
    hypers: Shared hypers for use in input.
    weights: Shared weights to use in input.
    states:  Sequence of CarfacState to use, one each per audio.
    open_loop: Whether to process as open_loop.

  Returns:
    A sequence of tuples of results, of the type that carfac.run_segment.
  """
  n_devices = input_waves_array.shape[0]
  devices = jmesh_utils.create_device_mesh(mesh_shape=(n_devices,))
  mesh = jsharding.Mesh(devices, axis_names=('i',))
  in_specs = jsharding.PartitionSpec(
      'i',
  )
  out_specs = jsharding.PartitionSpec(
      'i',
  )
  batch_state = _tree_stack(states)

  @functools.partial(
      jshard_map.shard_map,
      mesh=mesh,
      in_specs=in_specs,
      out_specs=out_specs,
      check_rep=False,
  )
  def parallel_helper(input_waves, state):
    """Internal helper that executes per sharded piece of data.

    Args:
      input_waves: Single unstacked input_waves array. The first index is [1]
        sized.
      state: The corresponding starting state for this wave.

    Returns:
    """
    input_waves = input_waves[0]
    state = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), state)
    naps, naps_fibers, ret_state, bm, seg_ohc, seg_agc = (
        carfac_jax.run_segment_jit(
            input_waves, hypers, weights, state, open_loop
        )
    )
    ret_state = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x).reshape((1, -1)), ret_state
    )
    return (
        naps[None],
        naps_fibers[None],
        ret_state,
        bm[None],
        seg_ohc[None],
        seg_agc[None],
    )

  (
      stacked_naps,
      stacked_naps_fibers,
      stacked_states,
      stacked_bm,
      stacked_ohc,
      stacked_agc,
  ) = parallel_helper(input_waves_array, batch_state)
  output_states = _tree_unstack(stacked_states)
  output = []
  # TODO(robsc): Modify this for loop to a jax.lax loop, and then JIT the
  # whole function rather than internal use of run_segment_jit.
  for i, output_state in enumerate(output_states):
    tup = (
        stacked_naps[i],
        stacked_naps_fibers[i],
        output_state,
        stacked_bm[i],
        stacked_ohc[i],
        stacked_agc[i],
    )
    output.append(tup)
  return output


# TODO(robsc): Consider deleting this function if it is not useful.
def run_multiple_segment_pmap(
    input_waves_array: jnp.ndarray,
    hypers: carfac_jax.CarfacHypers,
    weights: carfac_jax.CarfacWeights,
    state: carfac_jax.CarfacState,
    open_loop: bool = False,
) -> Sequence[
    Tuple[
        jnp.ndarray,
        jnp.ndarray,
        carfac_jax.CarfacState,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]
]:
  """Run multiple equal-length, segments in carfac, Jitted, in parallel.

  Args:
    input_waves_array: Stacked ndarray of equal length audio. First dimension is
      number of inputs (N).
    hypers: Shared hypers for use in input.
    weights: Shared weights to use in input.
    state: Shared state to use as input.
    open_loop: Whether to process as open_loop.

  Returns:
    A sequence of tuples of results, of the type that carfac.run_segment.
  """
  pmapped = jax.pmap(
      fun=carfac_jax.run_segment_jit,
      in_axes=(0, None, None, None, None),
      static_broadcasted_argnums=[1, 4],
  )
  (
      stacked_naps,
      stacked_naps_fibers,
      stacked_states,
      stacked_bm,
      stacked_ohc,
      stacked_agc,
  ) = pmapped(input_waves_array, hypers, weights, state, open_loop)

  output_states = _tree_unstack(stacked_states)
  output = []
  for i, output_state in enumerate(output_states):
    tup = (
        stacked_naps[i],
        stacked_naps_fibers[i],
        output_state,
        stacked_bm[i],
        stacked_ohc[i],
        stacked_agc[i],
    )
    output.append(tup)
  return output


## output_states is the unstacked list of carfac_states and can be directly applied to run_multiple_segments_states_shmap


def run_scan_for_grad(
    input_wave: jnp.array,
    hypers: carfac_jax.CarfacHypers,
    weights: carfac_jax.CarfacWeights,
    state: carfac_jax.CarfacState,
    scan_grad_hypers: ScanGradHypers,
):
    """This function chunks single audio stimulus and runs this through CARFAC using lax.scan.

    Args:
      input_wave: Single audio stimulus.
      hypers: Shared hypers for use in input.
      weights: Shared weights to use in input.
      state: Shared state to use as input.
      scan_grad_hypers: This includes the chunk number for using as static_argnums in jitted function.

    Returns:
      nap_output: NAP output reconstructed as if complete audio stimulus presented
      final_recon_input_wave: This is the input_wave reconstructed from stimulus_stacked

    """

    def __run_carfac_scan(carry, k):
      nap, weights, state, input = carry
      nap, _, state, _, _, _ = carfac_jax.run_segment(
          input[:, k], hypers, weights, state, open_loop=False
      )
      return (nap, weights, state, input), nap

    chunk_length = len(input_wave) // scan_grad_hypers.chunk_num
    length_to_test = chunk_length * scan_grad_hypers.chunk_num
    stimnum_to_test = jnp.arange(scan_grad_hypers.chunk_num)
    nap = jnp.zeros([chunk_length, hypers.ears[0].n_ch, 1])

    if (scan_grad_hypers.exactlen) and (length_to_test != len(input_wave)):
        raise Exception("Chunk_length * Chunk_number does not equal stimulus length")
    ### generate stimulus_stack with jnp.split
    stimulus_stacked = input_wave[:length_to_test].reshape(
        scan_grad_hypers.chunk_num, chunk_length
    )
    ### run CARFAC for stimulus stack with lax.scan
    (_, _, _, _), nap_final = jax.lax.scan(
        __run_carfac_scan, (nap, weights, state, stimulus_stacked.T), stimnum_to_test
    )
    ### reconstruct stacked NAP output as one long output
    nap_output = jnp.reshape(
        nap_final,
        [jnp.shape(nap_final)[0] * jnp.shape(nap_final)[1], hypers.ears[0].n_ch, 1],
    )
    final_recon_input = jnp.hstack(stimulus_stacked)
    return nap_output, final_recon_input