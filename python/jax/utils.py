"""Some utility functions that make dev/debug convenient."""
import copy

import jax
import numpy as np


def iter_perturbed(input_pytree, perturbs=(-1e-5, 1e-5)):
  """Iterates over pytrees with 1 value perturbed each from the input pytree.

  Args:
    input_pytree: the input pytree to perturb.
    perturbs: the perturb values. Numerical approximation of gradients usually
      needs both a positive and a negative perturbation. To ensure that the
      positive and negative perturbations follow the same order, we define
      `perturbs` to be a tuple or list so that users can iterate over positive
      and negative perturbed pytrees together.

  Yields:
    A tuple containing,
      - the pytree with 1 value perturbed
      - the index of perturbed variable in the array returned by `tree_flatten`.
      - the index of perturbed variable in the array. If it is no an array,
  """
  contents, pydef = jax.tree_util.tree_flatten(input_pytree)
  contents_for_loop = copy.deepcopy(contents)
  for i, content in enumerate(contents_for_loop):
    if not isinstance(content, float) and not isinstance(content, jax.Array):
      raise ValueError(f'{type(content)} cannot be perturbed.')
    if isinstance(content, float) or not content.shape:
      ret = []
      for ptb in perturbs:
        contents[i] = content + ptb
        ret.append((jax.tree_util.tree_unflatten(pydef, contents), i, None))
        contents[i] = content
      yield tuple(ret)
    else:
      array_iter = np.nditer(content, flags=['multi_index'])
      for _ in array_iter:
        ret = []
        for ptb in perturbs:
          new_array = content.at[array_iter.multi_index].add(ptb)
          contents[i] = new_array
          ret.append((
              jax.tree_util.tree_unflatten(pydef, contents),
              i,
              array_iter.multi_index,
          ))
          contents[i] = content
        yield tuple(ret)
