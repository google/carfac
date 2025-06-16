from absl.testing import absltest
import jax
import jax.flatten_util
import jax.numpy as jnp

from carfac.jax import utils as jax_utils


class CarfacJaxUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # The default tolerance used in `assertAlmostEqual` etc.
    self.default_delta = 1e-6

  def _is_similar_pytrees(self, pytree1, pytree2, delta=None):
    delta = delta or self.default_delta

    def help_checker(item1, item2):
      if isinstance(item1, jax.Array):
        self.assertIsInstance(item2, jax.Array)
        self.assertLess(abs((item1 - item2).max()), delta)
      else:
        self.assertIsInstance(item1, float)
        self.assertIsInstance(item2, float)
        self.assertAlmostEqual(item1, item2, delta=delta)

    jax.tree_util.tree_map(help_checker, pytree1, pytree2)

  def test_iter_perturbed(self):
    """Tests whether `iter_perturbed` works correctly.

    This functions tests a few things:
    1. Whether each element of the pytree gets perturbed. This is done by
      1.1 storing the element indices of `jax.Array` members into a `set` and
      count the set's size.
      1.2 count the number of `float` members in the pytree.
    Then the sum of the two numbers should equal to the totally number of
    elements.
    2. Whether the "recovered" pytree by "unperturbing" is equal to the original
    pytree.
    """
    input_pytree = {
        'Node1': jnp.array(1.0),
        'Node2': jnp.array([1.0, 2.0, 20.0, 10.0]),
        'Node3': 100.0,
        'Node4': [
            jnp.array(3.0),
            -2.0,
            jnp.array([[11.0, 12.0], [13.0, 14.0]]),
        ],
        'Node5': {
            'SubNode1': [jnp.array(-100.0), -1e3],
            'SubNode2': jnp.array([-2.0, 10.0]),
        },
    }

    perturbs = [-1e-5, 1e-5]
    num_of_elements = jax.flatten_util.ravel_pytree(input_pytree)[0].shape[0]
    elements_perturbed = set()
    num_of_floats = 0  # number of premitive floats
    for (neg_perturbed_pytree, neg_content_idx, neg_element_indices), (
        pos_perturbed_pytree,
        pos_content_idx,
        pos_element_indices,
    ) in jax_utils.iter_perturbed(input_pytree):
      self.assertEqual(neg_content_idx, pos_content_idx)
      self.assertEqual(neg_element_indices, pos_element_indices)

      # Note that this is based on the assumption that the `contents`` returned
      # by `jax.tree_util.tree_flatten` will have the same order for
      # `input_pytree`, `neg_perturbed_pytree` and `pos_perturbed_pytree`. This
      # should be a valid assumption because the only difference among them is
      # in the value of a particular element, that is, the shapes are always
      # exactly the same.
      neg_contents, neg_pydef = jax.tree_util.tree_flatten(neg_perturbed_pytree)
      pos_contents, pos_pydef = jax.tree_util.tree_flatten(pos_perturbed_pytree)

      if neg_element_indices:
        # The content is a `jax.Array`.
        neg_contents[neg_content_idx] = (
            neg_contents[neg_content_idx]
            .at[neg_element_indices]
            .add(-perturbs[0])
        )
        recovered_pytree = jax.tree_util.tree_unflatten(neg_pydef, neg_contents)
        self._is_similar_pytrees(recovered_pytree, input_pytree, delta=1e-12)

        pos_contents[pos_content_idx] = (
            pos_contents[pos_content_idx]
            .at[pos_element_indices]
            .add(-perturbs[1])
        )
        recovered_pytree = jax.tree_util.tree_unflatten(pos_pydef, pos_contents)
        self._is_similar_pytrees(recovered_pytree, input_pytree, delta=1e-12)

        elements_perturbed.add((neg_content_idx, neg_element_indices))
      else:
        # Else, the content should be a `float`.
        num_of_floats += 1

        neg_contents[neg_content_idx] = (
            neg_contents[neg_content_idx] - perturbs[0]
        )
        recovered_pytree = jax.tree_util.tree_unflatten(neg_pydef, neg_contents)
        self._is_similar_pytrees(recovered_pytree, input_pytree, delta=1e-12)

        pos_contents[pos_content_idx] = (
            pos_contents[pos_content_idx] - perturbs[1]
        )
        recovered_pytree = jax.tree_util.tree_unflatten(pos_pydef, pos_contents)
        self._is_similar_pytrees(recovered_pytree, input_pytree, delta=1e-12)

    self.assertEqual(num_of_elements, len(elements_perturbed) + num_of_floats)


if __name__ == '__main__':
  absltest.main()
