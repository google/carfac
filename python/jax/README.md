# Plain CARFAC-JAX

This folder implements CARFAC in plain JAX, i.e., making the dependencies as
little as possible.

## Overview

We group all the parameters/coefficients into 4 categories: `DesignParameters`,
`Hypers`, `Weights` and `State`. Each step and the whole CARFAC have their
versions of these 4 classes (e.g. `IhcDesignParameter`, `IhcHypers`,
`IhcWeights` and `IhcState` for inner hair cell and `CarfacDesignParameters`,
`CarfacHypers`, `CarfacWeights` and `CarfacState` for the whole CARFAC). The
meaning of the 4 classes are,

- DesignParameters: set by users and used in designing CARFAC. These numbers are
not used in any "Model Functions" (see definition below).
- Hypers: analogous to "hyperparameters" in ML. They are computed based on
DesignParameters and are not changed in inference. They can determine the
computational graph and are not changed in training/backprop either.
- Weights: the trainable weights of CARFAC.
- State: the state of the model. Every step of CARFAC needs an input state and
will output a new state.

We also classify the functions into 2 categories:

- Design and Init: used to design `Hypers`, init `Weights` and `State` based on
`DesignParameters`.
- Model Functions: carry out the computation of CARFAC.

The relation between the data classes and the model functions can be classified
as follows,

| Data Type        | In Design&Init Functions | In Model Functions               |
|------------------+--------------------------+----------------------------------|
| DesignParameters | Input, unchanged         | Not related                      |
| Hypers           | Output (designed)        | Input, static                    |
| Weights          | Output (initialised)     | Input, unchanged                 |
| State            | Output (initialised)     | Input and output, can be donated |


## Example Usage

Assume we want to fit the output NAP to a constant 1 for all the channels. The
following example shows how to obtain the gradient of `CarfacWeights`,

```python
import carfac as carfac_jax

## First, define a loss and optionally make it JITted.
@functools.partial(jax.jit, static_argnames=('hypers',))
def loss(weights, input_waves, hypers, state):
  # A loss function for tests.
  naps_jax, state_jax, _, _, _ = carfac_jax.run_segment(
      input_waves, hypers, weights, state, open_loop=False
  )
  # For testing, just fit `naps` to 1.
  return (
      jnp.sum((naps_jax - 1.0) ** 2)
      / jnp.prod(jnp.asarray(naps_jax.shape)),
      state_jax
  )

## Second, generate some random input audio signal.
n_samp = 200
n_ears = 1
run_seg_input = jax.random.normal(jax.random.PRNGKey(1), (n_samp, n_ears))

## Third, design and init CARFAC.
gfunc = jax.grad(loss, has_aux=True)
params_jax = carfac_jax.CarfacDesignParameters()
params_jax.ears[0].car.linear_car = False
hypers_jax, weights_jax, state_jax = carfac_jax.design_and_init_carfac(
    params_jax
)

## Then, computes gradients by `jax.grad`.
grad_jax, new_state = gfunc(weights_jax, run_seg_input, hypers_jax, state_jax)

## Now one can update `Weights` based on `grad_jax`.
# One can use a library like `Optax`. Here, for illustration, we can just do the
# gradient descent (stepsize=0.1) by,
new_weights = jax.tree_map(lambda x,y: x-0.1*y, weights_jax, grad_jax)
# Please note that: we currently put as many coefficients as possible into
# `Weights` but normally we don't need to train all of them (and some of the
# weights are much more sensitive than the others). One can selectively update
# part of the weights easily.
```
There is also a helper function `carfac_jax.run_segment_jit` which is the JITted
version of `carfac_jax.run_segment`.

## Relation with the numpy version

Compared with the numpy version, necessary differences have been made,

1. Instead of putting everything into 1 class like `CarfacParams`, we divide
them into 4 data classes. The reason is mainly that we want some of them to be
trainable, some of them to be static and some of them to be donatable.

2. JAX requires the JITted functions to be pure and not all python code can be
directly run in these functions (e.g. conditionals on non-static variables). So
many changes are made to get around this, including using `jax.lax.*` and
tuning the algorithms (e.g. in the AGC step).

Otherwise, we want to keep the 2 versions as similar as possible.
