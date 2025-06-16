# Python package for CARFAC

## Installing

This package comes with two extras: `tf` for Tensorflow dependencies, and `jax`
for JAX dependencies.

This repo can be installed using a [direct reference] to this Git repository.
Replace `jax` with any desired extras, and `master` with the desired commit:

```
carfac[jax] @ git+https://github.com/google/carfac.git@master#subdirectory=python
```

[direct reference]: https://packaging.python.org/en/latest/specifications/version-specifiers/#direct-references

## Development

This package uses [dependency groups] to specify dependencies required for
development.

Either use `uv` and sync with `--all-extras`:

```bash session
$ uv sync --all-extras
$ # run a test:
$ uv run python -m unittest carfac.tf.carfac_test -k "testMatchesMatlabOnBinauralData"
```

or use a virtualenv with pip 25.1+ and install all extras manually:

```bash session
(venv) $ pip install --upgrade pip  # the below requires pip 25.1+
(venv) $ pip install -e ".[tf,jax]"
(venv) $ pip install --group dev
(venv) $ # run a test:
(venv) $ python -m unittest carfac.tf.carfac_test -k "testMatchesMatlabOnBinauralData"
```

As we add the parent `test_data` directory to the package manually, you will
need to re-install the package to see any updates in `carfac.test_data` despite
the package being installed as editable.

[dependency groups]: https://packaging.python.org/en/latest/specifications/dependency-groups/
