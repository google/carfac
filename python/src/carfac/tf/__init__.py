"""TensorFlow implementation of CARFAC."""

import os

# Ensure that `tf.keras` stays at Keras 2 before importing TensorFlow,
# as Keras 3 does not support multi-dimensional RNN outputs as used in
# `CARFACCell`.
if "TF_USE_LEGACY_KERAS" not in os.environ:
  os.environ["TF_USE_LEGACY_KERAS"] = "1"
