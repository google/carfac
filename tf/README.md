# TF-CARFAC

This is a TensorFlow implementation of parts of CARFAC
(see http://dicklyon.com/hmh/ for more details).

The parts implemented are the CAR and FAC modules (Cascade of Asymmetric
Resonators and Fast Acting Compression), while the IHC, OHC, and AGC modules
(inner/outer hair cell, and adaptive gain control) are not yet present.

All parameters of the model are trainable variables, and a large focus is put
on performance to get reasonable training times.

## carfac.tf.car.CARCell

The carfac.tf.car.CARCell is a TF Keras RNN cell, suitable for wrapping with a
tf.keras.layers.RNN.

Conceptually it implements (at the moment) the top row of figure 15.2 in
"Human and Machine Hearing".

It takes as input a \[batch_size, 1\]-tensor of samples, and outputs a
\[batch_size, channel_size\]-tensor with the CARFAC channel outputs at that
step.

Wrapping it in a tf.keras.layers.RNN layer will allow the caller to provide an
entire sequence of samples, in the shape of a
\[batch_size, sequence_size, 1\]-tensor, and get as output a
\[batch_size, sequence_size, channel_size\]-tensor.
