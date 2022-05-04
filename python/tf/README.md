# TF-CARFAC

This is a TensorFlow implementation of CARFAC (see http://dicklyon.com/hmh/ for
more details).

It tracks the gradients of the output all the way back to the design parameters
used to design the model, and all parameters are trainable variables to allow
tuning them for the use case at hand.

Note that most DSP algorithms are designed once, based on user-specified
parameters, producing filter coefficients that enable a simple, fast
implementation on the incoming audio. The design process is non-linear,
expensive and is normally only done once, so the resulting DSP signal graph is
as simple as possible. A conventional ML approach might modify the coefficients,
but then you might be left with a design that is not representative of a real
auditory system.

In this implementation, we can backprop back through the entire design process.
By doing this we end up with changes that are more plausible.

A large focus is put on performance to get reasonable training times.

## carfac.python.tf.carfac.CARFACCell

The carfac.python.tf.carfac.CARFACCell is a TF Keras RNN cell, suitable for
wrapping with a tf.keras.layers.RNN.

It takes as input a \[batch_size, num_ears, 1\]-tensor of samples, and outputs a
\[batch_size, num_ears, channel_size, num_outputs\]-tensor with the CARFAC
channel outputs at that step.

Wrapping it in a tf.keras.layers.RNN layer will allow the caller to provide an
entire sequence of samples, in the shape of a
\[batch_size, sequence_size, num_ears, 1\]-tensor, and get as output a
\[batch_size, sequence_size, num_ears, channel_size, num_outputs\]-tensor.

## DISCLAIMER

This is research code, and it uses classes from tf.experimental. As such, it
is not guaranteed to stay compatible with the latest version of TensorFlow.

It also uses meta-programming magic to simplify the ways it follows the
limitations imposed by tf.function and tf.keras.layer.RNN cells. Don't see this
as an example of good TensorFlow code - it's not, it's just an example of
jumping through some very uncomfortable hoops.
