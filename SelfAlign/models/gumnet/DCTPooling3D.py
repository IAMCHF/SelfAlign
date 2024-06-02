import tensorflow as tf


class DCTPooling3D(tf.keras.layers.Layer):
    """
    Performs spectral pooling and filtering using DCT as a keras layer
    """

    def __init__(self, output_size, truncation, homomorphic=False, **kwargs):
        self.output_size = output_size
        self.truncation = truncation
        self.homomorphic = homomorphic
        super(DCTPooling3D, self).__init__(**kwargs)

    def build(self, input_shape):
        # No variables to build; this layer only performs operations on the inputs.
        pass

    def compute_output_shape(self, input_shape):
        length, height, width = self.output_size
        channels = input_shape[-1]
        return tf.TensorShape([None, length, height, width, channels])

    def call(self, inputs):
        if self.homomorphic:
            inputs = tf.math.log(inputs)

        x_dct = self._dct3D(inputs)
        x_crop = self._cropping3D(x_dct)
        x_idct = self._idct3D(x_crop)

        if self.homomorphic:
            x_idct = tf.exp(x_idct)

        return x_idct

    def _dct3D(self, x):
        x_perm = tf.transpose(x, perm=[0, 4, 1, 2, 3])
        output = tf.transpose(tf.signal.dct(tf.transpose(tf.signal.dct(
            tf.transpose(tf.signal.dct(x_perm, 2, norm='ortho'), perm=[0, 1, 2, 4, 3]),
            2, norm='ortho'), perm=[0, 1, 3, 4, 2]), 2, norm='ortho'), perm=[0, 4, 3, 2, 1])

        return output

    def _idct3D(self, x):
        x_perm = tf.transpose(x, perm=[0, 4, 3, 2, 1])
        output = tf.transpose(tf.signal.idct(tf.transpose(tf.signal.idct(
            tf.transpose(tf.signal.idct(x_perm, 2, norm='ortho'), perm=[0, 1, 4, 2, 3]),
            2, norm='ortho'), perm=[0, 1, 2, 4, 3]), 2, norm='ortho'), perm=[0, 2, 3, 4, 1])

        return output

    def _cropping3D(self, x):
        x_trunc = x[:, :self.truncation[0], :self.truncation[1], :self.truncation[2], :]
        paddings = tf.constant([[0, 0],
                    [0, self.output_size[0] - self.truncation[0]],
                    [0, self.output_size[1] - self.truncation[1]],
                    [0, self.output_size[2] - self.truncation[2]],
                    [0, 0]])

        output = tf.pad(x_trunc, paddings, "CONSTANT")

        return output
