import tensorflow as tf
from tensorflow.keras.layers import Layer


class FeatureCorrelation(Layer):
    """
    Performs feature correlation as a keras layer
    Modified for TensorFlow 2.x compatibility
    References
    ----------
    [1]  Convolutional neural network architecture for geometric matching, Ignacio Rocco, et al.
    """

    def __init__(self, **kwargs):
        super(FeatureCorrelation, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        length = input_shapes[1][1]
        height = input_shapes[1][2]
        width = input_shapes[1][3]
        num_channels = length * height * width
        return tf.TensorShape([None, length, height, width, num_channels])

    def call(self, inputs):
        f_A, f_B = inputs
        output = self._featurecorrelation(f_A=f_A, f_B=f_B)
        return output

    def get_config(self):
        base_config = super(FeatureCorrelation, self).get_config()
        return dict(list(base_config.items()))

    def _featurecorrelation(self, f_A, f_B):
        # ??????
        b = tf.shape(f_A)[0]
        l = tf.shape(f_A)[1]
        h = tf.shape(f_A)[2]
        w = tf.shape(f_A)[3]
        c = tf.shape(f_A)[4]
        l0_h0_w0 = tf.shape(f_A)[1] * tf.shape(f_A)[2] * tf.shape(f_A)[3]

        # ????????
        output_shape = tf.stack([-1, l, h, w, l0_h0_w0])

        f_A = tf.transpose(f_A, [0, 3, 2, 1, 4])
        f_A = tf.reshape(f_A, [-1, l0_h0_w0, c])

        f_B = tf.reshape(f_B, [-1, l0_h0_w0, c])
        f_B = tf.transpose(f_B, [0, 2, 1])

        f_mul = tf.matmul(f_A, f_B)
        correlation_tensor = tf.reshape(f_mul, output_shape)
        return correlation_tensor
