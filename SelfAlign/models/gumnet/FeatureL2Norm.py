import tensorflow as tf


def _featurenorm(feature):
    epsilon = 1e-6
    norm = tf.pow(tf.reduce_sum(tf.pow(feature, 2), 4) + epsilon, 0.5)
    norm = tf.expand_dims(norm, 4)
    norm = tf.tile(norm, [1, 1, 1, 1, tf.shape(feature)[-1]])
    norm = tf.divide(feature, norm)

    return norm


class FeatureL2Norm(tf.keras.layers.Layer):
    """
    Normalizing features using L2 norm
    Modified for TensorFlow 2.x
    References
    ----------
    [1]  Convolutional neural network architecture for geometric matching, Ignacio Rocco, et al.
    """

    def __init__(self, **kwargs):
        super(FeatureL2Norm, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        length, height, width, num_channels = input_shape[1:]
        return tf.TensorShape([None, length, height, width, num_channels])

    def call(self, feature):
        output = _featurenorm(feature=feature)
        return output

    def get_config(self):
        base_config = super(FeatureL2Norm, self).get_config()
        return dict(list(base_config.items()))
