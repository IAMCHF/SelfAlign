import numpy as np
import tensorflow as tf
from keras.backend import l2_normalize
from keras.initializers.initializers import GlorotUniform
from keras.layers import Lambda, Activation
from tensorflow.keras.layers import (
    Input, Conv3D, Dense, Concatenate, BatchNormalization, LeakyReLU, Flatten
)
from tensorflow.keras.models import Model
from SelfAlign.models.gumnet.DCTPooling3D import DCTPooling3D
from SelfAlign.models.gumnet.FeatureCorrelation import FeatureCorrelation
from SelfAlign.models.gumnet.FeatureL2Norm import FeatureL2Norm


def get_initial_weights(output_size):
    b = np.random.random((6, )) - 0.5
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights


def get_model(input_shape):
    if input_shape is None:
        input_shape = [None, 32, 32, 32, 1]
    source_volume_input = Input(shape=input_shape, name="source_volume_input")
    template_volume_input = Input(shape=input_shape, name="template_volume_input")
    channel_axis = -1
    # feature extractors share the same weights
    shared_conv1 = Conv3D(32, (3, 3, 3), padding='valid')
    shared_conv2 = Conv3D(64, (3, 3, 3), padding='valid')
    shared_conv3 = Conv3D(128, (3, 3, 3), padding='valid')
    shared_conv4 = Conv3D(256, (3, 3, 3), padding='valid')
    shared_conv5 = Conv3D(512, (3, 3, 3), padding='valid')

    v_a = shared_conv1(source_volume_input)
    v_a = Activation('relu')(v_a)
    v_a = BatchNormalization(axis=channel_axis)(v_a)
    v_a = DCTPooling3D((26, 26, 26), (22, 22, 22))(v_a)

    v_a = shared_conv2(v_a)
    v_a = Activation('relu')(v_a)
    v_a = BatchNormalization(axis=channel_axis)(v_a)
    v_a = DCTPooling3D((18, 18, 18), (15, 15, 15))(v_a)

    v_a = shared_conv3(v_a)
    v_a = Activation('relu')(v_a)
    v_a = BatchNormalization(axis=channel_axis)(v_a)
    v_a = DCTPooling3D((12, 12, 12), (10, 10, 10))(v_a)

    v_a = shared_conv4(v_a)
    v_a = Activation('relu')(v_a)
    v_a = BatchNormalization(axis=channel_axis)(v_a)
    v_a = DCTPooling3D((8, 8, 8), (7, 7, 7))(v_a)

    v_a = shared_conv5(v_a)
    v_a = BatchNormalization(axis=channel_axis)(v_a)
    v_a = FeatureL2Norm()(v_a)


    v_b = shared_conv1(template_volume_input)
    v_b = Activation('relu')(v_b)
    v_b = BatchNormalization(axis=channel_axis)(v_b)
    v_b = DCTPooling3D((26, 26, 26), (22, 22, 22))(v_b)

    v_b = shared_conv2(v_b)
    v_b = Activation('relu')(v_b)
    v_b = BatchNormalization(axis=channel_axis)(v_b)
    v_b = DCTPooling3D((18, 18, 18), (15, 15, 15))(v_b)

    v_b = shared_conv3(v_b)
    v_b = Activation('relu')(v_b)
    v_b = BatchNormalization(axis=channel_axis)(v_b)
    v_b = DCTPooling3D((12, 12, 12), (10, 10, 10))(v_b)

    v_b = shared_conv4(v_b)
    v_b = Activation('relu')(v_b)
    v_b = BatchNormalization(axis=channel_axis)(v_b)
    v_b = DCTPooling3D((8, 8, 8), (7, 7, 7))(v_b)

    v_b = shared_conv5(v_b)
    v_b = BatchNormalization(axis=channel_axis)(v_b)
    v_b = FeatureL2Norm()(v_b)

    # correlation layer
    c_ab = FeatureCorrelation()([v_a, v_b])
    c_ab = FeatureL2Norm()(c_ab)

    # correlation layer
    c_ba = FeatureCorrelation()([v_b, v_a])
    c_ba = FeatureL2Norm()(c_ba)

    c_ab = Conv3D(1024, (3, 3, 3))(c_ab)
    c_ab = BatchNormalization(axis=channel_axis)(c_ab)
    c_ab = Activation('relu')(c_ab)

    c_ab = Conv3D(1024, (3, 3, 3))(c_ab)
    c_ab = BatchNormalization(axis=channel_axis)(c_ab)
    c_ab = Activation('relu')(c_ab)

    c_ab = Flatten()(c_ab)

    c_ba = FeatureL2Norm()(c_ba)
    c_ba = Conv3D(1024, (3, 3, 3))(c_ba)
    c_ba = BatchNormalization(axis=channel_axis)(c_ba)
    c_ba = Activation('relu')(c_ba)

    c_ba = Conv3D(1024, (3, 3, 3))(c_ba)
    c_ba = BatchNormalization(axis=channel_axis)(c_ba)
    c_ba = Activation('relu')(c_ba)

    c_ba = Flatten()(c_ba)

    c = Concatenate()([c_ab, c_ba])
    c = tf.debugging.check_numerics(c, 'c has invalid values!')

    print("c_ba Shape:", c_ba.shape)
    print("c Shape:", c.shape)

    c = Dense(2000)(c)
    c = Dense(2000)(c)
    weights = get_initial_weights(2000)

    # estimated 3D rigid body transformation parameters
    c = Dense(6, weights=weights)(c)
    c = Activation('sigmoid')(c)

    pcrnet = Model(
        inputs={
            "source_volume_input": source_volume_input,
            "template_volume_input": template_volume_input
        },
        outputs={
            "combined_params": c
        }
    )
    return pcrnet


if __name__ == "__main__":
    print(" ")
