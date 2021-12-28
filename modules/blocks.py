from tensorflow.keras import layers
import tensorflow as tf


class Blocks:
    def __init__(self):
        pass

    def residual_block(n_filters, name, inputs, stride=(1,1)):
        x = layers.Conv2D(filters=n_filters, kernel_size=(3,3), strides=stride, activation="relu", name='Conv1ResidualBlock_' + name)(inputs)
        x = layers.BatchNormalization(scale=True)(x)

        x = layers.Conv2D(filters=n_filters, kernel_size=(3,3), strides=stride, name='Conv2ResidualBlock_' + name)(x)
        x = layers.BatchNormalization(scale=True)(x)

        # Crop the input to match sizes and add activations.
        inputs = tf.keras.layers.Cropping2D(cropping=((2,2),(2,2)), data_format=None)(inputs)
        x = layers.Add()([x, inputs])
        return x