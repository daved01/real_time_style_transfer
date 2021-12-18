import tensorflow as tf

from modules.reflection_padding import ReflectionPadding2D


def test_dimensions_reflection_padding2D():
    # Input
    # Shape: (batch, height, width, channels)
    x = tf.ones(shape=(1,256,256,3))
    
    padding = ReflectionPadding2D()
    y = padding(x)

    # Assert
    assert(y.shape[0] == 1)
    assert(y.shape[1] == 336)
    assert(y.shape[2] == 336)
    assert(y.shape[3] == 3)
