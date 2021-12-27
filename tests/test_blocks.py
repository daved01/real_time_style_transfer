import tensorflow as tf

import modules.blocks as blocks


def test_blocks():
    # Arrange
    x1 = tf.ones((1,8,8,3))
    x2 = tf.ones((4,8,8,3))
    x3 = tf.ones((16,12,12,1))

    y1 = blocks.Blocks.residual_block(3, "name1", x1)
    y2 = blocks.Blocks.residual_block(3, "name2", x2)
    y3 = blocks.Blocks.residual_block(1, "name3", x3)

    assert(y1.shape == (1,4,4,3))
    assert(y2.shape == (4,4,4,3))
    assert(y3.shape == (16,8,8,1))