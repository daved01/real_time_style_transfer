import tensorflow as tf

import modules.loss_functions as loss_functions


def test_gram_matrix():
    # Arrange
    x1 = tf.ones((1,10,10,1))
    x2 = tf.ones((1,8,8,3))
    x3 = tf.ones((9,11,11,3))
    x4 = tf.ones((8,10,20,3))
    x5 = tf.ones((8,20,10,3))

    # Test
    y1 = loss_functions.gram_matrix(x1)
    y2 = loss_functions.gram_matrix(x2)
    y3 = loss_functions.gram_matrix(x3)
    y4 = loss_functions.gram_matrix(x4)
    y5 = loss_functions.gram_matrix(x5)

    # Assert
    assert(x1.shape[3] == y1.shape[1] and x1.shape[3] == y1.shape[2])
    assert(x2.shape[3] == y2.shape[1] and x2.shape[3] == y2.shape[2])
    assert(x3.shape[3] == y3.shape[1] and x3.shape[3] == y3.shape[2])
    assert(x4.shape[3] == y4.shape[1] and x4.shape[3] == y4.shape[2])
    assert(x5.shape[3] == y5.shape[1] and x5.shape[3] == y5.shape[2])



def test_compute_content_loss():
    # Arrange
    generated = tf.ones((1,4,4)) * 2
    content = tf.ones((1,4,4))
    dimensions = [8,8,8]

    height, width, channels = dimensions[0], dimensions[1], dimensions[2]
    scaling_factor = (int(height/4) * int(width/4) * channels) # H, W, C

    # Sum over all elements, including the batch_size to get average loss over the batch.
    content_reconstruction_loss = tf.math.reduce_sum(tf.square(generated - content)) / (scaling_factor * generated.shape[0])
    assert(content_reconstruction_loss == tf.constant([0.5]))




def test_compute_style_loss():
    assert(1 == 2)

def test_compute_loss_and_grads():
    assert(1 == 2)


