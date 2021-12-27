import tensorflow as tf

import modules.networks as networks


## TRANSFORMATION NETWORK DECONVS
def test_get_transformation_model():
    # Prepare dummy input
    x1 = tf.ones((1,256,256,3))
    x2 = tf.ones((4,256,256,3))
    
    # Get model
    model = networks.get_transformation_model()
    
    # Tests
    y1 = model(x1)
    assert(y1.shape == (1,256,256,3))

    y2 = model(x2)
    assert(y2.shape == (4,256,256,3))


## TRANSFORMATION NETWORK CONVS



## MINI NETWORK DECONVS
def test_get_mini_network():
     # Prepare dummy input
    x1 = tf.ones((1,256,256,3))
    x2 = tf.ones((4,256,256,3))
    
    # Get model
    model = networks.get_mini_model()
    
    # Tests
    y1 = model(x1)
    assert(y1.shape == (1,256,256,3))

    y2 = model(x2)
    assert(y2.shape == (4,256,256,3))
