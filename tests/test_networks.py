import tensorflow as tf

import modules.networks as networks


def test_load_architecture():
    # Test the supported names 
    supported_models = ["transformNet","transformNetConvs","mediumNet","tinyNet"]

    # Load a false model
    model_to_check = networks.load_architecture("dummyModelThatDoesNotExistYet")
    assert(model_to_check == -1)


def test_load_architecture():
    # Test the supported names 
    supported_models = ["transformNet","transformNetConvs","mediumNet","tinyNet"]

    for model_name in supported_models:
        model = networks.load_architecture(model_name)
        assert(model != None)

    model = networks.load_architecture("NonExistingNameHopefully")
    assert(model == -1)


## TRANSFORMATION NETWORK DECONVS
def test_get_transform_net():
    # Prepare dummy input
    x1 = tf.ones((1,256,256,3))
    x2 = tf.ones((4,256,256,3))
    
    # Get model
    model = networks.get_transform_net()
    
    # Tests
    y1 = model(x1)
    assert(y1.shape == (1,256,256,3))

    y2 = model(x2)
    assert(y2.shape == (4,256,256,3))



## TRANSFORMATION NETWORK CONVS
def test_get_transform_net_convs():
    # Prepare dummy input
    x1 = tf.ones((1,256,256,3))
    x2 = tf.ones((4,256,256,3))
    
    # Get model
    model = networks.get_transform_net_convs()
    
    # Tests
    y1 = model(x1)
    assert(y1.shape == (1,256,256,3))

    y2 = model(x2)
    assert(y2.shape == (4,256,256,3))



## MEDIUM NETWORK
def test_get_medium_net():
    # Prepare dummy input
    x1 = tf.ones((1,256,256,3))
    x2 = tf.ones((4,256,256,3))
    
    # Get model
    model = networks.get_medium_net()
    
    # Tests
    y1 = model(x1)
    assert(y1.shape == (1,256,256,3))

    y2 = model(x2)
    assert(y2.shape == (4,256,256,3))



## TINY NETWORK
def test_get_tiny_net():
     # Prepare dummy input
    x1 = tf.ones((1,256,256,3))
    x2 = tf.ones((4,256,256,3))
    
    # Get model
    model = networks.get_tiny_net()
    
    # Tests
    y1 = model(x1)
    assert(y1.shape == (1,256,256,3))

    y2 = model(x2)
    assert(y2.shape == (4,256,256,3))
