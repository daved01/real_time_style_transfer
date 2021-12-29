import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from modules.reflection_padding import ReflectionPadding2D
from modules.blocks import Blocks

def get_supported_architecture_names():
    """
    List of supported network architectures.
    """
    models = {
        "transformNet": "1,682,435 parameters, deconvolutions, residual blocks", 
        "transformNetConvs": "Like transformNet with deconvs replaced by convs and bilinear upsampling", 
        "mediumNet": "201,475 parameters, convolutions and bilinear upsampling", 
        "tinyNet": "53,059 parameters, deconvolutions"
    }
    return models


def load_architecture(model):
    models = get_supported_architecture_names()
    if model not in models.keys():
        print("Error! Selected network architecture is not supported. Please check the configuration file.\nSupported are:\n")
        [print(model, ":\t\t", models[model]) for model in models]
        return -1
    # Load architecture
    if model == "transformNet":
        return get_transform_net()
    elif model == "transformNetConvs":
        return get_transform_net_convs()
    elif model == "mediumNet":
        return get_medium_net()
    elif model == "tinyNet":
        return get_tiny_net()



### Transform network ###
def get_transform_net():
    """
    Loads the transformation network.
    """

    inputs = keras.Input(shape=(256,256,3), name="InputLayer")

    # Reflection padding
    padding = ReflectionPadding2D()
    padding._name = "Reflection"
    x = padding(inputs)
   
    x = layers.Conv2D(filters=32, kernel_size=(9,9),strides=(1,1), activation="relu", padding='same', name="Conv1")(x)
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2D(filters=64, kernel_size=(3,3),strides=(2,2), activation="relu", padding='same', name="Conv2")(x) 
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2D(filters=128, kernel_size=(3,3),strides=(2,2), activation="relu", padding='same', name="Conv3")(x)
    x = layers.BatchNormalization(scale=True)(x)

    # Residual blocks
    x = Blocks.residual_block(128,"1", x)
    x = Blocks.residual_block(128,"2", x)
    x = Blocks.residual_block(128,"3", x)
    x = Blocks.residual_block(128,"4", x)
    x = Blocks.residual_block(128,"5", x)

    # Deconvolutions
    x = layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same', name="Deconv1")(x)
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same', name='Deconv2')(x)
    x = layers.BatchNormalization(scale=True)(x)

    outputs = layers.Conv2DTranspose(filters=3, kernel_size=(9,9), strides=(1,1), activation="tanh", padding='same', name="Conv4")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="transformNet")



### Transform network with convolutions ###
def get_transform_net_convs():
    """
    Loads the transformation network.
    """

    inputs = keras.Input(shape=(256,256,3), name="InputLayer")

    # Reflection padding
    padding = ReflectionPadding2D()
    padding._name = "Reflection"
    x = padding(inputs)
   
    x = layers.Conv2D(filters=32, kernel_size=(9,9),strides=(1,1), activation="relu", padding='same', name="Conv1")(x)
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2D(filters=64, kernel_size=(3,3),strides=(2,2), activation="relu", padding='same', name="Conv2")(x) 
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2D(filters=128, kernel_size=(3,3),strides=(2,2), activation="relu", padding='same', name="Conv3")(x)
    x = layers.BatchNormalization(scale=True)(x)

    # Residual blocks
    x = Blocks.residual_block(128,"1", x)
    x = Blocks.residual_block(128,"2", x)
    x = Blocks.residual_block(128,"3", x)
    x = Blocks.residual_block(128,"4", x)
    x = Blocks.residual_block(128,"5", x)

    # Convolutions with resizing and padding.
    paddings = tf.constant([[0,0],[2,2],[2,2],[0,0]])
    x = tf.image.resize(x, size=[253,253], method=tf.image.ResizeMethod.BILINEAR)
    x = tf.pad(x, paddings)
    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='valid', name="Deconv1")(x)
    x = layers.BatchNormalization(scale=True)(x)

    x = tf.image.resize(x, size=[510,510], method=tf.image.ResizeMethod.BILINEAR)
    x = tf.pad(x, paddings)
    x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='valid', name='Deconv2')(x)
    x = layers.BatchNormalization(scale=True)(x)

    outputs = layers.Conv2D(filters=3, kernel_size=(9,9), strides=(1,1), activation="tanh", padding='same', name="Conv4")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="transformNetConvs")



### Medium network ###
def get_medium_net():
    """
    Loads the medium network.
    """
    inputs = keras.Input(shape=(256,256,3), name="InputLayer")
    
    x = layers.Conv2D(filters=32, kernel_size=(9,9), strides=(1,1), activation="relu", padding='same', name="Conv1")(inputs)
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same', name="Conv2")(x)
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same', name="Conv3")(x)
    x = layers.BatchNormalization(scale=True)(x)

    x = tf.image.resize(x, [256,256], method=tf.image.ResizeMethod.BILINEAR)
    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same', name="Deconv1")(x)
    x = layers.BatchNormalization(scale=True)(x)

    x = tf.image.resize(x, [512,512], method=tf.image.ResizeMethod.BILINEAR)
    x = layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same', name="Deconv2")(x)
    x = layers.BatchNormalization(scale=True)(x)

    outputs = layers.Conv2D(filters=3, kernel_size=(9,9), strides=(1,1), activation="tanh", padding='same', name="Deconv3")(x) 
    
    return keras.Model(inputs=inputs, outputs=outputs, name="mediumNet")



### Tiny network ###
def get_tiny_net():
    inputs = keras.Input(shape=(256,256,3), name="InputLayer")
    
    x = layers.Conv2D(filters=32, kernel_size=(9,9), strides=(1,1), activation="relu", padding='same', name="Conv1")(inputs)
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same', name="Conv2")(x)
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same', name="Deconv1")(x)
    x = layers.BatchNormalization(scale=True)(x)

    outputs = layers.Conv2DTranspose(filters=3, kernel_size=(9,9), strides=(1,1), activation="tanh", padding='same', name="Deconv2")(x) 
    
    return keras.Model(inputs=inputs, outputs=outputs, name="tinyNet")