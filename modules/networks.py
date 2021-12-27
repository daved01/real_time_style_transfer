from tensorflow import keras
from tensorflow.keras import layers

from modules.reflection_padding import ReflectionPadding2D
from modules.blocks import Blocks


def get_supported_architecture_names():
    """
    List of supported network architectures.
    """
    return ["transformNetwork", "miniNetwork"]


### Transform network ###
def get_transformation_model():
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

    return keras.Model(inputs=inputs, outputs=outputs, name="Transform_network")


### Mini network ###
def get_mini_model():
    inputs = keras.Input(shape=(256,256,3), name="InputLayer")
    
    x = layers.Conv2D(filters=32, kernel_size=(9,9), strides=(1,1), activation="relu", padding='same', name="Conv1")(inputs)
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same', name="Conv2")(x)
    x = layers.BatchNormalization(scale=True)(x)

    x = layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same', name="Deconv1")(x)
    x = layers.BatchNormalization(scale=True)(x)

    outputs = layers.Conv2DTranspose(filters=3, kernel_size=(9,9), strides=(1,1), activation="tanh", padding='same', name="Deconv2")(x) 
    
    return keras.Model(inputs=inputs, outputs=outputs, name="Mini_network")