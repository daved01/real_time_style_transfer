"""
Generates style transfer images with trained models from an input image.
"""

# Supress warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image as ImagePIL
import argparse

import modules.networks as networks
import modules.utils as utils


def generate_image(content_image_path, generated_image_path, model_weights_path, image_name, model_name):
    """
    Runs inference with selected image and displays result.
    """
    
    # Parse model weights file for name.
    
    # <networkArchitecure>_<dataSet>_<style_image>_batchsize<batchSize>_epochs<numEpochs>.h5
    raw = model_name.split("_")
    model_weights_name = raw[0]

    try:
        image = keras.preprocessing.image.load_img(content_image_path + image_name + ".jpg")
    except FileNotFoundError:
        print("Error! Did not find image. Are your image name and content_image_path in configuration.yaml correct?")
        exit()
    image = keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  
    image = np.array([image]) # Adds batch dimension. Shape is (batch, height, width, colour)
    image = tf.convert_to_tensor(image)


    # Load the selected model.
    supported_network_architectures = networks.get_supported_architecture_names()

    if model_weights_name == "transformNetwork":
        network = networks.get_transformation_model()
    elif model_weights_name == "miniNetwork":
        network = networks.get_mini_model()
    else:
        print("Error! Model architecture name is invalid!\nValid architectures are:\n")
        [print(name) for name in supported_network_architectures]
        exit()


    try:
        network.load_weights(model_weights_path + model_name)
    except BaseException:
        print("Error! Did not find model. Are your model name and the model_weights_path in configuration.yaml correct?")
        exit()

    # Run forward pass
    generated_image = network(image, training=False)
    generated_image = generated_image.numpy()
    generated_image = ((generated_image * 0.5) + 0.5) 
    generated_image = generated_image * 255
    generated_image = generated_image.reshape((256,256,3)) # Remove batch dimension

    # Postprocess image and save.
    # <networkArchitecure>_<dataSet>_<style_image>_batchsize<batchSize>_epochs<numEpochs>.h5
    img = ImagePIL.fromarray(np.uint8(generated_image)).convert('RGB')
    style_image_name, epochs, batch_size = raw[2], raw[4].split(".")[0], raw[3]
    filename = image_name + "_" + style_image_name + "_" + epochs + "_" + batch_size
    img.save(generated_image_path + filename + ".png")
    


def show_image():
    pass



if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert an input image into a selected style.")
    parser.add_argument('--model', default=None, help="Provide the name of a model in the models folder or flag 'all' to iterate over all models.")
    parser.add_argument('--image', default=None, help="Provide the name of a content image in the folder data/content.")
    parser.add_argument('-showimage', default=False, help="Shows the generated image in a window if set.") # TODO: implement

    args = parser.parse_args()
    model_name = args.model
    image_name = args.image
    show_image_flag = args.showimage

    # Check inputs


    # Run model and save generated image to folder generated
    print("\n==============================\nTransforming the image...\n")
    content_image_path, generated_image_path, model_weights_path = utils.get_configuration_paths()
    if model_name != "all":
        generate_image(content_image_path, generated_image_path, model_weights_path, image_name, model_name)
    else:
        # Get all model names in folder and iterate over them
        model_names = [file for file in os.listdir(model_weights_path) if file.endswith(".h5")]
        print("Found {:.0f} models!".format(len(model_names)))
        for model_name in model_names:
            #print(model_name)
            generate_image(content_image_path, generated_image_path, model_weights_path, image_name, model_name)
    print("\n==============================\nDone!")