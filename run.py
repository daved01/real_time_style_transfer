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
        image = keras.preprocessing.image.load_img("".join([content_image_path, "/" + image_name, ".jpg"]))
    except FileNotFoundError:
        print("Error! Did not find image. Are your image name and content_image_path in configuration.yaml correct?")
        exit()
    image = keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  
    image = np.array([image]) # Adds batch dimension. Shape is (batch, height, width, colour)
    image = tf.convert_to_tensor(image)


    # Load the selected model.
    supported_network_architectures = networks.get_supported_architecture_names()
    
    if model_weights_name == "transformNet":
        network = networks.get_transform_net()
    elif model_weights_name == "transformNetConvs":
        network = networks.get_transform_net_convs()
    elif model_weights_name == "mediumNet":
        network = networks.get_medium_net()
    elif model_weights_name == "tinyNet":
        network = networks.get_tiny_net()
    else:
        print("Error! Model architecture name is invalid!\nValid architectures are:\n")   
        [print(name) for name in supported_network_architectures.keys()]
        exit()
    try:
        network.load_weights("".join([model_weights_path, "/", model_name, ".h5"]))
        
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
    filename = "".join([image_name, "_", style_image_name, "_", batch_size, "_", epochs])
    img.save("".join([generated_image_path, "/", filename, ".png"]))
    

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert an input image into a selected style.")
    parser.add_argument('--weights', default=None, help="Provide the name of a weight file in the models folder, without .h5 extension."+
                        "\nFormat: <networkArchitecure>_<dataSet>_<style_image>_batchsize<batchSize>_epochs<numEpochs>")
    parser.add_argument('--allWeights', default=None, help="Provide the name of a model folder which containts model weight files. Iterates through all models.")
    parser.add_argument('--image', default=None, help="Provide the name of a content image in the folder data/content.")
    
    args = parser.parse_args()
    model_name = args.weights
    model_folder_name = args.allWeights
    image_name = args.image
    
    # Check inputs
    if model_name != None and model_folder_name != None:
        print("Error! Provide either --model or --allModels.")
        exit()

    # Run model and save generated image to folder generated.
    print("\n==============================\nTransforming the image...\n")
    content_image_path, generated_image_path, model_weights_path = utils.get_configuration_paths()

    # Update model folder name if single model is given.
    if model_name != None:
        model_folder_name = "_".join(model_name.split("_")[:-1])

    # Update paths with model name for subfolder.
    generated_image_path = "".join([generated_image_path, "/", model_folder_name])
    model_weights_path = "".join([model_weights_path, "/", model_folder_name])
    
    # Create folder for generated images.
    try:
        os.mkdir(generated_image_path)
    except FileExistsError:
        pass

    # Run either single model or multiple models.
    if model_name != None:
        generate_image(content_image_path, generated_image_path, model_weights_path, image_name, model_name)
    else:
        # Get all model names in folder and iterate over them.
        model_names = [file.split(".")[0] for file in os.listdir(model_weights_path) if file.endswith(".h5")]
        print("Found {:.0f} models!".format(len(model_names)))
        for model_name in model_names:
            generate_image(content_image_path, generated_image_path, model_weights_path, image_name, model_name)
    print("\n==============================\nDone!")