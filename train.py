"""
Trains a model on the data in the folder data with a selected style image.
"""
# Supress warnings
import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras

from modules.loss_functions import get_loss_network
from modules.loss_functions import compute_loss_and_grads
import modules.networks as networks


def get_dataset(content_image_path, batch_size, image_size=(256,256)):
    # Make tf dataset from content images.
    print("Generating dataset...")
    dataset = keras.preprocessing.image_dataset_from_directory(
        content_image_path,
        validation_split=0.0,
        labels=None,
        seed=1581,
        image_size=image_size,
        batch_size=int(batch_size)
    )
    return dataset


def get_and_scale_image(image_path):
    """
    Loads and scales a .jpg image.
    Returns a tf.tensor with batch dimension, scaled to [0,1].
    """

    image = keras.preprocessing.image.load_img(image_path)
    image = keras.preprocessing.image.img_to_array(image)   
    # Scale
    image = image / 255.0
    # Adds batch dimension. Shape is (batch, height, width, colour)
    image = np.array([image])
    # Convert to tensorflow tensor
    image = tf.convert_to_tensor(image)
    return image


def run_training(dataset_name, model_architecture_name, loss_net_activations, style_image, style_image_name, epochs, epochs_ran, save_epoch_interval, 
                batch_size, model_weights_path, content_layers, style_layers):
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    num_samples = len(dataset) * batch_size
    
    ## Custom training loop from scratch
    for epoch in range(epochs_ran, epochs):
        print("Running epoch %d / %d" %(epoch+1, epochs))
        for step, image in enumerate(dataset):         
            # Scale image to range [0,1]
            image = image / 255.0
            loss = compute_loss_and_grads(image, style_image, transform_network, optimizer, loss_net_activations, batch_size, content_layers, style_layers)    
            # Print current batch-wise loss, 10 times per epoch.     
            if step > 10 and step % int(num_samples//10) == 0 or step <= 10:
                print("Current loss for one batch at step {:.0f}: {:.2f}".format(step, loss))

        if ((epoch+1) % save_epoch_interval == 0):
            transform_network.save_weights(model_weights_path + model_architecture_name+"_" + dataset_name + "_"+style_image_name+"_batchsize"+str(batch_size)+"_epochs"+str(epoch+1)+".h5", save_format='h5')
            print("Saved latest model at epoch", epoch+1)
    
  
    transform_network.save_weights(model_weights_path + model_architecture_name+"_" + dataset_name + "_"+style_image_name+"_batchsize"+str(batch_size)+"_epochs"+str(epoch+1)+".h5", save_format='h5')
    print("Training completed!")


if __name__ == "__main__":
    # train.py --style <style_image_name> --epochs <total_num_epoch> --batchsize <batch_size> --weights <model_name> -logs
    parser = argparse.ArgumentParser(description="Trains a model to transfer a content image into a style.")
    parser.add_argument('--style', default=None, help="Provide name of the style image.")
    parser.add_argument('--epochs', default=20, help="Set total number of epochs to train the model.")
    parser.add_argument('--batchsize', default=1, help="Batch size used for training.")
    parser.add_argument('--weights', default=None, help="Select weights of trained model to continue training.")
    parser.add_argument('--saveepochs', default=2, help="Set after how many epochs a model is saved.")
    parser.add_argument('-logs', default=False, action='store_const', const=True, help="Flag to enable logging of model training stats.")

    args = parser.parse_args()
    style_image_name = args.style
    total_num_epochs = int(args.epochs)
    batch_size = int(args.batchsize)
    save_epoch_interval = int(args.saveepochs)
    model_weights = args.weights
    write_logs = args.logs

    # Checks
    if model_weights == None and style_image_name == None:
        print("Error! No style image name given as argument.")
        exit()

    # Get configuration
    with open("./configuration.yaml") as file:
        config = yaml.load(file, yaml.FullLoader)
    content_image_path = config["content_image_path"]
    dataset_name = config["dataset_name"]
    style_image_path = config["style_image_path"]
    model_weights_path = config["model_weights_path"]
    model_architecture_config = config["model_architecture"]
    generated_image_path = config["generated_image_path"]
    raw = config["image_size"]['tuple']
    image_size = (int(raw[0]), int(raw[1]))
    content_layers = config["content_layers"]
    style_layers = config["style_layers"]

    # Parse model weights if given.
    if model_weights != None:
        raw = model_weights.split("_")
        model_architecture_weights, dataset_name, style_image_name, batch_size, weights_epochs = raw[0], raw[1], raw[2], int(raw[3][9:]), int(raw[4][6:])

        # Checks
        # Model architecture from the weights is different to the one in the config file.
        if model_architecture_weights != model_architecture_config:
            print("Architecture name given in config is different from the one used for the given weights.\n" +
            "Using architecture from the weights.")
            model_architecture = model_architecture_weights

        if weights_epochs >= total_num_epochs:
            print("Total number of epochs is less than the number of epochs already trained!")
            exit()
        print("Found weights. Loading with configuration:")
        print("Style image: " + str(style_image_name))
        print("Batch size: " + str(batch_size))
    
    # If model weights are given, ignore the model from the config and load the architecture from the weight's name.
    print("Loading model " + str(model_architecture) + " ...")

    transform_network = networks.load_architecture(model_architecture)
    
    # If loading failed exit.
    if transform_network == -1:
        exit()

    if model_weights != None:
        try:
             # Model name: <model_name>_<dataset_name>_<style_image>_batchsize<batch_size>_epochs<num_epochs>.h5
            print("Loading weights for model" + str(model_architecture) + " ...")
            transform_network.load_weights(model_weights_path + model_weights + ".h5")
        except OSError:
            print("Error! Could not load file \"" + str(model_weights_path + model_weights + ".h5") + "\". Does it exist?")
            exit()
    
    # Load loss network
    print("Loading loss network...")
    loss_net_activations = get_loss_network()

    # Get data
    dataset = get_dataset(content_image_path, batch_size, image_size=image_size)
    print("Loading style image...")
    try:
        style_image = get_and_scale_image(style_image_path + style_image_name + ".jpg") # tf.tensor, scaled to [0,1]
    except FileNotFoundError:
        print("Error! Could not find style image \"" + str(style_image_name) + "\". Does it exist?")
        exit()

    print("Starting training...")
    run_training(dataset_name, model_architecture, loss_net_activations, style_image, style_image_name, total_num_epochs, weights_epochs, 
                    save_epoch_interval, batch_size, model_weights_path, content_layers, style_layers)
