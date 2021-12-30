from threading import Barrier
import yaml
import os


def get_configuration_paths():
    """
    Gets all paths from the configuration.yaml.


    """
     # Load selected image from folder, prepare image format.
    with open("./configuration.yaml") as file:
        config = yaml.load(file, yaml.FullLoader)
    content_image_path = config["content_image_path"]
    generated_image_path = config["generated_image_path"]
    model_weights_path = config["model_weights_path"]
    
    return content_image_path, generated_image_path, model_weights_path



def create_folder_for_run(model_weights_path, model_architecture, dataset_name, style_image_name, batch_size):
    """
    Checks if the folder for the weights and logging exists. If not creates it.

    Name format:
    <model_name>_<dataset_name>_<style_image>_batchsize<batch_size>

    Returns the full path to the model.
    """
    # Generate folder name.
    batch_size = "".join(["batchsize", str(batch_size)])
    filename = "_".join([model_architecture, dataset_name, style_image_name, batch_size])

    # Create folder if it does not exist.
    try:
        os.mkdir(model_weights_path + "/" + filename)
    except FileExistsError:
        pass
    path = "".join([model_weights_path, "/", filename])
    return path
