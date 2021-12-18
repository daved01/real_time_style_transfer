import yaml


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