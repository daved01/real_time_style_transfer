import yaml


def test_config_available():
    with open("./configuration.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
         
        assert(config["dataset_name"] != None)
        assert(config["model_architecture"] != None)

        # Path format: No slash at end!
        assert(config["content_image_path"] != None and config["content_image_path"][-1] != "/")
        assert(config["style_image_path"] != None and config["style_image_path"][-1] != "/")
        assert(config["model_weights_path"] != None and config["model_weights_path"][-1] != "/")
        assert(config["generated_image_path"] != None and config["generated_image_path"][-1] != "/")


def test_create_folder_for_run():
    pass