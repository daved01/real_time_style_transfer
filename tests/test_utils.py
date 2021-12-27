import yaml


def test_config_available():
    with open("./configuration.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
        assert(config["content_image_path"] != None)
        assert(config["style_image_path"] != None)
        assert(config["model_weights_path"] != None)
        assert(config["generated_image_path"] != None)