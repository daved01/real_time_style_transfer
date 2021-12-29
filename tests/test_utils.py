import yaml


def test_config_available():
    with open("./configuration.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
        assert(config["content_image_path"] != None)
        assert(config["dataset_name"] != None)
        assert(config["style_image_path"] != None)
        assert(config["model_weights_path"] != None)
        assert(config["generated_image_path"] != None)
        assert(config["model_architecture"] != None)


def test_create_folder_for_run():
    assert(1 == 2)


def test_save_model_weights():
    assert(1 == 2)



def test_load_model_weights():
    assert(1 == 2)