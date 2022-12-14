import yaml


def load_from_yaml(yaml_file_path: str) -> dict:
    """
    Load yaml file and return it in dictionary format.
    """
    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_yaml(dict_data: dict, yaml_file_path: str):
    """
    Save the dictionary as a yaml file.
    """
    with open(yaml_file_path, "w") as f:
        yaml.safe_dump(dict_data, f)
