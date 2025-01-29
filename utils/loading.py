import yaml


def load_yaml(path):
    with open(path) as file:
        res = yaml.load(file, Loader=yaml.FullLoader)
    return res