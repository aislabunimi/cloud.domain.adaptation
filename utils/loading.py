import yaml


def load_yaml(path):
    with open(path) as file:
        res = yaml.load(file, Loader=yaml.FullLoader)
    return res

def sanitize_split_file(split):
    split = dict(split)
    for key, value in split.items():
        split[key] = [v.replace('data/scannet_frames_25k/scannet_frames_25k/', '') for v in value]

    return split