import os
import shutil
from pathlib import Path

import torch
import yaml
from models.semantic_segmentator import DeepLabV3Lightning
from pytorch_lightning import seed_everything
from utils.loading import load_yaml
from utils.paths import REPO_ROOT, RESULTS_PATH

parameters = load_yaml(os.path.join(REPO_ROOT, 'configs', 'pretrain_25k_validation.yml'))


seed_everything(0)

experiment_path = os.path.join(RESULTS_PATH, parameters['general']['name'])
if parameters["general"]["clean_up_folder_if_exists"]:
    shutil.rmtree(experiment_path, ignore_errors=True)

Path(experiment_path).mkdir(parents=True, exist_ok=True)

model = DeepLabV3Lightning(parameters)

# Restore pre-trained model
if parameters['model']['load_checkpoint']:
    checkpoint = torch.load(os.path.join(RESULTS_PATH, parameters["model"]["checkpoint_path"]), weights_only=True)
    print(checkpoint)
#model.load_state_dict(checkpoint, strict=True)