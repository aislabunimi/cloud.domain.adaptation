import os
import shutil
from pathlib import Path
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import yaml
from data_loaders.scannet.pretrain_data_module import DataModule25K
from pytorch_lightning import seed_everything, Trainer

from models.semantic_segmentator import SemanticsLightningNet
from utils.loading import load_yaml
from utils.paths import REPO_ROOT, RESULTS_PATH, DATASET_PATH

parameters = load_yaml(os.path.join(REPO_ROOT, 'configs', 'pretrain_25k_validation.yml'))


seed_everything(0)

experiment_path = os.path.join(RESULTS_PATH, parameters['general']['name'])
if parameters["general"]["clean_up_folder_if_exists"]:
    shutil.rmtree(experiment_path, ignore_errors=True)

Path(experiment_path).mkdir(parents=True, exist_ok=True)

####################################
# Load Model
###################################

model = SemanticsLightningNet(parameters, {'results': 'experiments',
                                           'scannet': DATASET_PATH,
                                           'scannet_frames_25k': 'scannet_frames_25k'})
print(model._env)



# Restore pre-trained model
if parameters['model']['load_checkpoint']:
    checkpoint = torch.load(os.path.join(RESULTS_PATH, parameters["model"]["checkpoint_path"]))
    checkpoint = checkpoint["state_dict"]
    # remove any aux classifier stuff
    removekeys = [
        key for key in checkpoint.keys()
        if key.startswith("_model._model.aux_classifier")
    ]
    print(removekeys)
    for key in removekeys:
        del checkpoint[key]
    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as e:
        print(e)
    model.load_state_dict(checkpoint, strict=False)

####################################
# Load dataset
####################################

datamodule = DataModule25K(parameters["data_module"])

trainer = Trainer(**parameters["trainer"],
                  default_root_dir=experiment_path,
                  strategy=DDPStrategy(find_unused_parameters=False),
)
print(type(model))
print(model._exp)

trainer.test(model, datamodule=datamodule)