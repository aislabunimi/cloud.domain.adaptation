import os
from pathlib import Path
import torch
from pytorch_lightning.strategies import DDPStrategy
from data_loaders.scannet.pretrain_data_module import DataModule25K
from pytorch_lightning import seed_everything, Trainer

from data_loaders.scannet.pretrain_data_module_different_images import DataModule25KTrainDifferent
from models.semantic_segmentator import SemanticsLightningNet
from utils.loading import load_yaml, get_wandb_logger
from utils.paths import REPO_ROOT, RESULTS_PATH, DATASET_PATH

parameters = load_yaml(os.path.join(REPO_ROOT, 'configs', 'pretrain_25k_validation.yml'))


seed_everything(123)

experiment_path = os.path.join(RESULTS_PATH, 'pretrain_25k_test_on_test_data_different_images')
if parameters["general"]["clean_up_folder_if_exists"]:
    pass
    #shutil.rmtree(experiment_path, ignore_errors=True)

Path(experiment_path).mkdir(parents=True, exist_ok=True)

####################################
# Load Model
###################################

model = SemanticsLightningNet(parameters, {'results': 'experiments',
                                           'scannet': DATASET_PATH,
                                           'scannet_frames_25k': 'scannet_frames_25k'}, experiment_path=experiment_path)

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

base_path = os.path.join(DATASET_PATH, 'scannet_frames_25k')
scenes = [directory for directory in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, directory))]
scenes.sort()

datamodule = DataModule25KTrainDifferent(parameters["data_module"], scene_list=[s for s in scenes if s < 'scene0458_00'])

trainer = Trainer(**parameters["trainer"],
                  default_root_dir=experiment_path,
                  strategy=DDPStrategy(find_unused_parameters=False),

)
trainer.test(model, datamodule=datamodule)