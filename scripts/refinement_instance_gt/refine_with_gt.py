import os

from utils.paths import RESULTS_PATH

scenes = ['scene0000_00']

pseudolabels_path = os.path.join(RESULTS_PATH, '')

for scene in scenes:
    predictions = 