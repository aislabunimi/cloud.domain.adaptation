import os

import cv2
import numpy as np
from tqdm import tqdm

from metrics.metrics import SemanticsMeter
from utils.colormaps import SCANNET_COLORS
from utils.paths import DATASET_PATH

scenes = ['scene0002_00']

for scene in scenes:
    semantic_instances_path = os.path.join(DATASET_PATH, 'scans', scene, 'instance_filt_scaled_segmented')
    metric = SemanticsMeter(number_classes=40)
    for semantic_instance_path in tqdm(os.listdir(semantic_instances_path), desc='Evaluating metric'):
        semantic_instance_image = cv2.imread(os.path.join(semantic_instances_path, semantic_instance_path))[:, :, ::-1]
        for i, color in enumerate(SCANNET_COLORS):

            semantic_instance_image[np.all(semantic_instance_image == color, axis=-1)] = [i, i, i]

        semantic_instance_image = semantic_instance_image[:, :, 0]
        gt = cv2.imread(os.path.join(DATASET_PATH, 'scans', scene, 'label_40_scaled', semantic_instance_path),
                        cv2.IMREAD_UNCHANGED)
        metric.update(semantic_instance_image, gt)
    print(metric.measure())
    print('ciao')





