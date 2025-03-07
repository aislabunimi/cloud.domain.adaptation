import os

import numpy as np
from utils.colormaps import SCANNET_COLORS
from utils.paths import DATASET_PATH
import cv2

scenes = [f'scene{i:04}_00' for i in range(10)]

os.makedirs(os.path.join(DATASET_PATH, 'scans', scene_id, 'label_40_colored'), exist_ok=True)

for scene_id in scenes:
    base_path = os.path.join(DATASET_PATH, 'scans', scene_id, 'label_40_colored')


    for image_name in os.listdir(os.path.join(DATASET_PATH, 'scans', scene_id, 'label_40')):
        image = cv2.imread(os.path.join(DATASET_PATH, 'scans', scene_id, 'label_40', image_name), cv2.IMREAD_UNCHANGED)
        image_semantic_colored = np.zeros((image.shape[0], image.shape[1], 3))
        for i in range(len(SCANNET_COLORS)):
            image_semantic_colored[image == i, :3] = SCANNET_COLORS[i][::-1]
        
        cv2.imwrite(os.path.join(DATASET_PATH, 'scans', scene_id, 'label_40_colored', image_name), image_semantic_colored)
                            