import os

import cv2
import numpy as np
from tqdm import tqdm

from utils.colormaps import SCANNET_COLORS
from utils.paths import RESULTS_PATH, DATASET_PATH

scenes = ['scene0009_00']



for scene in scenes:
    instances_labels = {}
    pseudolabels_path = os.path.join(RESULTS_PATH,
                                     f'pretrain_25k_test_{scene}_pseudo_labels',
                                     'visu', 'val_vis')

    instances_path = os.path.join(DATASET_PATH, 'scans', scene, 'instance_filt_scaled')
    semantic_instance_path = os.path.join(DATASET_PATH, 'scans', scene, 'instance_filt_scaled_segmented')
    os.makedirs(semantic_instance_path, exist_ok=True)

    for prediction in tqdm(os.listdir(pseudolabels_path), desc='Aggregating labels'):
        prediction_path = os.path.join(pseudolabels_path, prediction)

        prediction_image = cv2.imread(prediction_path, cv2.COLOR_BGR2RGB)[240:, 320:, ::-1]

        for i, color in enumerate(SCANNET_COLORS):

            prediction_image[np.all(prediction_image == color, axis=-1)] = [i, i, i]

        prediction_image = prediction_image[:, :, 0]

        instance_image =cv2.imread(os.path.join(instances_path, f'{prediction.split("_")[2]}.png'))[:, :, 0]
        for instance in np.unique(instance_image):
            # Skip instance 0, they are holes in the 3D mesh
            if instance == 0:
                continue
            labels, counts = np.unique(prediction_image[instance_image == instance], return_counts=True)
            if instance not in instances_labels.keys():
                instances_labels[instance.item()] = [0 for _ in range(41)]
            for l, c in zip(labels, counts):
                instances_labels[instance][l] += c.item()



    for prediction in tqdm(os.listdir(pseudolabels_path), desc='Saving semantic instance segmented'):
        instance_image =cv2.imread(os.path.join(instances_path, f'{prediction.split("_")[2]}.png'))[:, :, 0]
        prediction_path = os.path.join(pseudolabels_path, prediction)

        prediction_image = cv2.imread(prediction_path, cv2.COLOR_BGR2RGB)[240:, 320:, ::-1]

        for i, color in enumerate(SCANNET_COLORS):
            prediction_image[np.all(prediction_image == color, axis=-1)] = [i, i, i]

        prediction_image = prediction_image[:, :, 0]

        instance_image =cv2.imread(os.path.join(instances_path, f'{prediction.split("_")[2]}.png'))[:, :, 0]

        semantic_instance = np.zeros(instance_image.shape)
        semantic_instance[instance_image == 0] = prediction_image[instance_image == 0]
        for instance in np.unique(instance_image):
            if instance == 0:
                continue

            labels = instances_labels[instance.item()]
            m = labels.index(max(labels))
            semantic_instance[instance_image == instance] = labels.index(max(labels))

        # Save colored image
        semantic_instance_colored = np.zeros(semantic_instance.shape + (3,), dtype=np.uint8)
        for i, color in enumerate(SCANNET_COLORS):
            semantic_instance_colored[semantic_instance == i] = color[::-1]
        cv2.imwrite(os.path.join(semantic_instance_path, f'{prediction.split("_")[2]}.png'), semantic_instance_colored)
print(np.unique(prediction_image))





