import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from metrics.metrics import SemanticsMeter
from utils.colormaps import SCANNET_COLORS

base_path = '/home/antonazzi/myfiles/scannet_signorelli/'
voxel_size = [5]
methods = ['C']
for scene_number in range(8, 9):
    for voxel in voxel_size:
        for method in methods:
            path = os.path.join(base_path, f'scene000{scene_number}_00')
            colors = SCANNET_COLORS
            colors_dict = {v: l for v, l in enumerate(colors)}

            if not os.path.exists(os.path.join(path, f'visualization_sam{method}b{voxel}')):
                os.mkdir(os.path.join(path, f'visualization_sam{method}b{voxel}'))

            images = sorted(os.listdir(os.path.join(path, f'pseudo{voxel}')), key=lambda x: int(os.path.basename(x)[:-4]))
            for image in tqdm(images[:]):
                label_sam = cv2.imread(os.path.join(path, f'sam{method}b{voxel}', image), cv2.IMREAD_UNCHANGED)
                label_sam_rgb = np.zeros((*label_sam.shape, 3)).astype(np.uint8)

                label_kimera = cv2.imread(os.path.join(path, f'pseudo{voxel}', image), cv2.IMREAD_UNCHANGED)
                label_kimera_rgb = np.zeros((*label_kimera.shape, 3)).astype(np.uint8)

                label_deeplab = cv2.imread(os.path.join(path, 'deeplab', image), cv2.IMREAD_UNCHANGED)
                label_deeplab_rgb = np.zeros((*label_deeplab.shape, 3)).astype(np.uint8)

                gt = cv2.imread(os.path.join(path, 'gt', image), cv2.IMREAD_UNCHANGED)
                gt_rgb = np.zeros((*gt.shape, 3)).astype(np.uint8)

                rgb = cv2.imread(os.path.join(path, 'rgb', image.replace('png', 'jpg')))
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                for i, color in enumerate(colors):
                    gt_rgb[gt == i, :] = color[::-1]
                    label_deeplab_rgb[label_deeplab == i, :] = color[::-1]
                    label_kimera_rgb[label_kimera == i, :] = color[::-1]
                    label_sam_rgb[label_sam == i, :] = color[::-1]

                line1 = np.hstack([rgb, gt_rgb, np.zeros((*label_deeplab.shape, 3)).astype(np.uint8)])
                line2 = np.hstack([label_deeplab_rgb, label_kimera_rgb, label_sam_rgb])
                cv2.imwrite(os.path.join(path, f'visualization_sam{method}b{voxel}', image), np.vstack([line1, line2]))
                #cv2.waitKey()


