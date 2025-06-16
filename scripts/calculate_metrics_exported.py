import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from metrics.metrics import SemanticsMeter

path = '/home/antonazzi/myfiles/scennet_signorelli/scene0001_00'

metric_calculator_sam = SemanticsMeter(40)
metric_calculator_deeplab = SemanticsMeter(40)
metric_calculator_kimera = SemanticsMeter(40)

images = sorted(os.listdir(os.path.join(path, 'pseudo5')), key=lambda x: int(os.path.basename(x)[:-4]))[:839]
print(images)
for image in tqdm(images[:int(len(images)*0.8)]):
    label_sam = cv2.imread(os.path.join(path, 'samCs5', image), cv2.IMREAD_UNCHANGED)-1
    label_kimera = cv2.imread(os.path.join(path, 'pseudo5', image), cv2.IMREAD_UNCHANGED)-1
    label_deeplab = cv2.imread(os.path.join(path, 'deeplab', image), cv2.IMREAD_UNCHANGED)-1
    gt = cv2.imread(os.path.join(path, 'gt', image), cv2.IMREAD_UNCHANGED)-1
    #print(np.min(gt), np.min(label))

    metric_calculator_sam.update(torch.tensor(label_sam).unsqueeze(dim=0), torch.tensor(gt).unsqueeze(dim=0))
    metric_calculator_kimera.update(torch.tensor(label_kimera).unsqueeze(dim=0), torch.tensor(gt).unsqueeze(dim=0))
    metric_calculator_deeplab.update(torch.tensor(label_deeplab).unsqueeze(dim=0), torch.tensor(gt).unsqueeze(dim=0))

print('deeplab', metric_calculator_deeplab.measure())
print('kimera', metric_calculator_kimera.measure())
print('SAM', metric_calculator_sam.measure())


