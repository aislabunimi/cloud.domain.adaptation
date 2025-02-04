import os
import re

from utils.paths import DATASET_PATH

subsample_factor = 10

base_path = os.path.join(DATASET_PATH, 'scannet_frames_25k')
scenes = [directory for directory in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, directory))]
scenes.sort()

exclude_scenes = [i for i in range(10)]

for scene in scenes:
    scene_number = int(re.findall(r'\d+', scene)[0])

    if scene_number in exclude_scenes:
        continue

    scene_path = os.path.join(DATASET_PATH, 'scans', scene)
    # Remove useless zip
    if os.path.exists(f'{scene_path}/{scene}.sens'):
        os.remove(f'{scene_path}/{scene}.sens')
    if os.path.exists(f'{scene_path}/{scene}_2d-label.zip'):
        os.remove(f'{scene_path}/{scene}_2d-label.zip')
    if os.path.exists(f'{scene_path}/{scene}_2d-label-filt.zip'):
        os.remove(f'{scene_path}/{scene}_2d-label-filt.zip')
    if os.path.exists(f'{scene_path}/{scene}_2d-instance.zip'):
        os.remove(f'{scene_path}/{scene}_2d-instance.zip')

    # Subsample images
    folder = 'color'
    images = sorted([int(image.split('.')[0]) for image in os.listdir(os.path.join(scene_path, folder))])
    keep_images = images[::subsample_factor]
    for folder in ['color_scaled', 'label_40', 'label_40_scaled', 'label-filt']:
        extension = os.listdir(os.path.join(scene_path, folder))[0].split('.')[1]
        images = sorted([int(image.split('.')[0]) for image in os.listdir(os.path.join(scene_path, folder))])
        keep_images = images[::subsample_factor]
        for image in images:
            if image in keep_images:
                continue
            try:
                os.remove(os.path.join(scene_path, folder, f'{image}.{extension}'))
            except:
                pass





