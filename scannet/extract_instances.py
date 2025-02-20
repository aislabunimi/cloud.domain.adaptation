import os.path
import subprocess

import cv2

from utils.paths import DATASET_PATH

scenes = [f'scene{i:04}_00' for i in range(10)]

# Scale dimensions
W = 320
H = 240

for scene in scenes:
    command = f"unzip -o -q {DATASET_PATH}/scans/{scene}/{scene}_2d-instance-filt.zip -d {DATASET_PATH}/scans/{scene}"
    process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE)

    process.wait()

    if process.returncode == 0:
        print(f"Extraction of labels for {scene} completed successfully.")
    else:
        print(f"Extraction of labels for {scene} failed with error {process.returncode}")

    # Scale instances
    os.makedirs(os.path.join(DATASET_PATH, 'scans', scene, 'instance_filt_scaled'), exist_ok=True)
    for image_name in os.listdir(os.path.join(DATASET_PATH, 'scans', scene, 'instance-filt')):
        image_path = os.path.join(DATASET_PATH, 'scans', scene, 'instance-filt', image_name)
        instance_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        instance_image_scaled = cv2.resize(instance_image, (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(DATASET_PATH, 'scans', scene, 'instance_filt_scaled', image_name), instance_image_scaled)
